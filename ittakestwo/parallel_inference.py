import os
import sys
# Ensure repo root is in path when script is run directly (e.g. via torch.distributed.run)
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import json
import argparse
import logging
import multiprocessing as mp
import shutil
from pathlib import Path
from typing import Dict, List, Any
import PIL

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image
from omegaconf import OmegaConf
from ittakestwo.preprocess.action_visualization_utils import visualize_action_on_frames
from utils.video_utils import concat_videos, load_video_cv2

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from diffsynth.pipelines.wan_video_ittakestwo import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video
from utils import load_config, instantiate_from_config   
# --------------- logger ---------------
def setup_logger(rank: int):
    logger = logging.getLogger(f"Rank{rank}")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():   # Avoid duplicate handlers
        logger.handlers.clear()
    fmt = logging.Formatter(f"[Rank{rank} %(asctime)s] %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    return logger


# --------------- VideoGenerator ---------------
class VideoGenerator:
    """
    Responsibilities:
    1. Lazy initialization of WanVideoPipeline (ensures it runs on the correct subprocess & cuda:rank)
    2. Provide generate_video(dataset_item) -> List[np.ndarray]
    3. Save generated videos to disk
    """
    def __init__(self, rank: int, args: argparse.Namespace, config: Any):
        self.rank = rank
        self.args = args
        self.config = config
        self.logger = setup_logger(rank)
        self.device = torch.device(f"cuda:{rank}")
        self.pipe: WanVideoPipeline = None

    # ---------- Lazy model initialization ----------
    def initialize_model(self):
        self.logger.info("Initializing WanVideoPipeline ...")
        torch.cuda.set_device(self.device)
        # Patch for Wan2.2 VAE config
        if "vae_config" not in self.config.simulator_config:
            model_id="Wan-AI/Wan2.2-TI2V-5B"
            origin_file_pattern="Wan2.2_VAE.pth"
        else:
            model_id = self.config.simulator_config.vae_config.model_id
            origin_file_pattern = self.config.simulator_config.vae_config.origin_file_pattern
                
        self.pipe = WanVideoPipeline.from_pretrained(
            config=self.config , 
            torch_dtype=torch.bfloat16,
            device=self.device,
            model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                            origin_file_pattern="diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id=model_id,
                            origin_file_pattern=origin_file_pattern),
            ],
        )
        # load pretrained ckpt first 
        pretrained_ckpt_list = [
        "./models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
        "./models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
        "./models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors",
        self.args.model_path,
        ]
        self.pipe.load_from_checkpoint(pretrained_ckpt_list)
        self.logger.info("WanVideoPipeline loaded.")
        self.pipe.env_encoder.to(self.device) if self.pipe.env_encoder is not None else self.pipe.env_encoder
    # ---------- Core generation ----------
    def generate_video(self, example: Dict[str, Any]) -> List[np.ndarray]:
        """
        Input example from IttakestwoVideoActionDataset.__getitem__, fields:
        {
          'video': Tensor,               # [C,T,H,W] or [C,H,W] first frame
          'action': Dict[str,Tensor],    # Continuous/discrete actions
          'prompt': str,                 # Optional, reserved for extension
        }
        Returns List[np.ndarray] compatible with save_video
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not initialized.")
        input_image = example["video"][0] # List[PIL.Image]
        action = example["action"]        # Dict
        # Move action to current device & dtype
        if "left_player_action" in action: 
            action['left_player_action'] = {k: v.to(self.device, dtype=torch.bfloat16 if v.is_floating_point() else torch.long)
                  for k, v in action['left_player_action'].items() if isinstance(v,torch.Tensor)}
        if "right_player_action" in action:
            action['right_player_action'] = {k: v.to(self.device, dtype=torch.bfloat16 if v.is_floating_point() else torch.long)
                  for k, v in action['right_player_action'].items() if isinstance(v,torch.Tensor)}

        # Top-level discrete_action/continuous_action (used by V4/V5 action encoders)
        for k, v in list(action.items()):
            if isinstance(v, torch.Tensor):
                action[k] = v.to(self.device, dtype=torch.bfloat16 if v.is_floating_point() else torch.long)
        
        env_obv = example["env_obv"].to(self.device, dtype=torch.bfloat16)
        dataset_config = self.config.eval_dataset_config.params # type: ignore
        if dataset_config.return_view == "both":
            width = dataset_config.video_params.width
        else: 
            width = dataset_config.video_params.width // 2


        generated = self.pipe(
            input_image=input_image,
            action=action,
            env_obv=env_obv,
            seed=self.args.inference_seed,
            tiled=False,
            height=dataset_config.video_params.height,
            width=width,
            num_frames=dataset_config.video_params.num_frames,
            num_inference_steps=self.args.num_inference_steps,
        )
        
        # Return List[np.ndarray] (T,H,W,C) uint8
        return generated, example['video']

    # ---------- Save ----------
    def save(self, frames: List[np.ndarray], save_path: str):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_params = [
            '-vcodec', 'libx264',       # Video codec
            '-preset', 'medium',        # Balance between compression speed and ratio
            '-crf', '18',               # Quality factor, lower is better (commonly 18-28)
            '-pix_fmt', 'yuv420p',      # Most compatible pixel format
            '-movflags', '+faststart'   # Faster online playback (optional)
        ]
        save_fps = int(60 // self.config.eval_dataset_config.params.video_params.frame_skip)
        save_video(frames, str(save_path), fps=save_fps, quality=10, ffmpeg_params=ffmpeg_params)
        self.logger.info(f"Saved {len(frames)} frames -> {save_path}")

    # ---------- Autoregressive generation ----------
    def generate_video_autoregressive(self, example: Dict[str, Any], num_chunks: int):
        """Generate video autoregressively over multiple chunks.

        Expects example to have return_view='both' (full-width frame that is
        split into left/right halves).
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not initialized.")

        input_image = example["video"][0]  # First frame, PIL Image (full width)
        action = example["action"]
        env_obv = example["env_obv"].to(self.device, dtype=torch.bfloat16)

        # Split full-width input image into left/right views
        if isinstance(input_image, Image.Image):
            w, h = input_image.size
        else:
            # numpy array
            h, w = input_image.shape[:2]
            input_image = Image.fromarray(input_image)
            w, h = input_image.size

        left_input = input_image.crop((0, 0, w // 2, h))
        right_input = input_image.crop((w // 2, 0, w, h))

        # Move action tensors to device
        for k, v in list(action.items()):
            if isinstance(v, torch.Tensor):
                action[k] = v.to(self.device, dtype=torch.bfloat16 if v.is_floating_point() else torch.long)
            elif isinstance(v, dict):
                action[k] = {kk: vv.to(self.device, dtype=torch.bfloat16 if vv.is_floating_point() else torch.long)
                             for kk, vv in v.items() if isinstance(vv, torch.Tensor)}

        dataset_config = self.config.eval_dataset_config.params
        per_view_width = dataset_config.video_params.width // 2
        frames_per_chunk = getattr(self.config, '_ar_frames_per_chunk',
                                   dataset_config.video_params.num_frames)

        left_frames, right_frames = self.pipe.autoregressive_generate(
            left_input_image=left_input,
            right_input_image=right_input,
            action=action,
            env_obv=env_obv,
            num_chunks=num_chunks,
            frames_per_chunk=frames_per_chunk,
            seed=self.args.inference_seed,
            height=dataset_config.video_params.height,
            width=per_view_width,
            num_inference_steps=self.args.num_inference_steps,
        )

        return left_frames, right_frames, example['video']


# --------------- View-concatenation helpers (distributed) ---------------
def _get_video_info(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return num_frames, height, width


def _parse_video_pairs(video_dir: Path):
    """Group videos by base name (suffix after first '-') and pair them up."""
    files = [f for f in video_dir.iterdir() if f.suffix == ".mp4"]
    groups = {}
    for f in files:
        parts = f.name.split("-", 1)
        if len(parts) < 2:
            continue
        base_name = parts[1]
        groups.setdefault(base_name, []).append(f)

    pairs = []
    for base_name, file_list in groups.items():
        if len(file_list) < 2:
            continue
        file_list = sorted(file_list)
        for i in range(0, len(file_list) - 1, 2):
            pairs.append((file_list[i], file_list[i + 1], base_name))
    return pairs


def _distributed_concat_views(output_dir: str, subdir_name: str, rank: int, world_size: int, logger):
    """Concatenate paired view videos horizontally across ranks."""
    video_dir = Path(output_dir) / subdir_name
    if not video_dir.exists():
        return

    pairs = _parse_video_pairs(video_dir)
    if not pairs:
        return

    # Distribute pairs across ranks
    my_pairs = [p for idx, p in enumerate(pairs) if idx % world_size == rank]

    if my_pairs:
        num_frames, height, width = _get_video_info(str(my_pairs[0][0]))
        ffmpeg_params = [
            "-vcodec", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
        temp_dir = Path(output_dir) / f"_concat_{subdir_name}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        for left_path, right_path, base_name in my_pairs:
            left_frames = load_video_cv2(str(left_path), 0, 1, num_frames, height, width)
            right_frames = load_video_cv2(str(right_path), 0, 1, num_frames, height, width)

            left_pil = [Image.fromarray(f) for f in left_frames]
            right_pil = [Image.fromarray(f) for f in right_frames]
            concat_frames = concat_videos(left_pil, right_pil, dim="width")

            output_path = temp_dir / base_name
            save_video(concat_frames, str(output_path), fps=60, quality=10, ffmpeg_params=ffmpeg_params)
            logger.info(f"Concatenated views -> {output_path}")

    dist.barrier()

    if rank == 0:
        backup_dir = Path(output_dir) / f"{subdir_name}_backup"
        temp_dir = Path(output_dir) / f"_concat_{subdir_name}"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if temp_dir.exists():
            os.rename(str(video_dir), str(backup_dir))
            os.rename(str(temp_dir), str(video_dir))
            logger.info(f"Replaced {subdir_name} with concatenated views (backup: {backup_dir})")


# --------------- Single-GPU main loop ---------------
def run_one_rank(local_rank: int, world_size: int,
                 args: argparse.Namespace, config: Any):
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    gen = VideoGenerator(local_rank, args, config)
    gen.initialize_model()

    # dataset / sampler / dataloader
    eval_dataset_config = config.get("eval_dataset_config", None)
    if eval_dataset_config is None:
        raise ValueError("Please specify eval_dataset_config in the config file.")

    is_autoregressive = args.inference_mode == "autoregressive" and args.num_chunks > 1

    if is_autoregressive:
        # Override dataset to load both views and enough frames for all chunks
        base_num_frames = eval_dataset_config.params.video_params.num_frames
        total_frames = args.num_chunks * (base_num_frames - 1) + 1
        eval_dataset_config.params.video_params.num_frames = total_frames
        # Store base chunk size for the generator to use
        config._ar_frames_per_chunk = base_num_frames
        print(f"[Autoregressive] Overriding num_frames={total_frames} (base={base_num_frames}, "
              f"chunks={args.num_chunks}), return_view=both")

    dataset = instantiate_from_config(eval_dataset_config)
    sampler = DistributedSampler(dataset,
                                 num_replicas=world_size,
                                 rank=local_rank,
                                 shuffle=False,
                                 drop_last=False)

    # Key point: batch_size=1 and no stacking, return dict directly
    from ittakestwo.dataset.collate_functions import default_collate_fn
    collate_fn = default_collate_fn

    loader = DataLoader(dataset,
                        batch_size=1,
                        sampler=sampler,
                        num_workers=0,
                        pin_memory=True,
                        collate_fn=collate_fn)

    gen.logger.info(f"Start ItTakesTwo inference, {len(loader)} examples on this rank.")
    # Compute global start index for this rank
    start_idx = len(loader) * local_rank   # Valid when shuffle=False and drop_last=False
    for local_idx, example in enumerate(loader):
        # Compute global index for naming
        global_idx = start_idx + local_idx
        save_video_name = f"global_idx{global_idx:06d}-" + "-".join(example['video_name'].split("/"))[:-4]

        if is_autoregressive:
            left_frames, right_frames, gt = gen.generate_video_autoregressive(example, args.num_chunks)

            left_out = Path(args.output_dir) / "gen_left" / f"{save_video_name}.mp4"
            gen.save(left_frames, str(left_out))

            right_out = Path(args.output_dir) / "gen_right" / f"{save_video_name}.mp4"
            gen.save(right_frames, str(right_out))

            concat_lr = concat_videos(left_frames, right_frames, dim='width')
            concat_out = Path(args.output_dir) / "concat" / f"{save_video_name}.mp4"
            gen.save(concat_lr, str(concat_out))

            # Save ground truth
            if len(gt) == 1:
                gt = gt * len(left_frames)
            gt_out_path = Path(args.output_dir) / "gt" / f"{save_video_name}.mp4"
            gen.save(gt, str(gt_out_path))
        else:
            frames, gt = gen.generate_video(example)
            action = example['action']
            action = torch.concat([action['discrete_action'],action['continuous_action']],dim=-1).float().cpu().numpy()
            out_path = Path(args.output_dir) / "gen" / f"{save_video_name}.mp4"
            gen.save(frames, str(out_path))

            if len(gt) == 1:
                gt = gt * len(frames)
            gt_out_path = Path(args.output_dir) / "gt" / f"{save_video_name}.mp4"
            gen.save(gt, str(gt_out_path))

            concat_out_path = Path(args.output_dir) / "concat" / f"{save_video_name}.mp4"
            concat_frames = concat_videos(gt, frames, dim='height')
            gen.save(concat_frames, str(concat_out_path))

    dist.barrier()

    # For non-autoregressive inference with per-view samples, concatenate
    # left/right views horizontally so that the output matches the post-process script.
    if not is_autoregressive:
        for subdir in ["gen", "gt"]:
            _distributed_concat_views(args.output_dir, subdir, local_rank, world_size, gen.logger)
        dist.barrier()

    dist.destroy_process_group()
    gen.logger.info("Rank finished.")

# --------------- Main entry ---------------
def main():
    parser = argparse.ArgumentParser(description="Wan2.2 TI2V inference torchrun")
    parser.add_argument("--config-path", required=True, type=str)
    parser.add_argument("--eval-data-config-path",  default=None, type=str)
    parser.add_argument("--model-path",  required=True, type=str)
    parser.add_argument("--inference-mode", default="fixlength",
                        choices=["autoregressive", "fixlength"])
    parser.add_argument("--inference-seed",default=0,type=int)
    parser.add_argument("--num-inference-steps",default=50,type=int)
    parser.add_argument("--num-chunks", default=1, type=int,
                        help="Number of autoregressive chunks (only used with --inference-mode autoregressive)")
    parser.add_argument("--output-dir",  default="outputs", type=str)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Load config
    config = load_config(args.config_path)
    if args.eval_data_config_path is not None:
        eval_data_config = load_config(args.eval_data_config_path)
        # Direct replacement:
        if eval_data_config.get("target") is not None \
            and eval_data_config.get("params") is not None:
            config.eval_dataset_config = eval_data_config
        elif eval_data_config.get("eval_dataset_config") is not None:
            config.eval_dataset_config = eval_data_config.eval_dataset_config
        else:
            raise ValueError("Invalid eval_data_config format.")
    print(f"config {config}")
    # Environment variables injected by torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    run_one_rank(local_rank, world_size, args, config)


if __name__ == "__main__":
    if "RANK" not in os.environ:
        print("Please use torchrun to launch, e.g.")
        print("python -m torch.distributed.run parallel_inference.py --config-path xxx.yaml --model-path xxx.pth")
        exit(1)
    main()

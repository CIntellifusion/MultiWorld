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
from pathlib import Path
from typing import Dict, List, Any
import PIL

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image
from omegaconf import OmegaConf
from ittakestwo.preprocess.action_visualization_utils import visualize_action_on_frames
from utils.video_utils import concat_videos

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
        save_video_name = f"global_idx{global_idx:06d}-rank{local_rank:03d}"
        save_video_name = f"global_idx{global_idx:06d}-" + "-".join(example['video_name'].split("/"))[:-4] # Distinguish by rank
        frames, gt = gen.generate_video(example)
        action = example['action']
        action = torch.concat([action['discrete_action'],action['continuous_action']],dim=-1).float().cpu().numpy()  # (T, D) 
        out_path = Path(args.output_dir) / "gen" / f"{save_video_name}.mp4"
        gen.save(frames, str(out_path))
        
        if len(gt) == 1:
            gt = gt * len(frames) 
        gt_out_path = Path(args.output_dir) / "gt" / f"{save_video_name}.mp4"
        gen.save(gt, str(gt_out_path))
        
        # Concatenate gt & generated for easy comparison
        concat_out_path = Path(args.output_dir) / "concat" / f"{save_video_name}.mp4"
        concat_frames = concat_videos(gt, frames, dim='height')
        gen.save(concat_frames, str(concat_out_path))
        
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

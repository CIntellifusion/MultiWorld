import os
import sys
# Ensure repo root is in path when script is run directly (e.g. via torch.distributed.run)
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
import csv
import json
import random
import argparse
import torch
import multiprocessing as mp
import logging
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from typing import Dict, List, Optional, Tuple, Any
import numpy as np 
from diffsynth.pipelines.wan_video_robots import WanVideoPipeline,ModelConfig
from utils import load_config ,instantiate_from_config  
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from utils.video_utils import concat_videos
from diffsynth.utils.data import save_video
from diffsynth.diffusion.runner import move_action_to_device
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_logger(gpu_id: int):
    """Set up an independent logger for each GPU."""
    logger = logging.getLogger(f"GPU_{gpu_id}")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter with GPU identifier
    formatter = logging.Formatter(
        f'[GPU-{gpu_id} %(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_observation_relationship(relationship_path: str) -> Dict:
    """Load observation relationship data."""
    with open(relationship_path, 'r') as f:
        return json.load(f)

def load_metadata(metadata_path: str) -> Tuple[Dict[str, List[str]], int]:
    """Load metadata."""
    metadata = {}
    count = 0
    
    with open(metadata_path, 'r') as f:
        reader = csv.reader(f)
        fieldnames = next(reader)
        metadata = {name: [] for name in fieldnames}
        
        for row in reader:
            for name, value in zip(fieldnames, row):
                metadata[name].append(value)
            count += 1
    
    return metadata, count

class VideoGenerator:
    """Video generator class responsible for model initialization and video generation."""
    
    def __init__(self, gpu_id: int, args: argparse.Namespace, config: Any):
        self.gpu_id = gpu_id
        self.args = args
        self.config = config
        self.logger = setup_logger(gpu_id)
        self.device = f"cuda:{gpu_id}"
        self.pipe = None
        
    def initialize_model(self):
        """Lazy model initialization (executed in sub-process)."""
        self.logger.info("Initializing model...")
        torch.cuda.set_device(self.gpu_id)
        self.pipe = WanVideoPipeline.from_pretrained(
            config=self.config,
            torch_dtype=torch.bfloat16,
            device=self.device,
            model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                            origin_file_pattern="diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                            origin_file_pattern="Wan2.2_VAE.pth"),
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

    def generate_video(self, inputs: Dict) -> List:
        """Execute video generation."""
        if self.args.inference_mode == "fixlength":
            return self._generate_fix_length(inputs)
        else:
            return self._generate_autoregressive(inputs)
            
    def _generate_fix_length(self, example: Dict) -> List:
        """Fixed-length generation mode."""
        dataset_cfg = self.config.dataset_config
        input_image = example["video"][0]  # List[PIL.Image]
        width,height = input_image.size  # Get width and height of a single frame
        print(f"Env obv is None: {example['env_obv'] is None}")
        action = example["action"] # Dict
        gt_frames = example["video"]  # Note: may need to convert to list[PIL] or numpy
        action = move_action_to_device(action, self.device)
        width,height = input_image.size  # Get width and height of a single frame
        env_obv = example["env_obv"].to(self.device, dtype=torch.bfloat16)
        # Generate video
        video = self.pipe(
            input_image=input_image,
            action=action,
            env_obv=env_obv,
            seed=0,
            tiled=True,
            height=height,
            width=width,
            num_frames=81,
            num_inference_steps=35,
        )
        return video,gt_frames
        
    def _generate_autoregressive(self, inputs: Dict) -> List:
        """Autoregressive generation mode."""
        dataset_cfg = self.config.dataset_config
        action_latents = inputs["action"]["action"]
        print(f"action latent shape {action_latents.shape}")
        total_frames = action_latents.shape[2]
        
        # Calculate number of chunks
        chunk_size = self.config.dataset_config.params.video_params.num_frames
        num_chunks = (total_frames + chunk_size - 1) // chunk_size
        if num_chunks == 1 :
            return None, None 
        self.logger.info(f"Generating {total_frames} frames in {num_chunks} chunks")
        
        # Pad action latents to full length
        padding = num_chunks * chunk_size - total_frames
        if padding > 0:
            pad = torch.zeros(
                (*action_latents.shape[:2], padding, action_latents.shape[3]),
                device=action_latents.device,
                dtype=action_latents.dtype
            )
            action_latents = torch.cat([action_latents, pad], dim=2)
        
        video = []
        current_image = inputs["video"][0]  # Use main view as initial input
        # TODO: update observation result
        env_obv = inputs["env_obv"].to(self.device, dtype=torch.bfloat16)

        for chunk_idx in range(num_chunks):
            action_chunk_size = 21 
            start = chunk_idx * action_chunk_size
            end = min((chunk_idx + 1) * action_chunk_size, total_frames)
            
            # Compute action indices for current chunk
            need_extra = (end - start) % 4 != 1
            indices = list(range(start, end, 1))
            if need_extra:
                indices.append(end - 1)
            
            self.logger.info(
                f"Chunk {chunk_idx+1}/{num_chunks}: frames {start}-{end}, "
                f"action indices: {indices}"
            )
            
            # Extract current chunk data
            chunk_action_latent = action_latents[:, :, indices, :]
            chunk_action = {"action":chunk_action_latent,"camera":inputs['action']['camera']}
            chunk_action = move_action_to_device(chunk_action, self.device)
            width,height = current_image.size  # Get width and height of a single frame
            chunk_video = self.pipe(
                action=chunk_action,
                env_obv=env_obv,
                input_image=current_image,
                num_frames=chunk_size,
                height=height,
                width=width,
                seed=1,
                tiled=True,
            )
            
            video.extend(chunk_video)
            current_image = chunk_video[-1]  # Use last frame as input for next chunk
        
        return video,inputs["video"]
        
    def save_output(self, video: List, output_paths: List):
        """Save generated video."""
        for idx, video_path in enumerate(output_paths):
            # Build output path
            rel_path = os.path.relpath(video_path, self.args.base_path)
            output_path = os.path.join(self.args.output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            save_video(video, output_path, fps=25, quality=10)
            self.logger.info(f"Saved video to {output_path} ({len(video)} frames)")
        
    def save(self, frames: List[np.ndarray], save_path: str):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_params = [
            '-vcodec', 'libx264',       # Specify encoder
            '-preset', 'medium',        # Balance between compression speed and ratio
            '-crf', '18',               # Quality factor; lower is better (18-28 range commonly used)
            '-pix_fmt', 'yuv420p',      # Most compatible pixel format
            '-movflags', '+faststart'   # Enable fast start for online playback (optional)
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
    # Key: batch_size=1 without stacking, return dict directly
    from ittakestwo.dataset.collate_functions import default_collate_fn
    collate_fn = default_collate_fn

    loader = DataLoader(dataset,
                        batch_size=1,
                        sampler=sampler,
                        num_workers=0,
                        pin_memory=True,
                        collate_fn=collate_fn)
    gen.logger.info(f"Start Robotic inference, {len(loader)} examples on this rank.")
    # Calculate global start index for this rank
    start_idx = len(loader) * local_rank   # Valid only when shuffle=False and drop_last=False
    for local_idx, example in enumerate(loader):
        # Calculate global index for naming
        global_idx = start_idx + local_idx
        save_video_name = "-".join(example['video_path'].split("/"))
        frames, gt = gen.generate_video(example)
        if frames is None and gt is None: continue
        out_path = Path(args.output_dir) / "gen" / f"{save_video_name}.mp4"
        gen.save(frames, str(out_path))
        gt_out_path = Path(args.output_dir) / "gt" / f"{save_video_name}.mp4"
        gen.save(gt, str(gt_out_path))
        concat_out_path = Path(args.output_dir) / "concat" / f"{save_video_name}.mp4"
        concat_frames = concat_videos(gt, frames, dim='height')
        gen.save(concat_frames, str(concat_out_path))
def main():
    parser = argparse.ArgumentParser(description="Wan2.2 Video Generation")
    parser.add_argument("--config-path", default=None, type=str)
    parser.add_argument("--eval-data-config-path",  default=None, type=str)

    parser.add_argument("--model-path", 
                       default="models/train/Wan2.2-TI2V-5B_lora-action/epoch-0.safetensors",
                       type=str)
    parser.add_argument("--inference-mode", default="fixlength", type=str,
                       choices=["autoregressive", "fixlength"])
    parser.add_argument("--relationship-path", default=None, type=str)   
    parser.add_argument("--output-dir", default="output.mp4", type=str)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Load config
    config = load_config(args.config_path) if args.config_path else None

    if args.eval_data_config_path is not None:
        eval_data_config = load_config(args.eval_data_config_path)
        # direct replace: 
        if eval_data_config.get("target") is not None \
            and eval_data_config.get("params") is not None:
            config.eval_dataset_config = eval_data_config
        elif eval_data_config.get("eval_dataset_config") is not None:
            config.eval_dataset_config = eval_data_config.eval_dataset_config
        else:
            raise ValueError("Invalid eval_data_config format.")
    if args.inference_mode == "autoregressive":
        config.eval_dataset_config.params.video_params.num_frames = None 
    print(f"config {config}")
    # Environment variables injected by torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    run_one_rank(local_rank, world_size, args, config)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
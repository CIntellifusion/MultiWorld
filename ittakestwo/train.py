import os
import sys
# Ensure repo root is in path when script is run directly (e.g. via torch.distributed.run)
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
    
import random
import numpy as np
import torch, os, argparse, accelerate, warnings
from diffsynth.pipelines.wan_video_ittakestwo import WanVideoPipeline
from diffsynth.diffusion.logger import ModelTSLogger
from diffsynth.diffusion import *
from omegaconf import OmegaConf 
from utils import load_config
from utils import parse_unknown_to_dict, merge_dict_into_config
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ITTWanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        config,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        # ITT = ItTakesTwo
        # Image+Action to Video Pipeline Training Module 
        super().__init__()
        # Warning
        self.config = config 
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        
        # Text Encoder is removed 
        self.pipe = WanVideoPipeline.from_pretrained(config=config,torch_dtype=torch.bfloat16, device=device, model_configs=model_configs)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        assert "imageaction2video" in task , "This is a customized training module for Image+Action to Video tasks."
        self.task_to_loss = {
            "sft:imageaction2video-df": lambda pipe, inputs_shared, inputs_posi, inputs_nega: ImageAction2VideoFlowMatchDFSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:imageaction2video": lambda pipe, inputs_shared, inputs_posi, inputs_nega: ImageAction2VideoFlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:imageaction2video-causal-tf": lambda pipe, inputs_shared, inputs_posi, inputs_nega: ImageAction2CausalVideoFlowMatchTFLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:imageaction2video-causal-ode": lambda pipe, inputs_shared, inputs_posi, inputs_nega: ImageAction2CausalODERegressionLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        def index_frame_from_video(video,frame_idx):
            if isinstance(video, torch.Tensor):
                # video tensor [B,C,F,H,W]
                indexed_frames = video[:, :, frame_idx]  # [B,C,H,W]
                return indexed_frames
            else:
                pil_frame = video[frame_idx]
                return pil_frame
                
        has_video = "video" in data
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                if has_video:
                    inputs_shared["input_image"] = index_frame_from_video(data["video"], 0)
            elif extra_input == "end_image":
                if has_video:
                    inputs_shared["end_image"] = index_frame_from_video(data["video"], -1)
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                if has_video:
                    inputs_shared[extra_input] = index_frame_from_video(data["video"], data["start_point"].item())
            else:
                if extra_input in data:
                    inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"] if "prompt" in data else ""}
        inputs_nega = {}

        # ODE regression: no raw video, latents are pre-computed
        is_ode = "ode" in self.task

        if is_ode:
            width = self.config.dataset_config.params.video_params.get("width", 480)
            batch_size = data["clean_latent"].shape[0]
            input_video = None
        else:
            if self.config.dataset_config.params.return_view == "both":
                width = self.config.dataset_config.params.video_params.width
            else:
                width = self.config.dataset_config.params.video_params.width // 2
            batch_size = data["video"].shape[0] if isinstance(data["video"], torch.Tensor) else 1
            input_video = data["video"]

        inputs_shared = {
            "input_video": input_video,
            "batch_size": batch_size,
            "height": self.config.dataset_config.params.video_params.height,
            "width": width,
            "num_frames": self.config.dataset_config.params.video_params.num_frames,
            # Action2Video specific inputs
            "action": data["action"],
            "env_obv": data.get("env_obv", None),
            "env_processor_flag": self.pipe.env_processor_flag,
            # Pipeline control parameters
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "use_diffusion_forcing": "df" in self.task,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)

        # ODE regression: inject pre-computed latents after pipeline units
        if "ode" in self.task:
            inputs_shared = inputs[0]
            inputs_shared["input_latents"] = data["clean_latent"].to(self.pipe.device, dtype=self.pipe.torch_dtype)
            inputs_shared["ode_latent"] = data["ode_latent"].to(self.pipe.device, dtype=self.pipe.torch_dtype)
            inputs_shared["sigmas"] = data["sigmas"].to(self.pipe.device, dtype=torch.float32)
            inputs_shared["clean_latent"] = data["clean_latent"].to(self.pipe.device, dtype=self.pipe.torch_dtype)
            first_frame = data["clean_latent"][:, :, 0:1].to(self.pipe.device, dtype=self.pipe.torch_dtype)
            inputs_shared["first_frame_latents"] = first_frame
            inputs_shared["fuse_vae_embedding_in_latents"] = True

        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss

def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    return parser

def seed_everything(seed: int = 42, deterministic: bool = True, benchmark: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多卡
    print(f"[Seed Everything] Global seed set to {seed}, deterministic={deterministic}")
    return seed
if __name__ == "__main__":
    seed_everything(42)
    parser = wan_parser()
    parser.add_argument('--config_path', type=str, default='configs/base_config.yaml', help='Path to the config file.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    args, unknown = parser.parse_known_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    # load config from yaml
    config = load_config(args.config_path) 
    # update config with args 
    flat_dict = parse_unknown_to_dict(unknown)
    merge_dict_into_config(config, flat_dict)
    
    if args.debug:
        config.dataset_config.params.max_entries = 100
        config.trainer_config.validation_interval = 10
    
    # save merged config 
    config_output_path = os.path.join(args.output_path,"configs/training_config.yaml")
    os.makedirs(os.path.dirname(config_output_path),exist_ok=True)
    print(f"Merged Final Training config:  {OmegaConf.to_yaml(config)}")
    OmegaConf.save(config, config_output_path)
    print(f"Saving Training config to {config_output_path}")
    
    model = ITTWanTrainingModule(
        config=config,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=config.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    
    model_logger = ModelTSLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    
    launch_training_task(config.trainer_config,config, accelerator, model, model_logger)

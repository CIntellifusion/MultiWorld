from utils.tensor_utils import auto_match_dim
import math
import torch, types
import numpy as np
from PIL import Image
from einops import repeat,rearrange
from typing import Optional, Union
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal
from diffsynth.core.loader.file import load_state_dict 
from diffsynth.models.wan_env_encoder import WanEnvEncoder 
from utils.import_utils import instantiate_from_config

from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
class WanVideoPipeline(BasePipeline):

    def __init__(self,config, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.config = config
        self.scheduler = FlowMatchScheduler("Wan")
        self.dit: WanModel = None
        self.base_iteration_models = ["dit",]
        self.in_iteration_models = ("dit",)
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ActionCFG(),
            WanVideoUnit_EnvEmbedder(),  # TODO: Support VGGT and DINO environment encoders 
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
        ]
        self.post_units = [
            # WanVideoPostUnit_S2V(),
        ]

        self.model_fn = model_fn_wan_action2video
    def enable_usp(self):
        from ..utils.xfuser import get_sequence_parallel_world_size, usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True

    @staticmethod
    def from_pretrained(
        config,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        redirect_common_files: bool = True,
        use_usp: bool = False,
        vram_limit: float = None,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
                "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors"),
                "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to {redirect_dict[model_config.origin_file_pattern]}. You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(config=config,device=device, torch_dtype=torch_dtype)
        if use_usp:
            from ..utils.xfuser import initialize_usp
            initialize_usp(device)
        # for model_config in model_configs:
            # model_config.download_if_necessary()
        dit_model_config,vae_model_config = model_configs
        model_pool = pipe.download_and_load_models([vae_model_config], vram_limit)
        
        # Fetch models
        
        dit_config = config.simulator_config.dit_config
        dit = instantiate_from_config(dit_config)
        if isinstance(dit_config.model_path, str):
            model_path_list = [dit_config.model_path]
        else:
            model_path_list = list(dit_config.model_path) 
        dit_state_dict = load_state_dict(model_path_list)
        dit_load_statedict_result = dit.load_state_dict(dit_state_dict,strict=False)
        pipe.dit = dit.to(dtype=torch_dtype, device=device)
    
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        
        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        env_encoder_config = config.simulator_config.env_encoder_config
        if config.simulator_config.load_env_encoder and env_encoder_config is not None:
            pipe.env_encoder = instantiate_from_config(env_encoder_config)
            pipe.base_iteration_models.append("env_encoder")
            pipe.env_processor_flag = getattr(config.simulator_config,"env_processor_flag",None)
        else:
            pipe.env_encoder = None 
            pipe.env_processor_flag = None 
        # update the in-iteration models
        pipe.in_iteration_models = set(pipe.base_iteration_models)
        
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()

        return pipe
    
    def load_from_checkpoint(self,checkpoint_path: str|list):
        state_dict = load_state_dict(checkpoint_path)
        dit_load_result = self.dit.load_state_dict(state_dict, strict=False)
        if hasattr(self,"env_encoder") and self.env_encoder is not None:
            env_state_dict = {k.replace("pipe.env_encoder.connector.",""):v for k,v in state_dict.items() if 'env_encoder' in k}
            print(f"env_state_dict keys:{env_state_dict.keys()}")
            env_encoder_result = self.env_encoder.connector.load_state_dict(env_state_dict)
            # print(f"debug env encoder load result: {env_encoder_result}")

    @torch.no_grad()
    def __call__(
        self,
        action: torch.Tensor | dict,
        env_obv: torch.Tensor | dict,
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
        output_type: Optional[Literal["quantized", "floatpoint"]] = "quantized",
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "batch_size": 1,
            "input_image": input_image,
            "action": action,
            "env_obv":env_obv,
            "env_processor_flag": self.env_processor_flag, 
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "cfg_scale": 1,
            "use_diffusion_forcing": False,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT removed 
            # Timestep
            # Inference
            noise_pred = self.model_fn(self.config,**models, **inputs_shared, **inputs_posi, timestep=timestep)
            # No prompt since no cfg 
            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                # Preserve the first-frame latents.
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
        
        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        elif output_type == "floatpoint":
            pass
        # self.load_models_to_device([])
        return video

    # ------------------------------------------------------------------
    # Helper methods for encoding (shared across inference methods)
    # ------------------------------------------------------------------

    def _encode_env_context(self, env_obv):
        """Encode environment observations to context embeddings."""
        env_encoder = getattr(self, "env_encoder", None)
        if env_obv is None or env_encoder is None:
            return None

        env_processor_flag = getattr(self, "env_processor_flag", None)
        if env_processor_flag == "vae":
            self.load_models_to_device(['vae'])
            b, k = env_obv.shape[:2]
            env_obv = rearrange(env_obv, 'b k c t h w -> (b k) c t h w')
            env_obv = self.dit.patchify(env_obv)
            env_obv = rearrange(env_obv, '(b k) c t h w -> b k c t h w', b=b, k=k)
            env_obv = torch.mean(env_obv, dim=1)
            env_obv = rearrange(env_obv, 'b c f h w -> b (f h w) c').contiguous()

        device_type = self.device.type if isinstance(self.device, torch.device) else self.device
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            return env_encoder(env_obv)

    def _encode_action(self, action, is_null=False):
        """Encode action dict to embeddings. If is_null, replace with ones (CFG null token)."""
        if action is None or self.dit.action_encoder is None:
            return None

        if is_null:
            action_input = self._make_null_action(action)
        else:
            action_input = action

        action_embeds = self.dit.action_encoder(action_input)
        if self.config.simulator_config.dit_config.params.action_injection in ("causal_cross_attention", "ia2v_causal_cross_attention"):
            action_embeds = action_embeds.squeeze(2)
        return action_embeds

    @staticmethod
    def _make_null_action(action):
        """Recursively replace tensor values with ones (null token for CFG)."""
        if isinstance(action, torch.Tensor):
            return torch.ones_like(action)
        if isinstance(action, dict):
            return {k: WanVideoPipeline._make_null_action(v) for k, v in action.items()}
        return action

class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}

class WanVideoUnit_ActionCFG(PipelineUnit):
    """
    Action Classifier-Free Guidance dropout during training.
    With 10% probability, replaces all action values with ones (unconditional).
    All-zero actions are meaningful (no keys pressed); all-ones is the null token.
    Only active when the scheduler is in training mode.
    """
    def __init__(self):
        super().__init__(
            input_params=("action",),
            output_params=("action",),
        )

    def process(self, pipe: WanVideoPipeline, action):
        if pipe.scheduler.training and torch.rand(1).item() < 0.1:
            action = {k: torch.ones_like(v) for k, v in action.items()}
        return {"action": action}

class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image","batch_size"),
            output_params=("noise",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image,batch_size=1):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
            length += f
        shape = (batch_size, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        return {"noise": noise}
    

class WanVideoUnit_EnvEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("env_obv", "tiled", "tile_size", "tile_stride","env_processor_flag"),
            output_params=("env_obv"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, env_obv, tiled, tile_size, tile_stride,env_processor_flag):
        # Only used when encoding with VAE.
        if env_obv is None:
            return {"env_obv": None}
        pipe.load_models_to_device(self.onload_model_names)
        # rearrange 
        if env_processor_flag is None:
            # Keep raw tensor for VGGT to maintain backward compatibility.
            return {"env_obv": env_obv}
        elif env_processor_flag=="vae":
            b,k,c,t,h,w = env_obv.shape 
            env_obv = rearrange(env_obv,"b k c t h w -> (b k) c t h w")
            env_obv_latents = pipe.vae.encode(env_obv.to(dtype=pipe.torch_dtype, device=pipe.device), device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            env_obv_latents = rearrange(env_obv_latents, '(b k) c f h w -> b k c f h w',b=b,k=k).contiguous()
            return {"env_obv": env_obv_latents}
        else:
            raise ValueError(f"Wrong env_processor_flag {env_processor_flag}")
class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        # The original code accepts List[PIL.Image].
        # If the input is already a tensor, skip preprocessing.
        if isinstance(input_video, torch.Tensor):
            input_video = input_video
        else:
            input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}

class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "fuse_vae_embedding_in_latents", "first_frame_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if isinstance(input_image, torch.Tensor):
            assert input_image.dim() == 4, "When input_image is a tensor, it should be of shape (B,C,H,W)." 
            z = pipe.vae.encode(repeat(input_image,'b c h w -> b c f h w',f=1), device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        else:
            image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
            z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}


class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=(), output_params=("use_unified_sequence_parallel",))

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}

class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            output_params=("tea_cache",)
        )

    def process(self, pipe: WanVideoPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}

class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega

class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states

def model_fn_wan_action2video(
    config, 
    dit: WanModel,
    env_encoder: WanEnvEncoder = None,
    env_processor_flag: str=None, 
    action: dict = None,
    env_obv: torch.Tensor = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    fuse_vae_embedding_in_latents: bool = False,
    use_diffusion_forcing: bool = False,
    **kwargs,
):
    """
    Model function for action-to-video generation.
    
    This function:
    1. Prepares timestep tensor (handles diffusion forcing and separated timestep)
    2. Encodes environment observations if provided
    3. Encodes actions if provided
    4. Calls dit.forward() to perform the actual forward pass
    
    The forward pass logic is now implemented in WanModel.forward() for better code organization.
    """
    # Handle timestep preparation (diffusion forcing / separated timestep)
    if use_diffusion_forcing:
        # Per-token timestep for diffusion forcing
        batch_size, c, num_frames, h, w = latents.shape
        num_timestep_token_per_frame = h * w // 4
        timestep_prepared = timestep[:, None].repeat(1, num_timestep_token_per_frame).flatten()
    elif dit.seperated_timestep and fuse_vae_embedding_in_latents:
        # First frame has t=0, rest have t=timestep
        num_token_per_frame = latents.shape[3] * latents.shape[4] // 4
        timestep_prepared = torch.concat([
            torch.zeros((1, num_token_per_frame), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, num_token_per_frame), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
    else:
        timestep_prepared = timestep
    
    # --- Environment context encoding ---
    env_context = None
    if env_obv is not None and env_encoder is not None:
        # patchify env_obv latents: B,K, C, T, H, W
        if env_processor_flag == "vae":
            b, k = env_obv.shape[:2]
            env_obv = rearrange(env_obv, 'b k c t h w -> (b k) c t h w')
            env_obv = dit.patchify(env_obv)
            env_obv = rearrange(env_obv, '(b k) c t h w -> b k c t h w', b=b, k=k)
            env_obv = torch.mean(env_obv, dim=1)  # b c t h w
            env_obv = rearrange(env_obv, 'b c f h w -> b (f h w) c').contiguous()
        with torch.autocast(device_type=latents.device.type, dtype=torch.bfloat16):
            env_context = env_encoder(env_obv)
    
    # --- Action encoding ---
    action_embeds = None
    if action is not None and dit.action_encoder is not None:
        action_embeds = dit.action_encoder(action)
        if config.simulator_config.dit_config.params.action_injection == "adaln_zero":
            # Repeat action embeds for each token and add to t_mod in forward
            _, _, _, h, w = latents.shape
            num_token_per_frame = (h // dit.patch_size[1]) * (w // dit.patch_size[2])
            action_embeds = action_embeds.repeat_interleave(num_token_per_frame, dim=1)
        elif config.simulator_config.dit_config.params.action_injection in ("causal_cross_attention", "ia2v_causal_cross_attention"):
            action_embeds = action_embeds.squeeze(2)  # (B, F, D)
    # --- Call the model's forward method ---
    return dit(
        x=latents,
        timestep=timestep_prepared,
        context=None,  # Not used, we pass env_context separately
        action_embeds=action_embeds,
        env_context=env_context,
        clip_feature=None,  # Not used in action2video mode
        y=y,
        reference_latents=reference_latents,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        tea_cache=tea_cache,
        use_unified_sequence_parallel=use_unified_sequence_parallel,
    )

import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelTSLogger
from time import time
from utils.import_utils import import_class_from_string
from utils import instantiate_from_config
import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from typing import Optional, List
import numpy as np
import imageio 
from ittakestwo.preprocess.action_visualization_utils import visualize_action_on_frames
from utils.video_utils import concat_videos
from diffsynth.utils.data import save_video
# helper function 

def seed_worker(worker_id, base_seed=42):
    """Ensure deterministic randomness for each worker, compatible with distributed and non-distributed environments."""
    import random
    
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    worker_seed = base_seed + worker_id + rank * 1000
    
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)
    
def initialize_dataloader(config, batch_size, num_workers, global_seed=42, collate_fn=None):
    """
    Deterministic dataloader initialization.
    """
    # 1. initialize Dataset (random seed is not passed here; sampler controls epoch-level randomness)
    dataset = instantiate_from_config(config.dataset_config)
    print(f"Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}: Initialized dataset with {len(dataset) if hasattr(dataset, '__len__') else 'unknown'} samples.")
    
    # 2. check if distributed environment
    if torch.distributed.is_initialized():
        # key: use DistributedSampler with fixed seed
        # when shuffle=True, calling sampler.set_epoch(epoch) each epoch produces deterministic permutation based on seed
        sampler = DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=True,  # typically True during training
            seed=global_seed,  # key: fixed seed ensures consistency across runs
            drop_last=True,
        )
        shuffle = False  # when using sampler, DataLoader shuffle must be False
        # print sampler sequence 
        indices = list(sampler)
        print(f"current rank {torch.distributed.get_rank()}/{torch.distributed.get_world_size()} - DistributedSampler initialized with seed {global_seed}.")
        # print(f"DistributedSampler indices for epoch 0: {indices[:20]}... total {len(indices)} samples.")
    else:
        sampler = None
        shuffle = True
    
    # 3. build generator for DataLoader shuffle (when not using DistributedSampler)
    g = torch.Generator()
    g.manual_seed(global_seed)
    
    # 4. create DataLoader
    if isinstance(dataset, torch.utils.data.IterableDataset):
        # IterableDataset usually implements custom shard logic
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            worker_init_fn=lambda wid: seed_worker(wid, global_seed),
            generator=g,  # ensure deterministic shuffle
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,  # distributed sampler controls sample allocation
            shuffle=shuffle if sampler is None else False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=lambda wid: seed_worker(wid, global_seed),
            generator=g,
        )
        print(f"Dataloader batch size {batch_size}, num_workers {num_workers}, shuffle {shuffle}, drop_last True")
        print(f"Dataloader length: {len(dataloader)} batches per epoch.")
    
    return dataloader, sampler, dataset   # return sampler so set_epoch can be called each epoch
def print_unused_params(model):
    print("\n" + "="*80)
    print("Parameters not participating in loss computation:")
    print("="*80)
    
    unused_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"  {name:<60} | shape: {str(param.shape):<20}")
            unused_count += 1
    
    if unused_count == 0:
        print("All trainable parameters are participating in computation.")
    print("="*80 + "\n")

def save_video_(frames,save_path,save_fps):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_params = [
        '-vcodec', 'libx264',       # video codec
        '-preset', 'medium',        # balance between compression speed and ratio
        '-crf', '18',               # quality factor, lower is better (commonly 18-28)
        '-pix_fmt', 'yuv420p',      # most compatible pixel format
        '-movflags', '+faststart'   # enable fast start for online playback
    ]
    save_video(frames, str(save_path), fps=save_fps, quality=10, ffmpeg_params=ffmpeg_params)

def move_action_to_device(obj, device: str | torch.device, prefix: str = ""):
    """
    Recursively move Tensors in nested dicts to the target device and convert dtype.

    Args:
        obj: may be Tensor, Dict, or other types
        device: target device, e.g., f"cuda:{local_rank}"
        prefix: indentation prefix for printing (internal use)

    Returns:
        converted object (if Tensor or Dict, otherwise returned as-is)
    """
    indent = "  " * (len(prefix) // 2)  # indent based on hierarchy
    
    if isinstance(obj, torch.Tensor):
        # determine target dtype: floating-point -> bf16, integer -> long
        target_dtype = torch.bfloat16 if obj.is_floating_point() else torch.long
        
        # print(f"{indent}Tensor: shape={tuple(obj.shape)}, dtype={obj.dtype} -> {target_dtype}, "
            #   f"device={obj.device} -> {device}")
        
        return obj.to(device, dtype=target_dtype)
    
    elif isinstance(obj, dict):
        print(f"{indent}Dict:")
        # modify dict in-place to preserve reference
        for k, v in obj.items():
            print(f"{indent}  Key: '{k}'")
            obj[k] = move_action_to_device(v, device, prefix + "  ")
        return obj
    elif isinstance(obj,int):
        return [obj]
    else:
        # other types (e.g., list, tuple, int) are returned as-is for now; extend if needed
        print(f"{indent}Other: {type(obj).__name__}")
        return obj
    
def _get_inference_config(config):
    """Extract inference_config from training config with backward-compatible defaults."""
    if config is not None and hasattr(config, 'inference_config'):
        inf_cfg = config.inference_config
        return inf_cfg
    # Default: causal_kv (matches previous hardcoded behavior)
    return None


def _run_inference(pipe, method, example, action, height, width, num_frames, inf_cfg):
    """Dispatch to the appropriate pipeline inference method based on config."""
    env_obv = example.get("env_obv", None)

    if method == "standard":
        input_image = example["video"][0]
        return pipe(
            input_image=input_image,
            action=action,
            env_obv=env_obv,
            seed=0,
            tiled=True,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=int(inf_cfg.get("num_inference_steps", 50)) if inf_cfg else 50,
        )

    elif method == "causal_ode":
        video_frames = example["video"]
        return pipe.imageaction2causalvideo_ode(
            input_video_frames=video_frames,
            action=action,
            env_obv=env_obv,
            seed=0,
            num_inference_steps=int(inf_cfg.get("num_inference_steps", 50)) if inf_cfg else 50,
            guidance_scale=float(inf_cfg.get("guidance_scale", 1.0)) if inf_cfg else 1.0,
            sigma_shift=float(inf_cfg.get("sigma_shift", 5.0)) if inf_cfg else 5.0,
            height=height,
            width=width,
            tiled=True,
        )

    elif method == "dmd":
        input_image = example["video"][0]
        dmd_steps = list(inf_cfg.get("denoising_step_list", [1000, 750, 500, 250])) if inf_cfg else [1000, 750, 500, 250]
        ctx_noise = int(inf_cfg.get("context_noise", 0)) if inf_cfg else 0
        return pipe.imageaction2dmdvideo(
            input_image=input_image,
            action=action,
            env_obv=env_obv,
            seed=0,
            num_frames=num_frames,
            height=height,
            width=width,
            denoising_step_list=dmd_steps,
            context_noise=ctx_noise,
            tiled=True,
        )

    else:
        # Default: causal_kv
        input_image = example["video"][0]
        return pipe.imageaction2causalvideo(
            input_image=input_image,
            action=action,
            env_obv=env_obv,
            seed=0,
            tiled=True,
            height=height,
            width=width,
            num_frames=num_frames,
            num_frame_per_block=int(inf_cfg.get("num_frame_per_block", 1)) if inf_cfg else 1,
            independent_first_frame=bool(inf_cfg.get("independent_first_frame", True)) if inf_cfg else True,
            num_inference_steps=int(inf_cfg.get("num_inference_steps", 35)) if inf_cfg else 35,
        )


@torch.no_grad()
def evaluate_model(
    accelerator: Accelerator,
    eval_dataloader: torch.utils.data.DataLoader,
    model: DiffusionTrainingModule,
    output_dir: Path,
    global_step: int,
    dataset_config: dict,
    config=None,
):
    """Evaluation loop. Dispatches inference based on inference_config in config."""
    model.eval()
    local_rank = accelerator.local_process_index

    if accelerator.is_main_process:
        (output_dir / "gen").mkdir(parents=True, exist_ok=True)
        (output_dir / "gt").mkdir(parents=True, exist_ok=True)
        (output_dir / "concat").mkdir(parents=True, exist_ok=True)

    # Parse inference config
    inf_cfg = _get_inference_config(config)
    method = str(inf_cfg.get("method", "causal_kv")) if inf_cfg else "standard"
    print(f"Evaluation on local rank {local_rank} at step {global_step}, method={method}... output to {output_dir}")
    accelerator.wait_for_everyone()

    pipe = accelerator.unwrap_model(model).pipe
    if not hasattr(dataset_config.params,"return_view") or dataset_config.params.return_view == "both":
        width = dataset_config.params.video_params.width
    else:
        width = dataset_config.params.video_params.width // 2

    height = dataset_config.params.video_params.height
    num_frames = dataset_config.params.video_params.num_frames
    save_fps = int(60 //dataset_config.params.video_params.frame_skip)

    start_idx = len(eval_dataloader) * local_rank
    for local_idx, example in enumerate(eval_dataloader):
        input_image = example["video"][0]
        width,height = input_image.size
        print(f"Env obv is None: {example['env_obv'] is None}")
        action = example["action"]
        gt_frames = example["video"]

        action = move_action_to_device(action, f"cuda:{local_rank}")
        global_idx = start_idx + local_idx

        # Dispatch inference based on config
        generated_frames = _run_inference(
            pipe, method, example, action, height, width, num_frames, inf_cfg
        )

        # Save generated video
        save_video_name = f"global_idx{global_idx:06d}-" + "-".join(example['video_name'].split("/"))[:-4]
        gen_path  = output_dir / "gen" / f"{save_video_name}.mp4"
        save_video_(generated_frames, str(gen_path), save_fps=save_fps)

        # Save GT
        gt_path = output_dir / "gt" / f"{save_video_name}.mp4"
        if len(gt_frames) == 1:
            gt_frames = gt_frames * len(generated_frames)

        save_video_(gt_frames, str(gt_path), save_fps=save_fps)

        gt_with_action = gt_frames
        concat_frames = concat_videos(gt_with_action, generated_frames, dim='height')
        concat_path = output_dir / "concat" / f"{save_video_name}.mp4"
        save_video_(concat_frames, str(concat_path), save_fps=save_fps)

    pipe.scheduler.set_timesteps(training=True)
    model.train()
    accelerator.wait_for_everyone()

def launch_training_task(
    trainer_config,
    config,
    accelerator: Accelerator,
    model: DiffusionTrainingModule,
    model_logger: ModelTSLogger,
):
    # check trainable parameters
    num_trainable_params = sum(p.numel() for p in model.trainable_modules())
    num_trainable_params_in_B = num_trainable_params / 1e9
    print(f'launch_training_task: num training parameters: {num_trainable_params}({num_trainable_params_in_B:.3f} B)')
    trainable_state_dict = [k for k, v in model.named_parameters() if v.requires_grad]
    with open(os.path.join(model_logger.output_path, 'trainable_params.txt'), 'w') as f:
        for param_name in trainable_state_dict:
            f.write(f"{param_name}\n")
    
    # action2video collate function to support batch size > 1
    if 'collate_fn' not in config or config.collate_fn is None:
        print(f"Using default action2video_collate_fn as collate function.")
        collate_fn = import_class_from_string("ittakestwo.dataset.collate_functions.action2video_concatview_collate_fn")
    else:
        collate_fn = import_class_from_string(config.collate_fn)
    
    # create optimizer 
    optimizer_class = import_class_from_string(trainer_config.optimizer_config.target_class)
    optimizer_state_path = trainer_config.optimizer_config.get('optimizer_state_path', None)       
    optimizer = optimizer_class(model.trainable_modules(), **trainer_config.optimizer_config.params)

    if optimizer_state_path is not None:
        ckpt = torch.load(optimizer_state_path, map_location="cpu", weights_only=False)
        
        optimizer.load_state_dict(ckpt['optimizer'])
        
        for k, v in trainer_config.optimizer_config.params.items():
            for item in optimizer.param_groups:
                if k in item:
                    item[k] = v
                    print(f"Updating optimizer param_groups {k} to {v}")
        if "num_steps" in ckpt:
            model_logger.num_steps = ckpt["num_steps"]
            print(f"Resumed from step {model_logger.num_steps}")
    
    # create lr scheduler
    scheduler_config = trainer_config.scheduler_config 
    scheduler_class = import_class_from_string(scheduler_config.target_class)
    scheduler = scheduler_class(optimizer, **scheduler_config.get("params", {}) )
    
    # setup training parameters 
    batch_size = trainer_config.get('batch_size', 1)
    num_workers = trainer_config.get('dataset_num_workers', 1)
    num_epochs = trainer_config.get('num_epochs', 1)
    save_steps = trainer_config.get('save_steps', None)
    
    # logging hyper-parameters 
    print(f"launch_training_task: batch_size={batch_size}, num_workers={num_workers} ,num_epochs={num_epochs} save_steps={save_steps}")
    
    # initialize the dataset 
    global_seed = 42  # experiment seed
    
    dataloader, sampler,dataset  = initialize_dataloader(
        config, 
        batch_size=batch_size, 
        num_workers=num_workers,
        global_seed=global_seed,
        collate_fn=collate_fn
    )
    
    if "eval_dataset_config" in config:
        eval_dataset_config = config.eval_dataset_config
        eval_dataset = instantiate_from_config(eval_dataset_config)
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=1, shuffle=False, 
            num_workers=num_workers, collate_fn=lambda x: x[0]  # return single sample instead of batch
        )

    else:
        eval_dataset_config = None 
        eval_dataset = None
        eval_dataloader = None
    
    # print(f"debug {len(dataloader)} batches per epoch. dataset has {len(dataset)} samples.")
    print(f"Dataloader batch size {batch_size}, num_workers {num_workers}, shuffle {sampler is None}, drop_last True")
    print(f"before prepare len(dataloader)={len(dataloader) if hasattr(dataloader, '__len__') else 'unknown'} batches per epoch.")
    model, optimizer, eval_dataloader, scheduler = accelerator.prepare(model, optimizer, eval_dataloader, scheduler)
    print(f"after prepare len(dataloader)={len(dataloader) if hasattr(dataloader, '__len__') else 'unknown'} batches per epoch.")
    print(f"Evaluation dataset size: {len(eval_dataset)}")

    num_step_per_epoch = len(dataloader) if hasattr(dataset, '__len__') else "unknown"
    for epoch_id in range(num_epochs):
        time0 = time()
        if sampler is not None:
            # key: set epoch each epoch to ensure consistent shuffle order across runs
            # DistributedSampler uses seed + epoch as the random seed to generate indices for this epoch
            sampler.set_epoch(epoch_id)
        
        for data in dataloader:
            with accelerator.accumulate(model):
                time1=time()
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                # 3. raise error when loss > 2
                # if loss.item() > 2:
                    # raise RuntimeError(f"Loss exploded: {loss.item():.4f} > 2")
                accelerator.backward(loss)
                
                # print_unused_params(model) 
                # 1. grad clip + 2. compute grad norm (returns norm before clipping)
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
                optimizer.step()
                time2=time()
                model_logger.log('Training/grad_norm', grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                model_logger.log('Training/learning_rate', scheduler.get_last_lr()[0])
                model_logger.log('Profiling/data_loading_time', time1 - time0)
                model_logger.log('Profiling/step_compute_time', time2 - time1)
                model_logger.log('Epoch', epoch_id)
                start_point = data.get('start_point', None)
                real_idx = data.get('real_idx', None)
                model_logger.on_step_end(accelerator, model, optimizer,  save_steps,loss,num_step_per_epoch=num_step_per_epoch,
                                         epoch_id=epoch_id,step_time=time2-time1,start_point=start_point,real_idx=real_idx)
                scheduler.step()
                time0 = time() 
                global_step = model_logger.num_steps
                
            if eval_dataloader is not None and global_step % config.trainer_config.validation_interval == 0:
                print(f"\n>>> Step {global_step}: Running evaluation...")
                eval_output_dir = Path(model_logger.output_path) / "eval/videos" / f"step-{global_step}-videos"
                evaluate_model(
                    accelerator=accelerator,
                    eval_dataloader=eval_dataloader,
                    model=model,
                    output_dir=eval_output_dir,
                    global_step=global_step,
                    dataset_config=eval_dataset_config,
                    config=config,
                )
                print(f">>> Evaluation done. Resuming training...\n")
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, optimizer, epoch_id)
    model_logger.on_training_end(accelerator, model, optimizer, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelTSLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)

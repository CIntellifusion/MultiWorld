from .base_pipeline import BasePipeline
import torch
from utils.tensor_utils import auto_match_dim

def ImageAction2VideoFlowMatchDFSFTLoss(pipe: BasePipeline, **inputs):
    # Image + Action -> Video
    # Flow Matching SFT
    # Diffusion Forcing treats all frames identically.
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    num_frames = inputs["input_latents"].shape[2]
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (num_frames,))
    timestep_id[0] = max_timestep_boundary -1 
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)  # 1D tensor
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    # The first latent is kept clean. Latent shape: [B, C, F, H, W]
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(pipe.config,**models, **inputs, timestep=timestep)
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(), reduction='none')
    loss_weight = pipe.scheduler.training_weight(timestep).to(loss.device)
    loss = loss * auto_match_dim(loss_weight, loss)
    loss = loss.mean()
    return loss

def ImageAction2VideoFlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    # Keep the first latent clean. Latent shape: [B, C, F, H, W]
    inputs['latents'][:,:,0:1,:,:] = inputs['input_latents'][:,:,0:1,:,:]
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(pipe.config,**models, **inputs, timestep=timestep)
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss

def ImageAction2CausalVideoFlowMatchTFLoss(pipe: BasePipeline, **inputs):
    """
    Causal Teacher Forcing loss for image-action-to-video.
    Following the Causal-Forcing training paradigm:
    - All frames are noised with a uniform timestep
    - Clean (optionally augmented) latents are passed as context for teacher forcing
    - The causal model sees [clean_tokens | noisy_tokens] and predicts flow for noisy half
    - First frame kept clean (image conditioning)

    Ref: Causal-Forcing/model/diffusion.py CausalDiffusion.generator_loss
    """
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    noise_augmentation_max_timestep = int(inputs.get("noise_augmentation_max_timestep", 10))

    # Sample a single uniform timestep for all frames
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    clean_latent = inputs["input_latents"]  # [B, C, F, H, W]
    noise = torch.randn_like(clean_latent)

    # Add noise to all frames
    inputs["latents"] = pipe.scheduler.add_noise(clean_latent, noise, timestep)
    # Keep first frame clean (image conditioning)
    inputs["latents"][:, :, 0:1, :, :] = clean_latent[:, :, 0:1, :, :]
    training_target = pipe.scheduler.training_target(clean_latent, noise, timestep)
    
    # Noise augmentation for clean context (optional, adds small noise to TF context)
    if noise_augmentation_max_timestep > 0:
        aug_timestep_id = torch.randint(
            int(noise_augmentation_max_timestep / 1000 * len(pipe.scheduler.timesteps)),
            len(pipe.scheduler.timesteps), (1,))
        aug_timestep = pipe.scheduler.timesteps[aug_timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
        clean_latent_aug = pipe.scheduler.add_noise(clean_latent, noise, aug_timestep)
    else:
        clean_latent_aug = clean_latent
        aug_timestep = None

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    # print(f"debug timestep {timestep},{timestep.shape}")
    noise_pred = pipe.model_fn(
        pipe.config, **models, **inputs, timestep=timestep,
        clean_latents=clean_latent_aug,
        aug_t=aug_timestep,
    )

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def ImageAction2CausalODERegressionLoss(pipe: BasePipeline, **inputs):
    """ODE regression loss for Stage 2 distillation.

    Following Causal-Forcing reparameterization:
    1. Sample a random ODE step k per batch element
    2. Model predicts velocity (flow) from noisy ODE state
    3. Convert to x0: x0_pred = x_t - sigma * flow_pred
    4. Loss = MSE(x0_pred, clean_latent) masked where sigma > eps

    Required inputs:
        ode_latent: [B, K, C, F, H, W] pre-computed ODE trajectory states
        clean_latent: [B, C, F, H, W] clean target latents
        sigmas: [B, K] sigma values at each ODE step
        first_frame_latents: [B, C, 1, H, W] clean first-frame VAE latents
    """
    noise_augmentation_max_timestep = int(inputs.get("noise_augmentation_max_timestep", 10))

    ode_latent = inputs["ode_latent"]    # [B, K, C, F, H, W]
    clean_latent = inputs["clean_latent"]  # [B, C, F, H, W]
    sigmas_all = inputs["sigmas"]        # [B, K] timestep values (0~1000 range)
    B, K = sigmas_all.shape[:2]

    # 1. Sample random ODE step per batch element
    k_idx = torch.randint(0, K, (B,), device=ode_latent.device)

    # 2. Gather noisy input and timestep for each batch element
    noisy_input = ode_latent[torch.arange(B), k_idx]  # [B, C, F, H, W]
    # sigmas_all stores scheduler timestep values (= sigma * 1000), pass directly to model
    timestep = sigmas_all[torch.arange(B), k_idx]      # [B] timestep values
    timestep = timestep.to(dtype=pipe.torch_dtype, device=pipe.device)

    # 3. Set up latents: noisy ODE state with clean first frame
    inputs["latents"] = noisy_input.clone()
    inputs["latents"][:, :, 0:1, :, :] = clean_latent[:, :, 0:1, :, :]

    # 4. Optional noise augmentation on clean TF context
    if noise_augmentation_max_timestep > 0:
        aug_timestep_id = torch.randint(
            int(noise_augmentation_max_timestep / 1000 * len(pipe.scheduler.timesteps)),
            len(pipe.scheduler.timesteps), (1,))
        aug_timestep = pipe.scheduler.timesteps[aug_timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
        # Expand to [B] so it matches per-batch timestep in the model forward
        aug_timestep = aug_timestep.expand(B)
        aug_noise = torch.randn_like(clean_latent)
        clean_latent_aug = pipe.scheduler.add_noise(clean_latent, aug_noise, aug_timestep)
    else:
        clean_latent_aug = clean_latent
        aug_timestep = None

    # 5. Forward pass
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    flow_pred = pipe.model_fn(
        pipe.config, **models, **inputs, timestep=timestep,
        clean_latents=clean_latent_aug,
        aug_t=aug_timestep,
    )

    # 6. Convert flow prediction to x0 prediction: x0 = x_t - sigma * flow
    # Convert timestep values back to actual sigma: sigma = timestep / 1000
    sigma_for_x0 = timestep.float() / pipe.scheduler.num_train_timesteps
    sigma_for_x0 = sigma_for_x0.to(dtype=flow_pred.dtype, device=flow_pred.device)
    # Reshape sigma for broadcasting: [B] -> [B, 1, 1, 1, 1]
    sigma_for_x0 = sigma_for_x0[:, None, None, None, None]
    x0_pred = noisy_input - sigma_for_x0 * flow_pred

    # 7. Mask out near-zero sigma (already clean, no useful gradient)
    mask = (sigma_for_x0 > 1e-6).float()

    # 8. Compute MSE loss on x0 prediction
    loss = torch.nn.functional.mse_loss(
        (x0_pred * mask).float(),
        (clean_latent * mask).float(),
    )
    return loss


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(pipe.config,**models, **inputs, timestep=timestep)
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss

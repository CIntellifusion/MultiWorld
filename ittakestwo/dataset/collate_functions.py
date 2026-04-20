import torch 
def default_collate_fn(batch):
    return batch[0]

def action2video_concatview_collate_fn(batch):
    data = {}
    data['video'] = torch.cat([item['video'] for item in batch], dim=0)  # [B,C,F,H,W]
    # action = {'continouse_action': Tensor[B, action_dim], 'discrete_action': Tensor[B, num_discrete_actions]}
    data['action'] = {}
    data['action']['continuous_action'] = torch.cat([item['action']['continuous_action'] for item in batch], dim=0)
    data['action']['discrete_action'] = torch.cat([item['action']['discrete_action'] for item in batch], dim=0)
    if "left_player_action" in batch[0]['action']: 
        data['action']['left_player_action'] = {}
        data['action']['left_player_action']['continuous_action'] = torch.cat([item['action']['left_player_action']['continuous_action'] for item in batch], dim=0)
        data['action']['left_player_action']['discrete_action'] = torch.cat([item['action']['left_player_action']['discrete_action'] for item in batch], dim=0)
    if "right_player_action" in batch[0]['action']: 
        data['action']['right_player_action'] = {}
        data['action']['right_player_action']['continuous_action'] = torch.cat([item['action']['right_player_action']['continuous_action'] for item in batch], dim=0)
        data['action']['right_player_action']['discrete_action'] = torch.cat([item['action']['right_player_action']['discrete_action'] for item in batch], dim=0)
    
    if "env_obv" in batch[0]:
        data['env_obv'] = torch.cat([item['env_obv'] for item in batch], dim=0)
    data['start_point'] = torch.tensor([item['start_point'] for item in batch], dtype=torch.long)
    data['real_idx'] = torch.tensor([item['real_idx'] for item in batch], dtype=torch.long)
    return data 

def action2video_independent_collate_fn(batch, share_action=True):
    data = {}
    videos = torch.cat([item['video'] for item in batch], dim=0)  # [B, C, F, H, W]
    B, C, F, H, W = videos.shape
    
    # Cut video from W dim to two views: [B, C, F, H, W] -> [2B, C, F, H, W//2]
    # Assume W is even, split into left and right view
    w_half = W // 2
    left_view = videos[..., :w_half]   # [B, C, F, H, W//2]
    right_view = videos[..., w_half:]  # [B, C, F, H, W//2]
    data['video'] = torch.cat([left_view, right_view], dim=0)  # [2B, C, F, H, W//2]
    
    # Handle actions: duplicate for both views
    continuous = torch.cat([item['action']['continuous_action'] for item in batch], dim=0)  # [B, action_dim]
    discrete = torch.cat([item['action']['discrete_action'] for item in batch], dim=0)      # [B, num_discrete_actions]
    
    if share_action:
        # Same action for both views: duplicate
        data['action'] = {
            'continuous_action': continuous.repeat(2, 1, 1),  # [2B, action_dim]
            'discrete_action': discrete.repeat(2, 1, 1)       # [2B, num_discrete_actions]
        }
    else:
        # Independent actions per view (placeholder: duplicate for now)
        data['action'] = {
            'continuous_action': continuous.repeat(2, 1, 1),  # [2B, action_dim]
            'discrete_action': discrete.repeat(2, 1, 1)       # [2B, num_discrete_actions]
        }
    
    # Shuffle within this batch: create random order and permute
    # Total samples = 2B (B left views + B right views)
    total_samples = 2 * B
    perm = torch.randperm(total_samples)
    
    data['video'] = data['video'][perm]
    data['action']['continuous_action'] = data['action']['continuous_action'][perm]
    data['action']['discrete_action'] = data['action']['discrete_action'][perm]
    
    # Metadata: repeat for both views then permute
    start_points = torch.tensor([item['start_point'] for item in batch], dtype=torch.long).repeat(2)
    real_idxs = torch.tensor([item['real_idx'] for item in batch], dtype=torch.long).repeat(2)
    
    data['start_point'] = start_points[perm]
    data['real_idx'] = real_idxs[perm]
    
    return data

def action2video_independent_sharedaction_collate_fn(batch):
    return action2video_independent_collate_fn(batch,share_action=True)

def action2video_independent_sepaction_collate_fn(batch):
    return action2video_independent_collate_fn(batch,share_action=False)

def ode_regression_collate_fn(batch):
    """Collate function for ODE regression dataset.

    Stacks/concatenates ODE latents, clean latents, sigmas, actions, and env_obv.
    """
    data = {}
    data['ode_latent'] = torch.stack([item['ode_latent'] for item in batch], dim=0)  # [B, K, C, F, H, W]
    data['clean_latent'] = torch.stack([item['clean_latent'] for item in batch], dim=0)  # [B, C, F, H, W]
    data['sigmas'] = torch.stack([item['sigmas'] for item in batch], dim=0)  # [B, K]

    # Actions: cat on batch dim (each item has [1, F, 2, D])
    data['action'] = {}
    data['action']['discrete_action'] = torch.cat([item['action']['discrete_action'] for item in batch], dim=0)
    data['action']['continuous_action'] = torch.cat([item['action']['continuous_action'] for item in batch], dim=0)

    # Env observations: stack if present
    if batch[0].get('env_obv') is not None:
        data['env_obv'] = torch.stack([item['env_obv'] for item in batch], dim=0)
    else:
        data['env_obv'] = None

    data['start_point'] = torch.tensor([item['start_point'] for item in batch], dtype=torch.long)
    data['real_idx'] = torch.tensor([item['real_idx'] for item in batch], dtype=torch.long)
    return data

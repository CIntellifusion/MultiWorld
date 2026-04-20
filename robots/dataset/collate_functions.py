import torch 
def default_collate_fn(batch):
    return batch[0]
def action2video_variable_agents_collate_fn(batch):
    data = {}
    # should be tensor; 
    data['video'] = torch.cat([item['video'] for item in batch], dim=0)  # [B,C,F,H,W]
    
    # action = {'continouse_action': Tensor[B, action_dim], 'discrete_action': Tensor[B, num_discrete_actions]}
    action = {}
    action['camera'] = torch.cat([item['action']['camera'] for item in batch], dim=0)
    action['action'] = torch.cat([item['action']['action'] for item in batch], dim=0)
    action['num_agents'] = torch.tensor([item['action']['num_agents'] for item in batch], dtype=torch.long)
    data['action'] = action 
    if "env_obv" in batch[0] and batch[0]['env_obv'] is not None:
        data['env_obv'] =torch.cat([item['env_obv'] for item in batch], dim=0)
    else:
        data['env_obv'] = None
    return data 

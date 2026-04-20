import torch 
from torch import nn 
from diffsynth.models.vggt.models.vggt import VGGT 
from einops import rearrange
class WanEnvEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim = 2048, # defined by VGGT's latent dimension
        output_dim = 3072, # defined by Wan Video's latent dimension 
    ):
        super().__init__()
        self.env_encoder = VGGT.from_pretrained("facebook/VGGT-1B").eval()
        
        del self.env_encoder.camera_head
        del self.env_encoder.point_head
        del self.env_encoder.depth_head
        del self.env_encoder.track_head
        
        print("WanEnvEncoder Initializing WanEnvEncoder with VGGT backbone. Head is deleted")
        self.env_encoder.requires_grad_(False)
        self.connector = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, output_dim),
        ).train()
        num_params = sum(p.numel() for p in self.env_encoder.parameters() if p.requires_grad)
        print(f"WanEnvEncoder VGGT backbone total trainable parameters: {num_params}")
        connector_params = sum(p.numel() for p in self.connector.parameters() if p.requires_grad)
        print(f"WanEnvEncoder connector total trainable parameters: {connector_params}")
    
    def forward(self, images):
        """
        images: [B, F, K, 3, H, W]
        # B: batch size
        # F: number of frames
        # K: number of views
        # H: height
        # W: width
        # 3: RGB channels
        """
        B = images.shape[0] 
        F = images.shape[1]
        # Debug: images shape [B, F, K, C, H, W]
        # print(f"debug env images: {images.shape} {images.min()} {images.max()}")
        images = rearrange(images, "B F K C H W -> (B F) K C H W").contiguous()
        # import pdb; pdb.set_trace() 
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            with torch.amp.autocast(dtype=dtype,device_type='cuda'):
                env_states_list , _  = self.env_encoder.shortcut_forward(images) # ((BF) K 24 N_token L_Dim)
                env_states_list = [torch.mean(env_hidden_states,dim=1) for env_hidden_states in env_states_list] # [(BF) N D]
                env_states = torch.stack(env_states_list, dim=1) 
                # Default: mean pooling on K dim and then L dim
                env_states = env_states.mean(dim=1) # ((BF) 24 N D)
                
        env_states = rearrange(env_states, "(B F) N L  -> (B F N) L",B=B).contiguous()
        env_states = self.connector(env_states) # ( (BFN)  output_dim)
        # print(f"state after connector: {env_states.shape}")
        env_states = rearrange(env_states, "(B F N) L  -> B (F N) L", B=B,F=F).contiguous()
        return env_states

class WanDINOEnvEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim = 2048, # defined by VGGT's latent dimension
        output_dim = 3072, # defined by Wan Video's latent dimension 
    ):
        super().__init__()
        ckpt_path="/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"
        dinov2_repo = "/root/.cache/torch/hub/facebookresearch_dinov2_main"
        import sys 
        if dinov2_repo not in sys.path:
            sys.path.insert(0, dinov2_repo)

        # Local import and instantiate (no network operations)
        
        from dinov2.hub.backbones import dinov2_vitb14
        encoder = dinov2_vitb14(pretrained=False)  # Do not trigger download
        result = encoder.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        # print(result)
        encoder.head = torch.nn.Identity()
        self.env_encoder = encoder.eval()
        print("WanDINOEnvEncoder Initializing WanDINOEnvEncoder with VGGT backbone. Head is deleted")
        self.env_encoder.requires_grad_(False)
        self.connector = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, output_dim),
        ).train()
        num_params = sum(p.numel() for p in self.env_encoder.parameters() if p.requires_grad)
        print(f"WanEnvEncoder DINO backbone total trainable parameters: {num_params}")
        connector_params = sum(p.numel() for p in self.connector.parameters() if p.requires_grad)
        print(f"WanEnvEncoder connector total trainable parameters: {connector_params}")
    def forward(self, images):
        """
        images: [B, F, K, 3, H, W]
        # B: batch size
        # F: number of frames
        # K: number of views
        # H: height
        # W: width
        # 3: RGB channels
        """
        B = images.shape[0] 
        F = images.shape[1]
        K = images.shape[2]
        images = rearrange(images, "B F K C H W -> (B F K) C H W").contiguous()
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast(dtype=dtype,device_type='cuda'):
                env_states =  self.env_encoder(images) # ( (BF) K 24 N_token L_Dim)
                env_states = rearrange(env_states,"(B F K) D -> (B F) K D",B=B,F=F,K=K)
                env_states = torch.mean(env_states,dim=1).contiguous() # (B F) D 
                
        env_states = self.connector(env_states) # (BF)  output_dim)
        env_states = rearrange(env_states, "(B F) D  -> B F D", B=B,F=F).contiguous()
        return env_states
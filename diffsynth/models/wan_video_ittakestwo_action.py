import torch
import torch.nn as nn
from einops import rearrange
from .wan_video_robots_action import SoftmaxAgentPooling

class WanActionEncoder(nn.Module):
    """
    WanActionEncoder: handles dual-player action space, similar to WanUnifiedActionEncoder architecture.

    Input action format (dataset __getitem__ output):
        - discrete_action: [B, F, 2, discrete_dim] - 2 for left/right players
        - continuous_action: [B, F, 2, continuous_dim]

    Processing flow:
        1. Encode discrete actions with nn.Embedding (follow V2 design)
        2. Project continuous actions with nn.Linear
        3. Concatenate along agent dim: [B, 2, F, D]
        4. Agent-wise RoPE pooling
        5. Output projection to output_ratio * output_dim
    """
    def __init__(self, 
                 discrete_dim=10,      # discrete action dim per player
                 continuous_dim=2,     # continuous action dim per player
                 output_dim=1024,      # output feature dim D
                 output_ratio=6,       # output token ratio
                 adaptive_agent_pooling=False,
                 action_pe_config={},
        ):
        super().__init__()
        self.discrete_dim = discrete_dim
        self.continuous_dim = continuous_dim
        self.output_dim = output_dim
        self.output_ratio = output_ratio
        
        # 1. discrete action encoding: independent true/false embedding per token (follow V2)
        self.discrete_embedding = nn.Embedding(discrete_dim * 2, output_dim)
        
        # 2. continuous action projection: from continuous_dim to output_dim
        self.continuous_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(continuous_dim, output_dim)
        )
        
        # Agent-wise pooling with RoPE
        self.agent_pooling = SoftmaxAgentPooling(
            **action_pe_config,
            adaptive_agent_pooling=adaptive_agent_pooling
        )
        
        # final output projection
        self.output_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(output_dim, output_dim * output_ratio)
        )
        
        self.kaiming_init_selective()
    
    def kaiming_init_selective(self):
        """Selective Kaiming initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.kaiming_normal_(param, nonlinearity='linear')
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        # print(f"{self.__class__.__name__} all parameters Kaiming initialized")
    
    def forward(self, action):
        """
        Args:
            action: dict containing:
                - 'discrete_action': [B, F, 2, discrete_dim], values {0, 1}
                - 'continuous_action': [B, F, 2, continuous_dim], values [-1, 1]
            
        Returns:
            action_tokens: shape [B, F, output_ratio * output_dim]
        """
        discrete_action = action['discrete_action']      # [B, F, 2, discrete_dim]
        continuous_action = action['continuous_action']  # [B, F, 2, continuous_dim]
        
        B, F, N, _ = discrete_action.shape
        assert N == 2, f"Expected 2 agents, got {N}"
        
        # ===== 1. Discrete Action Embedding (follow V2 design) =====
        # discrete_action: [B, F, 2, discrete_dim], values {0, 1}
        # compute embedding index for each discrete token
        token_indices = torch.arange(self.discrete_dim, device=discrete_action.device).view(1, 1, 1, -1)  # [1, 1, 1, discrete_dim]
        embedding_indices = token_indices * 2 + discrete_action  # [B, F, 2, discrete_dim]
        discrete_embedded = self.discrete_embedding(embedding_indices)  # [B, F, 2, discrete_dim, output_dim]
        discrete_token = discrete_embedded.sum(dim=3)  # [B, F, 2, output_dim]
        
        # ===== 2. Continuous Action Projection =====
        # Continuous: [B, F, 2, continuous_dim] -> [B, F, 2, output_dim]
        continuous_token = self.continuous_projection(continuous_action)
        
        # ===== 3. Merge discrete and continuous (sum) =====
        action_token = discrete_token + continuous_token  # [B, F, 2, output_dim]
        
        # transpose to the format expected by agent pooling: [B, N, F, D]
        action_token = action_token.permute(0, 2, 1, 3)  # [B, 2, F, output_dim]
        
        # Agent-wise RoPE pooling
        pooled_token = self.agent_pooling(action_token)  # [B, F, output_dim]
        
        # output projection (expand to output_ratio times)
        output = self.output_projection(pooled_token)  # [B, F, output_ratio * output_dim]
        
        return output

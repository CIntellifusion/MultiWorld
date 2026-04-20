import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange    
class WanStateActionEncoderv1(nn.Module):
    def __init__(self, 
                 hidden_dim=48, 
                 input_shape=[80, 16],   # [T, L]
                 output_shape=[21, 16, 20]  # [F', H, W]
                ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        T, L = input_shape
        F_out, H_out, W_out = output_shape
        
        # Design a conv network mapping input to target output shape
        # Input: [B, 1, T, L, 1], Output: [B, hidden_dim, F_out, H_out, W_out]
        
        self.encoder = nn.Sequential(
            nn.Conv3d(1, hidden_dim//2, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim//2, hidden_dim, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            # Use adaptive pooling to adjust output size
            nn.AdaptiveAvgPool3d(output_shape)
        )

    def forward(self, x):
        """
        Args:
            x: [B,N,T,L]
        Returns:
            output: [B, hidden_dim, F', H, W]
        """
        B, T, L = x.shape
        # import pdb; pdb.set_trace()
        assert [T, L] == self.input_shape, f"Input shape mismatch. Expected {self.input_shape}, got {[T,L]}"
        
        # Add channel and width dimensions for 3D conv processing
        x = x.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, L, 1]
        
        out = self.encoder(x)  # [B, hidden_dim, F', H, W]
        return out

class SelfAttentionPooling(nn.Module):
    def __init__(self, feature_dim=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=1,
            batch_first=True
        )
    def forward(self, x):  # x: [b, n, f, 8]
        b, n, f, d = x.shape
        
        # Reshape to [b*f, n, d]
        x_reshaped = x.permute(0, 2, 1, 3).reshape(b*f, n, d)
        
        # Use mean vector as query for self-attention
        query = x_reshaped.mean(dim=1, keepdim=True)  # [b*f, 1, d]
        attn_output, attn_weights = self.attention(
            query, x_reshaped, x_reshaped
        )
        # Reshape output back to [b, f, 8]
        output = attn_output.reshape(b, f, d)
        return output
    
class WanStateActionEncoder(nn.Module):
    def __init__(self, 
                 action_dim=8,
                 token_num_perframe = 80 ,
                 output_dim = 1024,
        ):
        super().__init__()
        self.agent_pooling = SelfAttentionPooling(feature_dim=action_dim)
        self.output_dim = output_dim
        self.token_num_perframe = token_num_perframe
        assert output_dim % 6 ==0, "output_dim should be divisible by 6"
        self.action2token =  nn.Sequential(
            nn.SiLU(),
            nn.Linear(action_dim, token_num_perframe, bias=True)
        )
        # AdaLn-zero
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(token_num_perframe, output_dim, bias=True)
        )
        self.zero_init_selective()
        
    def zero_init_selective(self):
        """Selective zero initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.constant_(param, 0.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        print("WanStateActionEncoder all parameters zero initialized")
             
    def forward(self,x):
        # input shape: B N f 8 
        B,N,f,d = x.shape
        global_action = self.agent_pooling(x)  # B,f,8
        global_action_flatten = rearrange(global_action, 'b f d -> (b f) d') 
        global_action_tokens = self.action2token(global_action_flatten)  # (B*f),(h w)
        action_embedding = self.adaLN_modulation(global_action_tokens)  # (B*f), (6*D)
        # output shape: B, (f,h,w) , 6 , D 
        D = self.output_dim // 6 
        action_embedding = rearrange(action_embedding, '(b f) d  -> b f d', b=B)
        action_embedding = action_embedding.unflatten(2, (6, D))  # B, f, 6, D
        # duplicate on h,w dimension
        action_embedding = action_embedding.unsqueeze(2)
        action_embedding = action_embedding.repeat(1, 1, self.token_num_perframe, 1 , 1)  # B, f, 6, h*w/6, D
        action_embedding = rearrange(action_embedding, 'b f p c d -> b (f p) c d')
        return action_embedding  # B, (f,h,w) , 6 , D
        
class WanStateActionEncoderV3(nn.Module):
    def __init__(self, 
                 action_dim=8,
                 output_dim = 1024,
        ):
        super().__init__()
        self.agent_pooling = SelfAttentionPooling(feature_dim=action_dim)
        self.output_dim = output_dim
        self.action2token =  nn.Sequential(
            nn.SiLU(),
            nn.Linear(action_dim, output_dim, bias=True)
        )
        self.action_projection =  nn.Sequential(
            nn.SiLU(), nn.Linear(output_dim, output_dim * 6)
        )
        self.zero_init_selective()
        
    def zero_init_selective(self):
        """Selective zero initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.constant_(param, 0.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        print("WanStateActionEncoderV3 all parameters zero initialized")
        
    def forward(self,x):
        # input shape: B N f 8 
        B,N,f,d = x.shape
        global_action = self.agent_pooling(x)  # B,f,8
        global_action_flatten = rearrange(global_action, 'b f d -> (b f) d') 
        global_action_tokens = self.action2token(global_action_flatten)  # (B*f), D
        action_embedding = self.action_projection(global_action_tokens)  # (B*f), (6*D)
        # output shape: B, f , 6 , D 
        D = self.output_dim
        action_embedding = rearrange(action_embedding, '(b f) d  -> b f d', b=B)
        action_embedding = action_embedding.unflatten(2, (6, D))  # B, f, 6, D
        return action_embedding  # B, f , 6 , D

import math

class RelativePositionEmbedding(nn.Module):
    """Relative position embedding."""
    def __init__(self, max_positions=100, feature_dim=8, num_heads=1):
        super().__init__()
        self.max_positions = max_positions
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Learnable relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * max_positions - 1, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # Generate relative position indices
        coords = torch.arange(max_positions)
        relative_coords = coords[:, None] - coords[None, :]  # [max_positions, max_positions]
        relative_coords += max_positions - 1  # Shift to non-negative indices
        self.register_buffer('relative_position_index', relative_coords)
        
    def forward(self, seq_len):
        """Get relative position bias matrix [num_heads, seq_len, seq_len]."""
        relative_position_index = self.relative_position_index[:seq_len, :seq_len]
        bias = self.relative_position_bias_table[relative_position_index.view(-1)]
        bias = bias.view(seq_len, seq_len, self.num_heads).permute(2, 0, 1)  # [num_heads, seq_len, seq_len]
        return bias

class SelfAttentionWithRelativePE(nn.Module):
    """Self-attention with relative position encoding."""
    def __init__(self, embed_dim, num_heads=1, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Q, K, V linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Relative position encoding
        self.relative_pe = RelativePositionEmbedding(
            max_positions=100,
            feature_dim=embed_dim,
            num_heads=num_heads
        )
        
    def forward(self, query, key, value):
        """
        Args:
            query: [batch_size, query_len, embed_dim]
            key: [batch_size, key_len, embed_dim]
            value: [batch_size, value_len, embed_dim]
        """
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]
        
        # Linear projections
        q = self.q_proj(query)  # [batch_size, query_len, embed_dim]
        k = self.k_proj(key)    # [batch_size, key_len, embed_dim]
        v = self.v_proj(value)  # [batch_size, value_len, embed_dim]
        
        # Reshape to multi-head format [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add relative position bias
        if query_len == key_len:  # Only for self-attention
            relative_bias = self.relative_pe(query_len)  # [num_heads, query_len, key_len]
            attn_scores = attn_scores + relative_bias.unsqueeze(0)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Weighted sum
        attn_output = torch.matmul(attn_weights, v)
        
        # Restore shape [batch_size, query_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.embed_dim
        )
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights

class SelfAttentionPoolingWithPE(nn.Module):
    """Self-attention pooling with relative position encoding."""
    def __init__(self, feature_dim=8, use_relative_pe=True, max_positions=100):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_relative_pe = use_relative_pe
        
        if use_relative_pe:
            # Use custom attention with relative position encoding
            self.attention = SelfAttentionWithRelativePE(
                embed_dim=feature_dim,
                num_heads=1
            )
        else:
            # Use standard attention
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=1,
                batch_first=True
            )
            
    def forward(self, x):
        """x: [batch_size, num_agents, num_frames, feature_dim]"""
        b, n, f, d = x.shape
        
        # Reshape to [batch_size * num_frames, num_agents, feature_dim]
        x_reshaped = x.permute(0, 2, 1, 3).reshape(b * f, n, d)
        
        # Use mean vector as query
        query = x_reshaped.mean(dim=1, keepdim=True)  # [b*f, 1, d]
        
        # Self-attention computation
        attn_output, attn_weights = self.attention(
            query, x_reshaped, x_reshaped
        )
        
        # Reshape output back to [batch_size, num_frames, feature_dim]
        output = attn_output.reshape(b, f, d)
        return output

class WanCameraEncoder(nn.Module):
    """
    Camera Controller Module: 
    input: camera pose sequence: B,T,16
    output: B,T
    """
    def __init__(self, 
                 output_dim=1024, 
                 camera_dim=16,   # [T, L]
                ):
        super().__init__()
        self.output_dim = output_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(camera_dim, output_dim//4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim//4, output_dim//2),
            nn.ReLU(),
            nn.Linear(output_dim//2, output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [B, T, L]
        Returns:
            output: [b nv t d]
        """
        if x.dim() == 4:
            B, num_view, T, L = x.shape
            x = rearrange(x, 'b nv t l -> (b nv t) l')
        else:
            num_view = 1 
            B, T, L = x.shape
            x = rearrange(x, 'b t l -> (b t) l')
        out = self.encoder(x)  # [B, hidden_dim]
        out = rearrange(out, '(b nv t) d -> b nv t d', b=B, nv=num_view, t=T)  # [B, num_view, hidden_dim, T]
        return out

import torch
import torch.nn as nn
from einops import rearrange

class MultiAgentActionRoPE2D(nn.Module):
    """
    2D RoPE for Multi-Agent Action: [B, N, F, D]
    - Dimension split: N gets 1/4 (256d), F gets 3/4 (768d)
    """
    def __init__(self, dim=1024, max_n=8, max_f=252, base_n=2, base_f=10000.0):
        super().__init__()
        self.dim = dim
        self.dim_n = dim // 4      # 256 for N-axis subspace
        self.dim_f = dim - self.dim_n  # 768 for F-axis subspace
        # base_n: theta in llama2 10000
        # max_n: end in llama2  4096
        assert self.dim_n % 2 == 0 and self.dim_f % 2 == 0, "dimensions must be even"
        
        # ========== N-axis precompute (base=1.59, period ~6.28~10) ==========
        freqs_n = 1.0 / (base_n ** (torch.arange(0, self.dim_n, 2).float() / self.dim_n))
        pos_n = torch.arange(max_n, dtype=torch.float32)
        freqs_n = torch.outer(pos_n, freqs_n)  # [max_n, dim_n/2]
        self.register_buffer('freqs_cos_n', torch.cos(freqs_n))
        self.register_buffer('freqs_sin_n', torch.sin(freqs_n))
        
        # ========== F-axis precompute (base=10000, period ~6.28~62832) ==========
        freqs_f = 1.0 / (base_f ** (torch.arange(0, self.dim_f, 2).float() / self.dim_f))
        pos_f = torch.arange(max_f, dtype=torch.float32)
        freqs_f = torch.outer(pos_f, freqs_f)  # [max_f, dim_f/2]
        self.register_buffer('freqs_cos_f', torch.cos(freqs_f))
        self.register_buffer('freqs_sin_f', torch.sin(freqs_f))
    
    def rotate_half(self, x):
        """[-x2, x1, -x4, x3, ...]"""
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)
    
    def apply_1d_rope(self, x, cos, sin):
        """
        x: [..., seq_len, dim]
        cos/sin: [seq_len, dim/2]
        """
        # Broadcast to match x dimensions: [..., seq_len, dim/2]
        cos = cos.view(*((1,) * (x.ndim - 2)), cos.shape[0], cos.shape[1])
        sin = sin.view(*((1,) * (x.ndim - 2)), sin.shape[0], sin.shape[1])
        
        # Repeat to [..., seq_len, dim]
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        
        return x * cos + self.rotate_half(x) * sin
    
    def forward(self, x, n=None, f=None):
        """
        x: [B, N, F, 1024]
        n: actual agent count (default N)
        f: actual frame count (default F)
        """
        B, N, F, D = x.shape
        n = n or N
        f = f or F
        
        # 1. Split dimensions 1:3
        x_n = x[..., :self.dim_n]   # [B, N, F, 256]  - Agent subspace
        x_f = x[..., self.dim_n:]   # [B, N, F, 768]  - Frame subspace
        
        # 2. N-axis RoPE (short period): [B,N,F,D] -> [B*F, N, D] -> apply -> back
        x_n = rearrange(x_n, 'b n f d -> (b f) n d')
        x_n = self.apply_1d_rope(x_n, self.freqs_cos_n[:n], self.freqs_sin_n[:n])
        x_n = rearrange(x_n, '(b f) n d -> b n f d', b=B, f=F)
        
        # 3. F-axis RoPE (long period): [B,N,F,D] -> [B*N, F, D] -> apply -> back  
        x_f = rearrange(x_f, 'b n f d -> (b n) f d')
        x_f = self.apply_1d_rope(x_f, self.freqs_cos_f[:f], self.freqs_sin_f[:f])
        x_f = rearrange(x_f, '(b n) f d -> b n f d', b=B, n=N)
        
        # 4. Concatenate back to [B, N, F, 1024]
        return torch.cat([x_n, x_f], dim=-1)

class MultiAgentActionRoPE1D(nn.Module):
    """
    1D RoPE for Multi-Agent Action: [B, N, F, D]
    """
    def __init__(self, dim=1024, max_n=8, base_n=2):
        super().__init__()
        """
        dim: latent dimension for each feature, split into two subspaces in RoPE
        max_n: number of agents
        base_n: RoPE base
        """
        self.dim = dim
        freqs_n = 1.0 / (base_n ** (torch.arange(0, self.dim, 2).float() / self.dim))
        pos_n = torch.arange(max_n, dtype=torch.float32)
        freqs_n = torch.outer(pos_n, freqs_n)  # [max_n, dim_n/2]
        self.register_buffer('freqs_cos_n', torch.cos(freqs_n))
        self.register_buffer('freqs_sin_n', torch.sin(freqs_n))
        
    
    def rotate_half(self, x):
        """[-x2, x1, -x4, x3, ...]"""
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)
    
    def apply_1d_rope(self, x, cos, sin):
        """
        x: [..., seq_len, dim]
        cos/sin: [seq_len, dim/2]
        """
        # Broadcast to match x dimensions: [..., seq_len, dim/2]
        cos = cos.view(*((1,) * (x.ndim - 2)), cos.shape[0], cos.shape[1])
        sin = sin.view(*((1,) * (x.ndim - 2)), sin.shape[0], sin.shape[1])
        
        # Repeat to [..., seq_len, dim]
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        
        return x * cos + self.rotate_half(x) * sin
    
    def forward(self, x, n=None, f=None):
        """
        x: [B, N, F, 1024]
        n: actual agent count (default N)
        f: actual frame count (default F)
        """
        B, N, F, D = x.shape
        n = n or N
        f = f or F
        
        # N-axis RoPE (short period): [B,N,F,D] -> [B*F, N, D] -> apply -> back
        x = rearrange(x, 'b n f d -> (b f) n d')
        x = self.apply_1d_rope(x, self.freqs_cos_n[:n], self.freqs_sin_n[:n])
        x = rearrange(x, '(b f) n d -> b n f d', b=B, f=F)
        
        return x

class AgentWiseSelfAttention(nn.Module):
    """
    Self-attention only on N dimension (among Agents), F dimension remains independent.
    - Input: [B, N, F, D]
    - Agents within each frame are mutually visible (bidirectional)
    - No interaction between different frames (avoids temporal bidirectionality issues)
    - Uses 2D RoPE (N short period + F long period)
    
    Uses F.scaled_dot_product_attention + einops to avoid dimension confusion and CuBLAS warnings.
    """
    def __init__(self, dim=1024, num_heads=8, max_n=8, max_f=252, base_n=2, pe_type='relative'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 2D RoPE (N short period, F long period)
        if pe_type == 'relative':
            self.rope = MultiAgentActionRoPE2D(
                dim=dim, 
                max_n=max_n, 
                max_f=max_f,
                base_n=base_n
            )
        elif pe_type == 'relative1d':
            self.rope = MultiAgentActionRoPE1D(
                dim=dim, 
                base_n=base_n
            )
        elif pe_type == "identity":
            self.rope = nn.Identity()
        else:
            raise NotImplementedError(f"pe_type {pe_type} not implemented")
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
    def forward(self, x):
        """
        x: [B, N, F, D]
        """
        B, N, f, D = x.shape
        
        # 1. RoPE encoding
        x = self.rope(x)  # [B, N, F, D]
        
        # 2. Project and split heads -> [B, N, F, 3, heads, head_dim]
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n f (three h d) -> three b n f h d', 
                    three=3, h=self.num_heads, d=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is [B, N, F, heads, head_dim]
        
        # 3. Merge batch and frame dimensions for agent-wise attention
        # [B, N, F, heads, head_dim] -> [B*F, heads, N, head_dim]
        q = rearrange(q, 'b n f h d -> (b f) h n d')
        k = rearrange(k, 'b n f h d -> (b f) h n d')
        v = rearrange(v, 'b n f h d -> (b f) h n d')
        
        # 4. Use F.scaled_dot_product_attention (deterministic, no CuBLAS warnings)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        # out: [B*F, heads, N, head_dim]
        
        # 5. Restore dimensions -> [B, N, F, D]
        out = rearrange(out, '(b f) h n d -> b n f (h d)', b=B, f=f)
        
        return self.out_proj(out)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight

class AgentWiseSelfAttentionV2(nn.Module):
    """
    Self-attention only on N dimension (among Agents), F dimension remains independent.
    - Input: [B, N, F, D]
    - Agents within each frame are mutually visible (bidirectional)
    - No interaction between different frames (avoids temporal bidirectionality issues)
    - Uses 2D RoPE (N short period + F long period)
    
    Uses F.scaled_dot_product_attention + einops to avoid dimension confusion and CuBLAS warnings.
    """
    def __init__(self, dim=1024, num_heads=8, max_n=8, max_f=252, base_n=2, pe_type='relative'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 2D RoPE (N short period, F long period)
        if pe_type == 'relative':
            self.rope = MultiAgentActionRoPE2D(
                dim=dim, 
                max_n=max_n, 
                max_f=max_f,
                base_n=base_n
            )
        elif pe_type == 'relative1d':
            self.rope = MultiAgentActionRoPE1D(
                dim=dim, 
                base_n=base_n
            )
        elif pe_type == "identity":
            self.rope = nn.Identity()
        else:
            raise NotImplementedError(f"pe_type {pe_type} not implemented")
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        eps=1e-6
        self.norm_q = RMSNorm(dim, eps)
        self.norm_k = RMSNorm(dim, eps)
    def forward(self, x):
        """
        x: [B, N, F, D]
        """
        B, N, f, D = x.shape
        
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        
        q = self.rope(q)
        k = self.rope(k)
        
        # 3. Merge batch and frame dimensions for agent-wise attention
        # [B, N, F, heads, head_dim] -> [B*F, heads, N, head_dim]
        q = rearrange(q, 'b n f (h d) -> (b f) h n d',h=self.num_heads, d=self.head_dim)
        k = rearrange(k, 'b n f (h d) -> (b f) h n d',h=self.num_heads, d=self.head_dim)
        v = rearrange(v, 'b n f (h d) -> (b f) h n d',h=self.num_heads, d=self.head_dim)
        
        # 4. Use F.scaled_dot_product_attention (deterministic, no CuBLAS warnings)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        # out: [B*F, heads, N, head_dim]
        
        # 5. Restore dimensions -> [B, N, F, D]
        out = rearrange(out, '(b f) h n d -> b n f (h d)', b=B, f=f)
        out = self.out_proj(out)
        return out 
class SoftmaxAgentPooling(nn.Module):
    """
    Weighted pooling via attention weights.
    Input: [B, N, F, D]
    Output: [B, F, D] (single representation) or [B, K, F, D]
    """
    def __init__(self, dim=1024, num_heads=8, max_n=8, base_n=2,  max_f=252,pe_type='relative',adaptive_agent_pooling=True,qk_norm=False):
        super().__init__()
        
        # N-dim Self-Attention
        if qk_norm: 
            self.agent_attn = AgentWiseSelfAttentionV2(dim, num_heads, max_n, max_f,base_n=base_n,pe_type=pe_type)
        else:
            self.agent_attn = AgentWiseSelfAttention(dim, num_heads, max_n, max_f,base_n=base_n,pe_type=pe_type)
            
        self.adaptive_agent_pooling = adaptive_agent_pooling 
        if adaptive_agent_pooling: 
            # Project to single weight: [D] -> 1
            self.weight_proj = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, 1)
            )
    def forward(self, x):
        B, N, F, D = x.shape
        
        # 1. RoPE + N-dim Self-Attention
        x = self.agent_attn(x)  # [B, N, F, D]
        
        # 2. Transpose to [B, F, N, D] for pooling
        x = x.permute(0, 2, 1, 3)  # [B, F, N, D]
        
        # 3. Compute weights [B, F, N, 1] or [B, F, N, K]
        if self.adaptive_agent_pooling: 
            logits = self.weight_proj(x)  # [B, F, N, 1] or [B, F, N, K]
            weights = torch.softmax(logits, dim=2)  # Normalize over N dimension
            # [B, F, N, D] * [B, F, N, 1] -> sum -> [B, F, D]
            pooled = (x * weights).sum(dim=2)
        else:
            pooled = x.sum(dim=2)
        return pooled
class WanUnifiedActionEncoder(nn.Module):
    def __init__(self, 
                 action_dim=8,
                 camera_dim=16,
                 output_dim=1024,
                 output_ratio=6,
                 adaptive_agent_pooling=False,
                 action_pe_config={}):
        super().__init__()

        self.output_dim = output_dim
        
        self.camera2token = WanCameraEncoder(
            output_dim=output_dim,
            camera_dim=camera_dim,
        )
        
        self.action2token = nn.Sequential(
            nn.SiLU(),
            nn.Linear(action_dim, output_dim, bias=True)
        )
        
        self.agent_pooling = SoftmaxAgentPooling(
            **action_pe_config,
            adaptive_agent_pooling=adaptive_agent_pooling
        ) 
    
        self.action_projection = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(output_dim, output_dim * output_ratio)
        )
        
        self.output_ratio = output_ratio
        self.kaiming_init_selective()
        
    def kaiming_init_selective(self):
        """Selective Kaiming initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.kaiming_normal_(param, nonlinearity='linear')
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        print("WanUnifiedActionEncoder all parameters Kaiming initialized")

    def forward(self,action_with_camera, action=None,camera=None):
        """
        Args:
            action_with_camera: dict containing 'action' and 'camera' keys
            num_agent: number of agents per sample, used to mask out extra agents
            action: [B, N, f, 8] input tensor
            camera: [B, N, f, 8] optional camera tensor
            agent_ids: optional agent identifiers
        """
        if action_with_camera is not None:
            action = action_with_camera['action']
            B, N, f, d = action.shape
            camera = action_with_camera['camera']
            num_agents = action_with_camera.get('num_agents', [N]*B)
        action_flatten = rearrange(action, 'b n f d -> (b f n) d')
        action_token = self.action2token(action_flatten)  # (B*f*n), D 
        action_token = rearrange(action_token, '(b f n) d -> b n f d', b=B, f=f, n=N)  # [B, N, f, D]
        pooled_action = [self.agent_pooling(a[:,:n_agent]) for n_agent,a in zip(num_agents,action_token.split(1, dim=0))]  
        pooled_action = torch.concat(pooled_action, dim=0)  # [B, f, D]
        action_embedding_reduction = self.action_projection(pooled_action)  # B,f, (6*D)
        
        # Process camera tokens
        camera_token = self.camera2token(camera)  # [B, nv, f, D]
        camera_token_reduction = camera_token.mean(dim=1)  # [B, f, D] 
        
        unified_embedding = camera_token_reduction + action_embedding_reduction
        return unified_embedding


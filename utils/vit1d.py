"""
Spatial-Temporal Vision Transformer for 12-lead ECG.

Each lead is divided into N_time non-overlapping temporal patches.
Total patches N = num_leads × N_time.
Learnable lead and temporal position embeddings are added to each patch.
The encoder returns all patch-level embeddings for fine-grained alignment.
"""
import math
import torch
import torch.nn as nn
from einops import rearrange


class DropPath(nn.Module):
    def __init__(self, drop_prob: float, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob <= 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, drop_out_rate=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_out_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, heads: int = 8,
                 dim_head: int = 64, qkv_bias: bool = True,
                 drop_out_rate: float = 0., attn_drop_out_rate: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == input_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_drop_out_rate)
        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=qkv_bias)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, output_dim),
                                    nn.Dropout(drop_out_rate)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int,
                 heads: int = 8, dim_head: int = 32, qkv_bias: bool = True,
                 drop_out_rate: float = 0., attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.):
        super().__init__()
        attn = Attention(input_dim=input_dim, output_dim=output_dim, heads=heads,
                         dim_head=dim_head, qkv_bias=qkv_bias,
                         drop_out_rate=drop_out_rate, attn_drop_out_rate=attn_drop_out_rate)
        self.attn = PreNorm(dim=input_dim, fn=attn)
        self.droppath1 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        ff = FeedForward(input_dim=output_dim, output_dim=output_dim,
                         hidden_dim=hidden_dim, drop_out_rate=drop_out_rate)
        self.ff = PreNorm(dim=output_dim, fn=ff)
        self.droppath2 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.droppath1(self.attn(x)) + x
        x = self.droppath2(self.ff(x)) + x
        return x


class SpatialTemporalViT(nn.Module):
    """
    Spatial-Temporal ViT for 12-lead ECG.

    Input:  (B, num_leads, seq_len)
    Output: all patch embeddings (B, N, width) where N = num_leads × N_time
    """
    def __init__(self,
                 num_leads: int = 12,
                 seq_len: int = 5000,
                 patch_size: int = 500,   # temporal patch size per lead
                 width: int = 768,
                 depth: int = 12,
                 mlp_dim: int = 3072,
                 heads: int = 12,
                 dim_head: int = 64,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 **kwargs):
        super().__init__()
        assert seq_len % patch_size == 0, 'seq_len must be divisible by patch_size'

        self.num_leads = num_leads
        self.num_time_patches = seq_len // patch_size   # N_time
        self.num_patches = num_leads * self.num_time_patches  # N = L × N_time
        self.patch_size = patch_size
        self.width = width
        self.depth = depth

        # Linear patch embedding: project each (patch_size,) patch to (width,)
        self.to_patch_embedding = nn.Linear(patch_size, width)

        # Learnable spatial (lead) and temporal position embeddings
        self.lead_embedding = nn.Parameter(torch.randn(num_leads, width) * 0.02)
        self.time_embedding = nn.Parameter(
            torch.randn(self.num_time_patches, width) * 0.02)

        self.dropout = nn.Dropout(drop_out_rate)

        # Build transformer blocks
        self.depth = depth
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            block = TransformerBlock(input_dim=width, output_dim=width,
                                     hidden_dim=mlp_dim, heads=heads,
                                     dim_head=dim_head, qkv_bias=qkv_bias,
                                     drop_out_rate=drop_out_rate,
                                     attn_drop_out_rate=attn_drop_out_rate,
                                     drop_path_rate=drop_path_rates[i])
            self.add_module(f'block{i}', block)

        self.norm = nn.LayerNorm(width)
        # Classification head (Identity by default; replaced in linear probing)
        self.head = nn.Identity()

        self._build_position_indices()

    def _build_position_indices(self):
        """Pre-compute lead and time indices for all N patches."""
        L, T = self.num_leads, self.num_time_patches
        # patch p = l * T + t  →  lead l, time position t
        lead_idx = torch.arange(L).unsqueeze(1).expand(L, T).reshape(-1)  # (N,)
        time_idx = torch.arange(T).unsqueeze(0).expand(L, T).reshape(-1)  # (N,)
        self.register_buffer('lead_idx', lead_idx)
        self.register_buffer('time_idx', time_idx)

    def forward_patches(self, series: torch.Tensor) -> torch.Tensor:
        """
        Encode ECG and return all patch embeddings.

        Args:
            series: (B, num_leads, seq_len)
        Returns:
            H: (B, N, width) — fine-grained patch embeddings
        """
        B, L, T = series.shape
        N_t = self.num_time_patches

        # Reshape into patches: (B, L, N_t, patch_size)
        x = series.reshape(B, L, N_t, self.patch_size)
        # Flatten lead and time dims: (B, N, patch_size)
        x = x.reshape(B, L * N_t, self.patch_size)

        # Linear projection: (B, N, width)
        x = self.to_patch_embedding(x)

        # Add lead + temporal embeddings
        lead_emb = self.lead_embedding[self.lead_idx]   # (N, width)
        time_emb = self.time_embedding[self.time_idx]   # (N, width)
        x = x + lead_emb.unsqueeze(0) + time_emb.unsqueeze(0)  # (B, N, width)

        x = self.dropout(x)

        for i in range(self.depth):
            x = getattr(self, f'block{i}')(x)

        return self.norm(x)  # (B, N, width)

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        """
        Returns global ECG embedding via mean pooling over patches.
        Used for zero-shot classification and linear probing.
        """
        x = self.forward_patches(series)   # (B, N, width)
        x = x.mean(dim=1)                  # (B, width) — global average pool
        return self.head(x)

    def reset_head(self, num_classes: int = 1):
        del self.head
        self.head = nn.Linear(self.width, num_classes)


# ---------------------------------------------------------------------------
# Factory functions — N = num_leads × (seq_len // patch_size) total patches
# Default: 12 leads × (5000 // 500) = 12 × 10 = 120 patches
# ---------------------------------------------------------------------------

def spatial_temporal_vit_tiny(num_leads=12, seq_len=5000, patch_size=500, **kwargs):
    return SpatialTemporalViT(num_leads=num_leads, seq_len=seq_len, patch_size=patch_size,
                              width=192, depth=12, heads=3, mlp_dim=768, **kwargs)


def spatial_temporal_vit_small(num_leads=12, seq_len=5000, patch_size=500, **kwargs):
    return SpatialTemporalViT(num_leads=num_leads, seq_len=seq_len, patch_size=patch_size,
                              width=384, depth=12, heads=6, mlp_dim=1536, **kwargs)


def spatial_temporal_vit_middle(num_leads=12, seq_len=5000, patch_size=500, **kwargs):
    return SpatialTemporalViT(num_leads=num_leads, seq_len=seq_len, patch_size=patch_size,
                              width=512, depth=12, heads=8, mlp_dim=2048, **kwargs)


def spatial_temporal_vit_base(num_leads=12, seq_len=5000, patch_size=500, **kwargs):
    return SpatialTemporalViT(num_leads=num_leads, seq_len=seq_len, patch_size=patch_size,
                              width=768, depth=12, heads=12, mlp_dim=3072, **kwargs)

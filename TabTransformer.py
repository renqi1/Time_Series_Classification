# paper: https://arxiv.org/abs/2012.06678

import torch
import torch.nn.functional as F
from torch import nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self,
                 dim,   # input token dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Transformer(nn.Module):
    def __init__(self, n_features, depth=6, dim=64, num_heads=8, attn_dropout=0.1, ff_dropout=0.1):
        super().__init__()
        self.embeds = nn.Linear(n_features, dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim = dim, num_heads = num_heads, attn_drop_ratio = attn_dropout))),
                Residual(PreNorm(dim, FeedForward(dim, dropout = ff_dropout))),
            ]))

    def forward(self, x):
        x = self.embeds(x)
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)
            if is_last:
                continue
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class TabTransformer(nn.Module):
    def __init__(self, seq_len, n_features, n_classes, depth=6, dim=64, num_heads=8, attn_dropout=0.1, ff_dropout=0.1, hidden_dimensions=[128, 64]):
        super().__init__()

        self.transformer = Transformer(n_features, depth, dim, num_heads, attn_dropout, ff_dropout)
        self.norm = nn.LayerNorm(seq_len)
        all_dimensions = [seq_len, *hidden_dimensions, n_classes]
        self.downsample = nn.Linear(hidden_dimensions[-1]+n_features, 1)
        self.mlp = MLP(all_dimensions)

    def forward(self, x):
        # input x: [B, F, T],  where B = Batch size, F = features, T = Time sampels
        x1 = x.permute(0, 2, 1)     # [B, T, F]
        x1 = self.transformer(x1)
        x2 = self.norm(x)
        x2 = x2.permute(0, 2, 1)    # [B, T, F]
        x = torch.cat((x1, x2), dim=2)
        x = self.downsample(x).squeeze(-1)
        x = self.mlp(x)
        return x

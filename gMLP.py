# paper: https://arxiv.org/abs/2105.08050

from torch import nn
from torch.nn import functional as F

# not only get the information in sequence spatial location information
# but also get the information between channels
class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(seq_len//2)
        self.spatial_proj = nn.Conv1d(d_ffn, d_ffn, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)
        nn.init.normal_(self.spatial_proj.weight, std=1e-6)

    def forward(self, x):
        # x must be divided two value with the same shape
        if x.shape[-1] % 2 == 1:
            x = x[:, :, :-1]
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u * v
        return out


class gMLPBlock(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(seq_len)
        self.channel_proj1 = nn.Linear(seq_len, seq_len * 2)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len * 2)
        self.channel_proj2 = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        return out


class gMLP(nn.Module):
    def __init__(self, n_features, n_classes, seq_len, d_model=256, num_layers=6):
        super().__init__()
        self.patcher = nn.Conv1d(n_features, d_model, kernel_size=1, stride=1)
        self.model = nn.Sequential(
            *[gMLPBlock(d_model, seq_len) for _ in range(num_layers)]
        )
        self.classifier = nn.Sequential(              # define classifier depends on your task
            nn.Linear(d_model * seq_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes)
        )
    def forward(self, x):
        # input x: [B, F, T],  where B = Batch size, F = features, T = Time sampels
        x = self.patcher(x)
        x = self.model(x)
        x = x.reshape(x.size(0), -1)
        out = self.classifier(x)
        return out

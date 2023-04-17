# paper: https://arxiv.org/abs/1910.13051
# code: https://github.com/angus924/rocket

# Convolution layer is randomly generated, so I suggest you setting random seed

import torch
import torch.nn as nn
import numpy as np

class Rocket(nn.Module):
    def __init__(self, n_features, seq_len, n_classes, n_kernels=100, kss=[7, 9, 11]):
        super(Rocket, self).__init__()
        kss = [ks for ks in kss if ks < seq_len]
        convs = nn.ModuleList()
        for i in range(n_kernels):
            ks = np.random.choice(kss)
            dilation = 2**np.random.uniform(0, np.log2((seq_len - 1) // (ks - 1)))
            padding = int((ks - 1) * dilation // 2) if np.random.randint(2) == 1 else 0
            weight = torch.randn(1, n_features, ks)
            weight -= weight.mean()
            bias = 2 * (torch.rand(1) - .5)
            layer = nn.Conv1d(n_features, 1, ks, padding=2 * padding, dilation=int(dilation), bias=True)
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)
            convs.append(layer)
        self.convs = convs
        self.n_kernels = n_kernels
        self.feature_dim = 2 * n_kernels
        self.fc = nn.Linear(self.feature_dim, n_classes)

    def forward(self, x):
        # input x: [B, F, T],  where B = Batch size, F = features, T = Time sampels,
        _output = []
        for i in range(self.n_kernels):
            out = self.convs[i](x).cpu()
            _max = out.max(dim=-1)[0]
            _ppv = torch.gt(out, 0).sum(dim=-1).float() / out.shape[-1]
            _output.append(_max)
            _output.append(_ppv)
        output = torch.cat(_output, dim=1).cuda()      # [batch_size, feature_dim]
        output = self.fc(output)
        return output





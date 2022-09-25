# paper: https://hal.inria.fr/hal-03469487/document
# code: https://github.com/XAIseries/XCM

import torch
import torch.nn as nn

class XCM(nn.Module):
    def __init__(self, seq_len, n_features, n_classes, hidden_features=128, window_perc=1):

        window_size = int(round(seq_len * window_perc, 0))
        self.conv2dblock = nn.Sequential(
            nn.Conv2d(1, hidden_features, kernel_size=(1, window_size), padding='same'),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU()
        )
        self.conv2d1x1block = nn.Sequential(
            nn.Conv2d(hidden_features, 1, kernel_size=1),
            nn.ReLU(),
        )
        self.conv1dblock = nn.Sequential(
            nn.Conv1d(n_features, hidden_features, kernel_size=window_size, padding='same'),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU()
        )
        self.conv1d1x1block = nn.Sequential(
            nn.Conv1d(hidden_features, 1, kernel_size=1),
            nn.ReLU()
        )
        self.conv1d = nn.Sequential(
            nn.Conv1d(n_features + 1, hidden_features, kernel_size=window_size, padding='same'),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU()
        )
        self.linear = nn.Linear(hidden_features, n_classes)

    def forward(self, x):
        # input x : [B, F, T], where B = Batch size, F = features,  T = Time series
        x1 = x.unsqueeze(1)             # [B, 1, F, T]
        x1 = self.conv2dblock(x1)       # [B, hidden_F, F, T]
        x1 = self.conv2d1x1block(x1)    # [B, 1, F, T]
        x1 = x1.squeeze(1)              # [B, F, T]

        x2 = self.conv1dblock(x)        # [B, hidden_F, T]
        x2 = self.conv1d1x1block(x2)    # [B, 1, T]

        x = torch.cat((x2, x1), 1)      # [B, F+1, T]
        x = self.conv1d(x)              # [B, hidden_F, T]
        # you may get a better result if you use linear connection to replace the global average pooling
        x = torch.mean(x, 2)            # global average pooling -> [B, hidden_F]
        x = self.linear(x)
        x = nn.Softmax(x)
        return x


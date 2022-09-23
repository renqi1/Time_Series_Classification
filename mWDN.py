# paper: https://dl.acm.org/doi/abs/10.1145/3219819.3220060

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class mWDN(nn.Module):
    def __init__(self, batch_size, seq_len, num_features, output_size, hidden_size=32):
        super(mWDN, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.res_layer = self.make_layer(ResidualBlock, inputs=seq_len)

        self.mWDN1_H = nn.Linear(seq_len, seq_len)
        self.mWDN1_L = nn.Linear(seq_len, seq_len)
        self.mWDN2_H = nn.Linear(int(seq_len / 2), int(seq_len / 2))
        self.mWDN2_L = nn.Linear(int(seq_len / 2), int(seq_len / 2))
        self.a_to_x = nn.AvgPool1d(2)
        self.sigmoid = nn.Sigmoid()
        self.lstm_xh1 = nn.LSTM(num_features, hidden_size, batch_first=True)
        self.lstm_xh2 = nn.LSTM(num_features, hidden_size, batch_first=True)
        self.lstm_xl2 = nn.LSTM(num_features, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

        self.l_filter = [-0.0106, 0.0329, 0.0308, -0.187, -0.028, 0.6309, 0.7148, 0.2304]
        self.h_filter = [-0.2304, 0.7148, -0.6309, -0.028, 0.187, 0.0308, -0.0329, -0.0106]

        self.cmp_mWDN1_H = torch.from_numpy(self.create_W(seq_len, False, is_comp=True)).float().cuda()
        self.cmp_mWDN1_L = torch.from_numpy(self.create_W(seq_len, True, is_comp=True)).float().cuda()
        self.cmp_mWDN2_H = torch.from_numpy(self.create_W(int(seq_len / 2), False, is_comp=True)).float().cuda()
        self.cmp_mWDN2_L = torch.from_numpy(self.create_W(int(seq_len / 2), True, is_comp=True)).float().cuda()

        self.mWDN1_H.weight = nn.Parameter(torch.from_numpy(self.create_W(seq_len, False)).float(), requires_grad=True)
        self.mWDN1_L.weight = nn.Parameter(torch.from_numpy(self.create_W(seq_len, True)).float(), requires_grad=True)
        self.mWDN2_H.weight = nn.Parameter(torch.from_numpy(self.create_W(int(seq_len / 2), False)).float(), requires_grad=True)
        self.mWDN2_L.weight = nn.Parameter(torch.from_numpy(self.create_W(int(seq_len / 2), True)).float(), requires_grad=True)


    def forward(self, input):
        # input [B, F, T] where B = Batch size, F = features, T = Time sampels,
        ah_1 = self.sigmoid(self.mWDN1_H(input))
        al_1 = self.sigmoid(self.mWDN1_L(input))
        xh_1 = self.a_to_x(ah_1)     # [B, F, T] -> [B, F, T/2]
        xl_1 = self.a_to_x(al_1)     # [B, F, T] -> [B, F, T/2]

        ah_2 = self.sigmoid(self.mWDN2_H(xl_1))                 # Wavelet Decomposition
        al_2 = self.sigmoid(self.mWDN2_L(xl_1))

        xh_2 = self.a_to_x(ah_2)     # [B, F, T/2] -> [B, F, T/4]
        xl_2 = self.a_to_x(al_2)     # [B, F, T/2] -> [B, F, T/4]

        xh_1 = xh_1.transpose(1, 2)         # [B, T/2, F]
        xh_2 = xh_2.transpose(1, 2)         # [B, T/4, F]
        xl_2 = xl_2.transpose(1, 2)         # [B, T/4, F]

        h1, c1, h2, c2, h3, c3 = self.init_state()
        level1_lstm, (h1, c1) = self.lstm_xh1(xh_1, (h1, c1))       # [B, T/2, hidden_size]
        level2_lstm_h, (h2, c2) = self.lstm_xh2(xh_2, (h2, c2))     # [B, T/4, hidden_size]
        level2_lstm_l, (h3, c3) = self.lstm_xl2(xl_2, (h3, c3))     # [B, T/4, hidden_size]
        output = self.res_layer(torch.cat((level1_lstm, level2_lstm_h, level2_lstm_l), 1))      # [B, T, hidden_size]  ->  [B, 1, output_size]
        output = output.squeeze(1)    # [B, hidden_size]

        W_mWDN1_H = self.mWDN1_H.weight.data
        W_mWDN1_L = self.mWDN1_L.weight.data
        W_mWDN2_H = self.mWDN2_H.weight.data
        W_mWDN2_L = self.mWDN2_L.weight.data
        L_loss = torch.norm((W_mWDN1_L - self.cmp_mWDN1_L), 2) + torch.norm((W_mWDN2_L - self.cmp_mWDN2_L), 2)
        H_loss = torch.norm((W_mWDN1_H - self.cmp_mWDN1_H), 2) + torch.norm((W_mWDN2_H - self.cmp_mWDN2_H), 2)
        lossLH = 0.3 * L_loss + 0.3 * H_loss
        return output, lossLH

    def init_state(self):
        h1 = torch.zeros(1, self.batch_size, self.hidden_size).cuda()
        c1 = torch.zeros(1, self.batch_size, self.hidden_size).cuda()
        h2 = torch.zeros(1, self.batch_size, self.hidden_size).cuda()
        c2 = torch.zeros(1, self.batch_size, self.hidden_size).cuda()
        h3 = torch.zeros(1, self.batch_size, self.hidden_size).cuda()
        c3 = torch.zeros(1, self.batch_size, self.hidden_size).cuda()
        return h1, c1, h2, c2, h3, c3

    def create_W(self, P, is_l, is_comp=False):
        if is_l:
            filter_list = self.l_filter
        else:
            filter_list = self.h_filter

        list_len = len(filter_list)

        max_epsilon = np.min(np.abs(filter_list))
        if is_comp:
            weight_np = np.zeros((P, P))
        else:
            weight_np = np.random.randn(P, P) * 0.1 * max_epsilon

        for i in range(0, P):
            filter_index = 0
            for j in range(i, P):
                if filter_index < len(filter_list):
                    weight_np[i][j] = filter_list[filter_index]
                    filter_index += 1
        return weight_np

    def make_layer(self, block, inputs, stride=1):
        layers = []
        channels = [4, 16]
        for k in range(2):
            layers.append(block(inputs, channels[k], stride))
            inputs = channels[k]
        layers.append(nn.Conv1d(channels[-1], 1, kernel_size=1, stride=stride, bias=False))
        layers.append(nn.AdaptiveAvgPool1d(self.output_size))
        return nn.Sequential(*layers)

# paper: https://arxiv.org/abs/1812.07683
# reposity: https://github.com/NellyElsayed/GRU-FCN-model-for-univariate-time-series-classification

import torch
import torch.nn as nn

class GRU_FCN(nn.Module):
    def __init__(self, Batch_Size, N_Features, N_ClassesOut, N_GRU_Out=128, N_GRU_layers=1,
                 Conv1_NF=128, Conv2_NF=256, Conv3_NF=128, gruDropP=0.8, FC_DropP=0.3):
        super(GRU_FCN, self).__init__()

        self.Batch_Size = Batch_Size
        self.N_Features = N_Features
        self.N_ClassesOut = N_ClassesOut
        self.N_GRU_Out = N_GRU_Out
        self.N_GRU_layers = N_GRU_layers
        self.Conv1_NF = Conv1_NF
        self.Conv2_NF = Conv2_NF
        self.Conv3_NF = Conv3_NF
        self.gru = nn.GRU(self.N_Features, self.N_GRU_Out, self.N_GRU_layers, batch_first=True)
        self.C1 = nn.Conv1d(self.N_Features, self.Conv1_NF, 8)
        self.C2 = nn.Conv1d(self.Conv1_NF, self.Conv2_NF, 5)
        self.C3 = nn.Conv1d(self.Conv2_NF, self.Conv3_NF, 3)
        self.BN1 = nn.BatchNorm1d(self.Conv1_NF)
        self.BN2 = nn.BatchNorm1d(self.Conv2_NF)
        self.BN3 = nn.BatchNorm1d(self.Conv3_NF)
        self.relu = nn.ReLU()
        self.gruDrop = nn.Dropout(gruDropP)
        self.ConvDrop = nn.Dropout(FC_DropP)
        self.FC = nn.Linear(self.Conv3_NF + self.N_GRU_Out, self.N_ClassesOut)

    def init_hidden(self):
        h = torch.zeros(self.N_GRU_layers, self.Batch_Size, self.N_GRU_Out).cuda()
        return h

    def forward(self, x):
        # input x : [B, F, T], where B = Batch size, F = features, T = Time sampels
        h = self.init_hidden()
        x1 = x.transpose(2, 1)  # [B, T, F]
        x1, h = self.gru(x1, h)
        x1 = self.gruDrop(x1)
        x1 = x1[:, -1, :]  # [B, GRU_Out]

        x2 = self.ConvDrop(self.relu(self.BN1(self.C1(x))))
        x2 = self.ConvDrop(self.relu(self.BN2(self.C2(x2))))
        x2 = self.ConvDrop(self.relu(self.BN3(self.C3(x2))))
        x2 = torch.mean(x2, 2)  # Global average pooling --> [B, Conv3_NF]
        # x2 = x2[:, :, -1]     # you may get a better result if you use this to replace the global average pooling

        x_all = torch.cat((x1, x2), dim=1)  # [B, GRU_Out+Conv3_NF]
        x_out = self.FC(x_all)
        return x_out

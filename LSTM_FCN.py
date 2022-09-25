# paper: https://arxiv.org/abs/1801.04503
# reposity: https://github.com/titu1994/LSTM-FCN

import torch
import torch.nn as nn

class LSTM_FCN(nn.Module):
    def __init__(self, Batch_Size, N_Features, N_ClassesOut, N_LSTM_Out=128, N_LSTM_layers=1,
                 Conv1_NF=128, Conv2_NF=256, Conv3_NF=128, lstmDropP=0.8, FC_DropP=0.3):
        super(LSTM_FCN, self).__init__()

        self.Batch_Size = Batch_Size
        self.N_Features = N_Features
        self.N_ClassesOut = N_ClassesOut
        self.N_LSTM_Out = N_LSTM_Out
        self.N_LSTM_layers = N_LSTM_layers
        self.Conv1_NF = Conv1_NF
        self.Conv2_NF = Conv2_NF
        self.Conv3_NF = Conv3_NF
        self.lstm = nn.LSTM(self.N_Features, self.N_LSTM_Out, self.N_LSTM_layers, batch_first=True)
        self.C1 = nn.Conv1d(self.N_Features, self.Conv1_NF, 8)
        self.C2 = nn.Conv1d(self.Conv1_NF, self.Conv2_NF, 5)
        self.C3 = nn.Conv1d(self.Conv2_NF, self.Conv3_NF, 3)
        self.BN1 = nn.BatchNorm1d(self.Conv1_NF)
        self.BN2 = nn.BatchNorm1d(self.Conv2_NF)
        self.BN3 = nn.BatchNorm1d(self.Conv3_NF)
        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(lstmDropP)
        self.ConvDrop = nn.Dropout(FC_DropP)
        self.FC = nn.Linear(self.Conv3_NF + self.N_LSTM_Out, self.N_ClassesOut)

    def init_hidden(self):
        h0 = torch.zeros(self.N_LSTM_layers, self.Batch_Size, self.N_LSTM_Out).cuda()
        c0 = torch.zeros(self.N_LSTM_layers, self.Batch_Size, self.N_LSTM_Out).cuda()
        return h0, c0

    def forward(self, x):
        # input x : [B, F, T], where B = Batch size, F = features, T = Time sampels
        h0, c0 = self.init_hidden()
        x1 = x.transpose(2, 1)   # [B, T, F]
        x1, (ht, ct) = self.lstm(x1, (h0, c0))
        x1 = self.lstmDrop(x1)
        x1 = x1[:, -1, :]  # [B, LSTM_Out]

        x2 = self.ConvDrop(self.relu(self.BN1(self.C1(x))))
        x2 = self.ConvDrop(self.relu(self.BN2(self.C2(x2))))
        x2 = self.ConvDrop(self.relu(self.BN3(self.C3(x2))))
        x2 = torch.mean(x2, 2)  # Global average pooling --> [B, Conv3_NF]
        # x2 = x2[:, :, -1]     # you may get a better result if you use this to replace the global average pooling 
        
        x_all = torch.cat((x1, x2), dim=1)  # [B, LSTM_Out+Conv3_NF]
        x_out = self.FC(x_all)
        return x_out

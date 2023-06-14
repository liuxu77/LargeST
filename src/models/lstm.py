import torch.nn as nn
import torch.nn.functional as F
from src.base.model import BaseModel

class LSTM(BaseModel):
    def __init__(self, init_dim, hid_dim, end_dim, layer, dropout, **args):
        super(LSTM, self).__init__(**args)   
        self.start_conv = nn.Conv2d(in_channels=self.input_dim, 
                                    out_channels=init_dim, 
                                    kernel_size=(1,1))

        self.lstm = nn.LSTM(input_size=init_dim, hidden_size=hid_dim, num_layers=layer, batch_first=True, dropout=dropout)
        
        self.end_linear1 = nn.Linear(hid_dim, end_dim)
        self.end_linear2 = nn.Linear(end_dim, self.horizon)


    def forward(self, input, label=None):  # (b, t, n, f)
        x = input.transpose(1, 3)
        b, f, n, t = x.shape

        x = x.transpose(1,2).reshape(b*n, f, 1, t)
        x = self.start_conv(x).squeeze().transpose(1, 2)

        out, _ = self.lstm(x)
        x = out[:, -1, :]

        x = F.relu(self.end_linear1(x))
        x = self.end_linear2(x)
        x = x.reshape(b, n, t, 1).transpose(1, 2)
        return x
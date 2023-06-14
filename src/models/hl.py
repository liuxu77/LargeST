import torch.nn as nn
from src.base.model import BaseModel

class HL(BaseModel):
    def __init__(self, **args):
        super(HL, self).__init__(**args)   
        self.fake = nn.Linear(1, 1)


    def forward(self, input, label=None):  # (b, t, n, f)
        x = input[:,[-1],:,:].expand(-1, self.horizon, -1, -1)
        return x
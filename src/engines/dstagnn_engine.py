import torch
from src.base.engine import BaseEngine

class DSTAGNN_Engine(BaseEngine):
    def __init__(self, **args):
        super(DSTAGNN_Engine, self).__init__(**args)
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)
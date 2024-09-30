import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Any, Dict, Tuple, Optional

class LinearHead(nn.Module):
    def __init__(self,
                 d_model: int, ):
        super(LinearHead, self).__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.linear(x)
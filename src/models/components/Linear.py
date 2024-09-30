import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

class Model(nn.Module):
    def __init__(self,
                 task: [Optional, str],
                 input_len: int,
                 output_len: int,
                 num_channels: int,
                 individual: bool,
                 eps: float = 1e-5,
                 num_models: int = 1,):
        super(Model, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.randn(input_len, output_len))
        self.bias = nn.Parameter(torch.zeros(output_len))
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def init_params(self):
        pass

    def save_parameters(self):
        pass

    def predict(self, x):
        y_hat = torch.einsum('bcl,lo->bco', x, self.weight) + self.bias
        return y_hat

    def forward(self, x, y, x_mark=None, y_mark=None):
        result = {}
        y_hat = self.predict(x)
        result['y_hat'] = y_hat
        result['predloss'] = F.l1_loss(y_hat, y, reduction='none').mean(-1)
        result['recloss'] = F.l1_loss(torch.zeros_like(x), torch.zeros_like(x), reduction='none').mean(-1)
        result['normloss'] = (F.mse_loss(torch.zeros_like(self.weight), self.weight, reduction='none').mean(-1) ** 0.5).mean()
        # result['normloss'] += (F.mse_loss(torch.zeros_like(self.weight), self.weight, reduction='none').mean(0) ** 0.5).mean()
        # result['normloss'] = torch.zeros_like(result['normloss'])
        return result
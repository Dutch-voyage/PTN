import torch
import torch.nn as nn
import torch.nn.functional as F
from iTransformer import iTransformer

class Model(nn.Module):
    def __init__(self,
                 input_len: int,
                 output_len: int,
                 num_channels: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 task: str,
                 num_models: int = 1,
                 ifmap: bool = False):
        super(Model, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        if ifmap:
            self.num_channels = num_channels
        else:
            self.num_channels = num_channels
        self.itransformer = iTransformer(num_variates=self.num_channels,
                                         lookback_len=input_len,
                                         dim=d_model,
                                         depth=num_layers,
                                         heads=num_heads,
                                         dim_head=d_model // num_heads,
                                         pred_length=output_len,
                                         num_tokens_per_variate=1,
                                         use_reversible_instance_norm=True,
                                         flash_attn=False)

    def init_params(self):
        pass

    def predict(self, x):
        y_hat = self.itransformer(x.transpose(-1, -2))[self.output_len].transpose(-1, -2)
        return y_hat

    def forward(self, x, y, x_mark=None, y_mark=None, mode='train'):
        result = {}
        y_hat = self.predict(x)
        result['y_hat'] = y_hat
        result['predloss'] = F.l1_loss(y_hat, y, reduction='none').mean(-1)
        result['recloss'] = F.l1_loss(torch.zeros_like(x), torch.zeros_like(x), reduction='none').mean(-1)
        result['normloss'] = torch.zeros_like(result['predloss'])
        return result
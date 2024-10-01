import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional

class Model(nn.Module):
    def __init__(self,
                 d_model: int,
                 embedding: nn.Module,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 mapper: nn.Module,
                 task: Optional[str] = None):
        super(Model, self).__init__()
        self.embedding = embedding(d_model=d_model)
        self.encoder = encoder(d_model=d_model)
        self.decoder = decoder(d_model=d_model)
        self.mapper = mapper(task=task)

    def init_params(self):
        pass

    def forward(self, x, y, x_mark, y_mark):
        result = {}
        x_emb = self.embedding(x)
        x_enc = self.encoder(x_emb)
        x_dec = self.decoder(x_enc)
        y_emb = self.embedding(y)
        y_enc = self.encoder(y_emb)
        y_dec = self.decoder(y_enc)

        map_result = self.mapper(x_dec, y_dec)
        y_hat = map_result['y_hat']
        result['y_hat'] = y_hat
        '''consider to encourage convergence on small dataset
        res_rec = y_rec - y
        res_pred = y_hat - y_rec
        norm_loss = F.mse_loss(F.tanh(res_rec), F.tanh(res_pred))
        '''
        result['predloss'] = map_result['predloss']
        result['normloss'] = torch.zeros_like(result['predloss'])
        result['recloss'] = F.l1_loss(x_dec, x, reduction='none').mean(-1) + F.l1_loss(y_dec, y, reduction='none').mean(-1)

        return result
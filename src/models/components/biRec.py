import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional

class Model(nn.Module):
    def __init__(self,
                 d_model: int,
                 embedding: nn.Module,
                 reconstructor: nn.Module,
                 mapper: nn.Module,
                 task: Optional[str] = None):
        super(Model, self).__init__()
        self.embedding = embedding(d_model=d_model)
        self.reconstructor = reconstructor(d_model=d_model)
        self.mapper = mapper(task=task)

    def init_params(self):
        pass

    def forward(self, x, y, x_mark, y_mark):
        result = {}
        x_emb = self.embedding(x)
        x_rec = self.reconstructor(x_emb)
        y_emb = self.embedding(y)
        y_rec = self.reconstructor(y_emb)

        map_result = self.mapper(x_rec, y)
        y_hat = map_result['y_hat']
        result['y_hat'] = y_hat
        res_rec = y_rec - y
        res_pred = y_hat - y_rec
        norm_loss = F.mse_loss(F.tanh(res_rec), F.tanh(res_pred))
        result['predloss'] = map_result['predloss']  # + map_result_stu['predloss']
        result['normloss'] = torch.zeros_like(norm_loss)
        result['recloss'] = F.l1_loss(x_rec, x, reduction='none').mean(-1) + F.l1_loss(y_rec, y, reduction='none').mean(-1)
        # result['recloss'] = F.l1_loss(x_rec, x, reduction='none').mean(-1)
        # result['recloss'] += F.l1_loss(y_rec, y, reduction='none').mean(-1)

        return result
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Router(nn.Module):
    def __init__(self,
                 topk: int,
                 num_channels: int,
                 num_models: int,
                 input_len: int,
                 d_model: int):
        super(Router, self).__init__()
        self.num_channels = num_channels
        self.num_models = num_models
        self.model_embedding = nn.Parameter(torch.randn(num_models, d_model), requires_grad=True)
        self.proj = nn.Linear(input_len, d_model)
        self.topk = topk
        nn.init.xavier_uniform_(self.model_embedding)

    def forward(self, x_emb):
        B, C, L  = x_emb.shape
        scores = torch.einsum('bcd, nd -> bcn', self.proj(x_emb), self.model_embedding)
        scores = F.softmax(scores, dim=-1)
        # mask = torch.topk(scores, self.topk, dim=-1, largest=True, sorted=True)
        # mask = F.one_hot(mask.indices, num_classes=self.num_models).squeeze(-2)
        return scores
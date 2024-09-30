import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Any, Dict, Tuple, Optional

class Reconstructor(nn.Module):
    def __init__(self,
                 embedding_type: str,
                 attn_block: nn.Module,
                 d_model: int,
                 num_channels: int,
                 patch_len: Optional[int]):
        super(Reconstructor, self).__init__()
        self.embedding_type = embedding_type
        self.num_channels = num_channels
        self.attn = attn_block(d_model=d_model)
        self.patch_len = patch_len
        if self.embedding_type == 'patch':
            d_out = self.patch_len
        elif self.embedding_type == 'conv':
            d_out = 1
        self.linear = nn.Linear(d_model, d_out)

    def forward(self, x_emb):
        B = x_emb.shape[0]
        C = x_emb.shape[1]
        if self.embedding_type == 'patch':
            x_emb = rearrange(x_emb, 'b c l d-> (b c) l d')
        elif self.embedding_type == 'conv':
            x_emb = rearrange(x_emb, 'b c (l p) d -> (b c l) p d', p=self.patch_len)
        else:
            raise ValueError('Invalid embedding type')
        x_enc = self.attn(x_emb)
        if self.embedding_type == 'patch':
            x_dec = rearrange(x_enc, '(b c) l d -> b c l d', b=B, c=C)
        elif self.embedding_type == 'conv':
            x_dec = rearrange(x_enc, '(b c l) p d -> b c l p d', b=B, c=C)
        else:
            raise ValueError('Invalid embedding type')
        x_dec = self.linear(x_dec)
        if self.embedding_type == 'patch':
            x_dec = rearrange(x_dec, 'b c l d -> b c (l d)')
        elif self.embedding_type == 'conv':
            x_dec = rearrange(x_dec, 'b c l p d -> b c (l p d)')
        else:
            raise ValueError('Invalid embedding type')
        return x_dec
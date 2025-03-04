import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Any, Dict, Tuple, Optional

class Encoder(nn.Module):
    def __init__(self,
                 embedding_type: str,
                 attn_block: nn.Module,
                 d_model: int,
                 num_channels: int,
                 patch_len: Optional[int],
                 with_ch: bool,
                 with_tem: bool,
                 d_in: Optional[int]=None):
        super(Encoder, self).__init__()
        self.embedding_type = embedding_type
        self.num_channels = num_channels
        # self.attntem = attn_block(d_model=d_model)
        self.attn = attn_block(d_model=d_model)
        self.patch_len = patch_len
        self.with_ch = with_ch
        self.with_tem = with_tem

    def forward(self, x_emb):
        B = x_emb.shape[0]
        C = x_emb.shape[1]
        if not self.with_ch and not self.with_tem:
            return x_emb
        x_enc_tem = torch.zeros_like(x_emb)
        x_enc_ch = torch.zeros_like(x_emb)
        if self.with_tem:
            if self.embedding_type == 'patch':
                x_emb_tem = rearrange(x_emb, 'b c l d-> (b c) l d')
                x_enc_tem = self.attn(x_emb_tem)
                x_enc_tem = rearrange(x_enc_tem, '(b c) l d-> b c l d', b=B, c=C)
            elif self.embedding_type == 'conv':
                x_emb_tem = rearrange(x_emb, 'b c l d-> (b c) l d')
                x_enc_tem = self.attn(x_emb_tem)
                x_enc_tem = rearrange(x_enc_tem, '(b c) l d -> b c l d', b=B, c=C)
            elif self.embedding_type == 'patchedconv':
                x_emb_tem = rearrange(x_emb, 'b c (l p) d -> (b c l) p d', b=B, p=self.patch_len)
                x_enc_tem = self.attn(x_emb_tem)
                x_enc_tem = rearrange(x_enc_tem, '(b c l) p d -> b c (l p) d', b=B,  c=C)
            elif self.embedding_type == 'vanilla':
                pass
            else:
                raise ValueError('Invalid embedding type')
        if self.with_ch:
            if self.embedding_type == 'patch':
                x_emb_ch = rearrange(x_emb, 'b c l d -> (b l) c d', b=B)
                x_enc_ch = self.attn(x_emb_ch)
                x_enc_ch = rearrange(x_enc_ch, '(b l) c d -> b c l d', b=B)
            elif self.embedding_type == 'conv':
                x_emb_ch = rearrange(x_emb, 'b c l d -> (b l) c d', b=B, c=C)
                x_enc_ch = self.attn(x_emb_ch)
                x_enc_ch = rearrange(x_enc_ch, '(b l) c d -> b c l d', b=B)
            elif self.embedding_type == 'patchedconv':
                x_emb_ch = rearrange(x_emb, 'b c (l p) d -> (b l p) c d', b=B, p=self.patch_len)
                x_enc_ch = self.attn(x_emb_ch)
                x_enc_ch = rearrange(x_enc_ch, '(b l p) c d -> b c (l p) d', b=B, p=self.patch_len)
            elif self.embedding_type == 'vanilla':
                pass
            else:
                raise ValueError('Invalid embedding type')
        x_enc = x_enc_tem + x_enc_ch
        return x_enc
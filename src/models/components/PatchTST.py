# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from .modules.PatchTST_backbone import PatchTST_backbone
from .modules.PatchTST_layers import series_decomp


class Model(nn.Module):
    def __init__(self,
                 input_len: int,
                 output_len: int,
                 num_channels: int,
                 num_layers: int,
                 num_heads: int,
                 d_model: int,
                 d_ff: int,
                 patch_len: int,
                 stride: int,
                 max_seq_len: Optional[int] = 1024,
                 d_k: Optional[int] = None,
                 d_v: Optional[int] = None,
                 norm: str = 'BatchNorm',
                 attn_dropout: float = 0.,
                 act: str = "gelu",
                 key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None,
                 res_attention: bool = True,
                 pre_norm: bool = False,
                 store_attn: bool = False,
                 pe: str = 'zeros',
                 learn_pe: bool = True,
                 pretrain_head: bool = False,
                 head_type='flatten',
                 verbose: bool = False,
                 num_models: int = 1,
                 **kwargs):

        super().__init__()

        # load parameters
        c_in = num_channels
        context_window = input_len
        target_window = output_len

        n_layers = num_layers
        n_heads = num_heads
        d_model = d_model
        d_ff = d_ff
        dropout = 0.0
        fc_dropout = 0.0
        head_dropout = 0.0

        individual = False

        patch_len = patch_len
        stride = stride
        padding_patch = None

        revin = True
        affine = False
        subtract_last = False

        decomposition = False
        kernel_size = None

        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                                 patch_len=patch_len, stride=stride,
                                                 max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                                 n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                 attn_dropout=attn_dropout,
                                                 dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                                 padding_var=padding_var,
                                                 attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                                 store_attn=store_attn,
                                                 pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                                 head_dropout=head_dropout, padding_patch=padding_patch,
                                                 pretrain_head=pretrain_head, head_type=head_type,
                                                 individual=individual, revin=revin, affine=affine,
                                                 subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                               patch_len=patch_len, stride=stride,
                                               max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                               n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                               attn_dropout=attn_dropout,
                                               dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                               padding_var=padding_var,
                                               attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                               store_attn=store_attn,
                                               pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                               head_dropout=head_dropout, padding_patch=padding_patch,
                                               pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                               revin=revin, affine=affine,
                                               subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                           patch_len=patch_len, stride=stride,
                                           max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                           n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                           attn_dropout=attn_dropout,
                                           dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                           padding_var=padding_var,
                                           attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                           store_attn=store_attn,
                                           pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
                                           padding_patch=padding_patch,
                                           pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                           revin=revin, affine=affine,
                                           subtract_last=subtract_last, verbose=verbose, **kwargs)

    def init_params(self):
        pass

    def forward(self, x, y, x_mark=None, y_mark=None, mode='train'):
        result = {}
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init, trend_init
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            y_hat = res + trend
        else:
            y_hat = self.model(x)
        result['y_hat'] = y_hat
        result['predloss'] = F.l1_loss(y_hat, y, reduction='none').mean(-1)
        result['recloss'] = F.l1_loss(torch.zeros_like(x), torch.zeros_like(x), reduction='none').mean(-1)
        result['normloss'] = torch.zeros_like(result['predloss'])
        return result
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from einops import rearrange
from torch.func import functional_call, grad, stack_module_state, grad_and_value
from typing import Optional
from .utils.loss import residual_loss
from functools import partial

class Model(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_models: int,
                 patch_len: int,
                 embedding: nn.Module,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 mapper: nn.Module,
                 task: Optional[str] = None):
        super(Model, self).__init__()
        self.num_models = num_models
        self.d_model = d_model
        self.patch_len = patch_len
        self.task = task
        self.emb_base = embedding(d_model=d_model)
        self.enc_base = encoder
        self.dec_base = decoder
        self.map_base = mapper(task=task)

        self.params = {'Remb': {}, 'Renc': {}, 'Rdec': {}}
        self.init_params()

    def init_params(self):
        enc_models = nn.ModuleList([self.enc_base(d_model=self.d_model) for _ in range(self.num_models)])
        dec_models = nn.ModuleList([self.dec_base(d_model=self.d_model) for _ in range(self.num_models)])
        enc_params, _ = stack_module_state(enc_models)
        dec_params, _ = stack_module_state(dec_models)

        self.params['Renc'] = {name: nn.Parameter(param) for name, param in enc_params.items()}
        self.params['Rdec'] = {name: nn.Parameter(param) for name, param in dec_params.items()}

        # Register parameters
        for key, value in self.params.items():
            for name, param in value.items():
                safename = name.replace('.', '_')
                self.register_parameter(f"{key}_{safename}", param)
    def load_params(self):
        params = self.named_parameters()
        for name, param in params:
            name = name.replace('_', '.')
            if "Renc" in name:
                self.params['Renc'][name[5:]] = param
            elif "Rdec" in name:
                self.params['Rdec'][name[5:]] = param
            else:
                continue

    def call_single_x(self, model, params, x):
        return functional_call(model, (params,), (x,))

    def call_single_xy(self, model, params, x, y):
        return functional_call(model, (params,), (x, y))

    def loss_fn(self, x, x_rec, y, y_rec, y_hats):
        res_rec = y_rec - y[0]
        res_pred = y_hats - y_rec
        norm_loss = F.l1_loss(F.tanh(res_rec), F.tanh(res_pred))
        norm_loss = torch.zeros_like(norm_loss)
        pred_loss = F.l1_loss(y_hats, y_rec)
        rec_loss = F.l1_loss(x_rec, x[0])
        rec_loss += F.l1_loss(y_rec, y[0])

        return rec_loss, pred_loss, norm_loss

    def compute_loss(self, x, y):
        self.load_params()
        enc_params_tem = self.params['Renc']
        dec_params_tem = self.params['Rdec']

        B = x.shape[1]

        dec_model = self.dec_base(d_model=self.d_model)
        enc_model = self.enc_base(d_model=self.d_model)

        call_enc = partial(self.call_single_x, enc_model)
        call_dec = partial(self.call_single_x, dec_model)


        x_emb = self.emb_base(x[0])
        y_emb = self.emb_base(y[0])

        x_emb = rearrange(x_emb, 'b c (l n) d -> n b c l d ', n=self.num_models)
        y_emb = rearrange(y_emb, 'b c (l n) d -> n b c l d ', n=self.num_models)

        x_enc = torch.vmap(call_enc, (0, 0))(enc_params_tem, x_emb)
        y_enc = torch.vmap(call_enc, (0, 0))(enc_params_tem, y_emb)

        x_rec = torch.vmap(call_dec, (0, 0))(dec_params_tem, x_enc)
        y_rec = torch.vmap(call_dec, (0, 0))(dec_params_tem, y_enc)

        x_rec = rearrange(x_rec, 'n b c l -> b c (l n)')
        y_rec = rearrange(y_rec, 'n b c l -> b c (l n)')

        results = self.map_base(x_rec, y_rec)
        y_hats = results['y_hat']

        rec_loss, pred_loss, norm_loss = self.loss_fn(x, x_rec, y, y_rec, y_hats)
        return pred_loss, rec_loss, norm_loss, y_hats

    def forward(self, x, y, x_mark=None, y_mark=None):
        x = x.unsqueeze(0).repeat(self.num_models, 1, 1, 1)
        y = y.unsqueeze(0).repeat(self.num_models, 1, 1, 1)
        pred_loss, rec_loss, norm_loss, y_hats = self.compute_loss(x, y)
        result = {}
        result['predloss'] = pred_loss
        result['recloss'] = rec_loss
        result['normloss'] = norm_loss
        result['nonautoloss'] = torch.zeros_like(pred_loss)
        result['y_hat'] = y_hats
        return result
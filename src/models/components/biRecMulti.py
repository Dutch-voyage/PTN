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
                 linearhead: nn.Module,
                 router: nn.Module,
                 task: Optional[str] = None):
        super(Model, self).__init__()
        self.num_models = num_models
        # self.emb_models = nn.ModuleList([embedding(d_model=d_model) for _ in range(num_models)])
        # self.enc_models = nn.ModuleList([encoder(d_model=d_model) for _ in range(num_models)])
        # self.dec_models = nn.ModuleList([decoder(d_model=d_model) for _ in range(num_models)])
        # self.map_models = nn.ModuleList([mapper(task=task) for _ in range(num_models)])
        self.d_model = d_model
        self.patch_len = patch_len
        self.task = task
        # self.emb_base = embedding
        self.emb_base = embedding(d_model=d_model)
        self.enc_base = encoder
        self.dec_base = decoder
        # self.linearhead = linearhead(d_model=d_model)
        # self.map_base = mapper
        self.map_base = mapper(task=task)
        # self.map_base = nn.Identity()
        # self.router = router(num_models=num_models, d_model=d_model)
        self.params = {'Remb': {}, 'Renc': {}, 'Rdec': {}}
        self.init_params()

    def init_params(self):
        # emb_models = nn.ModuleList([self.emb_base(d_model=self.d_model) for _ in range(self.num_models)])
        enc_models = nn.ModuleList([self.enc_base(d_model=self.d_model) for _ in range(self.num_models)])
        dec_models = nn.ModuleList([self.dec_base(d_model=self.d_model) for _ in range(self.num_models)])
        # map_models = nn.ModuleList([self.map_base(task=self.task) for _ in range(self.num_models)])
        # emb_params, _ = stack_module_state(emb_models)
        enc_params, _ = stack_module_state(enc_models)
        dec_params, _ = stack_module_state(dec_models)
        # map_params, _ = stack_module_state(map_models)

        # self.params['Remb'] = {name: nn.Parameter(param) for name, param in emb_params.items()}
        self.params['Renc'] = {name: nn.Parameter(param) for name, param in enc_params.items()}
        self.params['Rdec'] = {name: nn.Parameter(param) for name, param in dec_params.items()}
        # self.params['Rmap'] = {name: nn.Parameter(param) for name, param in map_params.items()}

        # Register parameters
        for key, value in self.params.items():
            for name, param in value.items():
                safename = name.replace('.', '_')
                self.register_parameter(f"{key}_{safename}", param)
    def load_params(self):
        params = self.named_parameters()
        for name, param in params:
            name = name.replace('_', '.')
            if "Remb" in name:
                continue
                # self.params['Remb'][name[5:]] = param
            elif "Renc" in name:
                self.params['Renc'][name[5:]] = param
            elif "Rdec" in name:
                self.params['Rdec'][name[5:]] = param
            # elif "Rmap" in name:
            #     self.params['Rmap'][name[5:]] = param
            else:
                continue

    def call_single_x(self, model, params, x):
        # return functional_call(self.emb_models[0], (params,), (x,))
        return functional_call(model, (params,), (x,))

    def call_single_xy(self, model, params, x, y):
        return functional_call(model, (params,), (x, y))

    def loss_fn(self, x, x_rec, y, y_rec, y_hats):
        # x = torch.einsum('b c n, b c l -> n b c l', scores, x[0])
        # y = torch.einsum('b c n, b c l -> n b c l', scores, y[0])

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
        # emb_params_tem = self.params['Remb']
        enc_params_tem = self.params['Renc']
        dec_params_tem = self.params['Rdec']
        # map_params = self.params['Rmap']

        B = x.shape[1]
        # emb_model = self.emb_base(d_model=self.d_model)
        dec_model = self.dec_base(d_model=self.d_model)
        enc_model = self.enc_base(d_model=self.d_model)
        # map_model = self.map_base(task=self.task)
        # call_emb = partial(self.call_single_x, emb_model)
        call_enc = partial(self.call_single_x, enc_model)
        call_dec = partial(self.call_single_x, dec_model)
        # call_map = partial(self.call_single_xy, map_model)

        # x_emb = torch.vmap(call_emb, (0, 0))(emb_params_tem, x)
        # y_emb = torch.vmap(call_emb, (0, 0))(emb_params_tem, y)

        x_emb = self.emb_base(x[0])
        y_emb = self.emb_base(y[0])
        # scores = self.router(x[0])
        x_emb = rearrange(x_emb, 'b c (l n) d -> n b c l d ', n=self.num_models)
        y_emb = rearrange(y_emb, 'b c (l n) d -> n b c l d ', n=self.num_models)

        x_enc = torch.vmap(call_enc, (0, 0))(enc_params_tem, x_emb)
        y_enc = torch.vmap(call_enc, (0, 0))(enc_params_tem, y_emb)

        x_rec = torch.vmap(call_dec, (0, 0))(dec_params_tem, x_enc)
        y_rec = torch.vmap(call_dec, (0, 0))(dec_params_tem, y_enc)

        # x_rec = self.linearhead(x_enc).squeeze(-1)
        # y_rec = self.linearhead(y_enc).squeeze(-1)

        x_rec = rearrange(x_rec, 'n b c l -> b c (l n)')
        y_rec = rearrange(y_rec, 'n b c l -> b c (l n)')

        # x_rec = torch.cat([x[0], x_rec], dim=0)
        # y_rec = torch.cat([y[0], y_rec], dim=0)

        '''
        x_emb = torch.vmap(call_emb, (0, 0))(emb_params_ch, x_rec.transpose(-1, -2))
        y_emb = torch.vmap(call_emb, (0, 0))(emb_params_ch, y_rec.transpose(-1, -2))

        x_enc = torch.vmap(call_enc, (0, 0))(enc_params_ch, x_emb)
        y_enc = torch.vmap(call_enc, (0, 0))(enc_params_ch, y_emb)

        x_rec = torch.vmap(call_dec, (0, 0))(dec_params_ch, x_enc)
        y_rec = torch.vmap(call_dec, (0, 0))(dec_params_ch, y_enc)

        x_rec = x_rec.transpose(-1, -2)
        y_rec = y_rec.transpose(-1, -2)

        x_rec = x_rec_ + x_rec
        y_rec = y_rec_ + y_rec
        '''
        # x_rec = rearrange(x_rec.squeeze(0), '(n b) c l -> n b c l', b=B)
        # y_rec = rearrange(y_rec.squeeze(0), '(n b) c l -> n b c l', b=B)
        # call_map = partial(self.call_single_xy, map_model)
        # results = torch.vmap(call_map, (0, 0, 0))(map_params, x_rec, y_rec)
        # y_hats = results['y_hat']
        results = self.map_base(x_rec, y_rec)
        y_hats = results['y_hat']

        # x_rec = rearrange(x_rec, '(m b) c l -> m b c l', b=B)
        # y_rec = rearrange(y_rec, '(m b) c l -> m b c l', b=B)
        # y_hats = rearrange(y_hats, '(m b) c l -> m b c l', b=B)
        # x_rec = rearrange(x_rec, '(n b) c l -> n b c l', n=self.num_models + 1)
        # y_rec = rearrange(y_rec, '(n b) c l -> n b c l', n=self.num_models + 1)
        # y_hats = rearrange(y_hats, '(n b) c l -> n b c l', n=self.num_models + 1)
        # y_hats = torch.einsum('n b c l, b c n -> n b c l', y_hats, scores)

        rec_loss, pred_loss, norm_loss = self.loss_fn(x, x_rec, y, y_rec, y_hats)
        # y_hats_stu = result_stu['y_hat']
        # _, pred_loss_stu, _ = self.loss_fn(x, x_rec, y, y_rec, y_hats_stu)
        # weight = self.map_base.aux_model.weight
        # norm_loss += ((weight ** 2).mean(0) ** 0.5).mean()
        # norm_loss += ((weight ** 2).mean(1) ** 0.5).mean()
        # norm_loss = torch.zeros_like(pred_loss)
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
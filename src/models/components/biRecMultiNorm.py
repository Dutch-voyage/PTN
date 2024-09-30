import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from einops import rearrange
from torch.func import functional_call, grad, stack_module_state, grad_and_value
from typing import Optional
from .utils.loss import residual_loss

class Model(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_models: int,
                 embedding: nn.Module,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 mapper: nn.Module,
                 router: nn.Module,
                 task: Optional[str] = None):
        super(Model, self).__init__()
        self.num_models = num_models
        self.emb_models = nn.ModuleList([embedding(d_model=d_model) for _ in range(num_models)])
        self.enc_models = nn.ModuleList([encoder(d_model=d_model) for _ in range(num_models)])
        self.dec_models = nn.ModuleList([decoder(d_model=d_model) for _ in range(num_models)])

        self.map_models = nn.ModuleList([mapper(task=task) for _ in range(num_models)])
        # self.router = router(num_models=num_models, d_model=d_model)
        self.params = {'emb': {}, 'enc': {}, 'dec': {}, 'map': {}}
        # self.init_params()

    def save_parameters(self):
        # save params into self.rec_models
        emb_params = self.params['emb']
        enc_params = self.params['enc']
        dec_params = self.params['dec']
        map_params = self.params['map']
        emb_params = [{name: param[i, ...] for name, param in emb_params.items()} for i in range(self.num_models)]
        enc_params = [{name: param[i, ...] for name, param in enc_params.items()} for i in range(self.num_models)]
        dec_params = [{name: param[i, ...] for name, param in dec_params.items()} for i in range(self.num_models)]
        map_params = [{name: param[i, ...] for name, param in map_params.items()} for i in range(self.num_models)]

        for i, model in enumerate(self.emb_models):
            model.load_state_dict(emb_params[i])
        for i, model in enumerate(self.enc_models):
            model.load_state_dict(enc_params[i])
        for i, model in enumerate(self.dec_models):
            model.load_state_dict(dec_params[i])
        for i, model in enumerate(self.map_models):
            model.load_state_dict(map_params[i])

    def init_params(self):
        emb_params, _ = stack_module_state(self.emb_models)
        enc_params, _ = stack_module_state(self.enc_models)
        dec_params, _ = stack_module_state(self.dec_models)
        map_params, _ = stack_module_state(self.map_models)
        self.params['emb'] = {name: nn.Parameter(param) for name, param in emb_params.items()}
        self.params['enc'] = {name: nn.Parameter(param) for name, param in enc_params.items()}
        self.params['dec'] = {name: nn.Parameter(param) for name, param in dec_params.items()}
        self.params['map'] = {name: nn.Parameter(param) for name, param in map_params.items()}

        # Register parameters
        for param_dict in self.params.values():
            for name, param in param_dict.items():
                safename = name.replace('.', '_')
                self.register_parameter(f"{safename}", param)

    def call_single_emb_model(self, params, x):
        return functional_call(self.emb_models[0], (params,), (x,))

    def call_single_enc_model(self, params, x):
        return functional_call(self.enc_models[0], (params,), (x,))

    def call_single_dec_model(self, params, x):
        return functional_call(self.dec_models[0], (params,), (x,))

    def call_single_map_model(self, params, x, y):
        return functional_call(self.map_models[0], (params,), (x, y))

    def loss_fn(self, x, x_rec, y, y_rec, y_hats):
        # x_rec = torch.einsum('n b c l, b c n m -> m b c l', x_rec, scores).sum(0)
        # y_rec = torch.einsum('n b c l, b c n m -> m b c l', y_rec, scores).sum(0)
        pred_loss = F.l1_loss(y_hats, y_rec)
        rec_loss = F.l1_loss(x_rec.mean(0), x[0])
        rec_loss += F.l1_loss(y_rec.mean(0), y[0])

        return rec_loss, pred_loss

    def compute_loss(self, x, y):
        emb_params = self.params['emb']
        enc_params = self.params['enc']
        dec_params = self.params['dec']
        map_params = self.params['map']
        B = x.shape[1]
        # scores = self.router(x[0])  # B, C, N

        x_emb = torch.vmap(self.call_single_emb_model, (0, 0))(emb_params, x)
        y_emb = torch.vmap(self.call_single_emb_model, (0, 0))(emb_params, y)

        # print(scores.argmax(dim=-1)[0])

        # x_noised = torch.vmap(self.call_single_dec_model, (0, 0))(dec_params, x_emb)
        # y_noised = torch.vmap(self.call_single_dec_model, (0, 0))(dec_params, y_emb)
        # x_emb = rearrange(x_emb, 'n b c l d -> (n b) c l d')
        # y_emb = rearrange(y_emb, 'n b c l d -> (n b) c l d')
        x_enc = torch.vmap(self.call_single_enc_model, (0, 0))(enc_params, x_emb)
        y_enc = torch.vmap(self.call_single_enc_model, (0, 0))(enc_params, y_emb)
        x_rec = torch.vmap(self.call_single_dec_model, (0, 0))(dec_params, x_enc)
        y_rec = torch.vmap(self.call_single_dec_model, (0, 0))(dec_params, y_enc)
        # x_rec = rearrange(x_rec.squeeze(0), '(n b) c l -> n b c l', b=B)
        # y_rec = rearrange(y_rec.squeeze(0), '(n b) c l -> n b c l', b=B)
        results, results_stu = torch.vmap(self.call_single_map_model, (0, 0, 0))(map_params, x_rec, y_rec)
        y_hats = results['y_hat']
        y_hats_stu = results_stu['y_hat']
        # y_hats = torch.einsum('n b c l, b c n -> n b c l', y_hats, scores)
        # weight = map_params['aux_model.weight']
        # norm_loss = ((weight ** 2).mean(-1) ** 0.5).mean()
        # norm_loss += ((weight ** 2).mean(0) ** 0.5).mean()
        # norm_loss = F.l1_loss(weight, torch.zeros_like(weight))

        rec_loss, pred_loss = self.loss_fn(x, x_rec, y, y_rec, y_hats)
        _, pred_loss_stu = self.loss_fn(x, x_rec, y, y_rec, y_hats_stu)
        norm_loss = torch.zeros_like(pred_loss)
        return pred_loss + pred_loss_stu, rec_loss, norm_loss, y_hats_stu.mean(0)

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
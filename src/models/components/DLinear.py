import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.decomp import series_decomp
from typing import Any, Dict, Optional, Tuple

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, 
                 task: str, 
                 input_len: int, 
                 output_len: int,
                 moving_avg: int,
                 num_channels: int,
                 individual: bool,
                 with_revin: bool,
                 eps: float,
                 num_models: int = 1):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task = task
        self.input_len = input_len
        self.output_len = output_len
        self.num_channels = num_channels
        # Series decomposition block from Autoformer
        self.decomposition = series_decomp(moving_avg)
        self.individual = individual
        self.with_revin = with_revin
        self.eps = eps

        if self.individual:
            self.LinearSeasonal = nn.ModuleList()
            self.LinearTrend = nn.ModuleList()

            for i in range(self.channels):
                self.LinearSeasonal.append(
                    nn.Linear(self.input_len, self.output_len))
                self.LinearTrend.append(
                    nn.Linear(self.input_len, self.output_len))

                self.LinearSeasonal[i].weight = nn.Parameter(
                    (1 / self.input_len) * torch.ones([self.output_len, self.input_len]))
                self.LinearTrend[i].weight = nn.Parameter(
                    (1 / self.input_len) * torch.ones([self.output_len, self.input_len]))
        else:
            self.LinearSeasonal = nn.Linear(self.input_len, self.output_len)
            self.LinearTrend = nn.Linear(self.input_len, self.output_len)

            self.LinearSeasonal.weight = nn.Parameter(
                (1 / self.input_len) * torch.ones([self.output_len, self.input_len]))
            self.LinearTrend.weight = nn.Parameter(
                (1 / self.input_len) * torch.ones([self.output_len, self.input_len]))

    def encoder(self, x):
        seasonal_init, trend_init = self.decomposition(x)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.output_len],
                                          dtype=seasonal_init.dtype)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.output_len],
                                       dtype=trend_init.dtype)
            for i in range(self.channels):
                seasonal_output[:, :, i] = self.LinearSeasonal[i](
                    seasonal_init[:, :, i])
                trend_output[:, :, i] = self.LinearTrend[i](
                    trend_init[:, :, i])
        else:
            seasonal_output = self.LinearSeasonal(seasonal_init)
            trend_output = self.LinearTrend(trend_init)
        x = seasonal_output + trend_output
        return x

    def predict(self, x):
        if self.with_revin:
            means = x.mean(dim=-1, keepdim=True)
            stds = x.std(dim=-1, keepdim=True)
            x = (x - means) / (stds + self.eps)
        y_hat = self.encoder(x)
        if self.with_revin:
            y_hat = y_hat * (stds + self.eps) + means
        return y_hat

    def forward(self, x, y, x_mark=None, y_mark=None):
        results = {}
        y_hat = self.predict(x)
        results['y_hat'] = y_hat
        results['predloss'] = F.l1_loss(y_hat, y, reduction='none').mean(-1)
        results['recloss'] = F.mse_loss(torch.zeros_like(x), torch.zeros_like(x), reduction='none').mean(-1)
        # results['normloss'] = ((self.LinearSeasonal.weight ** 2).mean(0) ** 0.5).mean()
        # results['normloss'] += ((self.LinearTrend.weight ** 2).mean(0) ** 0.5).mean()
        results['normloss'] = F.mse_loss(self.LinearSeasonal.weight, torch.zeros_like(self.LinearSeasonal.weight))
        results['normloss'] += F.mse_loss(self.LinearTrend.weight, torch.zeros_like(self.LinearTrend.weight))
        return results
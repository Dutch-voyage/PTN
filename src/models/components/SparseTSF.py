import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,
                 input_len: int,
                 output_len: int,
                 num_channels: int,
                 period_len: int,
                 task: str,
                 num_models: int = 1,):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = input_len
        self.pred_len = output_len
        self.enc_in = num_channels
        self.period_len = period_len

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

        self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
    def init_params(self):
        pass

    def predict(self, x):
        B = x.shape[0]
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        x = x - mean
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # sparse forecasting
        y_hat = self.linear(x)  # bc,w,m

        # upsampling: bc,w,m -> bc,m,w -> b,c,s
        y_hat = y_hat.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)

        y_hat = y_hat + mean
        return y_hat

    def forward(self, x, y, x_mark=None, y_mark=None, mode='train'):
        y_hat = self.predict(x)
        result = {}
        result['y_hat'] = y_hat
        result['predloss'] = F.l1_loss(y_hat, y, reduction='none').mean(-1)
        result['recloss'] = F.l1_loss(torch.zeros_like(x), torch.zeros_like(x), reduction='none').mean(-1)
        result['normloss'] = torch.zeros_like(result['predloss'])
        return result
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback

class StackParametersManagement(Callback):
    def __init__(self):
        super(StackParametersManagement, self).__init__()

    # def on_fit_start(self, trainer, pl_module):
    #     pl_module.net.init_params()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        pl_module.net.save_parameters()
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MinMetric, MeanMetric
from torchmetrics import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError
from ..utils.SAM import SAM

class LitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        scale: bool,
        scaler: Any,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.scaler = scaler

        # metric objects for calculating and averaging accuracy across batches
        self.train_mse = MeanSquaredError()
        self.train_mae = MeanAbsoluteError()
        self.train_mape = MeanAbsolutePercentageError()

        self.val_mse = nn.ModuleList([MeanSquaredError() for _ in range(2)])
        self.val_mae = nn.ModuleList([MeanAbsoluteError() for _ in range(2)])
        self.val_mape = nn.ModuleList([MeanAbsolutePercentageError() for _ in range(2)])

        self.test_mse = MeanSquaredError()
        self.test_mae = MeanAbsoluteError()
        self.test_mape = MeanAbsolutePercentageError()

        # for averaging loss across batches
        self.train_loss, self.train_predloss, self.train_recloss, self.train_normloss = (
            MeanMetric(),
            MeanMetric(),
            MeanMetric(),
            MeanMetric()
        )
        self.val_loss, self.val_predloss, self.val_recloss, self.val_normloss = (
            nn.ModuleList([MeanMetric() for _ in range(2)]),
            nn.ModuleList([MeanMetric() for _ in range(2)]),
            nn.ModuleList([MeanMetric() for _ in range(2)]),
            nn.ModuleList([MeanMetric() for _ in range(2)])
        )

        self.test_loss, self.test_predloss, self.test_recloss, self.test_normloss = (
            MeanMetric(),
            MeanMetric(),
            MeanMetric(),
            MeanMetric()
        )

        # for tracking best so far validation accuracy
        self.val_mse_best = MinMetric()
        self.test_mse_best = MinMetric()
        self.test_mae_best = MinMetric()

    def forward(self, x: torch.Tensor,
                y: torch.Tensor,
                x_mark: Dict[str, torch.Tensor],
                y_mark: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.net(x, y, x_mark, y_mark)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for item in self.val_predloss:
            item.reset()
        for item in self.val_recloss:
            item.reset()
        for item in self.val_normloss:
            item.reset()
        for item in self.val_loss:
            item.reset()
        for item in self.val_mse:
            item.reset()
        for item in self.val_mae:
            item.reset()
        for item in self.val_mape:
            item.reset()
        self.val_mse_best.reset()

    # def on_after_backward(self) -> None:
    #     for name, param in self.net.named_parameters():
    #         if param.grad is None:
    #             print(name)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y, x_mark, y_mark = batch
        if self.hparams.scale:
            x = self.scaler.transform(x)
            y = self.scaler.transform(y)
        result = self.forward(x, y, x_mark, y_mark)
        result['predloss'] = result['predloss'].mean()
        result['recloss'] = result['recloss'].mean()
        result['normloss'] = result['normloss'].mean()
        return result, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        result, targets = self.model_step(batch)

        preds = result['y_hat']

        # update and log metrics
        self.train_predloss(result['predloss'])
        self.train_recloss(result['recloss'])
        self.train_normloss(result['normloss'])
        loss = result['predloss'] + result['recloss'] + result['normloss']
        self.train_loss(loss)

        self.train_mse(preds, targets)
        self.train_mae(preds, targets)
        self.train_mape(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/predloss", self.train_predloss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recloss", self.train_recloss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/normloss", self.train_normloss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mape", self.train_mape, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        label = {0: "val", 1: "test"}

        result, targets= self.model_step(batch)
        preds = result['y_hat']

        # update and log metrics
        self.val_predloss[dataloader_idx](result['predloss'])
        self.val_recloss[dataloader_idx](result['recloss'])
        self.val_normloss[dataloader_idx](result['normloss'])
        loss = result['predloss'] + result['recloss'] + result['normloss']
        self.val_loss[dataloader_idx](loss)
        # masking out zeros in targets
        self.val_mse[dataloader_idx](preds, targets)
        self.val_mae[dataloader_idx](preds, targets)
        self.val_mape[dataloader_idx](preds, targets)
        self.log(f"{label[dataloader_idx]}/loss", self.val_loss[dataloader_idx], on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        self.log(f"{label[dataloader_idx]}/predloss", self.val_predloss[dataloader_idx], on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        self.log(f"{label[dataloader_idx]}/recloss", self.val_recloss[dataloader_idx], on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        self.log(f"{label[dataloader_idx]}/normloss", self.val_normloss[dataloader_idx], on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        self.log(f"{label[dataloader_idx]}/mse", self.val_mse[dataloader_idx], on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        self.log(f"{label[dataloader_idx]}/mae", self.val_mae[dataloader_idx], on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        self.log(f"{label[dataloader_idx]}/mape", self.val_mape[dataloader_idx], on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        mse = self.val_mse[0].compute()  # get current val acc
        if mse < self.val_mse_best.compute():
            self.log("test/mse_best", self.val_mse[1].compute(), sync_dist=True)
            self.log("test/mae_best", self.val_mae[1].compute(), sync_dist=True)
        self.val_mse_best(mse)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/mse_best", self.val_mse_best.compute(), sync_dist=True, prog_bar=True)


    # def on_test_start(self):
    #     if self.trainer.world_size > 1:
    #         self.trainer.model.net.module.load_params()
    #     else:
    #         self.trainer.model.net.load_params()


    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        result, targets = self.model_step(batch)
        preds = result['y_hat']
        # update and log metrics
        self.test_predloss(result['predloss'])
        self.test_recloss(result['recloss'])
        self.test_normloss(result['normloss'])
        self.test_mse(preds, targets)
        self.test_mae(preds, targets)
        self.test_mape(preds, targets)
        self.log("test/predloss", self.test_predloss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recloss", self.test_recloss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/normloss", self.test_normloss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mape", self.test_mape, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        # self.trainer.model.module.net.init_params()
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        # optimizer = SAM(self.trainer.model.parameters(), optimizer)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = LitModule(None, None, None, None)

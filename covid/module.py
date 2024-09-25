from typing import Any
import torch
from torch.nn.modules.loss import _Loss
from torchmetrics import MetricCollection
from pytorch_lightning import LightningModule


class CovidModuleClassifier(LightningModule):

    def __init__(
        self,
        model: torch.nn.Module,
        *args: Any,
        metrics: MetricCollection | None = None,
        criterion: _Loss | None = None,
        learning_rate: float = 1e-3,
        **kwargs: Any
    ) -> None:

        super().__init__(*args, **kwargs)

        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, inputs):
        return self.model(inputs)

    def step(self, batch):
        images, targets = batch
        logits = self.forward(images)

        preds = torch.argmax(logits, dim=1)
        loss = self.criterion(logits, targets)

        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.log(name="train_loss", value=loss, on_step=False, on_epoch=True)
        self.train_metrics.update(preds, targets)

        return loss

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, on_epoch=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_ids):
        loss, preds, targets = self.step(batch)
        self.log(name="val_loss", value=loss, on_step=False, on_epoch=True)
        self.val_metrics.update(preds, targets)

        return loss

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, on_epoch=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_ids):
        loss, preds, targets = self.step(batch)
        self.log(name="test_loss", value=loss, on_step=False, on_epoch=True)
        self.test_metrics.update(preds, targets)

        return loss

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_epoch=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

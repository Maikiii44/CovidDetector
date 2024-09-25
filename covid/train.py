from datetime import datetime
from pathlib import Path

from pytorch_lightning import (
    Trainer,
    LightningDataModule,
    LightningModule,
    seed_everything,
)
from pytorch_lightning.loggers import Logger, MLFlowLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback


class ModelTrainer:

    def __init__(
        self,
        pl_model: LightningModule,
        pl_datamodule: LightningDataModule,
        run_datadir: str = f"./model_trainer",
    ):

        self.pl_model = pl_model
        self.pl_datamodule = pl_datamodule
        self.run_datadir = Path(run_datadir)

        self._logger = MLFlowLogger(
            experiment_name="CovidDetection",
            run_name=datetime.now().strftime("%Y%m%d_%H%M"),
            tracking_uri="./mlflow",
        )

        self._callbacks_list = [
            EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min"),
            ModelCheckpoint(
                dirpath=Path(self.run_datadir, "checkpoints"),
                filename="{epoch}-{val_loss:.3f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
        ]

        self.trainer = None

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def callback(self) -> Callback:
        return self._callbacks_list

    @classmethod
    def set_seed(cls, seed: int = 42):
        seed_everything(seed=seed)

    def train(self, epochs: int = 20, use_gpu: bool = True):

        self.set_seed()

        self.trainer = Trainer(
            accelerator="gpu" if use_gpu else "auto",
            logger=self.logger,
            callbacks=self.callback,
            max_epochs=epochs,
            devices="auto",
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
        )

        self.trainer.fit(model=self.pl_model, datamodule=self.pl_datamodule)

        return self.pl_model

    def test(self):
        if self.trainer is None:
            raise ValueError("The model has not been trained, Please call train first")

        results = self.trainer.test(dataloaders=self.pl_datamodule, ckpt_path="best")

        return results

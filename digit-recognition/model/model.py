from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adadelta
from torchmetrics.classification import MulticlassAccuracy, MulticlassROC


class CnnMNIST(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        # (0-9) classes
        self.n_classes = cfg.model.n_classes

        # define layers for MNIST
        # mnist images are (1, 28, 28) (channels, width, height)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, self.n_classes)

        # loss
        self.loss = nn.CrossEntropyLoss()

        # metric
        self.accuracy = MulticlassAccuracy(self.n_classes)
        self.roc = MulticlassROC(self.n_classes)

        # optimizer parameters
        self.lr = cfg.training.lr

        # save hyperparameters to self.hparams
        self.save_hyperparameters()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def training_step(self, batch, batch_idx):
        _, loss, accuracy = self._get_pred_loss_metric(batch)

        # Log loss and metric
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        pred, loss, accuracy = self._get_pred_loss_metric(batch)

        # Log loss and metric
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)

        # Let's return pred to use it in a custom callback
        return pred

    def test_step(self, batch, batch_idx):
        _, loss, accuracy = self._get_pred_loss_metric(batch)
        # Log loss and metric
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

    def configure_optimizers(self):
        return Adadelta(self.parameters(), lr=self.lr)

    def _get_pred_loss_metric(self, batch):
        data, target = batch
        output = self(data)
        pred = torch.argmax(output, dim=1)
        loss = self.loss(output, target)
        accuracy = self.accuracy(output, target)
        return pred, loss, accuracy


class LogPredictionsCallback(Callback):
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        logger = pl_module.logger
        if isinstance(logger, WandbLogger):
            if batch_idx == 0:
                n = 8
                data, target = batch
                images = [img for img in data[:n]]
                captions = [
                    f"Ground Truth: {target_i} - Prediction: {output_i}"
                    for target_i, output_i in zip(target[:n], outputs[:n])
                ]
                logger.log_image(
                    key="sample_images", images=images, caption=captions
                )

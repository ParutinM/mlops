from __future__ import print_function

import subprocess

import hydra
import torch
from dvc.repo import Repo
from ml_utils.model import CnnMNIST, LogPredictionsCallback
from ml_utils.utils import data_exists
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@hydra.main(
    config_path="configs", config_name="config", version_base="1.3"
)
def main(cfg: DictConfig):
    """
    Training model
    :param cfg:             config
    """
    # set seed
    torch.manual_seed(cfg.training.seed)

    # transformation of MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    if not data_exists():
        # pull data from DVC
        repo = Repo(".")
        repo.pull()

    # split to train and val parts
    dataset = datasets.MNIST(
        cfg.data.path, train=True, transform=transform
    )
    training_set, validation_set = random_split(dataset, [55000, 5000])

    # create loaders
    training_loader = DataLoader(
        training_set, batch_size=cfg.training.batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        validation_set, batch_size=cfg.training.batch_size
    )

    # create model
    model = CnnMNIST(cfg)

    # create loggers
    loggers = [
        MLFlowLogger(
            cfg.model.name,
            tracking_uri="file:./.logs/mlflow-logs",
            run_name=subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode(),
        ),
        WandbLogger(
            project=cfg.model.name,
            config={
                "learning_rate": cfg.training.lr,
                "epochs": cfg.training.epochs,
                "batch_size": cfg.training.batch_size,
            },
        ),
    ]

    # create callbacks
    callbacks = [
        LogPredictionsCallback(),
        ModelCheckpoint(monitor="val_accuracy"),
    ]

    # train model
    trainer = Trainer(
        logger=loggers,
        callbacks=callbacks,
        accelerator=cfg.training.accelerator,
        max_epochs=cfg.training.epochs,
    )

    trainer.fit(model, training_loader, validation_loader)

    # saving model
    if cfg.training.save_model:
        trainer.save_checkpoint(f"./results/{cfg.model.name}.ckpt")


if __name__ == "__main__":
    main()

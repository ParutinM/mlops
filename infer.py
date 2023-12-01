from __future__ import print_function

import subprocess
from pathlib import Path

import hydra
import torch
from dvc.repo import Repo
from ml_utils.model import CnnMNIST
from ml_utils.utils import data_exists
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@hydra.main(
    config_path="configs", config_name="config", version_base="1.3"
)
def main(cfg: DictConfig):
    """
    Inferring model
    :param cfg:         config
    """
    # set seed
    torch.manual_seed(cfg.training.seed)

    # transformation of MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    if (
        not data_exists()
        or not Path(f"./results/{cfg.model.name}.ckpt").exists()
    ):
        # pull data from DVC
        repo = Repo(".")
        repo.pull()

    # split to train and val parts
    inferring_set = datasets.MNIST(
        cfg.data.path, train=False, transform=transform
    )

    # create loaders
    inferring_loader = DataLoader(
        inferring_set, batch_size=cfg.training.batch_size
    )

    # create and load model
    model = CnnMNIST.load_from_checkpoint(
        f"./results/{cfg.model.name}.ckpt"
    )
    model.eval()

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
            version=subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode(),
            config={
                "learning_rate": cfg.training.lr,
                "epochs": cfg.training.epochs,
                "batch_size": cfg.training.batch_size,
            },
        ),
    ]

    # create callbacks
    callbacks = [ModelCheckpoint(monitor="test_accuracy", mode="max")]

    # infer model
    trainer = Trainer(
        logger=loggers,
        callbacks=callbacks,
        accelerator=cfg.training.accelerator,
        max_epochs=cfg.training.epochs,
    )

    trainer.test(model, inferring_loader)


if __name__ == "__main__":
    main()

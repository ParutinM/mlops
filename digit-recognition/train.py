from __future__ import print_function

import subprocess
from pathlib import Path

import hydra
import torch
from data.dataset import MNISTDataset, load_from_DVC
from model.model import CnnMNIST, LogPredictionsCallback
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


@hydra.main(config_path=".", config_name="config", version_base="1.3")
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

    # load data from DVC if it needs or data not exists
    load_from_DVC(
        folder_path=str(Path(cfg.data.path) / cfg.data.raw_folder),
        file_names=[cfg.data.train.images, cfg.data.train.labels],
        replace_if_exists=not cfg.training.use_local_data,
    )

    # create dataset
    dataset = MNISTDataset(
        root=cfg.data.path,
        image_file=cfg.data.train.images,
        label_file=cfg.data.train.labels,
        transform=transform,
    )

    # split to train and val parts
    training_set, validation_set = random_split(dataset, [55000, 5000])

    # create loaders
    training_loader = DataLoader(
        training_set,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.data_loader_num_workers,
        persistent_workers=True,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.data_loader_num_workers,
        persistent_workers=True,
    )

    # create model
    model = CnnMNIST(cfg)

    # create Wandb logger
    loggers = [
        WandbLogger(
            name=subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"]
            )
            .strip()
            .decode(),
            project=cfg.model.task,
        ),
        MLFlowLogger(
            experiment_name=cfg.model.task,
            tracking_uri=cfg.training.mlflow_server,
            run_name=subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"]
            )
            .strip()
            .decode(),
        ),
    ]

    # create callbacks
    # LogPredictionsCallback - to show images with labels in Wandb
    # ModelCheckpoint - to save best model according to best accuracy
    callbacks = [
        ModelCheckpoint(
            monitor="val_accuracy",
            mode="max",
            dirpath=Path(cfg.model.path) / cfg.model.checkpoints_path,
            filename="_".join([cfg.model.name, "best"]),
            save_top_k=1,
            enable_version_counter=False,
            save_on_train_epoch_end=True,
        ),
        LogPredictionsCallback(),
    ]

    # train model
    trainer = Trainer(
        logger=loggers,
        callbacks=callbacks,
        precision=32,
        accelerator=cfg.training.accelerator,
        max_epochs=cfg.training.epochs,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, training_loader, validation_loader)


if __name__ == "__main__":
    main()

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
from torch.utils.data import DataLoader
from torchvision import transforms


@hydra.main(config_path=".", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    Inferring model
    :param cfg:         config
    """
    # set seed
    torch.manual_seed(cfg.inferring.seed)

    # transformation of MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # load data from DVC if it needs or data not exists
    load_from_DVC(
        folder_path=str(Path(cfg.data.path) / cfg.data.raw_folder),
        file_names=[cfg.data.infer.images, cfg.data.infer.labels],
        replace_if_exists=not cfg.inferring.use_local_data,
    )

    # create dataset fot inferring
    inferring_set = MNISTDataset(
        root=cfg.data.path,
        image_file=cfg.data.infer.images,
        label_file=cfg.data.infer.labels,
        transform=transform,
    )

    # create loaders
    inferring_loader = DataLoader(
        inferring_set,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.data_loader_num_workers,
        persistent_workers=True,
    )

    # load model checkpoint from DVC if it needs or model not exists
    model_ckpt_path = (
        Path(cfg.model.path)
        / cfg.model.checkpoints_path
        / cfg.model.best_checkpoint_name
    )

    load_from_DVC(
        folder_path=str(Path(cfg.model.path) / cfg.model.checkpoints_path),
        file_names=[cfg.model.best_checkpoint_name],
        replace_if_exists=not cfg.inferring.use_local_data,
    )

    # load model from checkpoint
    model: CnnMNIST = CnnMNIST.load_from_checkpoint(model_ckpt_path)
    model.eval()

    # create loggers
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
    callbacks = [
        ModelCheckpoint(monitor="test_accuracy", mode="max"),
        LogPredictionsCallback(),
    ]

    # infer model
    trainer = Trainer(
        logger=loggers,
        callbacks=callbacks,
        accelerator=cfg.training.accelerator,
        max_epochs=cfg.training.epochs,
    )
    trainer.test(model, inferring_loader)

    # save to onnx
    dummy_input = torch.randn(1, 1, 28, 28)
    input_names = ["image"]
    output_names = ["class"]
    torch.onnx.export(
        model,
        dummy_input,
        f"{cfg.model.path}/mnist.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names,
    )


if __name__ == "__main__":
    main()

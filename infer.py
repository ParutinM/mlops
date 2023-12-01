from __future__ import print_function

import hydra
import pandas as pd
import torch
from dvc.repo import Repo
from ml_utils.nets import Net
from ml_utils.utils import test_model
from omegaconf import DictConfig
from torchvision import datasets, transforms


@hydra.main(
    config_path="configs", config_name="config", version_base="1.3"
)
def infer(cfg: DictConfig):
    """
    Inferring model
    :param cfg:         config
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    repo = Repo(".")
    repo.pull()

    dataset2 = datasets.MNIST(
        cfg.data.path, train=False, download=False, transform=transform
    )

    test_kwargs = {"batch_size": len(dataset2)}

    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net()
    model.load_state_dict(torch.load(f"results/{cfg.model.name}.pt"))

    output = test_model(model, test_loader)
    pd.DataFrame(output, columns=["class"]).to_csv(
        "results/predictions.csv", header=False, index=False
    )


if __name__ == "__main__":
    infer()

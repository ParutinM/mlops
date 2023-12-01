from __future__ import print_function

import hydra
import torch
import torch.optim as optim
from dvc.repo import Repo
from ml_utils.nets import Net
from ml_utils.utils import train_model
from omegaconf import DictConfig
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms


@hydra.main(
    config_path="configs", config_name="config", version_base="1.3"
)
def train(cfg: DictConfig):
    """
    Training model
    :param cfg:             config
    """
    use_cuda = not cfg.training.no_cuda and torch.cuda.is_available()
    use_mps = not cfg.training.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(cfg.training.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": cfg.training.batch_size}

    if use_cuda:
        cuda_kwargs = {
            "num_workers": 1,
            "pin_memory": True,
            "shuffle": True,
        }
        train_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    repo = Repo(".")
    repo.pull()

    dataset1 = datasets.MNIST(
        cfg.data.path, train=True, download=False, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=cfg.training.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=cfg.training.gamma)
    for epoch in range(1, cfg.training.epochs + 1):
        train_model(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            cfg.training.log_interval,
            cfg.training.dry_run,
        )
        scheduler.step()

    if cfg.training.save_model:
        torch.save(model.state_dict(), f"results/{cfg.model.name}.pt")


if __name__ == "__main__":
    train()

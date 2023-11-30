from __future__ import print_function

import fire
import torch
import torch.optim as optim
from ml_utils.nets import Net
from ml_utils.utils import train_model
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms


def train(
    batch_size: int = 64,
    epochs: int = 14,
    lr: float = 1.0,
    gamma: float = 0.7,
    no_cuda: bool = False,
    no_mps: bool = False,
    log_interval: int = 10,
    seed: int = 1,
    dry_run: bool = False,
    save_model: bool = True,
    model_name: str = "mnist_cnn",
):
    """
    Training model
    :param batch_size:      input batch size for training (default: 64)
    :param epochs:          number of epochs to train (default: 14)
    :param lr:              learning rate (default: 1.0)
    :param gamma:           learning rate step gamma (default: 0.7)
    :param no_cuda:         disables CUDA training
    :param no_mps:          disables macOS GPU training
    :param log_interval:    how many batches to wait before logging
    :param seed:            random seed (default: 1)
    :param dry_run:         quickly check a single pass
    :param save_model:      for saving the current model
    :param model_name:      model name
    """

    use_cuda = not no_cuda and torch.cuda.is_available()
    use_mps = not no_mps and torch.backends.mps.is_available()

    torch.manual_seed(seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": batch_size}

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

    dataset1 = datasets.MNIST(
        "data", train=True, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train_model(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            log_interval,
            dry_run,
        )
        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), f"results/{model_name}.pt")


if __name__ == "__main__":
    fire.Fire(train)

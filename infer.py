from __future__ import print_function

import fire
import pandas as pd
import torch
from ml_utils.nets import Net
from ml_utils.utils import test_model
from torchvision import datasets, transforms


def infer(model_name: str = "mnist_cnn.pt"):
    """
    Inferring model
    :param model_name:       model name
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset2 = datasets.MNIST(
        "data", train=False, download=True, transform=transform
    )

    test_kwargs = {"batch_size": len(dataset2)}

    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net()
    model.load_state_dict(torch.load(f"results/{model_name}"))

    output = test_model(model, test_loader)
    pd.DataFrame(output, columns=["class"]).to_csv(
        "results/predictions.csv", header=False, index=False
    )


if __name__ == "__main__":
    fire.Fire(infer)

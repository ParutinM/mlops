from __future__ import print_function

import argparse

import pandas as pd
import torch
from ml_utils.nets import Net
from ml_utils.utils import test
from torchvision import datasets, transforms


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST infer")
    parser.add_argument(
        "--model-name",
        type=str,
        default="mnist_cnn.pt",
        metavar="N",
        help="name of file with model",
    )

    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset2 = datasets.MNIST(
        "data", train=False, download=True, transform=transform
    )

    test_kwargs = {"batch_size": len(dataset2)}

    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net()
    model.load_state_dict(torch.load(f"results/{args.model_name}"))

    output = test(model, test_loader)
    pd.DataFrame(output, columns=["class"]).to_csv(
        "results/predictions.csv", header=False, index=False
    )


if __name__ == "__main__":
    main()

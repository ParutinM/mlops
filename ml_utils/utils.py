from pathlib import Path


def data_exists() -> bool:
    return (
        Path("./data/MNIST/raw/t10k-images-idx3-ubyte").exists()
        and Path("./data/MNIST/raw/t10k-labels-idx1-ubyte").exists()
        and Path("./data/MNIST/raw/train-images-idx3-ubyte").exists()
        and Path("./data/MNIST/raw/train-labels-idx1-ubyte").exists()
    )

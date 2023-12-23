import os
from pathlib import Path
from typing import Any, Callable, List, Tuple

from dvc.api import DVCFileSystem
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.mnist import read_image_file, read_label_file


def load_from_DVC(
    folder_path: str, file_names: List[str], replace_if_exists: bool = True
):
    fs = DVCFileSystem(folder_path)
    for file_name in file_names:
        path = Path(os.path.join(folder_path, file_name))
        if not path.exists() or replace_if_exists:
            # delete if exists
            path.unlink(missing_ok=True)

            # get from DVC
            fs.get_file(path, path)


class MNISTDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        image_file: str,
        label_file: str,
        transform: Callable[[...], Any] = None,
        target_transform: Callable[[...], Any] = None,
    ):
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
        )
        self.image_file = image_file
        self.label_file = label_file
        self.raw_folder = os.path.join(root, "MNIST/raw")
        self.train = "train" in image_file
        self.data, self.targets = self._load_data()

    def _load_data(self):
        data = read_image_file(
            os.path.join(self.raw_folder, self.image_file)
        )
        targets = read_label_file(
            os.path.join(self.raw_folder, self.label_file)
        )
        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

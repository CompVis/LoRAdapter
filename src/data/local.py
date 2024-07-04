import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path


def sort_key(p: Path):
    try:
        return int(p.stem)
    except:
        return p.stem


class ImageFolderDataset(Dataset):
    def __init__(self, directory: Path, transform, caption_from_name: bool, caption_prefix: str):
        """
        Args:
            directory (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transform
        self.image_paths = [directory / file for file in os.listdir(directory) if file.endswith(("jpg", "jpeg", "png"))]
        self.image_paths.sort(key=sort_key)
        self.caption_from_name = caption_from_name
        self.caption_prefix = caption_prefix

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        txt_path = image_path.with_suffix(".txt")

        if self.caption_from_name:
            label = self.caption_prefix + image_path.stem.split("_")[0].replace("-", " ")
        else:
            try:
                with open(txt_path, "r") as f:
                    label = f.read()
            except:
                label = ""

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"jpg": image, "caption": label}


class ZipDataset(Dataset):
    def __init__(self, datasets: list[ImageFolderDataset]):
        # Ensure all datasets have the same length
        assert all(len(datasets[0]) == len(d) for d in datasets), "Datasets must all be the same length!"
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx: int):
        # Return a tuple containing elements from each dataset at the given index
        if len(self.datasets) == 1:
            return self.datasets[0][idx]

        return tuple(d[idx] for d in self.datasets)


class ImageDataModule:
    def __init__(self, directories: list[str], transform: list, batch_size: int = 32, caption_from_name: bool = False, caption_prefix: str = ""):
        super().__init__()
        project_root = Path(os.path.abspath(__file__)).parent.parent.parent
        self.val_dataset = ZipDataset(
            [ImageFolderDataset(directory=Path(project_root, d), transform=transforms.Compose(transform), caption_from_name=caption_from_name, caption_prefix=caption_prefix) for d in directories]
        )
        self.batch_size = batch_size

    def train_dataloader(self):
        raise Exception("Not implemented")

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        raise Exception("Not implemented")

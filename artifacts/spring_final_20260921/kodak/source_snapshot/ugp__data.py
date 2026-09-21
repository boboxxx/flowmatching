from pathlib import Path
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class ImageDataset(Dataset):
    def __init__(self, root: str, crop_size: int = 256, train: bool = True, limit: int = 0):
        self.paths = sorted(
            p for p in Path(root).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if limit:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"no images found under {root}")
        self.crop_size = crop_size
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        width, height = image.size
        size = self.crop_size
        if min(width, height) < size:
            scale = size / min(width, height)
            image = image.resize((round(width * scale), round(height * scale)), Image.BICUBIC)
            width, height = image.size
        if self.train:
            left = random.randint(0, width - size)
            top = random.randint(0, height - size)
        else:
            left = (width - size) // 2
            top = (height - size) // 2
        image = TF.crop(image, top, left, size, size)
        if self.train and random.random() < 0.5:
            image = TF.hflip(image)
        return TF.to_tensor(image), self.paths[index].name


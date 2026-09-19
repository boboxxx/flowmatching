import hashlib
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def paths_at(root):
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Images(Dataset):
    def __init__(self, paths, training=False):
        self.paths, self.training = list(paths), training
        if not self.paths:
            raise ValueError("empty image split")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            image = source.convert("RGB")
            w, h = image.size
            if min(w, h) < 256:
                raise ValueError("No implicit resizing: image must support a 256x256 crop")
            left = random.randint(0, w-256) if self.training else (w-256)//2
            top = random.randint(0, h-256) if self.training else (h-256)//2
            image = image.crop((left, top, left+256, top+256))
            if self.training and random.random() < .5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            array = np.asarray(image, dtype=np.float32).copy() / 255
        return torch.from_numpy(array).permute(2, 0, 1), self.paths[index].name


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

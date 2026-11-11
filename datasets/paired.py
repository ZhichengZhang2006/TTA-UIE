import hashlib
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.manifest import RESAMPLE, read_csv


def source_rows(config, split):
    rows = [
        row for row in read_csv(config["data"]["manifest"])
        if row["dataset"].upper() == "LSUI" and row["split"] == split
        and row["valid_pair"].lower() == "true"
    ]
    if not rows:
        raise ValueError(f"No valid LSUI rows in split {split}")
    return rows


class PairedImages(Dataset):
    def __init__(self, config, split, augment=False, limit=None):
        self.rows = source_rows(config, split)[:limit]
        self.height, self.width = config["input"]["shape"][1:]
        self.augment = augment
        self.cache = Path(config["data"]["cache"]) if config["data"].get("cache") else None
        self.resized = None

    def resize(self, filename):
        with Image.open(filename) as image:
            resized = image.convert("RGB").resize((self.width, self.height), RESAMPLE.BILINEAR)
            return np.asarray(resized)

    def read_pair(self, row):
        return np.stack([self.resize(row[key]) for key in ("input_path", "reference_path")])

    def prepare_cache(self, workers):
        if self.cache is None:
            return
        digest = hashlib.sha256(f"pillow-{Image.__version__}:bilinear:{self.width}x{self.height}".encode())
        for row in self.rows:
            digest.update(f"{row['image_id']}:{row['sha256_input']}:{row['sha256_ref']}\n".encode())
        self.cache.mkdir(parents=True, exist_ok=True)
        final = self.cache / f"{digest.hexdigest()}.npy"
        if not final.is_file():
            temporary = self.cache / f"{uuid.uuid4().hex}.tmp"
            try:
                target = np.lib.format.open_memmap(
                    temporary, mode="w+", dtype=np.uint8,
                    shape=(len(self.rows), 2, self.height, self.width, 3),
                )
                # Bound outstanding decoded images as well as worker concurrency.
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    for start in range(0, len(self.rows), 8 * workers):
                        block = self.rows[start:start + 8 * workers]
                        for index, pair in enumerate(executor.map(self.read_pair, block), start):
                            target[index] = pair
                target.flush()
                del target
                os.replace(temporary, final)
            finally:
                if temporary.exists():
                    temporary.unlink()
        self.resized = np.load(final, mmap_mode="r")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.resized is None:
            pair = self.read_pair(row)
        else:
            pair = self.resized[index].copy()
        tensor = torch.from_numpy(pair).permute(0, 3, 1, 2)
        if self.augment:
            flags = int(torch.randint(8, ()))
            if flags & 1:
                tensor = tensor.flip(-1)
            if flags & 2:
                tensor = tensor.flip(-2)
            if flags & 4:
                tensor = tensor.transpose(-1, -2)
        return tensor[0], tensor[1], row["image_id"], row["group_id"]

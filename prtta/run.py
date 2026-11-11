import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.paired import PairedImages
from prtta.model import build_model


def load_config(path, seed=None):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if seed is not None:
        config["seed"] = seed
    return config


def configure(config):
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(config["hardware"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return device


def stable_seed(*values):
    raw = json.dumps(values, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_revision():
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    digest = hashlib.sha256()
    files = [*root.glob("*.py")]
    for directory in ("datasets", "prtta", "experiments", "tools", "tests"):
        files.extend((root / directory).rglob("*.py"))
    for path in sorted(files):
        digest.update(f"{path.relative_to(root).as_posix()}:{sha_file(path)}\n".encode())
    return {"git_commit": commit, "python_sources_sha256": digest.hexdigest()}


def run_info(config, device):
    packages = ["torch", "numpy", "Pillow", "PyYAML", "matplotlib", "pytest"]
    installed = {}
    for package in packages:
        try:
            installed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed[package] = None
    return {
        "config": config, **code_revision(),
        "manifest_sha256": sha_file(config["data"]["manifest"]),
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": installed, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
    }


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_save(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def checkpoint_path(config, suffix=""):
    return Path(config["paths"]["checkpoints"]) / f"source_seed{config['seed']}{suffix}.pt"


def load_static(config, device, checkpoint=None):
    checkpoint = checkpoint or checkpoint_path(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["config"]["static"] != config["static"]:
        raise ValueError("Static model/training configuration differs from the checkpoint")
    if state["manifest_sha256"] != sha_file(config["data"]["manifest"]):
        raise ValueError("Manifest differs from the one used for training")
    model = build_model(config, device)
    model.load_state_dict(state["model"], strict=True)
    return model.eval(), state


def loader(config, dataset, device, batch_size, shuffle=False, seed=None, workers=None):
    if workers is None:
        workers = config["data"]["loader_workers"]
    generator = torch.Generator().manual_seed(config["seed"] if seed is None else seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
        num_workers=workers, pin_memory=device.type == "cuda",
    )


def make_dataset(config, split, augment=False, limit=None):
    dataset = PairedImages(config, split, augment=augment, limit=limit)
    dataset.prepare_cache(config["data"]["manifest_workers"])
    return dataset


def tensors(batch, device):
    return tuple(
        part.to(device=device, memory_format=torch.channels_last, non_blocking=True).float().div_(255)
        for part in batch[:2]
    )

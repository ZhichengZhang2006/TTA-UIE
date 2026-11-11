from dataclasses import dataclass

import torch
from torch.nn import functional as F


def srgb_to_linear(value):
    return torch.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(value):
    power = value.clamp_min(0.0031308).pow(1 / 2.4)
    return torch.where(value <= 0.0031308, 12.92 * value, 1.055 * power - 0.055)


@dataclass(frozen=True)
class Physics:
    transmission: torch.Tensor
    background: torch.Tensor
    confidence: torch.Tensor


def min_pool(value, patch):
    return -F.max_pool2d(-value, patch, stride=1, padding=patch // 2)


@torch.no_grad()
def estimate(image, options):
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Physics estimator needs NCHW RGB")
    patch = options["dark_patch"]
    if patch < 1 or patch % 2 != 1:
        raise ValueError("Dark patch size must be positive and odd")
    image = image.detach().float()
    count = max(1, round(0.001 * image.shape[-2] * image.shape[-1]))
    gb = image[:, 1:3]
    dark = min_pool(gb.amin(dim=1, keepdim=True), patch)
    # Stable sort resolves the frequent dark-channel ties by raster index.
    indices = dark.flatten(2).argsort(dim=-1, descending=True, stable=True)[..., :count]
    selected = image.flatten(2).gather(2, indices.expand(-1, 3, -1))
    ordered = selected.sort(dim=-1, stable=True).values
    background = ordered[..., (count - 1) // 2][:, :, None, None].clamp_min(0.05)
    normalized = min_pool((gb / background[:, 1:3]).amin(dim=1, keepdim=True), patch)
    transmission = (1 - options["omega"] * normalized).clamp(*options["t_clip"])
    transmission = F.avg_pool2d(transmission, 5, stride=1, padding=2).clamp(*options["t_clip"])
    confidence = (image.amax(dim=1, keepdim=True) < 0.98).float()
    return Physics(transmission, srgb_to_linear(background), confidence)


def reproject(recovery_linear, transmission, background):
    t = transmission.detach()
    return t * recovery_linear + (1 - t) * background.detach()


def loss(input_srgb, output_srgb, physics, epsilon):
    reprojection = reproject(srgb_to_linear(output_srgb), physics.transmission, physics.background)
    residual = srgb_to_linear(input_srgb.detach()) - reprojection
    penalty = (residual.square() + epsilon**2).sqrt() - epsilon
    confidence = physics.confidence.detach()
    return (confidence * penalty.mean(dim=1, keepdim=True)).sum() / (confidence.sum() + 1e-12)

"""Controlled source-reference degradation and dual attenuation reprojection."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from prtta.physics import linear_to_srgb, srgb_to_linear


@dataclass(frozen=True)
class DualPhysics:
    transmission: torch.Tensor
    backscatter_transmission: torch.Tensor
    background: torch.Tensor
    confidence: torch.Tensor


def dual_loss(image, output, hypothesis, epsilon):
    td = hypothesis.transmission.detach()
    tb = hypothesis.backscatter_transmission.detach()
    background = hypothesis.background.detach()
    mask = hypothesis.confidence.detach()
    reconstructed = td * srgb_to_linear(output) + (1 - tb) * background
    residual = srgb_to_linear(image.detach()) - reconstructed
    penalty = (residual.square() + epsilon**2).sqrt() - epsilon
    return (mask * penalty.mean(1, keepdim=True)).sum() / (mask.sum() + 1e-12)


@torch.no_grad()
def synthesize(reference, settings, seed, dual):
    generator = torch.Generator().manual_seed(seed)
    size = settings["depth_grid"]
    low, high = settings["depth_range"]
    coarse = torch.rand(1, 1, size, size, generator=generator)
    depth = F.interpolate(coarse, size=reference.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
    depth = (low + (high - low) * depth).to(reference)
    lower = torch.tensor(settings["beta_low"]).view(1, 3, 1, 1)
    upper = torch.tensor(settings["beta_high"]).view(1, 3, 1, 1)
    beta = (lower + (upper - lower) * torch.rand(1, 3, 1, 1, generator=generator)).to(reference)
    beta_b = beta * (settings["dual_beta_ratio"] if dual else 1)
    b_low, b_high = settings["background_range"]
    background = (b_low + (b_high - b_low) * torch.rand(1, 3, 1, 1, generator=generator)).to(reference)
    td, tb = torch.exp(-beta * depth), torch.exp(-beta_b * depth)
    linear = td * srgb_to_linear(reference) + (1 - tb) * background
    noise = torch.randn(linear.shape, generator=generator).to(reference) * settings["noise_std"]
    noisy = linear + noise
    saturation = float(((noisy < 0) | (noisy > 1)).float().mean())
    image = linear_to_srgb(noisy.clamp(0, 1))
    hypothesis = DualPhysics(td, tb, background, torch.ones_like(depth))
    metadata = dict(
        generation_seed=seed, beta_direct=beta.flatten().tolist(),
        beta_backscatter=beta_b.flatten().tolist(), background_linear=background.flatten().tolist(),
        noise_std=settings["noise_std"], clipped_fraction=saturation,
    )
    return image, hypothesis, metadata

"""Small bounded shared/dual/effective operators and a matched residual control."""
import torch
from torch import nn
from torch.nn import functional as F

from prtta.physics import linear_to_srgb, srgb_to_linear
from prtta.synthetic import DualPhysics


def layer(source, target, stride=1):
    return nn.Sequential(nn.Conv2d(source, target, 3, stride, 1), nn.GroupNorm(4, target), nn.SiLU())


class OperatorEstimator(nn.Module):
    def __init__(self, mode, channels=(16, 32, 64), resolution=64):
        super().__init__()
        self.mode, self.resolution = mode, resolution
        a, b, c = channels
        self.encoder = nn.Sequential(layer(6 if mode in ("effective", "residual") else 3, a),
                                     layer(a, b, 2), layer(b, c, 2), layer(c, c))
        self.decoder = nn.Sequential(layer(c, b), layer(b, a))
        self.field = nn.Conv2d(a, 3 if mode in ("shared", "residual") else 6, 1)
        self.background = nn.Linear(c, 3)

    def forward(self, image, static=None):
        value = torch.cat((image, static), 1) if self.mode in ("effective", "residual") else image
        latent = self.encoder(F.interpolate(value, (self.resolution, self.resolution), mode="area"))
        features = self.decoder(F.interpolate(latent, (self.resolution, self.resolution), mode="bilinear", align_corners=False))
        field = F.interpolate(self.field(features).float(), image.shape[-2:], mode="bilinear", align_corners=False)
        if self.mode == "residual":
            return (static + 0.1 * field.tanh()).clamp(0, 1)
        field = 0.05 + 0.93 * field.sigmoid()
        td = field[:, :3]
        tb = td if self.mode == "shared" else field[:, 3:]
        background = self.background(latent.mean((-2, -1))).float().sigmoid()[:, :, None, None]
        return DualPhysics(td, tb, background, torch.ones_like(td[:, :1]))


def reproject(reference, hypothesis):
    return hypothesis.transmission * srgb_to_linear(reference) + (1 - hypothesis.backscatter_transmission) * hypothesis.background


def estimation_loss(predicted, truth, reference, image, weights):
    terms = [
        (predicted.transmission - truth.transmission).abs().mean(),
        (predicted.backscatter_transmission - truth.backscatter_transmission).abs().mean(),
        (predicted.background - truth.background).abs().mean(),
        (reproject(reference, predicted) - srgb_to_linear(image)).abs().mean(),
    ]
    return sum(weight * term for weight, term in zip(weights, terms))


@torch.no_grad()
def synthesize_batch(reference, settings, seed, dual):
    generator = torch.Generator(device=reference.device).manual_seed(seed)
    n = reference.shape[0]
    uniform = lambda shape: torch.rand(shape, device=reference.device, generator=generator)
    grid = uniform((n, 1, settings["depth_grid"], settings["depth_grid"]))
    depth = F.interpolate(grid, reference.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
    lo, hi = settings["depth_range"]
    depth = lo + (hi - lo) * depth
    lower = reference.new_tensor(settings["beta_low"]).view(1, 3, 1, 1)
    upper = reference.new_tensor(settings["beta_high"]).view(1, 3, 1, 1)
    beta = lower + (upper - lower) * uniform((n, 3, 1, 1))
    ratio = settings["dual_beta_ratio"] if dual else 1
    td, tb = (-beta * depth).exp(), (-beta * ratio * depth).exp()
    blo, bhi = settings["background_range"]
    background = blo + (bhi - blo) * uniform((n, 3, 1, 1))
    truth = DualPhysics(td, tb, background, torch.ones_like(depth))
    linear = reproject(reference, truth)
    noise = torch.randn(linear.shape, generator=generator, device=reference.device) * settings["noise_std"]
    image = linear_to_srgb((linear + noise).clamp(0, 1))
    return image, truth, dict(seed=seed, beta_direct=beta.flatten(1).tolist(),
        beta_backscatter=(beta * ratio).flatten(1).tolist(), background=background.flatten(1).tolist(),
        clipped_fraction=float(((linear + noise < 0) | (linear + noise > 1)).float().mean()))


@torch.no_grad()
def operator_fusion(image, static, hypothesis, budgets, settings):
    from prtta.output_match import Matched
    inverse = linear_to_srgb(((srgb_to_linear(image) - (1 - hypothesis.backscatter_transmission) * hypothesis.background)
                             / hypothesis.transmission).clamp(0, 1))
    distance = float((inverse - static).square().mean().sqrt())
    results = []
    for budget in budgets:
        valid = distance > 0 and budget <= distance
        alpha = budget / distance if valid else 0
        output = torch.lerp(static, inverse, alpha)
        drift = float((output - static).square().mean().sqrt())
        valid = valid and abs(drift - budget) <= settings["relative_tolerance"] * budget
        results.append(Matched(output if valid else static, valid, "" if valid else "drift_unreachable",
                               alpha if valid else 0, drift if valid else 0, 0, 1, min(distance, budget)))
    return results

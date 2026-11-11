from dataclasses import dataclass

import torch

from prtta.physics import loss


@dataclass(frozen=True)
class Candidate:
    output: torch.Tensor
    step: float
    drift: float
    gradient_norm: float
    direction_norm: float
    status: str

    @property
    def valid(self):
        return self.status in ("accepted", "static")


def snapshot(parameters):
    return tuple(p.detach().clone() for p in parameters)


@torch.no_grad()
def restore(parameters, original):
    for parameter, value in zip(parameters, original):
        parameter.copy_(value)


def block_direction(gradients):
    norms = torch.stack([torch.linalg.vector_norm(grad) for grad in gradients])
    normalized = tuple(
        grad / norm.clamp_min(1e-12) * (norm > 1e-12)
        for grad, norm in zip(gradients, norms)
    )
    return normalized, float(torch.linalg.vector_norm(norms))


def physics_directions(model, image, hypotheses, epsilon):
    parameters = tuple(model.adapter_parameters())
    if model.training or any(not p.requires_grad for p in parameters):
        raise ValueError("Call freeze_for_adaptation before generating a candidate")
    with torch.enable_grad():
        recovery = model(image)
        gradients = []
        for index, hypothesis in enumerate(hypotheses):
            objective = loss(image, recovery, hypothesis, epsilon)
            gradients.append(torch.autograd.grad(objective, parameters, retain_graph=index < len(hypotheses) - 1))
    return gradients


@torch.no_grad()
def apply_candidate(model, image, static, parameters, initial, gradients, options, eligible):
    direction, gradient_norm = block_direction(gradients)
    direction_norm = float(torch.linalg.vector_norm(torch.stack([part.norm() for part in direction])))
    if not eligible:
        return Candidate(static, 0.0, 0.0, gradient_norm, direction_norm, "low_confidence")
    if not torch.isfinite(torch.tensor([gradient_norm, direction_norm])).all():
        return Candidate(static, 0.0, 0.0, 0.0, 0.0, "nonfinite_gradient")
    if direction_norm == 0:
        return Candidate(static, 0.0, 0.0, gradient_norm, 0.0, "zero_direction")
    try:
        for trial in range(options["max_backtracking"] + 1):
            step = options["eta"] / (2**trial)
            restore(parameters, initial)
            for parameter, delta in zip(parameters, direction):
                parameter.add_(delta, alpha=-step)
            output = model(image)
            drift = float((output - static).square().mean().sqrt())
            if drift <= options["output_drift_rms_max"]:
                return Candidate(output, step, drift, gradient_norm, direction_norm, "accepted")
        return Candidate(static, 0.0, 0.0, gradient_norm, direction_norm, "trust_reject")
    finally:
        restore(parameters, initial)


def random_direction(physical_gradients, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = []
    for grad in physical_gradients:
        random_block = torch.randn(grad.shape, generator=generator).to(grad)
        result.append(random_block * (grad.norm() > 1e-12))
    return tuple(result)

"""Reference-free bounded search for a fixed sRGB output drift."""
import math
from dataclasses import dataclass, field

import torch

from prtta.candidate import block_direction, restore, snapshot
from prtta.physics import linear_to_srgb, srgb_to_linear


@dataclass
class Matched:
    output: torch.Tensor
    valid: bool
    reason: str
    alpha: float
    drift: float
    parameter_delta: float
    evaluations: int
    nearest_drift: float
    attempts: list = field(default_factory=list)


@torch.no_grad()
def match_direction(model, image, static, direction, budgets, settings):
    parameters = tuple(model.adapter_parameters())
    initial = snapshot(parameters)
    direction, _ = block_direction(direction)
    direction_norm = math.sqrt(sum(float(part.square().sum()) for part in direction))
    initial_norm = math.sqrt(sum(float(part.square().sum()) for part in initial))
    cap = min(settings["parameter_absolute_cap"], settings["parameter_relative_cap"] * initial_norm)
    if direction_norm == 0 or not math.isfinite(direction_norm):
        return [Matched(static, False, "zero_or_nonfinite_direction", 0, 0, 0, 0, 0) for _ in budgets]
    maximum = min(settings["alpha_max"], cap / direction_norm)
    if maximum < settings["alpha_min"]:
        return [Matched(static, False, "parameter_cap", 0, 0, 0, 0, 0) for _ in budgets]
    points = [(0.0, 0.0)]
    best = [None for _ in budgets]
    evaluations = 0
    attempts = []

    def evaluate(alpha):
        nonlocal evaluations
        restore(parameters, initial)
        for parameter, delta in zip(parameters, direction):
            parameter.add_(delta, alpha=-alpha)
        output = model(image).detach()
        drift = float((output - static).square().mean().sqrt())
        evaluations += 1
        attempts.append(dict(alpha=alpha, drift=drift if math.isfinite(drift) else None,
                             parameter_delta=alpha * direction_norm))
        if not math.isfinite(drift):
            return None
        for i, budget in enumerate(budgets):
            if best[i] is None or abs(drift - budget) < abs(best[i][1] - budget):
                best[i] = (alpha, drift, output)
        return drift

    try:
        grid = torch.logspace(math.log10(settings["alpha_min"]), math.log10(maximum),
                              settings["alpha_grid_steps"], dtype=torch.float64).tolist()
        for alpha in grid:
            drift = evaluate(alpha)
            if drift is not None:
                points.append((alpha, drift))
        # Refine only observed adjacent local crossings. Global monotonicity is not assumed.
        for budget_index, budget in enumerate(budgets):
            if best[budget_index] is not None and abs(best[budget_index][1] - budget) <= settings["relative_tolerance"] * budget:
                continue
            intervals = [
                (a, b) for a, b in zip(points, points[1:])
                if min(a[1], b[1]) <= budget <= max(a[1], b[1])
            ]
            if not intervals:
                continue
            left, right = min(intervals, key=lambda pair: pair[1][0] - pair[0][0])
            for _ in range(settings["refinement_steps"]):
                middle_alpha = (left[0] + right[0]) / 2
                middle_drift = evaluate(middle_alpha)
                if middle_drift is None:
                    break
                middle = (middle_alpha, middle_drift)
                if abs(middle_drift - budget) <= settings["relative_tolerance"] * budget:
                    break
                if min(left[1], middle_drift) <= budget <= max(left[1], middle_drift):
                    right = middle
                else:
                    left = middle
        results = []
        for budget, candidate in zip(budgets, best):
            if candidate is None:
                results.append(Matched(static, False, "nonfinite_output", 0, 0, 0, evaluations, 0, attempts))
                continue
            alpha, drift, output = candidate
            valid = abs(drift - budget) <= settings["relative_tolerance"] * budget
            results.append(Matched(
                output if valid else static, valid, "" if valid else "drift_unreachable",
                alpha if valid else 0, drift if valid else 0,
                alpha * direction_norm if valid else 0, evaluations, drift, attempts,
            ))
        return results
    finally:
        restore(parameters, initial)


@torch.no_grad()
def fusion_candidates(image, static, hypothesis, budgets, settings):
    t, background = hypothesis.transmission, hypothesis.background
    inverse_linear = ((srgb_to_linear(image) - (1 - t) * background) / t).clamp(0, 1)
    inverse = linear_to_srgb(inverse_linear)
    distance = float((inverse - static).square().mean().sqrt())
    outputs = []
    for budget in budgets:
        valid = distance > 0 and budget <= distance
        alpha = budget / distance if valid else 0
        output = torch.lerp(static, inverse, alpha)
        drift = float((output - static).square().mean().sqrt())
        valid = valid and abs(drift - budget) <= settings["relative_tolerance"] * budget
        outputs.append(Matched(output if valid else static, valid, "" if valid else "drift_unreachable",
                               alpha if valid else 0, drift if valid else 0, 0, 1, min(budget, distance)))
    return outputs

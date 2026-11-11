import inspect

import pytest
import torch
from torch import nn

from prtta.output_match import match_direction, fusion_candidates
from prtta.physics import Physics, linear_to_srgb, srgb_to_linear
from prtta.run import load_config
from prtta.synthetic import DualPhysics, dual_loss, synthesize


class LinearAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def adapter_parameters(self):
        yield self.weight

    def forward(self, image):
        return image * self.weight


def test_output_matching_and_full_restore():
    model = LinearAdapter()
    image = torch.ones(1, 3, 12, 12)
    settings = load_config("configs/e1b_output_matched.yaml")
    static = model(image).detach()
    budgets = [0.001, 0.02, 0.8]
    results = match_direction(model, image, static, (torch.ones(()),), budgets, settings)
    assert all(result.valid for result in results[:2])
    assert all(abs(result.drift - b) <= 0.1 * b for result, b in zip(results[:2], budgets))
    assert not results[-1].valid and results[-1].reason == "drift_unreachable"
    assert torch.equal(results[-1].output, static)
    assert model.weight.item() == 0.5
    assert max(result.parameter_delta for result in results) <= 0.125
    repeated = match_direction(model, image, static, (torch.ones(()),), budgets, settings)
    assert torch.equal(results[0].output, repeated[0].output)


def test_zero_direction_and_reference_isolation():
    assert "reference" not in inspect.signature(match_direction).parameters
    assert "reference" not in inspect.signature(fusion_candidates).parameters
    from experiments.e1b_output_matched import physical_gradient
    assert "reference" not in inspect.signature(physical_gradient).parameters
    model = LinearAdapter()
    image = torch.ones(1, 3, 12, 12)
    result, = match_direction(model, image, model(image).detach(), (torch.zeros(()),), [0.001],
                              load_config("configs/e1b_output_matched.yaml"))
    assert not result.valid and result.reason == "zero_or_nonfinite_direction"


def test_fusion_budget_and_unreachable():
    settings = load_config("configs/e1b_output_matched.yaml")
    image = torch.full((1, 3, 12, 12), 0.4)
    hypothesis = Physics(torch.ones(1, 1, 12, 12), torch.zeros(1, 3, 1, 1), torch.ones(1, 1, 12, 12))
    result, = fusion_candidates(image, image, hypothesis, [0.1], settings)
    assert not result.valid
    output, = fusion_candidates(image, image * 0.5, hypothesis, [0.05], settings)
    assert output.valid and abs(output.drift - 0.05) < 1e-6 and 0 <= output.alpha <= 1


def test_synthetic_dual_truth_and_replay():
    reference = torch.rand(1, 3, 32, 32)
    settings = load_config("configs/e1b_output_matched.yaml")["e2"]
    image, truth, _ = synthesize(reference, settings, 2026, True)
    repeated, _, _ = synthesize(reference, settings, 2026, True)
    assert torch.equal(image, repeated)
    assert dual_loss(image, reference, truth, 1e-3).item() < 1e-9
    assert not torch.equal(truth.transmission, truth.backscatter_transmission)
    tied = DualPhysics(truth.transmission, truth.transmission, truth.background, truth.confidence)
    assert dual_loss(image, reference, tied, 1e-3).item() > 1e-5


def test_dual_analytic_gradient():
    output = torch.rand(1, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    image = torch.rand_like(output)
    td = torch.full_like(output, 0.6)
    tb = torch.full_like(output, 0.8)
    b = torch.full((1, 3, 1, 1), 0.3, dtype=torch.float64)
    c = torch.ones(1, 1, 4, 5, dtype=torch.float64)
    actual, = torch.autograd.grad(dual_loss(image, output, DualPhysics(td, tb, b, c), 1e-3), output)
    residual = srgb_to_linear(image) - td * srgb_to_linear(output) - (1 - tb) * b
    slope = torch.where(output <= 0.04045, 1 / 12.92, (2.4 / 1.055) * ((output + 0.055) / 1.055).pow(1.4))
    expected = -td * residual / (residual.square() + 1e-6).sqrt() * slope / output.numel()
    torch.testing.assert_close(actual, expected)

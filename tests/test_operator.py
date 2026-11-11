import inspect

import torch

from prtta.operator_estimator import OperatorEstimator, estimation_loss, synthesize_batch, operator_fusion
from prtta.run import load_config


def test_estimator_bounds_training_and_budget():
    torch.set_num_threads(2)
    image = torch.rand(2, 3, 32, 32)
    for mode in ("shared", "dual", "effective", "residual"):
        net = OperatorEstimator(mode, resolution=16)
        assert sum(p.numel() for p in net.parameters()) < 2_000_000
        output = net(image, image)
        if mode == "residual":
            loss = (output - image).square().mean()
        else:
            assert output.transmission.shape == image.shape
            assert output.background.shape == (2, 3, 1, 1)
            assert 0.05 <= output.transmission.min() <= output.transmission.max() <= 0.98
            assert 0 <= output.background.min() <= output.background.max() <= 1
            loss = output.transmission.mean() + output.background.mean()
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


def test_batch_synthesis_reproducible_and_true_operator():
    settings = load_config("configs/e1b_output_matched.yaml")["e2"]
    reference = torch.rand(3, 3, 32, 32)
    for dual in (False, True):
        image, truth, _ = synthesize_batch(reference, settings, 2026, dual)
        replay, _, _ = synthesize_batch(reference, settings, 2026, dual)
        assert torch.equal(image, replay)
        assert not torch.equal(truth.background[0], truth.background[1])
        assert estimation_loss(truth, truth, reference, image, [1,1,1,1]) < 1e-6
    assert "reference" not in inspect.signature(operator_fusion).parameters

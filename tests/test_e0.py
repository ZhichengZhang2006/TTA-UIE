from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from datasets.manifest import assign_splits, duplicate_links, scan
from experiments.e1_gradient_audit import audit_one
from experiments.e1_report import deliver
from prtta.candidate import apply_candidate, physics_directions, snapshot
from prtta.metrics import SSIM
from prtta.model import UNet4
from prtta.physics import Physics, estimate, linear_to_srgb, loss, reproject, srgb_to_linear
from prtta.run import load_config


@pytest.fixture(scope="module")
def config():
    torch.set_num_threads(2)
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "v1.yaml")


@pytest.fixture
def model(config):
    torch.manual_seed(2026)
    adapter = config["static"]["adapter"]
    net = UNet4([8, 16, 24, 32], adapter["rank"], adapter["alpha"], adapter["positions"])
    # Give both factors nonzero physical gradients as in a trained model.
    with torch.no_grad():
        for parameter in net.adapter_parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    return net


def test_forward_shape_and_bounds(model):
    image = torch.rand(2, 3, 31, 35)
    recovery = model(image)
    assert recovery.shape == image.shape
    assert torch.all((0 <= recovery) & (recovery <= 1))
    assert torch.allclose(SSIM()(recovery, recovery), torch.ones(2))


def test_linear_rgb_conversion():
    srgb = torch.linspace(0, 1, 257, dtype=torch.float64)
    assert torch.allclose(linear_to_srgb(srgb_to_linear(srgb)), srgb, atol=1e-7)
    value = torch.tensor([0.0, 0.04, 0.5, 1.0], dtype=torch.float64)
    expected = torch.tensor([0.0, 0.04 / 12.92, 0.21404114048223255, 1.0], dtype=torch.float64)
    assert torch.allclose(srgb_to_linear(value), expected, atol=1e-12)


def test_physical_forward_and_loss(config):
    image = 0.1 + 0.6 * torch.rand(1, 3, 19, 21)
    physics = estimate(image.requires_grad_(), config["physics"])
    assert physics.transmission.shape == (1, 1, 19, 21)
    assert physics.background.shape == (1, 3, 1, 1)
    assert physics.confidence.shape == (1, 1, 19, 21)
    for tensor in (physics.transmission, physics.background, physics.confidence):
        assert not tensor.requires_grad
    assert 0.05 <= float(physics.transmission.min()) <= float(physics.transmission.max()) <= 0.980001
    linear = srgb_to_linear(image)
    ones = torch.ones_like(physics.transmission)
    assert torch.equal(reproject(linear, ones, physics.background), linear)
    assert torch.equal(reproject(linear, torch.zeros_like(ones), physics.background), physics.background.expand_as(image))
    assert loss(image, image, Physics(ones, physics.background, ones), 1e-3).item() == 0
    assert loss(image, 1 - image, physics, 1e-3).item() > 0


def test_charbonnier_gradient_matches_derivative_and_detaches_physics():
    generator = torch.Generator().manual_seed(5)
    inp = torch.rand(1, 3, 5, 7, dtype=torch.float64, generator=generator, requires_grad=True)
    recovery = torch.rand(1, 3, 5, 7, dtype=torch.float64, generator=generator, requires_grad=True)
    t = torch.full((1, 1, 5, 7), 0.55, dtype=torch.float64, requires_grad=True)
    b = torch.full((1, 3, 1, 1), 0.4, dtype=torch.float64, requires_grad=True)
    confidence = torch.rand(1, 1, 5, 7, dtype=torch.float64, generator=generator, requires_grad=True)
    epsilon = 1e-3
    actual = torch.autograd.grad(loss(inp, recovery, Physics(t, b, confidence), epsilon),
                                 (recovery, inp, t, b, confidence), allow_unused=True)
    residual = srgb_to_linear(inp) - reproject(srgb_to_linear(recovery), t, b)
    slope = torch.where(recovery <= 0.04045, 1 / 12.92, (2.4 / 1.055) * ((recovery + 0.055) / 1.055).pow(1.4))
    expected = -t * residual / (residual.square() + epsilon**2).sqrt() * slope * confidence / (3 * (confidence.sum() + 1e-12))
    torch.testing.assert_close(actual[0], expected, atol=1e-13, rtol=1e-11)
    assert all(gradient is None for gradient in actual[1:])
    recovery_lin = srgb_to_linear(recovery).detach().requires_grad_()
    residual = srgb_to_linear(inp) - reproject(recovery_lin, t, b)
    gradient, = torch.autograd.grad(residual.square().sum() / 2, recovery_lin)
    torch.testing.assert_close(gradient, -t * residual)


def candidate(model, image, config):
    parameters = tuple(model.adapter_parameters())
    initial = snapshot(parameters)
    physics = estimate(image, config["physics"])
    with torch.no_grad():
        static = model(image)
    grads, = physics_directions(model, image, (physics,), config["physics"]["charbonnier_eps"])
    return apply_candidate(model, image, static, parameters, initial, grads, config["candidate"], True)


def test_candidate_updates_only_adapter_and_restores_exactly(model, config):
    model.freeze_for_adaptation()
    image = torch.rand(1, 3, 24, 24)
    parameters = tuple(model.parameters())
    adapters = {id(p) for p in model.adapter_parameters()}
    assert len(adapters) == 4
    assert all(p.requires_grad == (id(p) in adapters) for p in parameters)
    saved = snapshot(parameters)
    observations = []

    def observe(_module, _input):
        observations.append([id(p) for p, value in zip(parameters, saved) if not torch.equal(p, value)])

    hook = model.register_forward_pre_hook(observe)
    try:
        result = candidate(model, image, config)
    finally:
        hook.remove()
    assert result.valid and result.step > 0 and result.drift > 0
    assert any(observations) and {p for call in observations for p in call} == adapters
    for value, parameter in zip(saved, parameters):
        assert value.detach().contiguous().numpy().tobytes() == parameter.detach().contiguous().numpy().tobytes()


def test_candidate_same_image_replays_after_other_image(model, config):
    model.freeze_for_adaptation()
    generator = torch.Generator().manual_seed(config["seed"])
    first, second = [torch.rand(1, 3, 24, 24, generator=generator) for _ in range(2)]
    a = candidate(model, first, config)
    candidate(model, second, config)
    b = candidate(model, first, config)
    assert torch.equal(a.output, b.output)
    assert (a.step, a.drift, a.status) == (b.step, b.drift, b.status)


def test_trust_reject_and_low_confidence_restore(model, config):
    model.freeze_for_adaptation()
    parameters = tuple(model.adapter_parameters())
    initial = snapshot(parameters)
    image = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        static = model(image)
    grads = tuple(torch.ones_like(parameter) for parameter in parameters)
    options = dict(config["candidate"], output_drift_rms_max=0)
    for eligible, status in ((True, "trust_reject"), (False, "low_confidence")):
        result = apply_candidate(model, image, static, parameters, initial, grads, options, eligible)
        assert result.status == status and not result.valid and result.step == 0
        assert torch.equal(result.output, static)
        assert all(torch.equal(x, y) for x, y in zip(parameters, initial))


def test_manifest_separates_groups_and_flags_unpaired(tmp_path):
    inp, ref = tmp_path / "input", tmp_path / "GT"
    inp.mkdir()
    ref.mkdir()
    rng = np.random.default_rng(2026)
    for i in range(12):
        pixels = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(inp / f"{i}.png")
        Image.fromarray(pixels // 2 + 24).save(ref / f"{i}.png")
    (inp / "copy.png").write_bytes((inp / "0.png").read_bytes())
    (ref / "copy.png").write_bytes((ref / "0.png").read_bytes())
    with Image.open(inp / "0.png") as image:
        image.save(inp / "reencode.bmp")
    with Image.open(ref / "0.png") as image:
        image.save(ref / "reencode.bmp")
    (inp / "orphan.png").write_bytes((inp / "1.png").read_bytes())
    rows = scan(tmp_path, "LSUI", 2)
    targets = dict(train=6, tune=2, utility_fit=2, risk_calib=2)
    counts, _, _ = assign_splits(rows, targets, 2026)
    by_id = {row["image_id"].split(":")[-1]: row for row in rows}
    assert sum(counts.values()) == 14 and set(counts) == set(targets)
    assert not by_id["orphan"]["valid_pair"] and by_id["orphan"]["split"] == ""
    assert len({by_id[key]["group_id"] for key in ("0", "copy", "reencode")}) == 1
    valid = [row for row in rows if row["valid_pair"]]
    for i, j, _ in duplicate_links(valid):
        assert valid[i]["split"] == valid[j]["split"]


def test_reference_only_changes_evaluation(model, config):
    model.freeze_for_adaptation()
    image = torch.rand(1, 3, 24, 24)
    alternate = torch.rand_like(image)
    backgrounds = {"donor": estimate(alternate, config["physics"]).background}
    metric = SSIM()
    a, _, a_images = audit_one(model, image, alternate, "sample", "g1", "donor", backgrounds, metric, config)
    b, _, b_images = audit_one(model, image, 1 - alternate, "sample", "g1", "donor", backgrounds, metric, config)
    assert torch.equal(a_images[2], b_images[2])
    for row_a, row_b in zip(a, b):
        assert row_a["adapter_step_size"] == row_b["adapter_step_size"]
        assert row_a["physics_loss_after"] == row_b["physics_loss_after"]
    assert a[0]["gradient_cosine"] != b[0]["gradient_cosine"]


def test_audit_reporting_pipeline(model, config, tmp_path):
    model.freeze_for_adaptation()
    inputs = [torch.rand(1, 3, 24, 24) for _ in range(2)]
    backgrounds = {f"sample{i}": estimate(inp, config["physics"]).background for i, inp in enumerate(inputs)}
    rows, probes, images = [], [], None
    for i, image in enumerate(inputs):
        records, output_probes, images = audit_one(
            model, image, torch.rand_like(image), f"sample{i}", f"group{i}", f"sample{1 - i}",
            backgrounds, SSIM(), config,
        )
        rows.extend(records)
        probes.extend(output_probes)
    report = tmp_path / "report.md"
    info = dict(config=config, checkpoint_epoch=1, tune_psnr=0.0, training_epochs=1)
    deliver(tmp_path, report, rows, probes, [(-0.1, "sample1", [image[0] for image in images])], info)
    assert "Q5" in report.read_text(encoding="utf-8")
    for name in ("gradient_cosine", "cosine_vs_delta", "control_distributions", "failure_cases"):
        assert (tmp_path / f"e1_{name}.png").stat().st_size > 0


def test_cuda_exact_replay(model, config, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", config["e1"]["cublas_workspace_config"])
    if not torch.cuda.is_available():
        pytest.skip("CUDA replay requires a currently accessible GPU")
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model.to(device="cuda", memory_format=torch.channels_last).freeze_for_adaptation()
        first, second = [torch.rand(1, 3, 32, 32, device="cuda") for _ in range(2)]
        saved = snapshot(model.parameters())
        a = candidate(model, first, config)
        candidate(model, second, config)
        b = candidate(model, first, config)
        assert torch.equal(a.output, b.output)
        assert all(torch.equal(p, value) for p, value in zip(model.parameters(), saved))
    finally:
        torch.use_deterministic_algorithms(previous)

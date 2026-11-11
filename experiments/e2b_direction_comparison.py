"""Development and once-opened source acceptance evaluation."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F

from datasets.manifest import read_csv, write_csv
from datasets.paired import PairedImages
from experiments.e1b_output_matched import physical_gradient, oracle_direction, objective
from experiments.e2b_factorial_error import setup, parameter_errors
from experiments.e2b_estimator_training import make_estimator, operator_path, fit_effective
from prtta.candidate import block_direction
from prtta.metrics import SSIM, psnr
from prtta.operator_estimator import operator_fusion
from prtta.output_match import match_direction, Matched
from prtta.physics import estimate
from prtta.physics import srgb_to_linear, linear_to_srgb
from experiments.matched_report import bootstrap
from prtta.run import sha_file, stable_seed, tensors, write_json, run_info, load_config
from prtta.synthetic import DualPhysics, synthesize


@torch.no_grad()
def predict_operators(image, static, estimators, physics, timing=False):
    started = time.perf_counter()
    h = estimate(image, physics)
    result = {"udcp": DualPhysics(h.transmission, h.transmission, h.background, h.confidence)}
    torch.cuda.synchronize(image.device)
    times = {"udcp": (time.perf_counter()-started)*1000}
    for mode, net in estimators.items():
        if mode != "residual":
            started = time.perf_counter()
            result[mode] = net(image, static)
            torch.cuda.synchronize(image.device)
            times[mode] = (time.perf_counter()-started)*1000
    return (result, times) if timing else result


def consistency_direction(model, image):
    targets = []
    with torch.no_grad():
        for flip in (False, True):
            for rotation in range(4):
                transformed = image.flip(-1) if flip else image
                transformed = transformed.rot90(rotation, (-2, -1))
                recovery = model(transformed).rot90(-rotation, (-2, -1))
                targets.append(recovery.flip(-1) if flip else recovery)
        target = torch.stack(targets).mean(0)
    return torch.autograd.grad((model(image) - target).square().mean(), tuple(model.adapter_parameters()))


@torch.no_grad()
def matched_refinement(static, prediction, budgets, settings):
    distance = float((prediction - static).square().mean().sqrt())
    results = []
    attempts = []
    for budget in budgets:
        valid = distance > 0 and budget <= distance
        alpha = budget / distance if valid else 0
        output = torch.lerp(static, prediction, alpha)
        drift = float((output - static).square().mean().sqrt())
        valid = valid and abs(drift - budget) <= settings["relative_tolerance"] * budget
        attempts.append(dict(target_rms=budget, alpha=alpha, drift=drift, parameter_delta=0.0))
        results.append(Matched(output if valid else static, valid, "" if valid else "drift_unreachable",
                               alpha if valid else 0, drift if valid else 0, 0, 1, min(budget, distance), attempts))
    return results


def score_image(model, image, reference, estimators, config, base, metadata, truth=None, effective_oracle=False):
    torch.cuda.reset_peak_memory_stats(image.device)
    began = time.perf_counter()
    with torch.no_grad():
        static = model(image)
    torch.cuda.synchronize()
    static_ms = (time.perf_counter() - began) * 1000
    operators, estimation_times = predict_operators(image, static, estimators, base["physics"], timing=True)
    if truth is not None:
        operators["true_operator"] = truth
    if effective_oracle:
        oracle_started = time.perf_counter()
        field, background = fit_effective(image, reference, config["effective"], base["physics"])
        value = 0.05 + 0.93 * F.interpolate(field, image.shape[-2:], mode="bilinear", align_corners=False).sigmoid()
        operators["effective_label_oracle"] = DualPhysics(value[:, :3], value[:, 3:], background.sigmoid(), operators["udcp"].confidence)
        torch.cuda.synchronize()
        estimation_times["effective_label_oracle"] = (time.perf_counter()-oracle_started)*1000
    settings = base["diagnostic"]
    budgets = [b / 255 for b in config["budgets_8bit"]]
    rows = []
    methods = list(operators)
    if truth is None:
        methods += ["consistency", "oracle"] + ["fusion_" + name for name in operators if name != "effective_label_oracle"]
        if "residual" in estimators:
            methods.append("residual")
    metric = SSIM().to(image.device)
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    for method in methods:
        torch.cuda.reset_peak_memory_stats(image.device)
        started = time.perf_counter()
        gradient_ms = 0.0
        estimator_name = method[7:] if method.startswith("fusion_") else method
        estimation_ms = estimation_times.get(estimator_name, 0.0)
        h = operators.get(estimator_name, operators["udcp"])
        if method.startswith("fusion_"):
            matches = operator_fusion(image, static, operators[method[7:]], budgets, settings)
        elif method == "residual":
            with torch.no_grad():
                estimator_start = time.perf_counter()
                prediction = estimators["residual"](image, static)
                torch.cuda.synchronize()
                estimation_ms = (time.perf_counter()-estimator_start)*1000
                matches = matched_refinement(static, prediction, budgets, settings)
        else:
            gradient_start = time.perf_counter()
            if method == "oracle":
                gradient = oracle_direction(model, image, reference)
            elif method == "consistency":
                gradient = consistency_direction(model, image)
            else:
                gradient = physical_gradient(model, image, h, base["physics"]["charbonnier_eps"])
            torch.cuda.synchronize()
            gradient_ms = (time.perf_counter() - gradient_start) * 1000
            matches = match_direction(model, image, static, gradient, budgets, settings)
        torch.cuda.synchronize()
        generation_ms = (time.perf_counter() - started) * 1000
        if method == "residual":
            generation_ms -= estimation_ms
        with torch.no_grad():
            outputs = torch.cat([static, *(item.output for item in matches)])
            references = reference.expand_as(outputs)
            output_start = time.perf_counter()
            p, s = psnr(outputs, references).tolist(), metric(outputs, references).tolist()
            output_ms = (time.perf_counter() - output_start) * 1000
            before = float(objective(image, static, h, base["physics"]["charbonnier_eps"]))
            for index, (pixel, candidate) in enumerate(zip(config["budgets_8bit"], matches), 1):
                rows.append(dict(
                    **metadata, method=method, target_rms_8bit=pixel, valid=candidate.valid, failure_reason=candidate.reason,
                    achieved_rms_8bit=candidate.drift * 255, nearest_rms_8bit=candidate.nearest_drift * 255,
                    alpha=candidate.alpha, parameter_delta_l2=candidate.parameter_delta,
                    psnr_static=p[0], psnr_candidate=p[index], delta_psnr=p[index]-p[0],
                    delta_ssim=s[index]-s[0], ssim_static=s[0], ssim_candidate=s[index],
                    physics_loss_before=before,
                    physics_loss_after=float(objective(image, candidate.output, h, base["physics"]["charbonnier_eps"])),
                    transmission_mean=float(h.transmission.mean()),
                    transmission_std=float(h.transmission.std()),
                    background_rgb=json.dumps(h.background.flatten().tolist()),
                    static_ms=static_ms, estimation_ms=estimation_ms, gradient_ms=gradient_ms,
                    search_ms=generation_ms-gradient_ms, output_ms=output_ms,
                    runtime_ms=static_ms+estimation_ms+generation_ms+output_ms,
                    peak_vram_bytes=torch.cuda.max_memory_allocated(image.device),
                    search_attempts=json.dumps(candidate.attempts) if index == 1 else "shared_with_first_budget",
                ))
    assert all(torch.equal(initial[key], value) for key, value in model.state_dict().items())
    return rows, operators


def run(phase):
    config, base, device, model = setup()
    gate = phase == "gate"
    if not gate and Path("results/e2b/gate/opened.json").exists():
        raise RuntimeError("Acceptance data already opened; development is frozen")
    effective = phase == "effective_development" or (gate and json.loads(Path("results/e2b/trigger.json").read_text())["trigger"])
    source = config["gate_manifest"] if gate else config["development_manifest"]
    root = Path("results/e2b") / phase
    root.mkdir(parents=True, exist_ok=True)
    modes = ["shared", "dual"] + (["effective", "residual"] if effective else [])
    lock = dict(config_sha256=sha_file("configs/e2b_operator_study.yaml"), manifest_sha256=sha_file(source),
                noise_audit_sha256=sha_file("configs/e2b_noise_audit.yaml"),
                estimator_hashes={f"{mode}_{seed}": sha_file(operator_path(mode, seed)) for mode in modes for seed in config["seeds"]})
    if gate:
        # Exclusive create prevents accidental repeated opening for method selection.
        opened = root / "opened.json"
        if opened.exists():
            assert json.loads(opened.read_text()) == lock, "Frozen acceptance settings changed"
            if (root / "run.json").exists():
                raise RuntimeError("Completed acceptance evaluation cannot be reopened")
            print("Resuming the same frozen acceptance attempt after interruption", flush=True)
        else:
            with opened.open("x") as handle:
                json.dump(lock, handle, indent=2)
    else:
        write_json(root / "lock.json", lock)
    data = PairedImages(base, "utility_fit")
    data.rows = read_csv(source)
    data.prepare_cache(base["data"]["manifest_workers"])
    records, errors = [], []
    for seed in config["seeds"]:
        nets = {mode: make_estimator(config, mode, seed, device) for mode in modes}
        for index, row in enumerate(data.rows):
            inp, ref, *_ = data[index]
            image, reference = tensors((inp[None], ref[None]), device)
            metadata = dict(image_id=row["image_id"], group_id=row["group_id"], seed=seed, domain="real")
            rows, _ = score_image(model, image, reference, nets, config, base, metadata, effective_oracle=effective)
            records.extend(rows)
            if not gate and not effective:
                for dual in (False, True):
                    generated, truth, _ = synthesize(reference, base["diagnostic"]["e2"],
                        stable_seed(2026, row["image_id"], "sspe_val"), dual)
                    synthetic_rows, operators = score_image(model, generated, reference, nets, config, base,
                        dict(metadata, domain="dual" if dual else "shared"), truth=truth)
                    records.extend(synthetic_rows)
                    for name, h in operators.items():
                        errors.append(dict(image_id=row["image_id"], seed=seed, model="dual" if dual else "shared", noise_std=0.0,
                                           method=name, **parameter_errors(h, truth)))
                    noise_config = load_config("configs/e2b_noise_audit.yaml")
                    for noise_std in noise_config["linear_noise_std"]:
                        generator = torch.Generator().manual_seed(stable_seed(2026, row["image_id"], noise_std, "noise_audit"))
                        noise = torch.randn(generated.shape, generator=generator).to(device) * noise_std
                        noisy = linear_to_srgb((srgb_to_linear(generated) + noise).clamp(0, 1))
                        with torch.no_grad():
                            noisy_static = model(noisy)
                            noisy_operators = predict_operators(noisy, noisy_static, nets, base["physics"])
                        for name, h in noisy_operators.items():
                            errors.append(dict(image_id=row["image_id"], seed=seed, model="dual" if dual else "shared",
                                noise_std=noise_std, method=name, **parameter_errors(h, truth)))
            if (index + 1) % 10 == 0:
                print(phase, seed, index+1, "/", len(data), flush=True)
                write_csv(root / "partial.csv", records, list(records[0]))
    write_csv(root / "per_image.csv", records, list(records[0]))
    if errors:
        write_json(root / "parameter_errors.json", errors)
    write_json(root / "run.json", dict(run_info(base, device), **lock, full_state_restore_exact=True))
    if phase == "development":
        reference = [e["mae_td"]+e["mae_tb"]+e["mae_b"] for e in errors if e["method"] == "udcp" and e["noise_std"] == 0]
        synthetic_pass = []
        for method in ("shared", "dual"):
            own = [e["mae_td"]+e["mae_tb"]+e["mae_b"] for e in errors if e["method"] == method and e["noise_std"] == 0]
            synthetic_pass.append(np.mean(own) < config["effective"]["trigger_synthetic_relative_mae"] * np.mean(reference))
        gains = [np.mean([r["delta_psnr"] for r in records if r["domain"] == "real" and r["method"] == method
                         and r["target_rms_8bit"] == budget])
                 for method in ("shared", "dual") for budget in config["primary_budgets_8bit"]]
        passes = []
        for method in ("shared", "dual"):
            for budget in config["primary_budgets_8bit"]:
                selected = [r for r in records if r["domain"] == "real" and r["target_rms_8bit"] == budget]
                ids = sorted({r["image_id"] for r in selected})
                values = {
                    name: {key: np.mean([r["delta_psnr"] for r in selected if r["method"] == name and r["image_id"] == key])
                           for key in ids}
                    for name in (method, "udcp", "fusion_" + method, "consistency")
                }
                group_ids = {r["image_id"]: r["group_id"] for r in selected}
                advantages = [
                    bootstrap([values[method][key]-values[other][key] for key in ids],
                              [group_ids[key] for key in ids], base["diagnostic"])[0] > 0
                    for other in ("udcp", "fusion_" + method, "consistency")
                ]
                own_rows = [r for r in selected if r["method"] == method]
                passes.append(np.mean(list(values[method].values())) > config["go"]["minimum_gain_db"]
                    and np.mean([r["delta_psnr"] < -0.2 for r in own_rows]) < config["go"]["maximum_harm_fraction"]
                    and all(advantages))
        trigger = any(synthetic_pass) and not any(passes)
        write_json("results/e2b/trigger.json", dict(trigger=bool(trigger), synthetic_pass=[bool(v) for v in synthetic_pass],
                   development_gains=[float(v) for v in gains], development_pass=[bool(v) for v in passes],
                   rule="synthetic MAE improvement and no development method satisfying preregistered mean/harm/control advantages"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["development", "effective_development", "gate"], default="development")
    run(parser.parse_args().phase)

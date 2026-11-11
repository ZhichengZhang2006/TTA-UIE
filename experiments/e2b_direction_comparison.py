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
from prtta.run import sha_file, stable_seed, tensors, write_json, run_info
from prtta.synthetic import DualPhysics, synthesize


@torch.no_grad()
def predict_operators(image, static, estimators, physics):
    h = estimate(image, physics)
    result = {"udcp": DualPhysics(h.transmission, h.transmission, h.background, h.confidence)}
    for mode, net in estimators.items():
        if mode != "residual":
            result[mode] = net(image, static)
    return result


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
    for budget in budgets:
        valid = distance > 0 and budget <= distance
        alpha = budget / distance if valid else 0
        output = torch.lerp(static, prediction, alpha)
        drift = float((output - static).square().mean().sqrt())
        results.append(Matched(output if valid else static, valid, "" if valid else "drift_unreachable",
                               alpha, drift if valid else 0, 0, 1, min(budget, distance)))
    return results


def score_image(model, image, reference, estimators, config, base, metadata, truth=None, effective_oracle=False):
    torch.cuda.reset_peak_memory_stats(image.device)
    began = time.perf_counter()
    with torch.no_grad():
        static = model(image)
    torch.cuda.synchronize()
    operators = predict_operators(image, static, estimators, base["physics"])
    torch.cuda.synchronize()
    estimation_ms = (time.perf_counter() - began) * 1000
    if truth is not None:
        operators["true_operator"] = truth
    if effective_oracle:
        field, background = fit_effective(image, reference, config["effective"], base["physics"])
        value = 0.05 + 0.93 * F.interpolate(field, image.shape[-2:], mode="bilinear", align_corners=False).sigmoid()
        operators["effective_label_oracle"] = DualPhysics(value[:, :3], value[:, 3:], background.sigmoid(), operators["udcp"].confidence)
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
        h = operators.get(method, operators["udcp"])
        if method.startswith("fusion_"):
            matches = operator_fusion(image, static, operators[method[7:]], budgets, settings)
        elif method == "residual":
            with torch.no_grad():
                prediction = estimators["residual"](image, static)
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
                    estimation_ms=estimation_ms, gradient_ms=gradient_ms,
                    search_ms=generation_ms-gradient_ms, output_ms=output_ms,
                    runtime_ms=estimation_ms+generation_ms+output_ms,
                    peak_vram_bytes=torch.cuda.max_memory_allocated(image.device),
                    search_attempts=json.dumps(candidate.attempts) if index == 1 else "shared_with_first_budget",
                ))
    assert all(torch.equal(initial[key], value) for key, value in model.state_dict().items())
    return rows, operators


def run(phase):
    config, base, device, model = setup()
    gate = phase == "gate"
    effective = phase == "effective_development" or (gate and json.loads(Path("results/e2b/trigger.json").read_text())["trigger"])
    source = config["gate_manifest"] if gate else config["development_manifest"]
    root = Path("results/e2b") / phase
    root.mkdir(parents=True, exist_ok=True)
    modes = ["shared", "dual"] + (["effective", "residual"] if effective else [])
    lock = dict(config_sha256=sha_file("configs/e2b_operator_study.yaml"), manifest_sha256=sha_file(source),
                estimator_hashes={f"{mode}_{seed}": sha_file(operator_path(mode, seed)) for mode in modes for seed in config["seeds"]})
    if gate:
        # Exclusive create prevents accidental repeated opening for method selection.
        with (root / "opened.json").open("x") as handle:
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
                        errors.append(dict(image_id=row["image_id"], seed=seed, model="dual" if dual else "shared",
                                           method=name, **parameter_errors(h, truth)))
            if (index + 1) % 10 == 0:
                print(phase, seed, index+1, "/", len(data), flush=True)
                write_csv(root / "partial.csv", records, list(records[0]))
    write_csv(root / "per_image.csv", records, list(records[0]))
    if errors:
        write_json(root / "parameter_errors.json", errors)
    write_json(root / "run.json", dict(run_info(base, device), **lock, full_state_restore_exact=True))
    if phase == "development":
        reference = [e["mae_td"]+e["mae_tb"]+e["mae_b"] for e in errors if e["method"] == "udcp"]
        synthetic_pass = []
        for method in ("shared", "dual"):
            own = [e["mae_td"]+e["mae_tb"]+e["mae_b"] for e in errors if e["method"] == method]
            synthetic_pass.append(np.mean(own) < config["effective"]["trigger_synthetic_relative_mae"] * np.mean(reference))
        gains = [np.mean([r["delta_psnr"] for r in records if r["domain"] == "real" and r["method"] == method
                         and r["target_rms_8bit"] == budget])
                 for method in ("shared", "dual") for budget in config["primary_budgets_8bit"]]
        trigger = any(synthetic_pass) and max(gains) < config["effective"]["trigger_real_mean_db"]
        write_json("results/e2b/trigger.json", dict(trigger=bool(trigger), synthetic_pass=[bool(v) for v in synthetic_pass],
                   development_gains=[float(v) for v in gains], rule="config-fixed development-only trigger"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["development", "effective_development", "gate"], default="development")
    run(parser.parse_args().phase)

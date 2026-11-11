"""Run frozen source-only output-matched diagnostics."""
import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from datasets.manifest import read_csv, write_csv
from datasets.paired import PairedImages
from prtta.candidate import block_direction, random_direction
from prtta.metrics import SSIM, psnr
from prtta.output_match import Matched, fusion_candidates, match_direction
from prtta.physics import estimate, loss
from prtta.run import configure, load_config, load_static, run_info, sha_file, stable_seed, tensors, write_json
from prtta.synthetic import DualPhysics, dual_loss, synthesize


def objective(image, output, hypothesis, epsilon):
    if isinstance(hypothesis, DualPhysics):
        return dual_loss(image, output, hypothesis, epsilon)
    return loss(image, output, hypothesis, epsilon)


def physical_gradient(model, image, hypothesis, epsilon):
    return torch.autograd.grad(objective(image, model(image), hypothesis, epsilon), tuple(model.adapter_parameters()))


def oracle_direction(model, image, reference):
    """Source-only diagnostic. This is the sole reference-aware direction API."""
    return torch.autograd.grad((model(image) - reference).square().mean(), tuple(model.adapter_parameters()))


def cosine(left, right):
    numerator = sum((a.double() * b.double()).sum() for a, b in zip(left, right))
    ln = sum(a.double().square().sum() for a in left).sqrt()
    rn = sum(b.double().square().sum() for b in right).sqrt()
    return float(numerator / (ln * rn + 1e-24))


def e2_hypotheses(image, truth, settings, dual):
    estimated = estimate(image, settings["physics"])
    if dual:
        td, tb, b = truth.transmission, truth.backscatter_transmission, truth.background
        denominator = (1 - td).square().sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        fitted_b = (b * ((1 - td) * (1 - tb)).sum(dim=(-2, -1), keepdim=True) / denominator).clamp(0, 1)
        from prtta.physics import Physics
        return {
            "dual_true": truth, "dual_wrong_t20": replace(truth, transmission=(td * 1.2).clamp(0.001, 0.999)),
            "dual_wrong_b20": replace(truth, background=(b * 1.2).clamp(0, 1)),
            "simplified_true_td_b": Physics(td, b, truth.confidence),
            "simplified_coefficient_fit": Physics(td, fitted_b, truth.confidence),
            "simplified_estimated": estimated,
        }
    from prtta.physics import Physics
    true = Physics(truth.transmission, truth.background, truth.confidence)
    hypotheses = {"true": true, "estimated": estimated,
                  "wrong_t": replace(true, transmission=1 - true.transmission),
                  "wrong_b": replace(true, background=true.background.roll(1, dims=1))}
    for percentage in settings["diagnostic"]["e2"]["perturbation_percent"]:
        for sign in (-1, 1):
            factor = 1 + sign * percentage / 100
            suffix = f"{sign * percentage:+d}"
            hypotheses[f"t{suffix}"] = replace(true, transmission=(true.transmission * factor).clamp(0.001, 0.999))
            hypotheses[f"b{suffix}"] = replace(true, background=(true.background * factor).clamp(0, 1))
    return hypotheses


def evaluate_candidates(model, image, reference, hypotheses, settings, metadata, include_controls, donor_background=None):
    diagnostic = settings["diagnostic"]
    budgets = [pixel / 255 for pixel in diagnostic["budgets_8bit"]]
    epsilon = settings["physics"]["charbonnier_eps"]
    metric = SSIM().to(image.device, memory_format=torch.channels_last)
    with torch.no_grad():
        static = model(image)
    oracle_start = time.perf_counter()
    oracle = oracle_direction(model, image, reference)
    if image.device.type == "cuda":
        torch.cuda.synchronize(image.device)
    oracle_ms = (time.perf_counter() - oracle_start) * 1000
    directions, gradient_times = {}, {}
    for name, hypothesis in hypotheses.items():
        began = time.perf_counter()
        directions[name] = physical_gradient(model, image, hypothesis, epsilon)
        if image.device.type == "cuda":
            torch.cuda.synchronize(image.device)
        gradient_times[name] = (time.perf_counter() - began) * 1000
    canonical = hypotheses[next(iter(hypotheses))]
    if include_controls:
        directions["random"] = random_direction(directions["physical"], stable_seed(settings["seed"], metadata["image_id"], "matched_random"))
        directions["oracle"] = oracle
        gradient_times["random"] = gradient_times["physical"]
        gradient_times["oracle"] = oracle_ms
    canonical_before = float(objective(image, static, canonical, epsilon))
    records, failures = [], []
    methods = list(directions)
    if include_controls:
        methods += ["fusion", "static"]
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    for method in methods:
        start = time.perf_counter()
        gradient = directions.get(method)
        hypothesis = hypotheses.get(method, canonical)
        output_leaf = static.detach().requires_grad_(True)
        output_gradient, = torch.autograd.grad(objective(image, output_leaf, hypothesis, epsilon), output_leaf)
        reference_gradient = 2 * (static - reference) / static.numel()
        output_cosine = cosine((output_gradient,), (reference_gradient,))
        if method == "oracle":
            output_cosine = cosine((reference_gradient,), (reference_gradient,))
        elif method in ("random", "fusion", "static"):
            output_cosine = None
        eligible = float(hypothesis.confidence.mean()) >= settings["physics"]["min_confidence_mean"]
        if method == "fusion":
            matches = fusion_candidates(image, static, canonical, budgets, diagnostic)
        elif method == "static":
            matches = [Matched(static, True, "", 0, 0, 0, 0, 0) for _ in budgets]
        elif eligible:
            matches = match_direction(model, image, static, gradient, budgets, diagnostic)
        else:
            matches = [Matched(static, False, "low_confidence", 0, 0, 0, 0, 0) for _ in budgets]
        search_runtime = (time.perf_counter() - start) * 1000
        runtime = search_runtime + gradient_times.get(method, 0)
        with torch.no_grad():
            predictions = torch.cat([static, *(match.output for match in matches)])
            references = reference.expand_as(predictions)
            p, s = psnr(predictions, references).tolist(), metric(predictions, references).tolist()
            norm = 0 if gradient is None else block_direction(gradient)[1]
            direction_cosine = 0 if gradient is None else cosine(block_direction(gradient)[0], block_direction(oracle)[0])
            for index, (pixel, matched) in enumerate(zip(diagnostic["budgets_8bit"], matches), 1):
                before = float(objective(image, static, hypothesis, epsilon))
                after = float(objective(image, matched.output, hypothesis, epsilon))
                records.append(dict(
                    **metadata, method=method, target_rms_8bit=pixel, achieved_rms_8bit=matched.drift * 255,
                    nearest_rms_8bit=matched.nearest_drift * 255, valid=matched.valid,
                    failure_reason=matched.reason, alpha=matched.alpha, parameter_delta_l2=matched.parameter_delta,
                    psnr_static=p[0], psnr_candidate=p[index], delta_psnr=p[index] - p[0],
                    ssim_static=s[0], ssim_candidate=s[index], delta_ssim=s[index] - s[0],
                    phy_loss_before=before, phy_loss_after=after,
                    canonical_phy_before=canonical_before,
                    canonical_phy_after=float(objective(image, matched.output, canonical, epsilon)),
                    gradient_norm=norm, parameter_gradient_cosine=direction_cosine,
                    output_gradient_cosine=output_cosine,
                    runtime_ms=runtime, search_ms=search_runtime, gradient_ms=gradient_times.get(method, 0),
                    forward_evaluations=matched.evaluations,
                ))
                if method in ("physical", "estimated", "simplified_estimated") and pixel == 1.0 and p[index] < p[0]:
                    failures.append((p[index] - p[0], metadata["image_id"],
                                     torch.cat((image, static, matched.output, reference)).cpu()))
    for name, value in model.state_dict().items():
        if not torch.equal(value, initial[name]):
            raise RuntimeError(f"Model state contaminated: {name}")
    return records, failures


def run(config_path, manifest, phase, synthetic=False):
    diagnostic = load_config(config_path)
    settings = load_config(diagnostic["base_config"])
    settings["seed"] = diagnostic["seed"]
    settings["diagnostic"] = diagnostic
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    device = configure(settings)
    if sha_file(diagnostic["checkpoint"]) != diagnostic["checkpoint_sha256"]:
        raise RuntimeError("Frozen checkpoint hash changed")
    model, _ = load_static(settings, device, Path(diagnostic["checkpoint"]))
    model.freeze_for_adaptation()
    dataset = PairedImages(settings, "utility_fit")
    dataset.rows = read_csv(manifest)
    for row in dataset.rows:
        if row["split"] != "diagnostic_consumed":
            raise ValueError("Only consumed diagnostic data may be evaluated")
    dataset.prepare_cache(settings["data"]["manifest_workers"])
    directory = Path("results") / ("e2" if synthetic else "e1b") / phase
    directory.mkdir(parents=True, exist_ok=True)
    lock = dict(config_sha256=sha_file(config_path), manifest_sha256=sha_file(manifest),
                checkpoint_sha256=sha_file(diagnostic["checkpoint"]))
    if phase == "fixed" and not Path("results/e1b/pilot/summary.json").exists():
        raise RuntimeError("Pilot must finish before Fixed diagnostics")
    lockpath = directory / "preregistration.json"
    if lockpath.exists():
        import json
        if json.loads(lockpath.read_text()) != lock:
            raise RuntimeError("Preregistered settings or manifest changed")
    else:
        write_json(lockpath, lock)
    background = {}
    if not synthetic:
        for index, row in enumerate(dataset.rows):
            inp, ref, *_ = dataset[index]
            image, _ = tensors((inp[None], ref[None]), device)
            background[row["image_id"]] = estimate(image, settings["physics"]).background
    rows, examples, generation = [], [], []
    started = time.perf_counter()
    for index, row in enumerate(dataset.rows):
        inp, ref, *_ = dataset[index]
        image, reference = tensors((inp[None], ref[None]), device)
        common = dict(image_id=row["image_id"], group_id=row["group_id"], seed=settings["seed"],
                      checkpoint_hash=diagnostic["checkpoint_sha256"])
        if synthetic:
            for dual in (False, True):
                seed = stable_seed(settings["seed"], row["image_id"], "synthetic")
                image, truth, params = synthesize(reference, diagnostic["e2"], seed, dual)
                generation.append(dict(**common, model="dual" if dual else "shared", **params))
                hypotheses = e2_hypotheses(image, truth, settings, dual)
                records, failures = evaluate_candidates(model, image, reference, hypotheses, settings,
                                                 dict(**common, model="dual" if dual else "shared"), False)
                examples.extend(failures)
                examples.sort(key=lambda entry: entry[0])
                del examples[6:]
                for record in records:
                    h = hypotheses[record["method"]]
                    record["transmission_mae"] = float((h.transmission - truth.transmission).abs().mean())
                    record["background_mae"] = float((h.background - truth.background).abs().mean())
                    tb = h.backscatter_transmission if isinstance(h, DualPhysics) else h.transmission
                    record["backscatter_mae"] = float((tb - truth.backscatter_transmission).abs().mean())
                rows.extend(records)
        else:
            original = estimate(image, settings["physics"])
            donors = [other for other in dataset.rows if other["group_id"] != row["group_id"]]
            donor = donors[stable_seed(settings["seed"], row["image_id"], "donor") % len(donors)]
            hypotheses = {
                "physical": original, "wrong_t": replace(original, transmission=(1 - original.transmission).clamp(*settings["physics"]["t_clip"])),
                "wrong_B": replace(original, background=background[donor["image_id"]]),
            }
            records, failures = evaluate_candidates(model, image, reference, hypotheses, settings,
                                                     dict(**common, background_donor=donor["image_id"]), True)
            rows.extend(records)
            examples.extend(failures)
            examples.sort(key=lambda entry: entry[0])
            del examples[6:]
        if (index + 1) % 10 == 0 or index + 1 == len(dataset):
            print(f"{phase} {'E2' if synthetic else 'E1b'} {index + 1}/{len(dataset)} in {time.perf_counter() - started:.1f}s", flush=True)
        if (index + 1) % 25 == 0:
            write_csv(directory / "partial.csv", rows, list(rows[0]))
    write_csv(directory / ("e2_per_image.csv" if synthetic else "e1b_per_image.csv"), rows, list(rows[0]))
    if generation:
        write_json(directory / "generation_parameters.json", generation)
    if examples:
        torch.save(examples, directory / "failure_tensors.pt")
    info = dict(run_info(settings, device), **lock, elapsed_seconds=time.perf_counter() - started,
                image_count=len(dataset), phase=phase, synthetic=synthetic, full_state_restoration_exact=True)
    write_json(directory / "run.json", info)
    from experiments.matched_report import summarize
    summarize(directory, rows, diagnostic)


def main(synthetic=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e1b_output_matched.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--phase", choices=["pilot", "fixed"], default="fixed")
    args = parser.parse_args()
    run(args.config, args.manifest, args.phase, synthetic)


if __name__ == "__main__":
    main()

"""Factorial t/B diagnostics on the already-consumed E2 image set."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.stats import spearmanr

from datasets.manifest import read_csv, write_csv
from datasets.paired import PairedImages
from experiments.e1b_output_matched import evaluate_candidates
from experiments.matched_report import bootstrap
from prtta.physics import estimate
from prtta.run import configure, load_config, load_static, run_info, sha_file, stable_seed, tensors, write_json
from prtta.synthetic import DualPhysics, synthesize


def setup():
    config = load_config("configs/e2b_operator_study.yaml")
    base = load_config(config["base_config"])
    matching = load_config(config["matching_config"])
    matching["budgets_8bit"] = config["budgets_8bit"]
    base["diagnostic"] = matching
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    device = configure(base)
    assert sha_file(config["checkpoint"]) == config["checkpoint_sha256"]
    model, _ = load_static(base, device, Path(config["checkpoint"]))
    model.freeze_for_adaptation()
    return config, base, device, model


def parameter_errors(predicted, truth):
    td = predicted.transmission.expand_as(truth.transmission)
    tb = predicted.backscatter_transmission.expand_as(truth.backscatter_transmission)
    difference = (td - truth.transmission).abs()
    low = truth.transmission < 0.2
    correlation = spearmanr(td.flatten().cpu().numpy(), truth.transmission.flatten().cpu().numpy()).statistic
    return dict(
        mae_td=float(difference.mean()), mae_tb=float((tb - truth.backscatter_transmission).abs().mean()),
        mae_b=float((predicted.background - truth.background).abs().mean()),
        td_channel_mae=[float(difference[:, c].mean()) for c in range(3)],
        low_t_mae=float(difference[low].mean()) if low.any() else None,
        spearman_td=float(correlation) if np.isfinite(correlation) else None,
        t_boundary_fraction=float(((td <= 0.050001) | (td >= 0.979999)).float().mean()),
        b_boundary_fraction=float(((predicted.background <= 1e-6) | (predicted.background >= 0.999999)).float().mean()),
        predicted_background=predicted.background.flatten().tolist(),
        transmission_spatial_std=float(td.std(dim=(-2, -1)).mean()),
    )


def main():
    config, base, device, model = setup()
    data = PairedImages(base, "utility_fit")
    data.rows = read_csv(config["factorial_manifest"])
    data.prepare_cache(base["data"]["manifest_workers"])
    rows, errors = [], []
    root = Path("results/e2b/factorial")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "preregistration.json", dict(config=config, manifest_sha256=sha_file(config["factorial_manifest"])))
    for index, meta in enumerate(data.rows):
        inp, ref, *_ = data[index]
        _, reference = tensors((inp[None], ref[None]), device)
        for dual in (False, True):
            seed = stable_seed(2026, meta["image_id"], "synthetic")
            image, truth, _ = synthesize(reference, base["diagnostic"]["e2"], seed, dual)
            h = estimate(image, base["physics"])
            predicted = DualPhysics(h.transmission, h.transmission, h.background, truth.confidence)
            hypotheses = {
                "P00": truth,
                "P10": DualPhysics(predicted.transmission, predicted.backscatter_transmission, truth.background, truth.confidence),
                "P01": DualPhysics(truth.transmission, truth.backscatter_transmission, predicted.background, truth.confidence),
                "P11": predicted,
            }
            records, _ = evaluate_candidates(model, image, reference, hypotheses, base,
                dict(image_id=meta["image_id"], group_id=meta["group_id"], model="dual" if dual else "shared"), False, record_attempts=True)
            rows.extend(records)
            errors.append(dict(image_id=meta["image_id"], model="dual" if dual else "shared",
                               **parameter_errors(predicted, truth)))
        if (index + 1) % 10 == 0:
            print(f"Factorial {index+1}/{len(data)}", flush=True)
    write_csv(root / "per_image.csv", rows, list(rows[0]))
    write_json(root / "parameter_errors.json", errors)
    estimates = []
    for mode in ("shared", "dual"):
        for budget in config["budgets_8bit"]:
            selected = {m: {r["image_id"]: r for r in rows if r["model"] == mode and r["target_rms_8bit"] == budget and r["method"] == m}
                        for m in ("P00", "P10", "P01", "P11")}
            keys = list(selected["P00"])
            for valid_only in (False, True):
                for label, coefficients in (
                    ("Delta_t", (1,-1,0,0)), ("Delta_B", (1,0,-1,0)), ("interaction", (1,-1,-1,1)),
                ):
                    required = [m for m, coefficient in zip(selected, coefficients) if coefficient]
                    ids = [key for key in keys if not valid_only or all(selected[m][key]["valid"] for m in required)]
                    groups = [selected["P00"][key]["group_id"] for key in ids]
                    values = [sum(coef * selected[m][key]["delta_psnr"] for m, coef in zip(selected, coefficients)) for key in ids]
                    ci = bootstrap(values, groups, base["diagnostic"])
                    estimates.append(dict(model=mode, budget=budget, contrast=label, both_valid=valid_only, n=len(ids),
                                          mean=float(np.mean(values)) if values else 0, ci_low=ci[0], ci_high=ci[1]))
    write_csv(root / "contrasts.csv", estimates, list(estimates[0]))
    write_json(root / "run.json", run_info(base, device))


if __name__ == "__main__":
    main()

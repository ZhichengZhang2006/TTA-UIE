"""Grouped descriptive inference for all preregistered budgets."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.manifest import read_csv, write_csv
from prtta.run import write_json


def bootstrap(values, groups, settings):
    sums, counts = defaultdict(float), defaultdict(int)
    for value, group in zip(values, groups):
        sums[group] += value
        counts[group] += 1
    keys = sorted(sums)
    if not keys:
        return [0.0, 0.0]
    totals = np.array([sums[key] for key in keys])
    sizes = np.array([counts[key] for key in keys])
    rng = np.random.default_rng(settings["seed"])
    draws = rng.integers(len(keys), size=(settings["bootstrap_groups"], len(keys)))
    estimates = totals[draws].sum(1) / sizes[draws].sum(1)
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def summarize(directory, rows, settings):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("model", "real"), row["method"], row["target_rms_8bit"]].append(row)
    summary = []
    for (model, method, budget), items in sorted(grouped.items()):
        delta = np.array([row["delta_psnr"] for row in items])
        groups = [row["group_id"] for row in items]
        ci = bootstrap(delta, groups, settings)
        oracle_ci = bootstrap(np.maximum(0, delta), groups, settings)
        summary.append(dict(
            model=model, method=method, budget=budget, n=len(items),
            mean=float(delta.mean()), median=float(np.median(delta)),
            p10=float(np.quantile(delta, 0.1)), p90=float(np.quantile(delta, 0.9)),
            ci_low=ci[0], ci_high=ci[1],
            positive=float((delta > 0).mean()), negative=float((delta < 0).mean()),
            harmful=float((delta < -0.2).mean()), worst10=float(np.sort(delta)[:math.ceil(len(delta) / 10)].mean()),
            oracle_accept_gain=float(np.maximum(0, delta).mean()),
            oracle_ci_low=oracle_ci[0], oracle_ci_high=oracle_ci[1],
            reachability=sum(row["valid"] for row in items) / len(items),
            achieved_rms_8bit=float(np.mean([r["achieved_rms_8bit"] for r in items])),
            delta_ssim=float(np.mean([r["delta_ssim"] for r in items])),
            parameter_delta_l2=float(np.mean([r["parameter_delta_l2"] for r in items])),
            runtime_ms=float(np.mean([r["runtime_ms"] for r in items])),
        ))
    write_csv(directory / "summary.csv", summary, list(summary[0]))
    write_json(directory / "summary.json", summary)
    comparisons = {
        "real": [("physical", other) for other in ("random", "wrong_t", "wrong_B", "fusion")],
        "shared": [("true", "estimated")],
        "dual": [("dual_true", other) for other in
                 ("dual_wrong_t20", "dual_wrong_b20", "simplified_true_td_b",
                  "simplified_coefficient_fit", "simplified_estimated")],
    }
    paired = []
    for model, pairs in comparisons.items():
        for left, right in pairs:
            for budget in settings["budgets_8bit"]:
                a = {r["image_id"]: r for r in grouped.get((model, left, budget), [])}
                b = {r["image_id"]: r for r in grouped.get((model, right, budget), [])}
                common = sorted(a.keys() & b.keys())
                if not common:
                    continue
                for valid_only in (False, True):
                    eligible = [key for key in common if not valid_only or (a[key]["valid"] and b[key]["valid"])]
                    differences = [a[key]["delta_psnr"] - b[key]["delta_psnr"] for key in eligible]
                    ci = bootstrap(differences, [a[key]["group_id"] for key in eligible], settings)
                    paired.append(dict(
                        model=model, comparison=f"{left} - {right}", budget=budget,
                        both_valid_only=valid_only, n=len(eligible), total=len(common),
                        mean=float(np.mean(differences)) if differences else 0, ci_low=ci[0], ci_high=ci[1],
                    ))
    if paired:
        write_csv(directory / "paired.csv", paired, list(paired[0]))
    plot(directory, summary, rows)


def plot(directory, summary, rows):
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    models = sorted({row["model"] for row in summary})
    for metric, ylabel in (
        ("mean", "Mean ΔPSNR (dB)"), ("negative", "Negative update fraction"),
        ("oracle_accept_gain", "Oracle acceptance gain (dB)"), ("reachability", "Reachability"),
        ("runtime_ms", "Search cost per direction (ms)"),
    ):
        fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 4), squeeze=False, constrained_layout=True)
        for ax, model in zip(axes[0], models):
            methods = sorted({r["method"] for r in summary if r["model"] == model})
            for method in methods:
                selected = sorted([r for r in summary if r["model"] == model and r["method"] == method], key=lambda r: r["budget"])
                x, y = [r["budget"] for r in selected], [r[metric] for r in selected]
                line, = ax.plot(x, y, marker=".", label=method)
                if metric in ("mean", "oracle_accept_gain"):
                    lo, hi = ("ci_low", "ci_high") if metric == "mean" else ("oracle_ci_low", "oracle_ci_high")
                    ax.fill_between(x, [r[lo] for r in selected], [r[hi] for r in selected], color=line.get_color(), alpha=0.12)
            ax.axhline(0, color="gray", linewidth=0.6)
            ax.set(xlabel="Target output RMS (8-bit units)", ylabel=ylabel, title=model)
            ax.legend(fontsize=6, ncol=2)
        fig.savefig(directory / f"{metric}_vs_rms.png", dpi=300)
        plt.close(fig)
    if any("transmission_mae" in row for row in rows):
        selected = [row for row in rows if row["target_rms_8bit"] == 1.0]
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
        for ax, field in zip(axes, ("transmission_mae", "background_mae")):
            ax.scatter([r[field] for r in selected], [r["delta_psnr"] for r in selected], s=3, alpha=0.15)
            ax.set(xlabel=field, ylabel="ΔPSNR at 1/255 (dB)")
        fig.savefig(directory / "parameter_error_vs_gain.png", dpi=300)
        plt.close(fig)
    casefile = directory / "failure_tensors.pt"
    if casefile.exists():
        cases = torch.load(casefile, weights_only=True)
        fig, axes = plt.subplots(len(cases), 4, figsize=(8, 2 * len(cases)), squeeze=False, constrained_layout=True)
        for i, (change, image_id, images) in enumerate(cases):
            for j, image in enumerate(images):
                axes[i, j].imshow(image.permute(1, 2, 0).clamp(0, 1))
                axes[i, j].set(xticks=[], yticks=[])
                if i == 0:
                    axes[i, j].set_title(["Input", "Static", "Candidate", "Reference"][j])
            axes[i, 0].set_ylabel(f"{image_id.split(':')[-1]}\n{change:+.3f} dB")
        fig.savefig(directory / "failure_cases.png", dpi=200)
        plt.close(fig)

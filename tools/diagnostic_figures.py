"""Focused comparisons from completed diagnostic CSV files."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    rows = read("results/e1b/fixed/summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for method, marker in (("physical", "o"), ("fusion", "s")):
        items = [r for r in rows if r["method"] == method]
        x = [float(r["budget"]) for r in items]
        y = [float(r["mean"]) for r in items]
        line, = axes[0].plot(x, y, marker=marker, label=method)
        axes[0].fill_between(x, [float(r["ci_low"]) for r in items],
                             [float(r["ci_high"]) for r in items], color=line.get_color(), alpha=0.15)
        axes[1].plot(x, [float(r["runtime_ms"]) for r in items], marker=marker, label=method)
    axes[0].set(xlabel="Target RMS (8-bit units)", ylabel="Mean ΔPSNR (dB)")
    axes[0].axhline(0, color="gray", linewidth=0.6)
    axes[1].set(xlabel="Target RMS (8-bit units)", ylabel="Shared six-budget generation cost (ms)", yscale="log")
    for ax in axes:
        ax.legend()
    fig.savefig("results/e1b/fixed/fusion_vs_adapter.png", dpi=300)
    plt.close(fig)
    rows = read("results/e2/fixed/summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for ax, parameter in zip(axes, ("t", "b")):
        data = [r for r in rows if r["model"] == "shared" and float(r["budget"]) == 1
                and (r["method"] == "true" or r["method"].startswith(parameter + "+") or r["method"].startswith(parameter + "-"))]
        data.sort(key=lambda r: 0 if r["method"] == "true" else int(r["method"][1:]))
        x = [0 if r["method"] == "true" else int(r["method"][1:]) for r in data]
        y = [float(r["mean"]) for r in data]
        ax.errorbar(x, y, yerr=[[v - float(r["ci_low"]) for v, r in zip(y, data)],
                               [float(r["ci_high"]) - v for v, r in zip(y, data)]],
                    marker="o", capsize=3)
        ax.set(xlabel=f"{parameter} perturbation (%)", ylabel="Mean ΔPSNR at 1/255 (dB)")
        ax.axhline(0, color="gray", linewidth=0.6)
    fig.savefig("results/e2/fixed/parameter_sensitivity.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

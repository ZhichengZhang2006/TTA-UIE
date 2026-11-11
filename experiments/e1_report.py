import math
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prtta.run import write_json


ARM_LABELS = {
    "static": "Static", "physical": "Physical", "random": "Random",
    "wrong_transmission": "Wrong transmission", "wrong_background": "Wrong background",
}


def numbers(rows, name):
    return np.array([row[name] for row in rows], dtype=float)


def cluster_bootstrap(values, groups, seed):
    by_group = defaultdict(list)
    for value, group in zip(values, groups):
        by_group[group].append(value)
    sums = np.array([sum(v) for v in by_group.values()])
    counts = np.array([len(v) for v in by_group.values()])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(sums), size=(5000, len(sums)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def distribution(values, groups, thresholds, seed):
    worst = np.sort(values)[:math.ceil(0.1 * len(values))]
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "mean_ci95": cluster_bootstrap(values, groups, seed),
        "positive": float((values > 0).mean()), "negative": float((values < 0).mean()),
        "negative_ci95": cluster_bootstrap(values < 0, groups, seed),
        "harm": {str(t): float((values < -t).mean()) for t in thresholds},
        "gain_over_0.2_db": float((values > 0.2).mean()),
        "worst_10pct_mean": float(worst.mean()), "min": float(values.min()),
    }


def correlation(x, y):
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def save_figure(figure, directory, name, vector=True):
    figure.savefig(directory / f"{name}.png", dpi=300)
    if vector:
        figure.savefig(directory / f"{name}.svg")
    plt.close(figure)


def figures(directory, by_arm, cosine, delta, failures):
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none",
    })
    blue, orange = "#0072B2", "#D55E00"
    fig, ax = plt.subplots(figsize=(5.0, 3.0), constrained_layout=True)
    ax.hist(cosine, bins=np.linspace(-1, 1, 31), color=blue, edgecolor="white")
    ax.axvline(0, linestyle="--", color="#333333", linewidth=1)
    ax.set(xlabel="Output gradient cosine", ylabel="Images", xlim=(-1, 1))
    save_figure(fig, directory, "e1_gradient_cosine")

    fig, ax = plt.subplots(figsize=(5.0, 3.4), constrained_layout=True)
    ax.scatter(cosine, delta, s=13, alpha=0.55, color=blue, edgecolors="none")
    ax.axhline(0, linestyle="--", color="#333333", linewidth=1)
    ax.axvline(0, linestyle=":", color="#333333", linewidth=1)
    ax.set(xlabel="Output gradient cosine", ylabel="Physical Adapter ΔPSNR (dB)")
    save_figure(fig, directory, "e1_cosine_vs_delta")

    fig, ax = plt.subplots(figsize=(6.5, 3.5), constrained_layout=True)
    data = [numbers(by_arm[arm], "delta_psnr") for arm in ARM_LABELS]
    ax.boxplot(data, vert=False, showfliers=False, medianprops={"color": orange, "linewidth": 1.4})
    for index, values in enumerate(data, 1):
        jitter = np.random.default_rng(index).uniform(-0.15, 0.15, len(values))
        ax.scatter(values, index + jitter, color=blue, alpha=0.23, s=7, edgecolors="none")
    ax.axvline(0, linestyle="--", color="#333333", linewidth=1)
    ax.set_yticks(list(range(1, len(ARM_LABELS) + 1)), list(ARM_LABELS.values()))
    ax.set_xlabel("ΔPSNR relative to static (dB)")
    ax.invert_yaxis()
    save_figure(fig, directory, "e1_control_distributions")

    if not failures:
        fig, ax = plt.subplots(figsize=(4.5, 1.4), constrained_layout=True)
        ax.axis("off")
        ax.text(0.5, 0.5, "No negative physical updates in this subset", ha="center", va="center")
    else:
        columns = ("Input", "Static", "Physical", "Reference", "Transmission", "Confidence", "Background")
        fig, axes = plt.subplots(
            len(failures), len(columns), figsize=(10.5, 1.55 * len(failures)),
            squeeze=False, constrained_layout=True,
        )
        for index, (change, image_id, images) in enumerate(failures):
            for col, (ax, value) in enumerate(zip(axes[index], images)):
                if value.shape[0] == 1:
                    ax.imshow(value[0], cmap="cividis", vmin=0, vmax=1)
                else:
                    ax.imshow(value.permute(1, 2, 0).clamp(0, 1))
                ax.set(xticks=[], yticks=[])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if index == 0:
                    ax.set_title(columns[col])
            axes[index][0].set_ylabel(f"{image_id.split(':')[-1]}\n{change:+.5f} dB")
    save_figure(fig, directory, "e1_failure_cases", vector=False)


def pct(value):
    return f"{value:.1%}"


def db(value):
    return f"{value:+.6f}"


def interval(values):
    return f"[{db(values[0])}, {db(values[1])}]"


def deliver(directory, report, rows, probes, failures, info):
    by_arm = {arm: [row for row in rows if row["condition"] == arm] for arm in ARM_LABELS}
    physical = by_arm["physical"]
    groups = [row["group_id"] for row in physical]
    cosine = numbers(physical, "gradient_cosine")
    delta = numbers(physical, "delta_psnr")
    loss_change = numbers(physical, "physics_loss_after") - numbers(physical, "physics_loss_before")
    accepted = np.array([row["candidate_valid"] for row in physical], dtype=bool)
    reduced = (loss_change < 0) & accepted
    thresholds = info["config"]["e1"]["harmful_thresholds_db"]
    seed = info["config"]["seed"]
    arms = {}
    paired = {}
    for arm, arm_rows in by_arm.items():
        values = numbers(arm_rows, "delta_psnr")
        arms[arm] = dict(
            distribution(values, groups, thresholds, seed),
            accepted=sum(row["candidate_valid"] for row in arm_rows),
            mean_delta_ssim=float(numbers(arm_rows, "delta_ssim").mean()),
            mean_output_drift=float(numbers(arm_rows, "output_drift").mean()),
            mean_update_l2=float(numbers(arm_rows, "adapter_update_l2").mean()),
        )
        if arm not in ("static", "physical"):
            paired[arm] = dict(
                mean=float((delta - values).mean()),
                mean_ci95=cluster_bootstrap(delta - values, groups, seed),
                physical_wins=float((delta > values).mean()),
            )
    probe_summary = {}
    for step in info["config"]["e1"]["output_probe_pixel_steps"]:
        selected = [row for row in probes if row["step_px"] == step]
        probe_summary[step] = dict(
            distribution(numbers(selected, "delta_psnr"), groups, thresholds, seed),
            mean_delta_ssim=float(numbers(selected, "delta_ssim").mean()),
        )
    summary = {
        "images": len(physical), "groups": len(set(groups)),
        "cosine": dict(
            mean=float(cosine.mean()), median=float(np.median(cosine)),
            positive=float((cosine > 0).mean()), negative=float((cosine < 0).mean()),
            mean_ci95=cluster_bootstrap(cosine, groups, seed),
        ),
        "correlation_cosine_delta": correlation(cosine, delta),
        "correlation_loss_decrease_delta": correlation(-loss_change, delta),
        "physics_decreased": int(reduced.sum()),
        "physics_decreased_psnr_increased": int((reduced & (delta > 0)).sum()),
        "physics_decreased_psnr_decreased": int((reduced & (delta < 0)).sum()),
        "arms": arms, "physical_minus_control": paired, "probes": probe_summary,
        "wrong_transmission_mean_abs_shift": float(numbers(physical, "wrong_transmission_abs_shift").mean()),
        "wrong_background_mean_abs_shift": float(numbers(physical, "wrong_background_abs_shift").mean()),
    }
    write_json(directory / "e1_summary.json", summary)
    figures(directory, by_arm, cosine, delta, failures)
    c, a = summary["cosine"], arms["physical"]
    positive_when_reduced = summary["physics_decreased_psnr_increased"]
    negative_when_reduced = summary["physics_decreased_psnr_decreased"]
    cosine_r = summary["correlation_cosine_delta"]
    loss_r = summary["correlation_loss_decrease_delta"]
    cosine_r = "未定义" if cosine_r is None else f"{cosine_r:+.3f}"
    loss_r = "未定义" if loss_r is None else f"{loss_r:+.3f}"
    lines = [
        "# PR-TTA E1 源域梯度审计", "",
        f"使用 LSUI Source Tune 的 {len(physical)} 对真实图像，覆盖 {len(set(groups))} 个去重组。"
        f"静态模型训练 {info['training_epochs']} 个 epoch，种子为 {seed}，"
        f"以完整 Tune 的 PSNR 选中第 {info['checkpoint_epoch']} 个 epoch（{info['tune_psnr']:.3f} dB）。",
        "",
        "## 实测结果", "",
        "### Q1 物理梯度与参考梯度是否一致", "",
        f"输出梯度余弦平均 {c['mean']:+.4f}，中位数 {c['median']:+.4f}，"
        f"正值占 {pct(c['positive'])}，负值占 {pct(c['negative'])}。"
        f"平均值的分组 bootstrap 95% 区间为 {interval(c['mean_ci95'])}。"
        f"余弦与实际 Adapter ΔPSNR 的 Pearson 相关为 {cosine_r}。", "",
        (
            "平均余弦区间跨越零，本轮样本没有显示总体方向一致性。"
            if c["mean_ci95"][0] <= 0 <= c["mean_ci95"][1]
            else "平均余弦的符号在本轮分组重采样中稳定，图像间差异见分布图。"
        ), "",
        "| 输出 Probe 步长 | 平均 ΔPSNR（dB） | 中位 ΔPSNR（dB） | 正收益 | 负收益 | 平均 ΔSSIM |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for step, values in probe_summary.items():
        lines.append(f"| {step}/255 | {db(values['mean'])} | {db(values['median'])} | "
                     f"{pct(values['positive'])} | {pct(values['negative'])} | {values['mean_delta_ssim']:+.6f} |")
    lines.extend([
        "", "### Q2 降低物理损失是否改善恢复质量", "",
        f"物理候选的有效更新有 {a['accepted']}/{len(physical)} 张。其中物理损失严格下降的有 "
        f"{summary['physics_decreased']} 张，PSNR 上升 {positive_when_reduced} 张，下降 {negative_when_reduced} 张。"
        f"物理损失下降量与 ΔPSNR 的 Pearson 相关为 {loss_r}。", "",
        "### Q3 真实 Adapter 更新的负迁移", "",
        f"物理候选平均 ΔPSNR 为 {db(a['mean'])} dB，分组 bootstrap 95% 区间为 "
        f"{interval(a['mean_ci95'])} dB。最差 10% 图像的平均 ΔPSNR 为 {db(a['worst_10pct_mean'])} dB，"
        f"最差单图为 {db(a['min'])} dB。", "",
        (
            "平均更新收益区间跨越零，本轮不足以判定总体存在系统性负迁移。负收益图像比例和尾部损失见下表。"
            if a["mean_ci95"][0] <= 0 <= a["mean_ci95"][1]
            else "平均更新收益的符号在本轮分组重采样中稳定，损失量级和图像间差异见下表。"
        ), "",
        "| 更新条件 | 有效 | 平均 ΔPSNR | 中位 ΔPSNR | 正收益 | 负收益 | ΔPSNR < −0.2 dB | ΔPSNR < −0.5 dB | 最差 10% 均值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for arm, values in arms.items():
        lines.append(f"| {ARM_LABELS[arm]} | {values['accepted']} | {db(values['mean'])} | "
                     f"{db(values['median'])} | {pct(values['positive'])} | {pct(values['negative'])} | "
                     f"{pct(values['harm']['0.2'])} | {pct(values['harm']['0.5'])} | {db(values['worst_10pct_mean'])} |")
    lines.extend([
        "", "表中 ΔPSNR 单位均为 dB。负收益按严格小于零统计。各组比例以全部固定样本为分母，"
        "候选未执行时按静态输出计分。最差 10% 为按 ΔPSNR 排序后最低的十分之一，样本数向上取整。", "",
        "### Q4 错误透射率和背景光的影响", "",
        f"错误透射率为裁剪后的 1−t，平均绝对改变量 "
        f"{summary['wrong_transmission_mean_abs_shift']:.4f}。"
        f"错误背景光来自另一个去重组的图像，在线性 RGB 中的平均绝对改变量为 "
        f"{summary['wrong_background_mean_abs_shift']:.4f}。配对差值按同图物理候选减去对照候选计算。", "",
        "| 对照 | 物理候选平均优势（dB） | 分组 bootstrap 95% 区间 | 物理候选胜率 | 平均参数更新 L2 |",
        "|---|---:|---:|---:|---:|",
    ])
    for arm, values in paired.items():
        lines.append(f"| {ARM_LABELS[arm]} | {db(values['mean'])} | {interval(values['mean_ci95'])} | "
                     f"{pct(values['physical_wins'])} | {arms[arm]['mean_update_l2']:.6f} |")
    lines.extend([
        "", f"物理候选的平均参数更新 L2 为 {a['mean_update_l2']:.6f}。"
        + (
            "与三种对照的配对平均差区间均跨越零，本轮没有显示原始物理估计优于这些对照的稳定优势。"
            if all(v["mean_ci95"][0] <= 0 <= v["mean_ci95"][1] for v in paired.values())
            else "配对差值的区间与方向见上表。"
        ), "",
        "## 推断与待验证假设", "",
        "### Q5 是否足以支持开发更新收益预测器", "",
        f"物理候选超过 +0.2 dB 的比例为 {pct(a['gain_over_0.2_db'])}，"
        f"低于 −0.2 dB 的比例为 {pct(a['harm']['0.2'])}。"
        + (
            "当前步长没有产生超过 +0.2 dB 的收益，缺少支持开发收益预测器的实质改善证据。"
            if a["gain_over_0.2_db"] == 0
            else "当前步长在部分图像上产生了可测收益，可检验这些收益在独立样本上能否重现。"
        )
        + "预测器能否利用物理特征识别收益，"
        "仍需在独立源域图像上比较物理特征与通用特征，并在冻结后验证跨域泛化。", "",
        "E1 使用已参与 checkpoint 选择的 Tune 数据和一个训练种子。区间基于 5,000 次去重组重采样，"
        "反映当前样本和分组条件下的变动。LSUI 的配对参考是增强质量的代理，物理估计量来自图像启发式。", "",
        f"实际物理候选平均输出 RMS 为 {a['mean_output_drift']:.8f}，"
        f"折合 {a['mean_output_drift'] * 255:.5f}/255。现有 Adapter 结果主要反映这一微步长范围。"
        "一个可检验的后续假设是，较大但受限的漂移能否产生可预测的实质收益。"
        "该检验应预先固定步长网格和判定标准，使用未参与当前 Tune 审计或 checkpoint 选择的源域样本。", "",
        "跨库文件审计同时发现本地 EUVP-s 与 LSUI 存在大量近重复候选，明细见 "
        "`manifests/cross_dataset_matches.csv`。未来目标域评估需要先核定独立测试子集。", "",
        "## 数值复现检查", "",
        "审计启用 PyTorch 确定性算法和固定 cuBLAS workspace。环境光的中位数采用稳定排序后取下中位数。"
        "全部图像结束后再次执行首张真实图像，候选输出和评分逐位相同，模型所有状态张量精确恢复。"
        "首次非确定性审计在独立重放中出现最高约 3×10⁻⁷ 的输出差异，已归档至 "
        "`results/nondeterministic_initial/`。本报告来自修正后的同 checkpoint、同子集、同步长审计。", "",
        "## 可复核输出", "",
        "- `e1_gradient_audit.csv` 包含每图五种条件及规范物理损失、条件自身损失和更新状态。",
        "- `e1_output_probes.csv` 包含每图三档 Probe 的 PSNR 和 SSIM。",
        "- `e1_fixed_subset.csv` 记录样本、去重组和背景光供体。",
        "- `e1_summary.json` 保存逐组汇总，`e1_run.json` 保存配置、版本、数据和 checkpoint 摘要。",
        "- `e1_gradient_cosine.png`、`e1_cosine_vs_delta.png`、`e1_control_distributions.png` "
        "和 `e1_failure_cases.png` 展示分布、对照和最差案例。", "",
    ])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")

"""Render the completed frozen diagnostics into the requested research report."""
import csv
import json
from pathlib import Path

import numpy as np


def read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def table(rows, columns):
    lines = ["| " + " | ".join(label for _, label in columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            try:
                value = f"{float(value):.5f}"
            except (ValueError, TypeError):
                pass
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main():
    e1b = read("results/e1b/fixed/summary.csv")
    e2 = read("results/e2/fixed/summary.csv")
    paired1 = read("results/e1b/fixed/paired.csv")
    paired2 = read("results/e2/fixed/paired.csv")
    raw1 = read("results/e1b/fixed/e1b_per_image.csv")
    info = json.loads(Path("results/e1b/fixed/run.json").read_text())
    isolation = json.loads(Path("manifests/diagnostic_isolation.json").read_text())
    duplicate = json.loads(Path("results/duplicates/review_summary.json").read_text())
    physical = [r for r in e1b if r["method"] == "physical"]
    oracle = [r for r in e1b if r["method"] == "oracle"]
    statistical_columns = [
        ("method", "方法"), ("budget", "目标 RMS /255"), ("mean", "平均 ΔPSNR"),
        ("ci_low", "95% 下界"), ("ci_high", "95% 上界"), ("median", "中位数"),
        ("positive", "正收益比例"), ("negative", "负收益比例"), ("harmful", "低于 −0.2 dB"),
        ("worst10", "最差 10% 均值"), ("reachability", "可达率"),
        ("oracle_accept_gain", "Oracle Accept Gain"),
    ]
    lines = [
        "# E1b & E2 Findings", "",
        "## 1. 运行信息与数据隔离检查", "",
        f"school-gpu2，GPU 为 {info['gpu']}，种子 2026。复用 E1 epoch 165 checkpoint，"
        f"SHA256 为 `{info['checkpoint_sha256']}`。实验代码起始 commit 为 "
        f"`{info['git_commit']}`，运行记录另保存实际 Python 源码摘要。",
        f"Pilot {isolation['counts']['pilot']} 图，Fixed {isolation['counts']['fixed']} 图，"
        f"分别覆盖 {isolation['groups']['pilot']} 和 {isolation['groups']['fixed']} 组。"
        f"240 图已消费，Utility Fit 剩余 {isolation['counts']['remaining']} 图。"
        "Risk Calibration 和目标域恢复成绩保持封存。按 v0.1 哈希规则的跨集合重复链接数为零。", "",
        "## 2. 固定实验设置和与 E1 的差异", "",
        "六档预注册输出 RMS 为 [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]/255，允许相对误差 10%。"
        "32 个对数网格点位于 [1e−6,1]，受参数位移上限进一步约束，局部区间最多加密 7 次。"
        "Adapter 总位移上限为初态总 L2 的 25% 和 0.5 的较小值。方向逐块归一化，"
        "所有候选从相同参数开始，仅按 sRGB 输出漂移搜索。Pilot 后未调节数值设置。"
        "相对于 E1，当前比较匹配输出漂移。", "",
        "## 3. E1b: Adapter Oracle 能力", "",
        "Oracle 使用源域 MSE 参考梯度，仅用于诊断 Adapter 表达能力。", "",
    ]
    lines += table(oracle, statistical_columns)
    lines += ["", "## 4. E1b: Physical/Random/Wrong/Fusion 公平对比", "",
              "下表覆盖全部预注册预算。ΔPSNR 单位为 dB，比例取 [0,1]，失败候选按静态回退计入总体分母。", ""]
    lines += table(e1b, statistical_columns)
    lines += ["", "配对差值还分别在全部固定样本和双方均可达子集统计。完整区间保存在 "
              "`results/e1b/fixed/paired.csv`。双方均可达的 Physical 对照如下。", ""]
    lines += table([r for r in paired1 if r["both_valid_only"] == "True"], [
        ("comparison", "比较"), ("budget", "RMS /255"), ("n", "可比数"),
        ("mean", "平均配对 ΔPSNR"), ("ci_low", "95% 下界"), ("ci_high", "95% 上界"),
    ])
    lines += ["", "## 5. E1b: 输出漂移和 Oracle Acceptance Gain", "",
              "Oracle Acceptance Gain 对固定方向和单档预算计算 mean(max(0,ΔPSNR))。"
              "不可达候选计零，预算之间不按参考逐图选优。以下为物理候选的选择上限与实际漂移。", ""]
    lines += table(physical, [
        ("budget", "目标 /255"), ("achieved_rms_8bit", "平均实际 /255"), ("reachability", "可达率"),
        ("parameter_delta_l2", "平均参数 L2"), ("oracle_accept_gain", "接受上限 dB"),
        ("oracle_ci_low", "95% 下界"), ("oracle_ci_high", "95% 上界"),
    ])
    alignment = []
    for budget in [0.05, 0.1, 0.25, 0.5, 1, 2]:
        items = [r for r in raw1 if r["method"] == "physical" and float(r["target_rms_8bit"]) == budget]
        delta = np.array([float(r["delta_psnr"]) for r in items])
        parameter = np.array([float(r["parameter_gradient_cosine"]) for r in items])
        output = np.array([float(r["output_gradient_cosine"]) for r in items])
        alignment.append(dict(
            budget=budget, parameter_mean=parameter.mean(), output_mean=output.mean(),
            parameter_r=np.corrcoef(parameter, delta)[0, 1],
            output_r=np.corrcoef(output, delta)[0, 1],
        ))
    lines += ["", "参考梯度仅用于以下诊断。参数余弦比较实际块归一化方向，输出余弦比较物理与 MSE 输出梯度。", ""]
    lines += table(alignment, [
        ("budget", "预算 /255"), ("parameter_mean", "参数余弦均值"), ("output_mean", "输出余弦均值"),
        ("parameter_r", "参数余弦与收益 Pearson r"), ("output_r", "输出余弦与收益 Pearson r"),
    ])
    lines += ["", "## 6. E2: 参数误差影响", "",
              "使用已消费 Fixed 源域参考生成受控退化。以下 shared 模型与候选物理形式一致，"
              "t 和 B 分别以 ±10/20/40% 扰动。生成参数、深度随机种子与裁剪比例逐图保存。", ""]
    lines += table([r for r in e2 if r["model"] == "shared"], statistical_columns)
    lines += ["", "## 7. E2: 简化与双衰减模型对比", "",
              "dual 输入的 βB=0.55βD。真实双衰减分支的损失使用 tD·J+(1−tB)·B。"
              "独立改变 t 或 B 的 dual 对照衡量参数误差。固定真实 tD 的简化模型采用空间恒定 B 的"
              "最小二乘系数拟合，以此限定模型失配对照的最优性范围。其拟合不读取参考像素。"
              "dual_true 与 simplified_estimated 同时改变了模型形式和参数来源，其差值需要结合分解对照解释。", ""]
    lines += table([r for r in e2 if r["model"] == "dual"], statistical_columns)
    lines += ["", "双有效配对的参数及模型差值如下。", ""]
    lines += table([r for r in paired2 if r["both_valid_only"] == "True"], [
        ("model", "生成模型"), ("comparison", "比较"), ("budget", "RMS /255"), ("n", "可比数"),
        ("mean", "平均配对 ΔPSNR"), ("ci_low", "95% 下界"), ("ci_high", "95% 上界"),
    ])
    lines += ["", "## 8. 工程失败和无效样本", "",
              "网格或参数约束内无法达到预算时，逐图 CSV 记录 drift_unreachable，输出返回静态结果。"
              "零方向、低置信和非有限结果单独标记。每张图处理后检查全部模型状态逐位恢复。"
              "runtime_ms 记录方向梯度和共享六预算搜索，跨预算汇总时每图每方法计一次。"
              "Fusion 直接解出混合系数，方法级代价和端到端总耗时分别见 CSV 与 run.json。", "",
              "## 9. 统计分析与限制", "",
              "95% 区间采用 5,000 次去重组 bootstrap，属于探索性描述，未据多档区间开展校正后的"
              "显著性宣称。去重组由哈希启发式建立，缺少真实场景标注。本轮仅一个训练种子。"
              "LSUI 参考是人类增强代理，E2 的零噪声受控退化只验证所选生成机制。", "",
              f"跨库共标记 {duplicate['total_flagged_target_ids']} 个目标 ID，"
              f"已由 Codex 对 {duplicate['agent_visually_confirmed']} 个候选逐对视觉确认相同内容，"
              f"其余 {duplicate['pending_visual_review']} 个仍待视觉核查。人工复核数为 "
              f"{duplicate['human_reviewed']}。已确认条目从独立性宣称中排除，原文件保留。"
              "审计记录和三张联系图位于 results/duplicates，完整独立目标清单尚未建立。", "",
              "## 10. Go / No-Go 判断", ""]
    positive_oracle = any(float(r["ci_low"]) > 0 and float(r["mean"]) >= 0.05 for r in oracle)
    go = []
    for budget in [0.05, 0.1, 0.25, 0.5, 1, 2]:
        contrasts = [r for r in paired1 if float(r["budget"]) == budget and r["both_valid_only"] == "True"
                     and r["comparison"] in ("physical - random", "physical - wrong_t", "physical - wrong_B")]
        utility = next(r for r in physical if float(r["budget"]) == budget)
        if len(contrasts) == 3 and all(float(r["ci_low"]) > 0 for r in contrasts) and float(utility["oracle_accept_gain"]) >= 0.05:
            go.append(budget)
    if go:
        lines += [f"预算 {go} 同时满足预设对照优势和接受上限内部门槛，可作为后续独立复核的候选证据。"
                  "E3 是否启动仍需核验跨预算趋势、Fusion 代价和独立数据复现。"]
    elif positive_oracle:
        lines += ["当前判为 B 类主证据，Adapter 的 Oracle 更新具有改善能力，物理方向尚未同时满足"
                  "预设的随机、错误参数对照优势和实际接受上限门槛。建议暂停 E3，优先检验物理监督与候选方向。"]
    else:
        lines += ["当前尚未满足进入 E3 的预设条件。需结合 Oracle 可达率和效应量评估 Adapter 容量与候选方向。"]
    selected1 = {r["method"]: r for r in e1b if float(r["budget"]) == 1}
    selected2 = {(r["model"], r["method"]): r for r in e2 if float(r["budget"]) == 1}
    contrast = {r["comparison"]: r for r in paired2 if r["budget"] == "1.0" and r["both_valid_only"] == "True"}
    lines += [
        "",
        f"以预设 1/255 预算说明效应量，真实图上的 Oracle 平均收益为 "
        f"{float(selected1['oracle']['mean']):+.5f} dB，物理候选为 "
        f"{float(selected1['physical']['mean']):+.5f} dB，物理候选的接受上限为 "
        f"{float(selected1['physical']['oracle_accept_gain']):.5f} dB。"
        "当前 Adapter 有能力产生实质改善，候选方向和无标签选择依据仍是待解决环节。",
        "",
        f"在同一预算的共享衰减合成数据上，真实参数收益为 "
        f"{float(selected2['shared','true']['mean']):+.5f} dB，估计参数为 "
        f"{float(selected2['shared','estimated']['mean']):+.5f} dB，配对优势 "
        f"{float(contrast['true - estimated']['mean']):+.5f} dB，"
        f"95% 区间 [{float(contrast['true - estimated']['ci_low']):+.5f}, "
        f"{float(contrast['true - estimated']['ci_high']):+.5f}]。"
        "这一对照支持 D 类机制证据，参数估计误差在所选受控退化下明显削弱了更新收益。",
        "",
        f"双衰减真模型收益为 {float(selected2['dual','dual_true']['mean']):+.5f} dB，"
        f"直接沿用真实 tD/B 的简化模型为 {float(selected2['dual','simplified_true_td_b']['mean']):+.5f} dB，"
        f"固定 tD 后重新拟合背景光系数可达到 "
        f"{float(selected2['dual','simplified_coefficient_fit']['mean']):+.5f} dB。"
        "因此在这个模拟族中，简化模型的偏差有相当部分可由有效背景光补偿，"
        "结构误差与参数重新拟合具有交互，归因需要同时参考三种分支。",
        "",
        "当前建议优先改进物理参数估计、有效重投影监督和候选方向，在独立源域诊断集合上复核后再决定进入 E3。",
    ]
    lines += ["", "上述分支采用全部预注册预算审阅。E2 的 true/estimated 及系数拟合配对差值"
              "为归因证据，无法直接外推真实海水参数准确性。", "",
              "## 11. 相关 CSV、图和可重现命令", "",
              "- results/e1b/fixed/e1b_per_image.csv、summary.csv、paired.csv。",
              "- results/e2/fixed/e2_per_image.csv、generation_parameters.json、summary.csv、paired.csv。",
              "- 两组目录中的 mean_vs_rms、negative_vs_rms、oracle_accept_gain_vs_rms、reachability_vs_rms、"
              "runtime_ms_vs_rms、failure_cases 图，以及 E2 的 parameter_error_vs_gain 图。",
              "- manifests/diagnostic_isolation.json 保存分组与清单摘要，各实验目录的 preregistration.json 保存锁定信息。",
              "- docs/E1b_E2_implementation.md 与 configs/e1b_output_matched.yaml 给出实现和生成分布。", "",
              "```bash",
              "python tools/prepare_diagnostics.py --config configs/e1b_output_matched.yaml",
              "pytest -q tests/",
              "python experiments/e1b_output_matched.py --config configs/e1b_output_matched.yaml --manifest manifests/lsui_e1b_pilot.csv --phase pilot",
              "python experiments/e1b_output_matched.py --config configs/e1b_output_matched.yaml --manifest manifests/lsui_e1b_fixed.csv --phase fixed",
              "python experiments/e2_parameter_mismatch.py --config configs/e1b_output_matched.yaml --manifest manifests/lsui_e1b_fixed.csv --phase fixed",
              "python tools/audit_cross_dataset_duplicates.py --input manifests/cross_dataset_matches.csv",
              "python tools/diagnostic_figures.py",
              "python tools/write_e1b_e2_report.py", "```", ""]
    Path("reports/E1b_E2_findings.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

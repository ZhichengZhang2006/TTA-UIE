"""Summarize all three seeds and grouped paired acceptance evidence."""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "axes.prop_cycle": plt.cycler(color=[
        "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#000000", "#999999"] )})

from datasets.manifest import read_csv, write_csv
from experiments.matched_report import bootstrap
from prtta.run import load_config, write_json


def main():
    config = load_config("configs/e2b_operator_study.yaml")
    boot = dict(seed=2026, bootstrap_groups=5000)
    root = Path("results/e2b")
    rows = read_csv(root / "gate/per_image.csv")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["seed"], row["method"], float(row["target_rms_8bit"])].append(row)
    summary = []
    for (seed, method, budget), items in sorted(grouped.items()):
        delta = np.array([float(r["delta_psnr"]) for r in items])
        ci = bootstrap(delta, [r["group_id"] for r in items], boot)
        summary.append(dict(seed=seed, method=method, budget=budget, n=len(items), mean=float(delta.mean()),
            ci_low=ci[0], ci_high=ci[1], median=float(np.median(delta)),
            positive=float((delta>0).mean()), harmful=float((delta < -0.2).mean()),
            negative=float((delta<0).mean()), worst10=float(np.sort(delta)[:math.ceil(len(delta)/10)].mean()),
            oracle_accept_gain=float(np.maximum(delta,0).mean()),
            reachability=sum(r["valid"]=="True" for r in items)/len(items),
            delta_ssim=float(np.mean([float(r["delta_ssim"]) for r in items])),
            runtime_ms=float(np.mean([float(r["runtime_ms"]) for r in items]))))
    write_csv(root / "gate/summary.csv", summary, list(summary[0]))
    seed_summary = []
    timing_summary = []
    for method in sorted({r["method"] for r in summary}):
        for budget in config["budgets_8bit"]:
            items = [r for r in summary if r["method"] == method and r["budget"] == budget]
            seed_summary.append(dict(method=method, budget=budget,
                mean=float(np.mean([r["mean"] for r in items])),
                seed_sd=float(np.std([r["mean"] for r in items], ddof=1)),
                harmful=float(np.mean([r["harmful"] for r in items])),
                delta_ssim=float(np.mean([r["delta_ssim"] for r in items]))))
        items = [r for r in rows if r["method"] == method and float(r["target_rms_8bit"]) == 1]
        timing_summary.append(dict(method=method, **{key: float(np.mean([float(r[key]) for r in items]))
            for key in ("static_ms", "estimation_ms", "gradient_ms", "search_ms", "output_ms", "runtime_ms", "peak_vram_bytes")}))
    write_csv(root / "gate/seed_summary.csv", seed_summary, list(seed_summary[0]))
    write_csv(root / "gate/timing_summary.csv", timing_summary, list(timing_summary[0]))
    failures = sorted([r for r in rows if r["method"] in ("shared", "dual", "effective")
                       and float(r["target_rms_8bit"]) == 1], key=lambda r: float(r["delta_psnr"]))[:30]
    write_csv(root / "gate/worst_cases.csv", failures, list(failures[0]))
    development_summary = []
    for phase in ("development", "effective_development"):
        path = root / phase / "per_image.csv"
        if not path.exists():
            continue
        groups = defaultdict(list)
        for row in read_csv(path):
            groups[row["seed"], row["domain"], row["method"], float(row["target_rms_8bit"])].append(row)
        for (seed, domain, method, budget), items in sorted(groups.items()):
            delta = [float(r["delta_psnr"]) for r in items]
            ci = bootstrap(delta, [r["group_id"] for r in items], boot)
            development_summary.append(dict(phase=phase, seed=seed, domain=domain, method=method, budget=budget,
                mean=float(np.mean(delta)), ci_low=ci[0], ci_high=ci[1], harmful=float(np.mean(np.array(delta) < -.2))))
    write_csv(root / "development_summary.csv", development_summary, list(development_summary[0]))
    per_image = defaultdict(list)
    for row in rows:
        per_image[row["image_id"], row["method"], float(row["target_rms_8bit"])].append(row)
    aggregate = {key: dict(
        delta=np.mean([float(r["delta_psnr"]) for r in values]),
        group=values[0]["group_id"], valid=all(r["valid"]=="True" for r in values),
    ) for key, values in per_image.items()}
    contrasts = []
    methods = sorted({r["method"] for r in rows})
    deployable = [m for m in methods if m in ("shared","dual","effective")]
    for method in deployable:
        for budget in config["budgets_8bit"]:
            ids = sorted(key[0] for key in aggregate if key[1:]==(method,budget))
            for control in ("udcp", "fusion_"+method, "consistency", "residual"):
                if control not in methods:
                    continue
                for valid_only in (False,True):
                    selected = [key for key in ids if not valid_only or (
                        aggregate[key,method,budget]["valid"] and aggregate[key,control,budget]["valid"])]
                    values = [aggregate[key,method,budget]["delta"]-aggregate[key,control,budget]["delta"] for key in selected]
                    ci = bootstrap(values,[aggregate[key,method,budget]["group"] for key in selected],boot)
                    contrasts.append(dict(method=method, control=control, budget=budget, both_valid=valid_only,
                        n=len(selected), mean=float(np.mean(values)) if values else 0, ci_low=ci[0],ci_high=ci[1]))
    write_csv(root / "gate/paired_seed_average.csv",contrasts,list(contrasts[0]))
    fig, axes = plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">")
    for method_index, method in enumerate(methods):
        if "oracle" in method:
            continue
        selected = [r for r in summary if r["method"]==method]
        x = config["budgets_8bit"]
        means = [np.mean([r["mean"] for r in selected if r["budget"]==b]) for b in x]
        harms = [np.mean([r["harmful"] for r in selected if r["budget"]==b]) for b in x]
        oag = [np.mean([r["oracle_accept_gain"] for r in selected if r["budget"]==b]) for b in x]
        for ax, y in zip(axes,(means,harms,oag)):
            ax.plot(x,y,marker=markers[method_index % len(markers)],label=method, markersize=3)
    for ax,label in zip(axes,("Seed-average ΔPSNR (dB)","Harm fraction (< -0.2 dB)","Oracle acceptance gain (dB)")):
        ax.set(xlabel="Output RMS (8-bit units)",ylabel=label);ax.legend(fontsize=6)
    fig.savefig(root / "acceptance_curves.png",dpi=300);plt.close(fig)
    factorial = read_csv(root / "factorial/contrasts.csv")
    fig, axes = plt.subplots(1,2,figsize=(8,3),constrained_layout=True)
    for ax, model in zip(axes,("shared","dual")):
        for label in ("Delta_t","Delta_B","interaction"):
            selected=[r for r in factorial if r["model"]==model and r["contrast"]==label and r["both_valid"]=="False"]
            x=[float(r["budget"]) for r in selected]; y=[float(r["mean"]) for r in selected]
            line,=ax.plot(x,y,marker=markers[("Delta_t","Delta_B","interaction").index(label)],label=label,markersize=3)
            ax.fill_between(x,[float(r["ci_low"]) for r in selected],[float(r["ci_high"]) for r in selected],alpha=.15,color=line.get_color())
        ax.set(title=model,xlabel="Output RMS (8-bit units)",ylabel="Paired effect (dB)");ax.legend()
    fig.savefig(root / "factorial_effects.png",dpi=300)
    fig.savefig(root / "factorial_effects.svg");plt.close(fig)
    decision=[]
    for method in deployable:
        for budget in config["primary_budgets_8bit"]:
            own=[r for r in summary if r["method"]==method and r["budget"]==budget]
            comparisons=[r for r in contrasts if r["method"]==method and r["budget"]==budget
                         and r["control"] in ("udcp","fusion_"+method,"consistency") and not r["both_valid"]]
            decision.append(dict(method=method,budget=budget,
                mean=float(np.mean([r["mean"] for r in own])),harm=float(np.mean([r["harmful"] for r in own])),
                pass_internal=bool(np.mean([r["mean"] for r in own])>.05
                    and np.mean([r["harmful"] for r in own])<.15 and len(comparisons)==3
                    and all(r["ci_low"]>0 for r in comparisons)),
                seed_means=[r["mean"] for r in own]))
    write_json(root / "decision.json",decision)
    def table(items,columns):
        output=["| "+" | ".join(columns)+" |","|"+"|".join("---" for _ in columns)+"|"]
        for row in items:
            vals=[]
            for key in columns:
                value=row[key]
                if isinstance(value,(float,np.floating)):
                    value=f"{value:.6f}"
                vals.append(str(value))
            output.append("| "+" | ".join(vals)+" |")
        return output
    isolation=json.loads(Path("manifests/e2b_isolation.json").read_text())
    trigger=json.loads((root / "trigger.json").read_text())
    parameter_rows=json.loads((root / "development/parameter_errors.json").read_text())
    parameter_groups=defaultdict(list)
    for row in parameter_rows:
        parameter_groups[row["seed"],row["model"],row["method"],row["noise_std"]].append(row)
    parameter_summary=[]
    for (seed,model,method,noise),items in sorted(parameter_groups.items()):
        bg=np.array([r["predicted_background"] for r in items])
        parameter_summary.append(dict(seed=seed,model=model,method=method,noise_std=noise,
            mae_td=float(np.mean([r["mae_td"] for r in items])),
            mae_tb=float(np.mean([r["mae_tb"] for r in items])),
            mae_b=float(np.mean([r["mae_b"] for r in items])),
            t_boundary_fraction=float(np.mean([r["t_boundary_fraction"] for r in items])),
            b_boundary_fraction=float(np.mean([r["b_boundary_fraction"] for r in items])),
            transmission_spatial_std=float(np.mean([r["transmission_spatial_std"] for r in items])),
            background_between_image_std=float(bg.std(0).mean())))
    write_csv(root / "parameter_summary.csv",parameter_summary,list(parameter_summary[0]))
    lines=["# E2b Findings","",
        "## 执行范围与隔离","",
        f"沿用 epoch 165 的增强模型。估计器使用三个训练种子 2026/2027/2028，Shared/Dual 均各训练 60 epoch。"
        f"本轮验收集 {isolation['gate']} 图、{isolation['gate_groups']} 去重组，未来 Utility Fit {isolation['future']} 图。"
        "原 Risk Calibration 摘要未变。增强 Backbone 仍为单个固定训练种子，三种子变化来自新估计器和精修器。",
        "", "## H1 参数误差与交互","",
        "使用原 E2 消费的源域参考重建同一合成机制。四个角点保持同一置信掩码；Delta_t、Delta_B 和交互"
        "以逐图实际候选收益计算。区间按组重采样 5000 次，完整不可达分母与双方可达子集均保存。",""]
    lines+=table(factorial,["model","budget","contrast","both_valid","n","mean","ci_low","ci_high"])
    lines+=["","## H2 合成估计与真实更新","",
        "轻量 64×64 输入 CNN 输出有界平滑透射率和全局背景光，Shared 绑定 tD/tB，Dual 单独预测。"
        "模型选择只使用已消费开发集的损失。训练合成来源、每批种子、衰减和背景光记录在 training 目录。"
        "parameter_errors.json 另外记录真实参数误差、通道与低透射区误差、Spearman、边界比例及固定线性噪声敏感性。",
        ""]
    lines+=table(parameter_summary,["seed","model","method","noise_std","mae_td","mae_tb","mae_b","background_between_image_std"])
    lines += ["", "开发集真实/合成恢复收益单独保存为 results/e2b/development_summary.csv，"
              "合成域的参数改善不能直接替代真实域的更新收益。", ""]
    lines+=["", "## H3 有效参数条件分支","",
        f"开发触发结果为 {trigger['trigger']}，合成参数改进标记为 {trigger['synthetic_pass']}。"
        "判定在开启验收集前完成。触发时拟合有界 8×8 有效参数教师，训练 I/J0 参数预测器和等宽残差精修器。"
        "标签拟合 Oracle 独立标记，推理方法只读取输入及静态输出。",
        "", "## H4 一次性验收与全部种子","",
        "以下 ΔPSNR 单位为 dB。每个种子单独报告，失败候选以静态输出计入分母。",""]
    lines+=table(summary,["seed","method","budget","mean","ci_low","ci_high","median","negative","harmful","worst10","reachability","oracle_accept_gain"])
    lines += ["", "三个种子的均值和样本标准差如下；标准差仅衡量估计器初始化变化，不表示数据抽样置信区间。", ""]
    lines += table(seed_summary, ["method", "budget", "mean", "seed_sd", "harmful", "delta_ssim"])
    lines+=["","## 跨种子配对比较","",
        "先在每个图像内平均三个种子的收益差，再按去重组重采样，避免将同图不同种子当作独立样本。"
        "两种可达性口径均保留，原始每种子记录可单独复核。",""]
    lines+=table(contrasts,["method","control","budget","both_valid","n","mean","ci_low","ci_high"])
    lines += ["", "## 计算开销", "", "时间单位 ms，显存单位 bytes。一次搜索共享四个预算的候选，"
        "runtime 包含静态输出、该方法估计、梯度、四预算搜索及指标计算；不是单预算独立执行的时延。"
        "标签参数拟合 Oracle 的估计耗时包含 80 步优化。", ""]
    lines += table(timing_summary, ["method", "estimation_ms", "gradient_ms", "search_ms", "output_ms", "runtime_ms", "peak_vram_bytes"])
    lines+=["","## Go / Partial / No-Go","",
        "内部推进条件同时考虑主预算平均收益、对 UDCP/自身融合/非物理一致性的配对优势和有害率。"
        "逐种子效应量与区间如上，跨预算区间作为探索性描述。",""]
    lines+=table(decision,["method","budget","mean","harm","pass_internal","seed_means"])
    passed=[r for r in decision if r["pass_internal"]]
    if passed:
        lines+=["", "主预算中存在满足内部推进条件的方法。应结合逐种子一致性和独立数据复核，再决定是否启动 E3。"]
    else:
        lines+=["", "所有可部署估计器均未同时满足预注册推进条件，本轮判为暂缓 E3。"
                 "合成参数改善与真实更新效果需要分别解释，后续方案需重新预注册并使用独立样本。"]
    lines+=["", "在 1/255 主预算上，逐种子平均的恢复变化如下。"]
    for method in deployable:
        selected=[r for r in summary if r["method"]==method and r["budget"]==1.0]
        mean=np.mean([r["mean"] for r in selected])
        harmful=np.mean([r["harmful"] for r in selected])
        paired=[r for r in contrasts if r["method"]==method and r["budget"]==1.0 and not r["both_valid"]]
        text=f"{method} 平均 ΔPSNR 为 {mean:+.6f} dB，有害率为 {harmful:.1%}。"
        for r in paired:
            text+=f" 相对 {r['control']} 的平均优势 {r['mean']:+.6f} dB，95% 区间 [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}]。"
        lines+=["",text]
    lines+=["","## 限制与复现","",
        "单个固定增强 Backbone、一个核心合成分布和基于哈希的去重组限制外推。"
        "有效参数仅表示与恢复目标兼容的成像算子。标签 Oracle 与接受上限均为诊断。"
        "候选 API 的方向与漂移匹配不使用参考，搜索尝试、权重与配置摘要、估计/梯度/搜索/输出时延及显存逐图保存。",
        "目标库采用保守排除策略，全部 1723 个候选暂时排除，其中 24 个经 Codex 视觉确认，1699 个仍待核查。"
        "保留样本 UIEB 889、EUVP-d 5549、EUVP-s 464，不据此宣称场景独立，本阶段未计算目标恢复成绩。",
        "", "主要产物位于 results/e2b/{factorial,training,development,effective_development,gate}。"
        "配置位于 configs/e2b_operator_study.yaml 和 configs/e2b_noise_audit.yaml，权重位于 checkpoints/e2b。",
        "", "```bash","python tools/prepare_e2b.py","python experiments/e2b_factorial_error.py",
        "python experiments/e2b_estimator_training.py","python experiments/e2b_pipeline.py",
        "python tools/prepare_conservative_targets.py","python tools/report_e2b.py","```",""]
    at_one = {r["method"]: r for r in seed_summary if r["budget"] == 1.0}
    residual_reach = np.mean([r["reachability"] for r in summary if r["method"] == "residual" and r["budget"] == 1.0]) if "residual" in at_one else None
    overview = ["## 主要结论", "", "本轮新物理梯度方法均未达到预注册推进标准：物理梯度 TTA 路线判为 No-Go，暂不进入 E3。"
        if not passed else "存在达到内部标准的方法，需结合逐种子与独立数据复核。", ""]
    for method in deployable:
        item = at_one[method]
        overview += [f"- {method} 在 1/255 的三种子平均 ΔPSNR 为 {item['mean']:+.6f} dB，种子标准差 {item['seed_sd']:.6f} dB，有害率 {item['harmful']:.1%}。"]
    if "effective_label_oracle" in at_one:
        overview += [f"- 源域标签拟合有效参数 Oracle 为 {at_one['effective_label_oracle']['mean']:+.6f} dB，监督 Adapter Oracle 为 {at_one['oracle']['mean']:+.6f} dB。可部署预测器未实现该诊断潜力。"]
    if "residual" in at_one:
        overview += [f"- 残差对照为 {at_one['residual']['mean']:+.6f} dB，匹配成功率 {residual_reach:.1%}；不可达候选按协议回退静态，因而该数值包含大量不更新样本。成功匹配子集的配对比较另表报告，不能把整体差值都归因为物理建模。"]
    overview += ["", "合成参数改善支持选定生成分布中的估计能力，尚不支持真实域稳定更新收益。"
        "SSPE/有效参数采用全图重投影，UDCP 使用自身置信权重，实域比较衡量完整候选方案；"
        "参数因子诊断则固定同一置信掩码。目标域未计算恢复分数，未来 290 图和风险校准集仍封存。", ""]
    lines[2:2] = overview
    Path("reports/E2b_findings.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    main()

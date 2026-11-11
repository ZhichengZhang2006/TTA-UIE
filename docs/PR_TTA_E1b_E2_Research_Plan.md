# PR-TTA：E1b 与 E2 可执行研究计划

**版本**：V1.1  
**日期**：2026-11-11  
**目标期刊**：Pattern Recognition  
**研究任务**：验证物理伪监督的候选更新是否具有真实、可利用且可归因的恢复收益  
**执行优先级**：E1b → E2 → Go / No-Go → 决定是否启动 E3  
**现有依据**：`E1_findings.md`、`PR_TTA_Research_Protocol_v1.md`、`configs/v1.yaml`

> 本文件是后续 Codex 编码和实验执行规范，不是已经获得的 E1b 或 E2 实验结果。所有判断均以固定协议下的真实测量为准。此次任务不得训练更新收益预测器，也不得使用目标域参考图像来调节算法。

---

## 1. E1 已观察到的事实

E1 使用 **200 对 LSUI Source Tune 图像**，覆盖 **187 个去重组**。静态 U-Net 训练 200 epochs，在完整 Source Tune 上以 PSNR 选择 epoch 165，Tune PSNR 为 **30.084 dB**。随机种子为 2026。

| 指标 | E1 数值 | 严格解释 |
|---|---:|---|
| 物理梯度与参考梯度的平均输出空间余弦 | −0.0083 | 未观察到总体正向对齐 |
| 余弦的分组 Bootstrap 95% CI | [−0.0630, +0.0462] | 区间跨零 |
| 输出 Probe，步长 1/255，平均 ΔPSNR | −0.168594 dB | 输出空间梯度跟随并非稳定收益 |
| 输出 Probe，步长 2/255，平均 ΔPSNR | −0.500969 dB | 增大 Probe 步长后更差 |
| 输出 Probe，步长 4/255，平均 ΔPSNR | −1.424206 dB | 不能简单将步长放大当作解决方案 |
| 物理损失下降的图像 | 200/200 | 只证明物理目标被优化 |
| 真实 Adapter 更新后 PSNR 增加 | 97/200 | 48.5% 样本正收益 |
| 真实 Adapter 更新后 PSNR 下降 | 103/200 | 51.5% 样本负收益 |
| 平均 Adapter ΔPSNR | −0.000615 dB | 95% CI [−0.001878, +0.000622]，无稳定总体收益证据 |
| 平均真实输出 RMS | 0.00005581 | 等价于约 **0.01423/255**，更新极微弱 |
| 正常物理更新优于随机方向的平均差 | −0.000530 dB | 区间跨零，尚无可靠优势 |
| 余弦与实际 ΔPSNR 的 Pearson r | +0.838 | 值得诊断，但余弦需要参考标签，且当前收益太小 |

**可支持的结论**：本轮实验尚未证明简化物理重投影损失可以带来可靠的真实恢复改善。也没有证明物理估计优于错误参数对照。

**不能支持的结论**：不能宣布物理引导 TTA 全面无效。当前 Adapter 更新幅度过小，实验又发生在与基础模型训练同分布的 Source Tune。它既不是有效的较大更新评估，也不是严格独立的跨域验证。

**本轮最大的工程风险**：本地 EUVP-s 与 LSUI 存在大量近重复候选。`manifests/cross_dataset_matches.csv` 中的候选必须先人工核查，未经核查不能把 EUVP-s 作为独立目标域测试。

---

## 2. 新阶段的核心科学问题

**RQ1：优化器可行性。** 若使用真实参考损失作为诊断梯度，当前冻结 Backbone + Adapter 能否通过合理幅度的单次更新改善图像质量？

**RQ2：物理监督方向。** 在**相同输出变化预算**下，真实物理梯度是否优于随机方向、错误物理参数及简单输出融合？

**RQ3：收益空间。** 对既定的候选生成策略，完美接受/拒绝的理论选择上限（Oracle Acceptance Gain）有多大？若几乎为零，收益预测器没有足够的性能空间。

**RQ4：失效来源。** 物理更新失败主要由透射率或后向散射参数估计误差导致，还是由简化成像模型假设或参考图像的目标差异导致？

**RQ5：后续方向。** E1b + E2 结束后应进入收益预测、修正物理监督、调整 Adapter，还是停止该 TTA 技术路线？

### 2.1 主要记号

- 输入水下图像：`I`，sRGB，[0,1]。
- 配对参考图像：`Y`，仅用于源域诊断和实验评分。
- 固定 Backbone 参数：`θ`。
- 源域训练好的 Adapter 初始参数：`φ₀`。
- 静态输出：`J₀ = f_{θ,φ₀}(I)`。
- 给定方向 `d` 与步长系数 `α` 的输出：`J(α,d) = f_{θ,φ₀−αd}(I)`。
- 输出变化：`D(J,J₀) = sqrt(mean((J−J₀)²))`。
- 候选更新收益：`u(I,d,α) = PSNR(J(α,d),Y) − PSNR(J₀,Y)`。

仅对开发/诊断图像计算 `u`。**候选更新、漂移匹配和物理参数估计过程中，绝不允许读取 Y**。只有 `Oracle` 对照允许显式使用 Y 生成更新方向，且该对照永远不能作为可部署方法。

---

## 3. 固定实验资源、模型与数据划分

### 3.1 环境和模型

- GPU 优先使用 **NVIDIA A100-PCIE-40GB**。
- 使用既有 E1 静态 U-Net Checkpoint，**不重新选择 epoch，不根据 E1b/E2 评分重训练**。
- Backbone 为 4-stage U-Net，宽度 `[32,64,128,256]`。
- Adapter 为解码器低秩 1×1 卷积，rank=4，alpha=4。
- 只修改 Adapter，Backbone 保持冻结。
- 输入统一 256×256，RGB [0,1]。明确记录插值与 sRGB→linear RGB 转换。
- 单图独立更新。每张图像都从完全一致的 `φ₀` 开始，结束后恢复全部模型状态。
- 固定 `seed=2026` 用于首轮验证。重要定性结论至少在其他静态模型训练种子上复查，不能以一个种子宣称模型普遍有效。
- 对照组使用相同模型、相同样本、相同输出变化预算与相同评分代码。

### 3.2 新的开发和固定验证样本

原 LSUI 分组方案为 `Train / Tune / Utility Fit / Risk Calibration`。E1 用过 Tune，又用 Tune 选过静态 Checkpoint。因此：

1. **E1b-Pilot**：从原 `Utility Fit` 中预先选择约 40 张，优先按组抽样。只用于代码与数值调试。
2. **E1b-Fixed**：从尚未使用的 `Utility Fit` 中选择约 200 张，与 Pilot 完全按去重组隔离。在执行前锁定图像 ID、组、配置和全部评价规则。
3. 这约 240 张在后续**不得重新当作 Utility Predictor 的无偏训练/校准/验证样本**。应将其单独标记为 `diagnostic_consumed`，并相应缩小或重新规划 Utility Fit 样本。
4. **Risk Calibration** 保持封存，不能因 E1b 样本不足而占用。
5. **UIEB/EUVP 参考图像保持封存**。E1b/E2 期间不得观察其定量测试成绩并据此选步长或方法。
6. 允许对 E1b-Fixed 的结果开展一次预注册的科学假设检验。若基于结果追加新方案，需明确标注为探索性研究，后续使用新的按组独立数据复核。

示意清单：

```text
manifests/
  lsui_all.csv
  lsui_e1b_pilot.csv
  lsui_e1b_fixed.csv
  lsui_diagnostic_consumed.csv
  lsui_remaining_utility_fit.csv
  risk_calibration_sealed.csv
  cross_dataset_matches.csv
  target_clean_independent.csv      # 仅去重完成后生成
```

> 配对图像是否来自同一场景，应以分组检查为准。数量服从分组，不强求恰好 40、200。原 Source Train 与这些开发图像也需要无近重复重叠。

---

## 4. E1b：Output-Matched Adaptation Feasibility Study

### 4.1 为什么要做输出变化匹配

E1 中不同对照的 Adapter 参数改变量 L2 相同，但它们在网络输出空间造成的变化未必相同。输入图像依赖的 Jacobian 会让相同参数位移生成不同的图像位移，因此用参数 L2 匹配不能保证方向比较公平。

对每张图像、每个更新方向，预设目标输出 RMS：

\[
\rho\in\left\{\frac{0.05}{255},\frac{0.10}{255},\frac{0.25}{255},\frac{0.50}{255},\frac{1.00}{255},\frac{2.00}{255}\right\}.
\]

这是一组**探索性且预先固定的有限预算**，不是根据结果挑选出来的最优步长。后续应绘制全曲线，不能只报告曲线上的最佳点。

目标为：

\[
\left|D(J(\alpha,d),J_0)-\rho\right|\leq\varepsilon_\rho,
\]

其中 `ε_ρ = 0.10 × ρ` 为建议的相对容差。每个实验还须设定独立的 Adapter 参数位移上限，防止追求输出漂移而走向极端权重值。建议以参数初始范数和有限绝对更新上限双重约束，并在 Pilot 上事先确定具体实现。

**不能假定输出 RMS 对参数步长全局单调。** 推荐在有限步长网格上先搜索可用局部区间，再在有效区间内二分或加密采样，最后选择漂移误差最小的合法点。不可仅凭无条件二分便声称找到全局最优。

### 4.2 对照组

| ID | 方向/操作 | 是否读取参考 Y | 目的 |
|---|---|---:|---|
| S0 | 不更新 Static | 否 | 静态基准 |
| S1 | 正常物理梯度 | 否 | 验证物理伪监督 |
| S2 | 随机 Adapter 参数方向，使用固定 hash 种子 | 否 | 检查任意扰动的效应 |
| S3a | 使用错误透射率 `t_wrong = clip(1−t)` 的物理梯度 | 否 | 参数有效性对照 |
| S3b | 使用来自不同去重组图像的背景光 `B_wrong` 的物理梯度 | 否 | 参数有效性对照 |
| S4 | Oracle 参考损失的 Adapter 梯度 | **是，仅诊断** | 测试 Adapter 表达能力与更新可行性 |
| S5 | `J₀` 与物理反演结果之间的输出融合 | 否 | 判断是否必须反向传播更新网络 |

Oracle 不计入可部署方法效果。Oracle 参考梯度采用与 PSNR 一致的 MSE：

\[
\mathcal L_{ref}(\phi)=\operatorname{MSE}(f_{\theta,\phi}(I),Y),\qquad
 d_{oracle}=\nabla_\phi\mathcal L_{ref}(\phi_0).
\]

物理方向采用此前定义的 Charbonnier 重投影损失：

\[
\mathcal L_{phy}=\frac{\sum_xc(x)\frac13\sum_{k=1}^3\left(\sqrt{r_k(x)^2+\epsilon^2}-\epsilon\right)}{\sum_x c(x)+\varepsilon},
\quad r=I_{lin}-[t\odot J_{lin}+(1-t)\odot B].
\]

`I_lin=srgb_to_linear(I)`，`J_lin=srgb_to_linear(J)`。估计得到的 `t,B,c` 均 detach。输出变化匹配应基于统一的 **sRGB 输出 J**，评分也使用相同的 sRGB 图像。

### 4.3 物理反演及公平融合基线

对于简化模型 `I_lin = t J_lin + (1−t)B`，当 `t>0` 时：

\[
J_{phys,lin}=\operatorname{clip}\left(\frac{I_{lin}-(1-t)B}{t},0,1\right),
\quad J_{phys}=\operatorname{linear\_to\_srgb}(J_{phys,lin}).
\]

定义输出融合：

\[
J_{fusion}(\lambda)=(1-\lambda)J_0+\lambda J_{phys},\quad \lambda\in[0,1].
\]

通过仅依赖 `D(J_fusion,J₀)` 的搜索选择 `λ`，使融合方法达到与 Adapter 更新相同的输出预算。若 `J_phys` 太接近 `J₀` 导致目标漂移不可达，记录为不可达，而不是修改参数或偷偷扩大 `λ` 到 `[0,1]` 之外。

### 4.4 输出漂移匹配的实现伪代码

```python
import math
import torch

@torch.no_grad()
def match_output_drift(
    model, image, adapter_snapshot, direction,
    target_rms, alpha_grid, relative_tolerance=0.10,
    max_param_delta_norm=None
):
    """
    输入:
      direction: 按 adapter_parameters 的固定顺序排列的方向张量
      alpha_grid: 预先固定、由小到大的正步长网格
    输出:
      J_candidate, valid, metrics

    只使用 image, J0 和模型参数。绝不读取 GT reference。
    """
    restore_full_model_state(model, adapter_snapshot)
    J0 = model(image).detach().clone()
    params = list(adapter_parameters(model))
    assert len(params) == len(direction)

    best = None
    for alpha in alpha_grid:
        restore_full_model_state(model, adapter_snapshot)
        squared_norm = 0.0
        for p, d in zip(params, direction):
            update = alpha * d
            squared_norm += update.square().sum().item()
            p.add_(-update)

        delta_norm = math.sqrt(squared_norm)
        if (max_param_delta_norm is not None and
                delta_norm > max_param_delta_norm):
            continue

        J = model(image).detach().clone()
        if not torch.isfinite(J).all():
            continue
        drift = (J - J0).square().mean().sqrt().item()
        error = abs(drift - target_rms)
        if best is None or error < best["error"]:
            best = {
                "image": J, "alpha": alpha,
                "drift": drift, "error": error,
                "parameter_delta_l2": delta_norm,
            }

    restore_full_model_state(model, adapter_snapshot)
    valid = (best is not None and
             best["error"] <= relative_tolerance * target_rms)
    if not valid:
        return J0, False, {"reason": "drift_unreachable"}
    return best["image"], True, best
```

说明：此段仅示意固定网格方案。正式实现可以用有限区间加密提高精度，但须保证搜索过程的目标始终只是输出 RMS，而不是 PSNR。`restore_full_model_state` 应恢复完整的 Adapter 参数和必要的非参数状态。每张图像处理完后再次进行状态哈希与精确恢复检查。

### 4.5 方向生成伪代码

```python
def get_candidate_direction(model, I, Y, mode, hyp, seed):
    # 执行前恢复模型初态，主干和非 Adapter 模块必须冻结
    params = list(adapter_parameters(model))
    J = model(I)

    if mode == "physical":
        loss = physics_loss(I, J, hyp)
    elif mode == "wrong_t":
        loss = physics_loss(I, J, hyp.with_wrong_transmission())
    elif mode == "wrong_B":
        loss = physics_loss(I, J, hyp.with_wrong_background())
    elif mode == "oracle":
        assert Y is not None   # 仅限源域诊断
        loss = ((J - Y) ** 2).mean()
    elif mode == "random":
        return fixed_random_unit_directions(params, seed)
    else:
        raise ValueError(mode)

    gradients = torch.autograd.grad(
        loss, params, allow_unused=True, retain_graph=False
    )
    directions = replace_none_with_zeros(gradients, params)
    return normalize_by_parameter_block(directions)
```

真实代码应先统一使用 `eval()`，禁用会影响统计的随机层，验证每一类方向非零且数值有限。若 `d=0` 则标为 `zero_direction`，不得通过随机方向替代。

### 4.6 E1b 评分和关键指标

对所有合法候选，逐张报告：

\[
\Delta PSNR_i=PSNR(J_{1,i},Y_i)-PSNR(J_{0,i},Y_i).
\]

主要指标包括：

1. 平均、中位与分位数 ΔPSNR，平均 ΔSSIM。
2. 正收益比例、`ΔPSNR < 0` 比例、`ΔPSNR < −0.2 dB` 比例。
3. 最差 10% 图像的平均 ΔPSNR。
4. 各预算下的更新可达率、失败率、平均输出 RMS、参数位移、运行时间。
5. **同图配对**差值：`Physical − Random`、`Physical − Wrong t`、`Physical − Wrong B`、`Physical − Fusion`。
6. 与参考梯度相关性和实际 Adapter 方向一致性，明确区分输出空间和参数空间。

#### 完美选择器上限

给定方向与预算 `ρ`，定义：

\[
\operatorname{OracleAcceptGain}(\rho)=\frac1N\sum_{i=1}^N\max(0,\Delta PSNR_i(\rho)).
\]

对不可达的候选，按预先约定返回 `J₀`，令 `ΔPSNR=0`，同时另外报告不可达比例。该指标是对**固定候选生成方法**，允许参考图像决定“接受或拒绝”时的平均 PSNR 收益上限，**不是对所有可能 TTA 方法的全局上限**。不同 `ρ` 的指标应单独报告，不可利用参考图像跨 `ρ` 逐图挑最优更新后再宣称是可部署结果。

#### 统计方法

- 使用去重组为单位的 paired/bootstrap，固定 5,000 次重采样，报告 95% CI。
- 主要结论对**全部预先指定漂移预算**报告，不得仅汇报最佳预算。
- 若对多个预算或多种方向进行显著性检验，报告多重比较处理或明确说明是探索性结果。
- 同时保留每张图像、每个对照、每档预算的逐行数据，以便复核配对差值。

---

## 5. E2：物理估计误差与成像模型失配

### 5.1 要区分的两个错误来源

1. **参数误差**：物理模型结构正确，但从单张图像估计的 `t,B` 不准。
2. **模型失配**：即使参数估计正确，使用相同衰减描述直接透射与后向散射也不适用。

E2 不能只观察物理重投影损失，必须观察实际恢复参考质量和候选更新收益。

### 5.2 合成协议

仅从**已标记为诊断用途的公开源域参考图像**生成受控退化图，不采集外部图像。先转换到近似线性 RGB，再按前向模型生成，最后转换回 sRGB 输入网络。

\[
I_c(x)=t_c^D(x)Y_{lin,c}(x)+[1-t_c^B(x)]B_c+n_c(x),
\]

\[
t_c^D(x)=\exp[-\beta_c^D z(x)],\qquad
 t_c^B(x)=\exp[-\beta_c^B z(x)].
\]

其中 `z(x)` 使用固定种子生成的平滑深度代理场，`β_D` 与 `β_B` 为按通道设置的正系数，`B` 为固定的非负背景光。具体随机分布需在 Pilot 固定到配置，并记录每张样本的生成参数与随机种子。

**物理解释限制**：公开 LSUI 的增强参考是人类增强质量代理，并不是相机测得的真实无水介质场景辐射。合成实验仅能验证已知生成机制下的数值敏感性，不能证明真实海水物理参数估计正确。

### 5.3 E2 固定实验组

| ID | 生成模型 | 候选使用的模型/参数 | 主要问题 |
|---|---|---|---|
| P0 | 简化 `tD=tB` | 与生成过程一致的真实参数 | 算法实现与已知参数下的监督可行性 |
| P1 | 简化 `tD=tB` | UDCP-like 估计参数 | 单图估计误差影响 |
| P2 | 简化 `tD=tB` | 真实参数分别扰动 ±10%、±20%、±40% | 参数敏感性曲线 |
| P3 | 简化 `tD=tB` | 错误透射率或交换背景光 | 假物理控制 |
| P4 | 双衰减 `tD≠tB` | 双衰减真实生成参数 | 准确建模是否提高收益 |
| P5 | 双衰减 `tD≠tB` | 简化模型 + 估计参数 | 真实模型不匹配时的影响 |

**注意**：P4 的重投影损失也必须采用对应的双衰减模型，不能仅在数据生成端换公式，却在优化损失中沿用单透射率形式。

### 5.4 关键对比和公式

同图、同预算下定义：

\[
\Delta_{est}=u(\text{真实参数})-u(\text{估计参数}).
\]

\[
\Delta_{model}=u(\text{正确双衰减模型})-u(\text{简化模型}).
\]

由于 `u` 是真实候选更新的 ΔPSNR，上式能够衡量在固定实验设置下参数错误和模型错误对恢复收益的影响。均应报告组 Bootstrap 区间。

E2 只在实验中确实存在两种可比方案的样本上定义上述差值，并记录不可达率。对比 P4 与 P5 时要清楚说明同时改变了物理模型和参数来源，不能把差异全部归因于模型结构。必须另做 **正确双衰减模型 + 错误参数** 与 **简化模型 + 最优匹配参数** 的分解消融，才能严格拆分两类效应。

### 5.5 合成器伪代码

```python
@torch.no_grad()
def synthesize_underwater(
    Y_srgb, depth, beta_direct, beta_backscatter,
    background_linear, noise_std, seed
):
    # shape: Y=[1,3,H,W], depth=[1,1,H,W]
    Y_lin = srgb_to_linear(Y_srgb)
    t_direct = torch.exp(-beta_direct * depth)
    t_backscatter = torch.exp(-beta_backscatter * depth)

    clean_degraded = (
        t_direct * Y_lin
        + (1.0 - t_backscatter) * background_linear
    )

    noise = seeded_gaussian_like(clean_degraded, seed, noise_std)
    I_lin = (clean_degraded + noise).clamp(0, 1)
    I_srgb = linear_to_srgb(I_lin)

    params = {
        "t_direct": t_direct.detach(),
        "t_backscatter": t_backscatter.detach(),
        "background": background_linear.detach(),
        "depth": depth.detach(),
        "beta_direct": beta_direct.detach(),
        "beta_backscatter": beta_backscatter.detach(),
    }
    return I_srgb, params
```

### 5.6 双衰减物理损失

```python
def dual_attenuation_loss(I_srgb, J_srgb, hyp, mask, eps=1e-3):
    I_lin = srgb_to_linear(I_srgb)
    J_lin = srgb_to_linear(J_srgb)
    td = hyp["t_direct"].detach()
    tb = hyp["t_backscatter"].detach()
    B = hyp["background"].detach()
    c = mask.detach()

    reconstruction = td * J_lin + (1 - tb) * B
    residual = I_lin - reconstruction
    penalty = torch.sqrt(residual.square() + eps**2) - eps
    per_pixel = penalty.mean(dim=1, keepdim=True)
    return (per_pixel * c).sum() / (c.sum() + 1e-8)
```

原有物理估计器可能只输出一个 `t`。在 P4 的真实参数分支使用生成记录中的 `t_direct`、`t_backscatter`，但在实际部署分支中不得读取合成真值。

---

## 6. 重要补充对照：为什么物理方向可能从根本上不合适

简化成像方程对固定 `t>0,B` 的零残差解为：

\[
J_{phys,lin}=\frac{I_{lin}-(1-t)B}{t}.
\]

所以物理重投影梯度实际上推动模型输出靠近该物理反演解，而不是直接推动输出靠近数据集增强参考 `Y`。这两者一般不等价。

因此，**E1b 必须同时比较输出空间简单融合与 Adapter 更新**。若输出融合在相同漂移预算下始终不劣于参数更新，还应评估是否值得继续使用 TTA 架构。

还需预防一种常见伪结论：`g_phy` 与 `g_ref` 的输出余弦在极小更新下和 ΔPSNR 高相关，并不证明存在可部署的风险预测器。`g_ref` 依赖参考图像，推理时不可用。对于实际任务，必须验证**不含 Y 的可观测特征**能否预测具有实质幅度的更新收益。

---

## 7. 工程任务与新增目录

继承 E0–E1 仓库，不重复创建基础模型。建议增加：

```text
PR-TTA/
  configs/
    e1b_output_matched.yaml
    e2_synthetic_physics.yaml
  adaptation/
    oracle_direction.py
    direction_controls.py
    output_drift_match.py
    physical_output_fusion.py
  physics/
    synthetic_underwater.py
    dual_attenuation.py
  experiments/
    e1b_output_matched.py
    e2_parameter_mismatch.py
  evaluation/
    group_bootstrap.py
    gain_risk_curves.py
  tests/
    test_output_drift_matching.py
    test_adapter_full_reset.py
    test_oracle_reference_isolation.py
    test_synthetic_forward.py
    test_dual_physics_grad.py
    test_dataset_split_isolation.py
  reports/
    E1b_E2_findings.md
  results/
    e1b/
    e2/
```

### 7.1 建议命令契约

```bash
pytest -q tests/

python experiments/e1b_output_matched.py \
  --config configs/e1b_output_matched.yaml \
  --manifest manifests/lsui_e1b_pilot.csv \
  --phase pilot

python experiments/e1b_output_matched.py \
  --config configs/e1b_output_matched.yaml \
  --manifest manifests/lsui_e1b_fixed.csv \
  --phase fixed

python experiments/e2_parameter_mismatch.py \
  --config configs/e2_synthetic_physics.yaml \
  --manifest manifests/lsui_e1b_fixed.csv

python tools/audit_cross_dataset_duplicates.py \
  --input manifests/cross_dataset_matches.csv
```

命令仅是接口建议。Codex 应实现后以 `--help` 输出核验真实参数。

### 7.2 配置建议

```yaml
project: pr_tta_e1b_e2
seed: 2026
checkpoint: checkpoints/source_unet_best.pt  # 以仓库实际路径为准
model:
  backbone: unet4
  adapter_rank: 4
  adapter_alpha: 4
input:
  resolution: 256
  data_space: srgb
  physics_space: linear_rgb
splits:
  pilot_manifest: manifests/lsui_e1b_pilot.csv
  fixed_manifest: manifests/lsui_e1b_fixed.csv
  sealed_risk_calibration: manifests/risk_calibration_sealed.csv
e1b:
  target_output_rms_8bit: [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]
  relative_tolerance: 0.10
  alpha_grid_steps: 32
  alpha_search_strategy: bounded_grid_and_refine
  include_oracle_diagnostic: true
  include_direct_fusion: true
  controls: [static, physical, random, wrong_t, wrong_B, oracle, fusion]
  invalid_candidate_policy: static_with_failure_flag
  bootstrap_groups: 5000
e2:
  models: [shared_attenuation, dual_attenuation]
  perturbation_percent: [10, 20, 40]
  compare_true_vs_estimated_parameters: true
  compare_model_mismatch: true
  fixed_output_budgets_from_e1b: true
  save_generation_params: true
report:
  per_image_csv: true
  plots: true
  failure_cases: true
```

**须在 Pilot 期间固定**：`alpha_grid` 绝对上下界、Adapter 更新上限、随机深度场生成分布、通道衰减系数范围、噪声标准差、无法达到目标 RMS 的处理和统计测试列表。固定前不得开始正式 E1b-Fixed。

---

## 8. 单元测试与工程验收

必须至少通过以下检查：

1. **方向一致性**：物理梯度方向对 Adapter 参数非零、有限，Oracle 方向确实来自 MSE，并且计算图正确。
2. **参考隔离**：除 `oracle` 和最终 `evaluation` 外，所有候选生成 API 的签名都不接收 `reference`。
3. **无图像污染**：同一图像两次独立运行结果逐位一致，不同图像之间参数和状态不传递。
4. **输出漂移**：能达到目标时误差满足预设容差，不能达到时明确失败，不伪造匹配。
5. **参数约束**：超出允许 Adapter 参数位移则拒绝，拒绝后恢复初态。
6. **有效性标记**：匹配失败样本出现在逐图 CSV 和总体分母，不得只统计成功样本。
7. **物理空间**：sRGB 与 linear RGB 的编码和反编码测试通过，双衰减公式按通道广播正确。
8. **已知合成真值**：在无噪声、使用正确参数时，双衰减前向重投影残差接近数值精度。
9. **错误模型对照**：简化和双衰减的差异真实生效，不得仅更换配置标签。
10. **样本互斥**：Source Train、E1 Source Tune、E1b-Pilot、E1b-Fixed、Risk Calibration 跨组不发生泄漏。
11. **重采样**：Bootstrap 以 `group_id` 而非单张图像为最小单元。
12. **日志可复现**：每次运行保存源代码 commit、Checkpoint hash、配置 hash、样本清单 hash、随机种子、硬件与依赖版本。

---

## 9. 输出表格、图片与报告模板

### 9.1 逐图记录

保存 `results/e1b/e1b_per_image.csv`，每行对应一张图、一个方向和一个预算。字段至少包括：

```text
image_id,group_id,method,target_rms_8bit,achieved_rms_8bit,
valid,failure_reason,alpha,parameter_delta_l2,
psnr_static,psnr_candidate,delta_psnr,
ssim_static,ssim_candidate,delta_ssim,
phy_loss_before,phy_loss_after,gradient_norm,
runtime_ms,seed,checkpoint_hash
```

E2 另存每张合成图像的真实 `β_D`、`β_B`、背景光、深度生成种子、参数估计误差和模型类型，且保证可从配置复现。

### 9.2 必须生成的图

1. **ΔPSNR vs Output RMS**：所有方向的均值、Bootstrap 区间和负迁移比例。
2. **Oracle / Physical / Random / Wrong-Physics vs Output RMS**：判别更新机制与物理监督价值。
3. **Oracle Acceptance Gain vs Output RMS**：判断收益预测器的可实现空间。
4. **物理参数误差 vs ΔPSNR**：包括置信区间和失败案例。
5. **Fusion vs Adapter TTA**：相同预算下的收益与计算代价。
6. **EUVP–LSUI 近重复核查**：已确认重复、未确认候选、已排除样本数和人工复核依据。

### 9.3 `reports/E1b_E2_findings.md` 的固定结构

```text
# E1b & E2 Findings
1. 运行信息与数据隔离检查
2. 固定实验设置和与 E1 的差异
3. E1b: Adapter Oracle 能力
4. E1b: Physical/Random/Wrong/Fusion 公平对比
5. E1b: 输出漂移和 Oracle Acceptance Gain
6. E2: 参数误差影响
7. E2: 简化与双衰减模型对比
8. 工程失败和无效样本
9. 统计分析与限制
10. Go / No-Go 判断
11. 相关 CSV、图和可重现命令
```

---

## 10. Go / No-Go：停止、修正还是进入 E3

| 结果类型 | 实测模式 | 下一步 |
|---|---|---|
| A：物理方向有价值 | 在预注册预算范围内，Physical 比 Random/Wrong-Physics 更好，且 Oracle Acceptance Gain 有实际量级 | 可进入 E3 更新收益预测研究 |
| B：Adapter 有能力，物理方向无效 | Oracle 有清晰正收益，Physical 无效或持续负收益 | 暂停 E3，优先修正物理监督或候选生成 |
| C：Adapter 本身受限 | Oracle 在合理输出预算内也难以改善 | 优先做 Adapter rank/插入位置/更新层的容量诊断 |
| D：只有真物理参数有效 | Synthetic + true physical parameters 有效，但估计参数失效 | 研究参数估计与不确定性建模，不直接训练更新收益预测器 |
| E：直接融合已经更好 | Fusion 在同预算下持续优于 Adapter TTA 且成本低 | 认真考虑将核心方法从 TTA 转向物理先验融合 |

### 10.1 建议的内部继续标准

对 E1b-Fixed：

- 至少某个**预先指定**漂移预算下，`Physical − Random` 与 `Physical − Wrong-Physics` 的组 Bootstrap 95% CI 下界大于零，或出现跨多个相邻预算一致的正向趋势并在独立数据中复核。
- 在同一候选产生机制下，`OracleAcceptGain` 不是极小数值扰动。可将 `≥0.05 dB` 作为**内部探索性实用性门槛**，但不是自然定律或 PR 录用标准。
- 如收益来自极少数个例，还必须报告覆盖率及最差 10% 图像损失，不能只强调平均值。
- 候选生成代价必须与直接融合比较，避免以非常大的推理开销换取极小收益。

若多个预算均失败，不能通过事后加更多预算、选择最优子集或用参考图像决定下一次候选来伪造可靠收益。可以发起新的探索性机制，但应使用独立数据验证。

---

## 11. 给 Codex 的执行次序与权限

**执行顺序：**

1. 阅读 `E1_findings.md`、原 V1 研究规范和实际仓库代码。
2. 检查 E1 checkpoint、manifest、模型状态快照和参数列表次序。
3. 完成 Source Train/Utility Fit/E1b/Risk Calibration 的按组数据隔离审核。
4. 新增必要模块，先运行全部单元测试。
5. 先跑约 40 张 E1b-Pilot，检查输出漂移可达率、数值稳定性、Oracle 更新和配置边界。
6. 锁定 E1b-Fixed 配置与样本清单，保存 hash 后运行约 200 张固定实验。
7. 生成全部逐图 CSV、统计表与图。
8. 用同一已消费诊断集合运行 E2 可控合成实验。
9. 核查 EUVP–LSUI 跨数据集近重复情况并保存审计记录。
10. 编写 `E1b_E2_findings.md`，根据事先规定的证据作出继续或转向建议。

**Codex 可以自行**：编写代码、增补测试、修复环境问题、在已定义范围内进行 Pilot 调试和正式实验。

**Codex 不可自行**：改变数据集分工、泄漏目标域标签、使用 GT 调整非 Oracle 更新、擅自放宽检验阈值、删除不利样本、伪造实验结果或提前实现 E3 作为主结论。

**交付最低要求**：可运行源码、锁定配置、固定验证清单、单元测试结果、E1b/E2 CSV、至少 5 张规定图、跨库重复核查、可复现命令以及完整研究报告。未能完成的项目必须明确标记原因。

---

## 12. 最终研究建议

从 E1 现有事实出发，对原始简化物理重投影损失作为唯一 TTA 监督保持审慎。它目前没有稳定优于错误参数方向，也没有显示总体恢复质量收益。同时，现有 Adapter 输出变化过小，尚未完成真正有辨识力的候选方向比较。

**下一个最优实验不是训练风险预测器，而是在同等输出变化预算下确认“Adapter 有能力”和“物理方向有价值”是否同时成立。** 若两者不能同时成立，就应先修复物理监督或改变更新机制，而不是继续叠加可靠性网络。

**一句话验收：E1b/E2 必须说明，当前方法究竟缺的是有效更新幅度、正确更新方向、准确物理参数，还是合理的物理模型。**

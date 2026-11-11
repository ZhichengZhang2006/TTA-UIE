# PR-TTA: Physics-Conditioned Update Utility for Underwater Image Enhancement

**研究执行规范 V1.0 | 2026-11-11 | 目标：Pattern Recognition | 状态：待实验验证的方法提案**

> 本文件供 Codex / PyTorch 工程实现使用。参数均为建议初始值，不代表已获得的实验结果。遵守 source-only 训练与目标域盲测。以真实实验结果决定是否继续。

## 0. 核心目标与不应宣称的创新

提出 **Physics-conditioned Update Utility Estimation (P-UUE)**：给定预训练 UIE 网络的一次物理引导候选参数更新，利用 *无需目标标签的候选前后信息* 预测该更新对有参考恢复质量的相对收益，并决定接受或回退。

- 已有工作：DCTTA/CVPR 2026 研究重新退化一致性与重要参数更新，CSTTA/Remote Sensing 2026 研究置信度和敏感参数选择，GALA/AAAI 2025 研究梯度对齐，PGAC-Net/EAAI 2026 研究物理置信融合，UDAPA/IET 2026 研究轻量 UIE 适配器。
- 不是创新：LoRA、单图 TTA、cosine 梯度一致性、多物理假设、物理损失、confidence gate。
- 待验证创新：**候选更新的参考相对收益**是训练监督标签；特征结合物理参数不确定性及候选响应；经 source-only 训练与独立校准后在多个未知目标水体具有增量判别能力。
- 注意：源域校准无法保证任意目标域的风险安全，物理图并不当然是真实物理量。

## 1. 数据协议（优先完成）

只使用公开数据。

- LSUI 作者页面：https://lintaopeng.github.io/code/ （页面报告 4,279 对），作为唯一主要源域。
- UIEB 官方：https://li-chongyi.github.io/proj_benchmark（890 对 + C60 无参考），作为配对独立目标与挑战测试。
- EUVP 官方：https://irvlab.cs.umn.edu/resources/euvp-dataset ，只在核对文件对齐后使用合法 paired test 子集。
- U45：版本、文件来源与 hash 固定后仅作无参考辅助。

若 LSUI 完整可用、无可疑重复组，建议近似数量：train=3000, tune=300, utility_fit=650, risk_calib=329。**真实分组约束优先于精确比例**。根据相同场景/序列、文件元信息、pHash、近重复图像进行分组，并检查跨数据集泄漏。使用 `sha256` 标记文件。若分组不足，缩减最小试验样本量而不拆组。

`manifest.csv`: `id,input_path,reference_path,dataset,split,group_id,sha256_input,sha256_ref,width,height,valid_pair`。

`train` 训练基础网络；`tune` 定候选步长、图像漂移上限、物理估计器和特征清单；`utility_fit` 训练收益预测器；`risk_calib` 校准预测器及选择门槛。目标 UIEB/EUVP **绝不用于优化模型、调参或选门槛**，其 GT 只供冻结后离线评分。

## 2. 静态模型基线

第一 Backbone：简洁四级 U-Net，特征宽度 `[32,64,128,256]`。每层 `3x3 Conv + GN(8 groups) + SiLU`，下采样 stride=2。解码器 bilinear upsample、skip concat、双卷积，最后 `3x3 Conv + Sigmoid`。在第 2、3 解码尺度 1x1 通道投影增加并行低秩残差（rank=4）。第二 Backbone（方法跨模型核查）：同源域训练的 MambaCue。

LoRA 对于一个 1x1 卷积 `W`：`z = W*h + (alpha/r) * B(A(h))`，其中 `A: C_in->r`，`B:r->C_out`，`alpha=r`。`A` Kaiming 初始化，`B` 零初始化。训练期间 backbone 与 Adapter 同时学习。保存 `theta0,phi0`，TTA 阶段 `theta0` 冻结，只修改 `phi`。

- 输入：RGB 0-1，256x256，原始文件 resize 方式和 range 必须记录。
- Train: 200 epochs, batch 8, accum 2, AdamW(lr=2e-4, weight_decay=1e-4), warmup 5 epochs, cosine -> 1e-6, AMP BF16（可用时），gradient norm 1.0, seeds 2026/2027/2028。
- 损失：`L_train = mean(abs(J-Y)) + 0.2*(1-SSIM(J,Y))`；确保 SSIM 在 `[0,1]`、窗口和边界处理固定。
- 数据增强：水平/垂直翻转 + 90° 旋转；禁止 RGB 通道顺序随机打乱。Train/Tune 与 RiskCalib 不共享图像/序列。
- 按 `tune` 的 PSNR 选静态 checkpoint，记录 epoch，禁止以目标数据集表现选 checkpoint。

## 3. 物理伪监督

### 3.1 正向模型

对用于形成损失的输入 `I_lin` 和恢复输出 `J_lin`，使用简化形式：

`F(J;t,B) = t * J_lin + (1-t) * B`, 其中 `t` 形状为 `[1,3,H,W]` 或 `[1,1,H,W]`，`B` 为 `[1,3,1,1]`。

`r_m=I_lin-F(J;t_m,B_m)`。

`rho_e(r)=sqrt(r^2+e^2)-e`，`e=1e-3`。

`L_phy_m = sum_x c_m(x) * mean_c rho_e(r_m(x)) / (sum_x c_m(x)+eps)`，`c_m` detach, `[0,1]`，强制 `mean(c_m)>=0.05`，否则该物理假设不更新。

不要对 J、I 使用不一致的色域：`I_lin=srgb_to_linear(I_srgb)`，`J_lin=srgb_to_linear(J_srgb)`；该近似不适用于全部真实相机响应，但比直接将 sRGB 当线性光强更自洽。所有参数估计器声明自己的工作色域，`B` 必须转换至损失使用的色域。`t,B,c` 均固定 `detach()`，禁止它们吸收所有重投影误差。

简化形式不同于 CVPR 2018 Akkaynak–Treibitz 修正模型：直接传输和后向散射衰减参数可不同。需在 E2 显式检验该模型失配。

### 3.2 可复现物理假设

V1: 先实现 **UDCP 类估计器**，输出 `t0,B0,c0`；若需要替代估计，增加红通道先验或不同窗口/约束的独立估计器。任何估计器需输出 `I,t,B,c` 可视化及数值范围。

- 估计程序中所有启发式细节（dark patch 15x15、omega=0.95、top-0.1% 选环境光、t clamp `[0.05,0.98]`）仅为初始实现建议，不等同于原论文的唯一实现。
- M=1 为主试验，避免先叠加昂贵梯度。
- M=3 消融：theta0；theta1 为平滑、受限的 `log(t)` 扰动和 B 扰动；theta2 为不同估计器结果（不能实现时用独立扰动，但需明确相关性）。
- 随机性通过 `seed = stable_hash(image_id, hypothesis_id, global_seed)` 固定。不能让训练测试中的随机假设对应同一测试样本变化，使收益标签不稳定。

### 3.2.1 UDCP 类估计器的实现约定

初版采用透明、可复现的启发式实现，不声称与 UDCP 原论文的官方代码逐行相同。估计 `t` 时输入 sRGB，重投影使用近似线性 RGB。`B` 经过 sRGB-to-linear 转换。

```python
# I_srgb [1,3,H,W] 0~1
def min_pool(x, k):
    return -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k//2)

@torch.no_grad()
def estimate_udcp_like(I_srgb, patch=15, omega=.95):
    n, c, h, w = I_srgb.shape
    assert n == 1 and c == 3
    gb_dark = min_pool(I_srgb[:,1:3].amin(1, keepdim=True), patch)
    K = max(1, round(.001 * h * w))
    idx = gb_dark.flatten(2).topk(K, dim=-1).indices
    selected = I_srgb.flatten(2).gather(2, idx.expand(1,3,K))
    B_srgb = selected.median(-1).values.view(1,3,1,1).clamp_min(.05)
    gb_normalized = I_srgb[:,1:3] / B_srgb[:,1:3]
    dark_norm = min_pool(gb_normalized.amin(1,keepdim=True), patch)
    t = (1 - omega * dark_norm).clamp(.05,.98)
    t = F.avg_pool2d(t, 5, stride=1, padding=2).clamp(.05,.98)
    # V1 仅过滤明显饱和区域，不把 c 当作物理置信真值
    confidence_mask = (I_srgb.amax(1,keepdim=True) < .98).float()
    return t.detach(), srgb_to_linear(B_srgb).detach(), confidence_mask.detach()
```

`physical_loss` 必须执行 `I_lin = srgb_to_linear(I_srgb)` 和 `J_lin = srgb_to_linear(J_srgb)`。`B` 已经处于线性空间。切勿让估计的 `t`、`B` 在同一次 Adapter 更新中反向传播。UDCP-like 透射率来自 sRGB 统计，只能称启发式物理线索，不等于仪器测量的真实参数。

### 3.3 物理损失不等于恢复有效性

对 L2 重投影 `L=1/2||I-tJ-(1-t)B||²`，在 t、B 固定时：

`grad_J L = -t * [I-tJ-(1-t)B]`。

因此只减小 L 会把输出推向 `J_inv=[I-(1-t)B]/t`，但 `J_inv` 的质量完全依赖 t、B 的正确性。`t` 太小会放大反演误差，简化物理损失也可能拟合错误物理模型。必须加入错误物理/随机物理对照。

## 4. Candidate Adapter Update (CAU) —— 必须与校准标记完全一致

定义 `J0=f(theta0,phi0)(I)`。对一个固定候选操作 `a`（V1 为一次 Adapter 物理梯度下降）：

`g_m = grad_phi L_phy_m(f(theta0,phi0)(I))`。

`g_bar = mean_m normalize_per_block(g_m)`，方向 `d = normalize_per_block(g_bar)`。

`phi1=phi0-eta*d`。

必要时仅使用无标签的 `RMS(J1-J0)` 设置 **输出漂移信任域**，例如 `<= 4/255` 或 `<= 8/255`（具体阈值由 `tune` 选），超限则将 eta 折半并重新推理，最多 3 次。**此信任域本身不保证恢复改善**。

`J1=f(theta0,phi1)(I)`。保存 `J0,J1,q`，然后立即将 Adapter 恢复为 `phi0`。

注意：若 V1 要用 PyTorch optimizer，可等价采用 SGD，但同一候选操作必须在 utility_fit / risk_calib / target 测试中精确一致。`model.eval()` 且禁用 BN running statistics/Dropout 波动。初始 Adapter 由 source Train 共同训练，不使用随机未训练 Adapter 作为正式候选。

```python
# 关键伪代码，张量 I:[1,3,H,W], 单张图像独立重置
# 所有 torch autograd 在 FP32 完成，第一阶段关闭 AMP

def candidate_step(model, phi0, I, hypotheses, eta, max_drift):
    restore_adapter(model, phi0)
    model.eval()
    with torch.no_grad():
        J0 = model(I).detach()
    ps = list(adapter_parameters(model))
    with torch.enable_grad():
        J = model(I)
        losses = [physical_loss(I, J, h) for h in hypotheses]
        G = []
        for m, L in enumerate(losses):
            gm = torch.autograd.grad(
                L, ps, retain_graph=(m < len(losses)-1),
                allow_unused=True)
            gm = replace_none_with_zeros(gm, ps)
            G.append(gm)
    d = aggregate_block_normalized_gradients(G)
    q_before = extract_before_features(I, J0, hypotheses, G)
    for trial in range(4):
        step = eta / (2 ** trial)
        with torch.no_grad():
            restore_adapter(model, phi0)
            for p, dp in zip(ps, d):
                p.add_(dp, alpha=-step)
            J1 = model(I).detach()
            drift = torch.mean((J1-J0)**2).sqrt()
        if drift <= max_drift: break
    # 如果所有候选仍超限，保持 J0，标记 action_status='trust_reject'
    if drift > max_drift:
        J1 = J0
        step = 0.0
    q_after = extract_candidate_response_features(I, J0, J1)
    restore_adapter(model, phi0)
    return J0, J1, concat(q_before,q_after), {'step':step, 'drift':drift}
```

## 5. 更新收益与风险标签

严格在 `utility_fit` 与 `risk_calib` 的 **公开 source reference** 上计算以下量：

`u_i = PSNR(J1_i,Y_i)-PSNR(J0_i,Y_i)`，单位 dB，正收益为好。

`h_i = 1[u_i < -delta]`，`delta=0.2 dB` 是首版建议；必须附加 `0`、`0.5 dB` 阈值敏感性分析。

候选特征 `q` 两类：

1. **pre-update**: mean/percentile t，low-t ratio，B dispersion，初始重投影残差，梯度范数，梯度 cosine（M=1 时无），初始输出 saturation/clipping/边缘统计。
2. **candidate-response**: `RMS(J1-J0)`，`mean_abs(J1-J0)`，局部结构变化、剪切像素比例变化、物理残差变化 `L_phy(J1)-L_phy(J0)`。**这些特征在测试阶段也可用，不需要目标 reference**，但需要先执行一次候选更新。其成本应计入 latency。

既比较 `q_pre`，也比较 `q_pre+q_response`，避免声称“无需候选更新即可决定”而实际上先执行了更新。

`u_i` 是**参考相对恢复收益**，不等于真实水下场景的辐射准确性。

## 6. P-UUE 预测器与校准器

使用 `utility_fit` 训练两个轻量模型：

- `h_u(q)`: Pipeline(StandardScaler, Ridge/HuberRegressor) => `u_hat`，用组内 5 折 CV 选正则化。
- `h_h(q)`: Pipeline(StandardScaler, LogisticRegression) => `p_raw=P(u<-delta|q)`，用 5 折 GroupKFold 进行超参选择。

训练模型仅使用 `utility_fit`。`risk_calib` 完全不进入训练、特征挑选或调参。对于几百样本，优先 logistic + sigmoid 概率再校准，避免不受限制的 isotonic 过拟合。

在 `utility_fit` 内通过 GroupKFold out-of-fold 预测拟合一维 Platt 概率校准，再以整个 `utility_fit` 重训基分类器。`risk_calib` 的标签**只用于接受门槛选择**，不能再次拟合概率校准器，以保证门槛评价标签与预测器拟合相互独立。预先固定 21 个风险阈值，选择满足 source calibration 单侧 Clopper-Pearson 有害率上界 <= 0.10 的最大覆盖率阈值，对 21 个阈值使用 Bonferroni alpha/21，接受样本数量至少 30。无合格阈值则返回静态结果。注意 OOF Platt 到 full-fit 分数会有分布偏移，需报告独立 calibration 校准误差，但不在其上再训练校准器。

该风险上界只适用于同分布的源域 calibration 样本及相同候选更新策略，**不能保证目标跨域安全**。

测试时 `if u_hat>0 and p_harm<tau and physics_valid and candidate_valid: output J1; else output J0`。源域和目标域调用完全相同候选生成器。候选参数全部恢复，不跨图累积。若所有回溯步长仍违反输出漂移约束，标记 `trust_reject` 并输出 J0。此类候选并未真正执行更新，不应被当作无害更新负例混入风险预测训练。

```python
def collect_utility_records(model, src_loader, eta, risk_cfg):
    rows = []
    for I, Y, meta in src_loader:
        H = construct_hypotheses(I, meta['id'], M=risk_cfg.M)
        J0, J1, q, stats = candidate_step(
            model, ADAPTER_PHI0, I, H, eta, risk_cfg.max_drift)
        u = float(psnr(J1, Y) - psnr(J0, Y))
        rows.append(dict(id=meta['id'],group=meta['group_id'],
                         q=q,utility=u,harm=int(u < -0.2),**stats))
    return rows

def train_predictors(fit_records, cal_records):
    predictor_u = fit_ridge_with_grouped_cv(fit_records)
    predictor_h = fit_logistic_with_grouped_cv(fit_records)
    # Platt 在 fit_records 的 grouped OOF 分数上拟合
    calibrated_h = fit_platt_on_grouped_oof(fit_records, predictor_h)
    # risk_calib 仅用于选阈值，不再调整概率模型
    p_cal = calibrated_h.predict_proba(cal_records.features)[:, 1]
    tau = choose_risk_threshold_with_simultaneous_bound(
        probabilities=p_cal, utility_pred=predictor_u(cal_records.features),
        harm_labels=cal_records.harm, alpha=0.05, candidate_count=21,
        max_risk=0.10, min_accepted=30)
    return predictor_u, calibrated_h, tau

@torch.no_grad()
def output_decision(J0, J1, q, pred_u, pred_h, tau):
    return J1 if (pred_u(q)>0 and pred_h(q)<tau) else J0
```

**重要数学限制**：若 `u_hat` 很差，则 `p_harm` 可能只学会拒绝所有样本，必须同时报告 coverage、净改善以及 accepted-only risk；拒绝样本按 `u=0` 计入整体净收益。

### 6.1 门槛选择的统计实现

令事先固定的候选阈值集合为 `T={0.05,...,0.95}`（21 个点）。每个阈值的 accepted mask 由**预先固定**的 `p_harm`、`u_hat`、候选有效状态产生。定义 `n_tau = accepted.sum()`、`k_tau = harmful[accepted].sum()`。在 IID 独立样本前提下，对固定 tau，单侧 Clopper–Pearson 上界为 `BetaPPF(1-alpha/K, k_tau+1, n_tau-k_tau)`（k=n 则为 1）。采用 Bonferroni alpha/K 对预设的 K 个阈值作同时控制。

```python
from scipy.stats import beta
import numpy as np

def choose_tau(p, uhat, harm, eligible, alpha=.05, risk_max=.10):
    grid = np.linspace(.05, .95, 21)  # 预先固定，不窥视 cal 标签
    feasible = []
    for tau in grid:
        accept = (p < tau) & (uhat > 0) & eligible
        n = int(accept.sum())
        if n < 30:
            continue
        k = int(harm[accept].sum())
        ub = 1.0 if k == n else beta.ppf(1-alpha/len(grid), k+1, n-k)
        if ub <= risk_max:
            feasible.append((n,tau,ub))
    return max(feasible)[1] if feasible else None
```

**有效性边界**：风险上界只针对与源域 calibration 同分布的独立样本，**无法保证任意未知目标域安全**。若数据来自相关视频或同场景连续帧，不能按帧 IID 计算 Clopper–Pearson，应按独立场景/序列进行风险评估或退回描述性置信区间。`risk_calib` 不能同时用于训练 Platt 并宣称上述置信上界有效。

## 7. E1--E6：必做实验矩阵

### E0 代码、数据及物理数学单元测试

- 真实 paired 文件校验、resize 对齐、sRGB/linear 互逆误差、物理张量广播维度、`t=1` 时 `F=J`、t/B 错误参数范围、loss 非零梯度。
- 物理平方损失的 autograd 对 J 梯度与解析梯度 `-t * r` 一致。
- `candidate_step` 后所有 Adapter 参数逐字节恢复；同图同 seed 产生完全相同 J1；另一张测试图像不得受上一张影响。
- 校准和部署的 `features_schema.json` 顺序、统计窗口、物理函数与输出空间完全一致。

### E1 梯度方向审计（gatekeeper，源域 Tune）

`g_ref = grad_J MSE(J0,Y)`；`g_phy = grad_J physical_loss(I,J0,h)`；计算 cosine 与 `Jprobe=clip(J0 - (eta/RMS(g_phy))*g_phy)` 的 ΔPSNR，eta 为 `[1,2,4]/255`。只作机制研究，参考梯度 **绝不可进入部署模型**。

保存 `e1_source.csv`：id、GT PSNR、candidate PSNR、cos、phy loss、t stats；绘制分布/相关和 harmful proportions。

### E2 已知参数物理仿真（源域 Tune 或独立 source 机制子集）

使用公开 source reference 图像，`I=tD*J+(1-tB)*B+n`，`tD=exp(-betaD*z)`，`tB=exp(-betaB*z)`。matched 为 `betaD=betaB`，mismatched 为不同系数。扰动参数 ±10/20/40%，并比较 spatial shuffled t、cross-image B、random matched range physics。

记录 true-vs-estimated-vs-shuffled 在 ΔPSNR、梯度 cosine、候选损失、风险预测上的区别。合成 source 机制数据不应当再放入独立最终测试。

### E3 候选更新效用可预测性（主研究判断）

训练 650 条或更多合法 `utility_fit` 记录、独立 risk calib 记录，比较五组特征：

- F0: image quality only (UIQM/UCIQE/简单输出统计)
- F1: generic TTA / output response (no physics params)
- F2: physical params and residual only
- F3: F1+F2
- F4: F3+cross-hypothesis gradient stats (M=3)

指标：harm AUROC/AUPRC/Brier/ECE，utility MAE/Spearman，校准曲线，risk@coverage=25/50/75%，每图 bootstrap CI。**关键**：F3/F4 必须在相同 backbone、相同候选更新、相近特征维度下优于 F1，才支持“physics-conditioned utility”主张。

### E4 选择性更新消融

Same backbone / same candidate:

- static J0
- unconditional J1
- random gate with coverage-matched
- mean-t gate
- physics residual gate
- generic-only utility predictor
- physics-conditioned utility predictor
- physics-conditioned predictor + risk-calibrated gate
- direct physics inversion and J0/physics-inversion fusion (same source tune, no target tune)

在相同 coverage 下比较 accepted harm rate 与总体净 ΔPSNR。仅靠拒绝所有样本得到的低风险不算改进。

### E5 跨域最终评测

主协议 LSUI->UIEB + EUVP；附加 UIEB-source->LSUI，必须重训 checkpoint 与预测器，不能同一个预测器混合泄漏。另选 U45/C60 做无参考辅助观察。

与 DCTTA、PGAC-Net、UDAPA 及机制性 CSTTA/GALA 对照时，说明哪些为官方可复现代码，哪些是按机制自建的 same-backbone ablation，不允许冒充官方复现。

指标：PSNR, SSIM, LPIPS, DeltaPSNR, harmonic negative transfer, worst-10% delta, per-image runtime, total backward passes, peak VRAM, adapter parameter count。至少 3 seeds，各数据集单独报告 95% grouped bootstrap CI。

### E6 风险–覆盖率与校准迁移

`coverage = mean(accept)`；`accepted_harm = sum(accept & (dPSNR < -0.2))/max(sum(accept),1)`；`net_utility = sum(accept * dPSNR)/N`。

比较 source-calibration Brier/ECE 与 target observed Brier/ECE。目标 GT 只在离线评估使用，不允许以目标 Brier/ECE 重新校准。若性能在 source->target 大幅下降，降低“可靠性保证”宣称，报告校准迁移失效与拒绝更新的 OOD 机制。

## 8. 运行命令（建议 Codex 实现的 CLI 契约，非已存在脚本）

```bash
# 0 数据 manifest（待 Codex 创建）
python tools/build_manifest.py --dataset LSUI --root data/LSUI --out manifests/lsui_all.csv --check-pairs --hash
python tools/check_leakage.py --source manifests/lsui_all.csv --target manifests/uieb_all.csv --sha256 --phash
python tools/split_source.py --manifest manifests/lsui_all.csv --train 3000 --tune 300 --fit 650 --cal 329 --group-key group_id --seed 2026

# 1 Source-only 基线
python train_source.py --config configs/v1.yaml --split train --seed 2026
python eval_static.py --config configs/v1.yaml --split tune

# 2 机制 E1/E2
python experiments/e1_gradient_audit.py --config configs/v1.yaml --split tune --max-images 200
python experiments/e2_physics_mismatch.py --config configs/v1.yaml --split tune --max-images 200

# 3 数据生成、拟合与校准
python experiments/e3_collect_utility.py --config configs/v1.yaml --split utility_fit --out results/fit.parquet
python experiments/e3_collect_utility.py --config configs/v1.yaml --split risk_calib --out results/cal.parquet
python experiments/e3_fit_utility.py --fit results/fit.parquet --cal results/cal.parquet --out checkpoints/utility_model.joblib

# 4 目标盲测：先冻结所有选择，再运行
python experiments/e5_target_test.py --config configs/v1.yaml --model checkpoints/source_seed2026.pt --utility checkpoints/utility_model.joblib --manifest manifests/uieb_target.csv --out results/uieb_final.csv
python experiments/e5_target_test.py --config configs/v1.yaml --model checkpoints/source_seed2026.pt --utility checkpoints/utility_model.joblib --manifest manifests/euvp_target.csv --out results/euvp_final.csv

# 5 统计
python analysis/report.py --results results/uieb_final.csv results/euvp_final.csv --bootstrap 10000 --grouped
```

## 9. 推荐 V1 YAML

```yaml
project: pr_tta_v1
seed: 2026
hardware: cuda
input:
  shape: [3, 256, 256]
  range: [0.0, 1.0]
  alignment: resize_square_256
  physics_linear_rgb: true
static:
  backbone: unet4
  channels: [32, 64, 128, 256]
  norm: group_norm
  adapter:
    type: low_rank_1x1
    rank: 4
    alpha: 4
    positions: [decoder_stage2, decoder_stage3]
  train:
    epochs: 200
    batch_size: 8
    grad_accum: 2
    optimizer: adamw
    lr: 0.0002
    min_lr: 0.000001
    warmup_epochs: 5
    weight_decay: 0.0001
    amp: bf16
    loss_l1: 1.0
    loss_ssim: 0.2
physics:
  first_estimator: udcp
  hypotheses: 1
  hypothesis_ablation: 3
  dark_patch: 15
  omega: 0.95
  t_clip: [0.05, 0.98]
  charbonnier_eps: 0.001
  min_confidence_mean: 0.05
candidate:
  steps: 1
  optimizer: manual_normalized_sgd
  # 从 source tune 对 [1e-4, 3e-4, 1e-3] 做离散选择，以下只作占位默认
  eta: 0.0003
  output_drift_rms_max: 0.0156862745
  max_backtracking: 3
predictor:
  continuous: ridge
  harm: logistic
  harm_delta_db: 0.2
  group_cv_folds: 5
  feature_sets: [generic, physics, combined]
calibration:
  one_sided_alpha: 0.05
  source_harm_max: 0.1
  candidate_threshold_count: 21
  min_accepted: 30
metrics:
  - psnr
  - ssim
  - lpips
  - harm_rate
  - risk_coverage
  - total_runtime
  - peak_vram
  - brier
  - auroc
```

## 10. 质量控制和决策门槛

- **Go-1**：E1 证明在有意义的子集上，物理候选实际更新有超过静态的潜力，而非仅下降自身物理损失；且误差对候选效用具有规律。
- **Go-2**：E3 的 `generic + physics` 预测器在 source held-out、UIEB 和 EUVP 上，优于 generic-only 的风险排序与相同 coverage 选择。
- **Go-3**：full gate 的总体 ΔPSNR 为正，至少一个跨域数据集得到稳定改善，在不同数据集上不出现显著加剧的 tail harms；未来论文要更高要求。
- **Stop/Pivot**：若物理特征并无增量价值，停止 physics-conditioned 主张；若 gate 无效但直接 fusion 有效，转向非 TTA 的可信物理融合；若源域校准完全跨域失效，不得宣称安全或概率保证。

## 11. 相关文献原始入口

- DCTTA, CVPR 2026: https://openaccess.thecvf.com/content/CVPR2026/html/Tang_Degradation-Consistent_Test-Time_Adaptation_for_All-in-One_Image_Restoration_CVPR_2026_paper.html
- CSTTA, Remote Sensing 2026: https://doi.org/10.3390/rs18162801
- GALA, AAAI: https://doi.org/10.1609/aaai.v39i19.34229
- PGAC-Net, EAAI 2026: https://doi.org/10.1016/j.engappai.2026.114714
- UDAPA, IET IP 2026: https://doi.org/10.1049/ipr2.70424
- AETTA, CVPR 2024: https://doi.org/10.1109/CVPR52733.2024.02706
- Monitoring Risks in TTA, NeurIPS 2025: https://proceedings.neurips.cc/paper_files/paper/2025/hash/746960ad49ddb47248970a0e1404230c-Abstract-Conference.html
- Akkaynak & Treibitz, CVPR 2018: https://openaccess.thecvf.com/content_cvpr_2018/html/Akkaynak_A_Revised_Underwater_CVPR_2018_paper.html
- Physics-Shaped, Not Physical (2026 preprint): https://arxiv.org/abs/2610.19763

**验收条件**：未提供所有目标域 GT 的模型决策脚本能够独立运行；训练/校准划分可复现；所有候选更新可重放且不污染下一张图像；`results/*.csv` 含每图的静态结果、候选结果、门控决策及预测风险；统计报告同时显示实际收益和拒绝比例。

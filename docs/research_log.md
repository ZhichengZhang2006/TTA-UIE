# PR-TTA E0–E1 运行记录

## 2026-11-11 · school-gpu2

- 项目在 `/home/user/PR-TTA`。GPU 0 为 A100，任务环境可见 20 GiB。开跑前显存占用为 0 MiB。使用现有 `/home/user/Conda/CDF_FFT`，Python 3.8.20、PyTorch 2.1.0+cu118、驱动 535.216.01。服务器 `/home/user` 可用磁盘约 3.2 TiB。
- LSUI 三个原始子目录共 4,279 对，均通过解码和尺寸校验。3,873 个去重组中有 252 个多样本组，最大组 42 对。Train、Tune、Utility Fit、Risk Calibration 分别为 3,000、300、650、329 对。数据源见 [LSUI 作者页面](https://lintaopeng.github.io/code/)。
- 跨库审计标记了 UIEB 1 对、EUVP-d 1 对、EUVP-s 1,721 对。EUVP-s 的本地 test 子目录中有 174 个目标 ID 被标记，118 个与 LSUI Source Train 匹配。按记录顺序等距抽查 12 个匹配，对齐到 64×64 的灰度图在匹配角色上的 Pearson 相关范围为 0.9160–1.0000。该核查仅比较数据文件。后续跨域评估需要先确认独立的 EUVP 测试子集。
- E0 单元测试 10 项通过。真实 LSUI 训练小批次的 Smoke Test 损失从 0.388170 降至 0.197432，梯度正常，保存并重新加载的 FP32 输出完全一致。
- 正式训练使用 2026 种子、BF16、200 个 epoch，按完整 Source Tune 的 FP32 PSNR 选 checkpoint。E1 的 200 个图像 ID 和背景供体已在正式审计前冻结，见 `results/e1_fixed_subset.csv`。
- 训练占用 GPU 时，独立 CUDA 进程报告 `bind: Address already in use`。正式训练、静态评分与 E1 通过完成标记串行执行。
- 已完成全部 200 个 epoch，第 165 个 epoch 的 Tune PSNR 最高，为 30.08400455 dB，SSIM 为 0.93045577。正式 checkpoint 保存在 `/home/user/PR-TTA/checkpoints/source_seed2026.pt`。
- 首次 CUDA 重放出现最高约 3×10⁻⁷ 的输出差异。已启用确定性算法和固定 cuBLAS workspace，用等价稳定排序实现环境光下中位数。初次结果归档，原样重做固定子集审计，首张真实图像在全部图像之后重放的输出和评分逐位一致，完整模型状态精确恢复。
- 最终单元测试 11 项全部通过，包含 CUDA 精确重放。正式 E1 包含 200 图、187 组、1,000 条更新条件记录和 600 条 Probe。四种实际更新的参数 L2 均约为 0.0006。
- 物理候选对全部 200 图降低重投影损失，PSNR 改善 97 图、下降 103 图。平均 ΔPSNR 为 −0.00061486 dB，95% 分组 bootstrap 区间约为 [−0.00187765, +0.00062227]。负收益比例为 51.5%，低于 −0.2 dB 的比例为 0%，最差 10% 均值为 −0.01706577 dB。
- 梯度余弦均值为 −0.00833559，正值占 48.5%，与实际 Adapter ΔPSNR 的 Pearson 相关为 0.83751。三个输出 Probe 的平均 ΔPSNR 分别为 −0.16859、−0.50097、−1.42421 dB。当前冻结候选的实际输出漂移很小，本轮证据不足以支持进入收益预测器开发。

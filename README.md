# PR-TTA v0.1 · E0–E1

在 LSUI 上训练带低秩 Adapter 的四级 U-Net，并检验水下重投影的物理梯度方向和实际单步更新收益。

v0.1 的实验目的是验证无标签物理重投影监督与参考恢复目标的方向关系，以及实际 Adapter 更新是否改善图像质量。已完成 seed 2026 的 200 epoch 源域训练与固定 200 图审计，11 项测试通过。物理候选平均 ΔPSNR 为 −0.000615 dB，负收益比例为 51.5%，分组区间跨零。详见 [E1 报告](reports/E1_findings.md)。

## 在 school-gpu2 运行

服务器项目目录为 `/home/user/PR-TTA`，现有公开 LSUI 文件位于 `/home/user/UnderWaterDataset/LSUI`。先查看 GPU 占用，按实际空闲卡设置 `CUDA_VISIBLE_DEVICES`。本次检查可见的 GPU 0 是 A100，CUDA 实际可见显存为 20 GiB。复用独立环境 `/home/user/Conda/CDF_FFT`。在其他环境按对应驱动安装支持 BF16 的 PyTorch，再安装 `requirements.txt` 中列出的依赖。

```bash
cd /home/user/PR-TTA
nvidia-smi
export PATH="/home/user/Conda/CDF_FFT/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1

python tools/build_manifest.py --help
python tools/build_manifest.py --config configs/v1.yaml
pytest tests/ -q
python experiments/e1_gradient_audit.py --config configs/v1.yaml --split tune --max-images 200 --freeze-subset
python train_source.py --config configs/v1.yaml --smoke
python train_source.py --config configs/v1.yaml
python eval_static.py --config configs/v1.yaml --split tune
python experiments/e1_gradient_audit.py --config configs/v1.yaml --split tune --max-images 200
```

训练中断后，使用同一配置、代码和清单执行 `python train_source.py --config configs/v1.yaml --resume`。它从最后一个完整 epoch 恢复。正式 E1 会核验 200 个 epoch 已全部完成，以及 checkpoint 确为其中 Tune PSNR 最高的一次。

## 数据审计

源域清单和组划分统计位于 `manifests/manifest.csv` 和 `manifests/manifest.json`。现有目标配对数据可以用文件哈希和图像指纹检查重叠。

```bash
python tools/check_leakage.py --target-root \
  /home/user/UnderWaterDataset/UIEB \
  /home/user/UnderWaterDataset/EUVP-d \
  /home/user/UnderWaterDataset/EUVP-s
```

## 输出

- `checkpoints/source_seed2026.pt` 是 Tune 选择的静态模型，`source_seed2026_latest.pt` 用于恢复。Smoke checkpoint 独立保存。
- `results/smoke.json` 和 `logs/e0_tests.log` 保存真实小批量训练验证和合成张量单元测试结果。
- `results/source_seed2026_epochs.csv` 保存逐 epoch 训练损失、Tune PSNR/SSIM 和训练时间。
- `results/e1_gradient_audit.csv` 按每图五个条件记录候选收益。`results/e1_output_probes.csv` 记录三种输出 Probe。
- `results/e1_fixed_subset.csv` 固定 E1 的 200 张图及背景供体。`results/e1_run.json` 记录复现信息。
- `results/e1_summary.json`、四幅诊断图和 `reports/E1_findings.md` 给出正式结果。

研究依据和实现选择见 [研究规范](docs/PR_TTA_Research_Protocol_v1.md) 及 [实现约定](docs/implementation_decisions.md)。

## E1b–E2 假设验证

[v0.1 分支](https://github.com/ZhichengZhang2006/TTA-UIE/tree/v0.1) 固定 E0–E1 基线。后续实验继承同一 checkpoint，从 Utility Fit 中按完整去重组划出 40 张 Pilot 和 200 张 Fixed，240 张登记为 diagnostic_consumed，剩余 410 张用于后续独立规划。

本阶段检验 Adapter 的参考 Oracle 能力、相同输出预算下的物理/随机/错误参数/融合方向，以及已知参数误差和双衰减模型失配。设置、生成分布和统计解释见 [E1b–E2 实现约定](docs/E1b_E2_implementation.md)，研究要求见 [执行计划](docs/PR_TTA_E1b_E2_Research_Plan.md)。

```bash
python tools/prepare_diagnostics.py --config configs/e1b_output_matched.yaml
python experiments/e1b_output_matched.py --config configs/e1b_output_matched.yaml --manifest manifests/lsui_e1b_pilot.csv --phase pilot
python experiments/e1b_output_matched.py --config configs/e1b_output_matched.yaml --manifest manifests/lsui_e1b_fixed.csv --phase fixed
python experiments/e2_parameter_mismatch.py --config configs/e1b_output_matched.yaml --manifest manifests/lsui_e1b_fixed.csv --phase fixed
python tools/audit_cross_dataset_duplicates.py --input manifests/cross_dataset_matches.csv
python tools/diagnostic_figures.py
python tools/write_e1b_e2_report.py
```

E1b 和 E2 在此服务器串行运行。各阶段输出保存于 `results/e1b/{pilot,fixed}` 和 `results/e2/fixed`，包括逐图 CSV、全部预算曲线、组 bootstrap 区间和失败案例。`preregistration.json` 固定配置、checkpoint 和清单摘要。最终报告为 `reports/E1b_E2_findings.md`。

## E2b 参数估计与候选方向验证

E2b 检验 t/B 误差归因、合成监督估计是否改善真实更新、有效成像参数的条件分支，以及物理梯度相对解析融合和翻转旋转一致性的独立优势。保留 v0.1 基线，固定增强模型；新估计器分别训练三个种子。实现设置见 [E2b 执行设置](docs/E2b_implementation.md)。

```bash
python tools/prepare_e2b.py
python experiments/e2b_factorial_error.py
python experiments/e2b_estimator_training.py
python experiments/e2b_pipeline.py
python tools/prepare_conservative_targets.py
python tools/audit_e2b.py
python tools/report_e2b.py
```

配置固定于 `configs/e2b_operator_study.yaml`。流水线先运行测试与开发诊断，再按预设条件决定是否训练有效参数预测器及残差对照，最后只开启一次 120 张验收集。剩余 290 张 Utility Fit 留待后续，329 张 Risk Calibration 封存。失败候选保留为静态输出，统计按去重组 bootstrap，并报告三个估计器种子的差异。报告保存为 `reports/E2b_findings.md`；本阶段不启动 E3，不计算最终目标域恢复成绩。

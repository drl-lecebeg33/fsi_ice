# M6 当前源码 10 秒正式重跑

日期：2026-10-13。用户选定实验终点为 10 s。本工作包使用已通过707项回归/2项跳过的分区停止条件修复源码，从静止状态重新运行加权网格 IQN FSI2；不使用旧第922步断点续算。

执行状态：已从零接受1000步/10 s，正式连续越过第923步，全程物理门槛通过；周期因振幅仍增长保持INCOMPLETE。已在选定终点停止。证据见[实际验证](../validation/m6_10s_run.md)。

## 冻结条件

- 空间网格 h=0.02 m，时间步 dt=0.01 s，BDF2（第一步BDF1），目标1000个接受步/10 s。
- 物理、加权网格和IQN参数沿用 `cases/fsi2_weighted.json`，仅将 `end_time` 从20改为10；完整配置冻结在 `results/m6/physical_gate/iqn_10s/config.json`。
- 包源码SHA256为 `7590751f6107bcc4d6d05bbcb3f7ea8cb70f340cd476df7563c1e1e85969c3a6`。实验期间不修改求解源码。
- 全残差≤1e-7、连续性≤1e-10、面积比≥0.2、网格质量≥0.1；保留原周期与参考误差判据。
- 使用项目 `.venv`，BLAS/OMP各1线程；独立输出 `results/m6/physical_gate/iqn_10s/trajectory/`，每个接受步保存完整断点。原922步断点、失败记录和监测账本的哈希在新实验中留存。

## 实际命令与判断

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u -m fsi_ice fsi2 \
  --config results/m6/physical_gate/iqn_10s/config.json \
  --output results/m6/physical_gate/iqn_10s/trajectory
```

检查是否连续越过第923步并达到1000步，核对每步物理门槛、断点身份、监测顺序和终态。若数值失败，保留最后接受断点并记录实际失败原因，不放宽门槛或隐式切换算法。完成10 s只代表本次目标时长达到；稳定周期及参考精度仍由原判据决定，不能把达到终点视为FSI2基准通过。完成后只读分析并生成真实时程与周期图，更新实际验证记录和M6状态。

# M6 当前源码 10 秒正式重跑验证

日期：2026-10-13。用户选定终点10 s；按[实验计划](../development/m6_10s_run_plan.md)从静止状态重新运行，没有使用旧922步断点。求解源码与模型未在运行期间修改。单一执行者运行与自查，本轮提交上传已获用户授权，数值结论仍按下述实际证据判定。

## 配置与运行

加权网格 IQN FSI2，h=0.02 m、dt=0.01 s、BDF2（第一步BDF1），目标1000步。沿用 `cases/fsi2_weighted.json`，仅覆盖终点为10 s；其他材料、网格权重、求解控制和物理/周期/参考门槛保持原值。项目 `.venv`、Python 3.12.3，BLAS/OMP各1线程。包源码SHA256为 `7590751f6107bcc4d6d05bbcb3f7ea8cb70f340cd476df7563c1e1e85969c3a6`，与前一轮707 passed/2 skipped的源码相同；本轮没有代码功能改动，未重复运行数值单元测试。

实际命令（冻结配置已存入独立实验目录）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u -m fsi_ice fsi2 \
  --config results/m6/physical_gate/iqn_10s/config.json \
  --output results/m6/physical_gate/iqn_10s/trajectory
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/fsi2_analyze.py \
  results/m6/physical_gate/iqn_10s/trajectory \
  --output results/m6/physical_gate/iqn_10s/analysis
```

也可用仓库配置加 `--end-time 10` 再生相同实验配置。求解和后处理退出码均为3，表示周期INCOMPLETE；本次配置目标时长已完成，不是分段提前结束或求解异常。

## 正式连续轨迹与物理检查

已接受 **1000步/10 s**，`target_time_complete=true`；不存在 `failure.json`。第923步是本次从零轨迹的正式接受步，以13次耦合评估达到全残差 `1.97020391112801e-8`，连续性 `4.364e-15`、面积比 `0.415956`、网格质量 `0.510987`，0次IQN重启。第924步及其后一直连续接受至第1000步，证明前一轮停止条件修复在这段正式轨迹中有效。

| 全程量 | 实测极值 | 原门槛 | 结果 |
| --- | ---: | ---: | --- |
| 归一化全残差 | 最大7.4833e-8 | ≤1e-7 | 通过 |
| 弱连续性 | 最大3.2359e-12 | ≤1e-10 | 通过 |
| 网格面积比 | 最小0.365021 | ≥0.2 | 通过 |
| 网格质量 | 最小0.482438 | ≥0.1 | 通过 |
| 固体变形Jacobian | 最小0.954284 | >0 | 通过 |

断点经正常 `load_checkpoint` 核对当前源码和实际网格身份，1000步监测顺序与时间完整；`solution.npz` 的全部六组末态数组与断点逐数组相同。旧922步的断点、失败记录和监测账本SHA256与运行前清单全部相同。机器可读检查见[verification.json](artifacts/m6_10s/verification.json)。

接受步累计求解计时4481.8 s（约74.7分钟），17950次耦合评估、430次IQN重启。该计时只包括接受步求解，不是完整墙钟或公平重复计时，不能用于算法排名。

## 周期结论

报告为 **INCOMPLETE：Periodic response has not stabilized**。末段三个完整梁尖周期的周期波动1.113%已低于2%，但振幅变化 **30.863%**，明显超过2%；升力振幅变化26.410%，阻力振幅变化9.633%，也超过原稳定门槛。实际曲线显示振幅仍增长，因此10 s足以检验本次连续运行修复，尚不足以证明稳定周期精度。

暂定的梁尖振幅0.06287 m、频率2.219 Hz及同一梁尖周期窗口内阻力均值177.36 N/m均标为provisional，不作为正式参考精度结论。M6稳定周期、网格/时间独立性与四算法公平成本仍未完成；本次到达选定的10 s终点后已停止计算。

产物：[求解报告](artifacts/m6_10s/report.json)、[分析摘要](artifacts/m6_10s/summary.md)、[真实时程图](artifacts/m6_10s/history.png)、[末段周期窗口图](artifacts/m6_10s/cycles.png)。两张图已实际打开检查，标注了INCOMPLETE/provisional，坐标单位与曲线一致；图表来自本次保存的1000步监测数据。

文档检查：`git diff --check`、修改的Markdown本地链接、原理页HTML标签闭合、唯一id、193个链接目标及同页导航锚点检查通过。响应式布局规则保持原有设置；当前环境没有可用浏览器，未重新做宽窄屏截图检查。

远端随文档保存必要实验摘要与两张实测图，见[文件校验清单](artifacts/m6_10s/manifest.json)。归档报告仅移除了本机解释器绝对路径和临时工作树文件列表，数值与配置保持原样；原报告SHA256留存于归档说明。归档周期CSV仅统一为LF换行，数值未变，原文件哈希记录在清单中。完整断点、监测账本和运行日志保留在本地 `results/`。

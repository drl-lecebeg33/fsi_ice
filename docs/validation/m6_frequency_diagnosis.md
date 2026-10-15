# M6 主频偏高诊断：已完成的检查与运行中的对照

日期：2026-10-13。用户授权“你去查一下吧”。这是单一执行者的诊断，不声称独立人员复核。现有12秒基线周期稳定，但2.18418 Hz相对1.93 Hz偏高13.170%。

## 已完成的检查

1. **参考与参数核对**：重新读取[官方FSI2结果](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_tests/fsi_fsi_tests.html)和[定义](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_definitions.html)。梁尖y频率为1.93 Hz；阻力频率3.86 Hz属于另一个量，未混用。固体密度10000 kg/m³、泊松比0.4、剪切模量500000 Pa，对应E=1400000 Pa；流体密度1000 kg/m³、运动黏度0.001 m²/s，对应动力黏度1 Pa·s，入口平均速度1 m/s。配置及源码与这些参数一致；平面应变、StVK、入口启动和梁尖位置沿用已核验约定。这不等于已经排除所有离散或边界误差。
2. **独立频率提取**：在原12秒基线10.51–12.00 s的监测数据上，用包含常数及前三谐波的最小二乘拟合、穿过均值的上升时刻线性插值，复核生产用的细化峰周期统计。三个结果最大相对差约0.0083%，不能解释13.170%的偏差。只读分析，不改物理轨迹。

| 方法 | 主频（Hz） |
| --- | ---: |
| 生产用细化峰间隔 | 2.18418178 |
| 三谐波最小二乘 | 2.18428636 |
| 上升均值穿越间隔 | 2.18436304 |

拟合RMS残差0.00010089 m，约为梁尖振幅的0.137%；这些方法共用同一段物理数据，构成统计交叉检查，不是独立物理参考。

3. **单独梁的空间敏感性**：使用项目实际Gmsh双域网格中的固体域、相同平面应变材料与圆柱连接处固支边界，求解未变形、小应变、无流体的广义特征值Kφ=ω²Mφ，再对自由端均布总竖向载荷1 N求线性静力解。频率f=ω/(2π)，单位Hz；挠度/载荷为柔度，单位m/N。1 N为诊断载荷，不是标准FSI2载荷，也不改FSI配置。

| h（m） | 固体单元 | 第1固有频率（Hz） | 第2固有频率（Hz） | 梁尖柔度（m/N） | 总质量（kg，1 m厚） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.02 | 148 | 0.446616 | 2.726445 | 0.00743566 | 70.101021 |
| 0.01 | 734 | 0.355830 | 2.188925 | 0.01175659 | 70.070851 |
| 0.005 | 2730 | 0.342899 | 2.110392 | 0.01264357 | 70.068433 |

特征对相对残差最大4.625e-8；静力自由自由度相对残差最大8.571e-10，三档总外载荷均为1 N（浮点误差范围）。相对最细诊断网格，粗网格柔度低约41.19%，等效刚度高约70.04%，质量差约0.0465%；h=0.02→0.01时第1/第2固有频率分别下降约20.3%/19.7%。因此已有直接证据表明粗网格结构响应显著偏硬。

**当前判断**：空间离散误差是值得优先检查的原因；P1三角形对细梁弯曲的分辨能力可能相关。上述无流体线性梁不包含附加质量、涡脱落、大变形和流固反馈，不能用其固有频率替代FSI主频，也不能据此认定某一种锁定效应、某个模态或空间误差已解释全部13.170%。时间误差仍待完整对照。

## 原两个完整对照的结束状态

| 算例 | h（m） | dt（s） | 终点 | 启动方式 | 当前结论 |
| --- | ---: | ---: | ---: | --- | --- |
| 时间步减半 | 0.02 | 0.005 | 12 s / 2400步 | 从零 | FAIL，未到终点 |
| 网格尺寸减半 | 0.01 | 0.01 | 12 s / 1200步 | 从零 | FAIL，未到终点 |

输入、求解源码及门槛冻结，两个配置各只改基线的一个字段。源码SHA256为6792c3e517e1e8ba5af74ab20a6423943a2b7d39e0604595f844c537f2f46023。两次实际启动命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u -m fsi_ice fsi2 \
  --config results/m6/frequency_accuracy/time_half/config.json \
  --output results/m6/frequency_accuracy/time_half/trajectory
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u -m fsi_ice fsi2 \
  --config results/m6/frequency_accuracy/space_half/config.json \
  --output results/m6/frequency_accuracy/space_half/trajectory
```

实际执行采用两个独立进程并行，各1个BLAS/OMP线程；机器8核，启动时可用内存约6.7 GiB，两进程初始RSS合计约0.6 GiB。相对原串行计划，此处调整只用于缩短等待，不增加线程、不修改配置；并发耗时不得用于算法性能排名。细网格计算预计需数小时，最终时长以实际日志为准。

原进程与本地`results/m6/frequency_accuracy/watch_experiments.py`均已结束，已调用既有只读后处理，分别生成`analysis/summary.json`、真实曲线和周期表，以及根目录`comparison.json`/`comparison.md`。已有12秒FAIL断点的两份副本用于本地实测结束分支：两份均正确返回稳定周期但参考FAIL，1200步及主频逐值吻合；没有误记PASS。若未到终点、求解失败、未稳定或后处理错误，报告保留真实状态，不放宽门槛。原结果分别停在2.57/2.07秒，未形成周期数据；后续故障修复和新重跑见[本轮记录](m6_refinement_repair.md)。

## 证据、复现与边界

- [本地检查与运行前缀快照](artifacts/m6_frequency_diagnosis/verification.json)：时间细化94步/0.47 s、空间细化27步/0.27 s快照的接受前缀均通过物理门槛；这些是当时运行中快照，原实验随后失败，不是12秒终点结果。
- 已完成诊断的[文件清单与SHA256](artifacts/m6_frequency_diagnosis/manifest.json)、[频率检查](artifacts/m6_frequency_diagnosis/frequency_extraction.json)、[模态数据](artifacts/m6_frequency_diagnosis/solid_modes.json)、[柔度数据](artifacts/m6_frequency_diagnosis/solid_compliance.json)。
- 复现脚本：[频率](artifacts/m6_frequency_diagnosis/check_frequency_extraction.py)、[模态](artifacts/m6_frequency_diagnosis/check_solid_modes.py)、[柔度](artifacts/m6_frequency_diagnosis/check_solid_compliance.py)。从仓库根目录用`.venv/bin/python`执行；模态脚本先生成本地诊断网格，柔度脚本读取这些网格。脚本均已实际执行，生产求解包未改。
- 本轮没有修改物理求解代码；此前当前源码全量712 passed/2 skipped证据仍见[12秒验证](m6_12s_run.md)，本轮不冒称重新执行全量测试。
- 两档细化只用于筛查误差来源，不是三网格/三步长收敛验收；无最终FSI细化结果前，M6仍未完成。原10秒和12秒物理结果不覆盖。

文档检查：HTML结构、导航、本地链接和响应式CSS静态检查通过；环境无可用浏览器，本轮未做实际宽窄窗口截图检查。没有新增求解器功能，没有重跑全量数值回归。

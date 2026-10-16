# M6 12秒续算与周期积分窗口修复验证

日期：2026-10-13。单一执行者运行与自查。按[续算计划](../development/m6_12s_continue_plan.md)从正式1000步/10 s断点新增200步，达到1200步/12 s；随后发现并按[统计修复计划](../development/m6_cycle_window_plan.md)修复周期积分窗口。原报告与全部物理数据保留；本轮尚未提交或上传。

## 正式续算与数据完整性

求解基于提交 `63ea9af`，包源码SHA256 `7590751f…`；h=0.02 m、dt=0.01 s、BDF2、加权网格IQN及全部物理门槛保持原值，只把end_time从10延长为12。使用项目`.venv/bin/python`（Python 3.12.3），BLAS/OMP各1线程。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u -m fsi_ice fsi2 \
  --config results/m6/physical_gate/iqn_12s/config.json \
  --output results/m6/physical_gate/iqn_12s/trajectory \
  --resume results/m6/physical_gate/iqn_10s/trajectory/checkpoint.npz
```

正常`--resume`核对源码、网格、配置和完整BDF历史后恢复第1000步，新增200步全部接受，`resumed_steps=1000`、`accepted_steps=1200`、`target_time_complete=true`，没有求解失败文件。原1000步监测前缀逐字段相同，1200步账本顺序/时间完整，末态六组数组与正常校验的断点逐数组相同。原10秒报告、断点、监测和图表共8项SHA256未变。核对结果见[verification.json](artifacts/m6_12s/verification.json)。

| 全1200步量 | 实测极值 | 原门槛 | 结果 |
| --- | ---: | ---: | --- |
| 归一化全残差 | 最大9.3693e-8 | ≤1e-7 | 通过 |
| 弱连续性 | 最大3.2359e-12 | ≤1e-10 | 通过 |
| 网格面积比 | 最小0.363905 | ≥0.2 | 通过 |
| 网格质量 | 最小0.477227 | ≥0.1 | 通过 |
| 固体变形Jacobian | 最小0.953815 | >0 | 通过 |

新增200步累计求解879.26 s（约14.65分钟）、2783次耦合评估、6次IQN重启；非公平重复计时，不用于算法排名。

## 周期均值误判与数学修复

原报告为INCOMPLETE：三个信号的均值变化约2.2%–2.3%。故障原因是频率使用二次插值后的峰时刻，但均值积分仍采用采样峰下标，周期窗口交替为45/46个步长（阻力为22/23步），人为多算或少算了峰值附近的一小段。

独立反例：完全稳定的解析信号 `0.0735*sin(2*pi*2.184*t)`，dt=0.01 s，在10秒或12秒数据上被旧算法算出约2.209%的均值变化；其真实均值恒为0。新增四个不同频率/相位/偏置的解析测试在旧算法下全部失败，实际结果4 failed/1 passed；真实均值漂移测试通过，说明原算法的漂移问题不能归因于物理波形。

修复后，相邻细化峰时刻作为同一周期的边界；在边界插入线性插值信号，再在此窗口内计算梯形时间积分和半峰峰振幅。周期长度、均值及振幅描述同一段时间，原2%稳定门槛和归一化尺度保持不变。修复只改`validation/fsi2.py::cycle_statistics`，未改变流体、固体、ALE、界面、BDF或耦合残差。

## 修正后实测结论

只读分析命令将结果写入新目录：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/fsi2_analyze.py \
  results/m6/physical_gate/iqn_12s/trajectory \
  --output results/m6/physical_gate/iqn_12s/analysis_corrected
```

| 12秒信号 | 周期变化 | 振幅变化 | 均值变化 | 稳定门槛 |
| --- | ---: | ---: | ---: | --- |
| 梁尖y | 0.038% | 0.194% | 0.046% | 三项均≤2%，通过 |
| 阻力 | 0.127% | 0.965% | 0.019% | 三项均≤2%，通过 |
| 升力 | 0.096% | 0.657% | 0.078% | 三项均≤2%，通过 |

梁尖稳定窗口为10.55761–11.93112 s。稳定周期成立后，按原参考门槛得到：

| 量 | 实测 | 参考 | 相对误差 | 门槛与结论 |
| --- | ---: | ---: | ---: | --- |
| 梁尖y振幅（m） | 0.0735307 | 0.0816 | 9.889% | ≤10%，通过 |
| 主频（Hz） | 2.18418 | 1.93 | **13.170%** | ≤5%，**失败** |
| 阻力均值（N/m） | 195.186 | 215.06 | 9.241% | ≤10%，通过 |

修正后后处理退出码1、总体**FAIL**，原因是稳定周期参考主频精度不合格，不是运行异常。原求解器退出码3和INCOMPLETE报告原样保留为历史证据；不能把统计修复写成基准通过。用新统计重新只读处理旧10秒数据，振幅变化30.879%，仍为INCOMPLETE。[前后对照](artifacts/m6_12s/comparison.json)表明真正在增长的信号仍被拒绝。

## 实际回归与断点身份

- 新解析测试修复后5 passed；FSI2/后处理/断点/瞬态/CLI相关回归126 passed / 19.05 s。
- 当前分析源码全量 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests -q`：**712 passed、2 skipped / 157.56 s**；两个既有跳过仍为T23历史产物与真实FSI1整体冒烟。
- 求解源码SHA256为 `7590751f…`，修复后分析包SHA256为 `6792c3e5…`。后处理记录分别保存两个身份，不执行续算。原求解源码53个Python文件已冻结到本地 `results/m6/physical_gate/iqn_12s/frozen_solver_src/`，实际用该包导入核对摘要仍为7590751f…。如需续算，必须使用匹配的冻结版本或重新计算；不得修改旧断点标识，用新包绕过正常身份守卫。

产物：[原求解报告](artifacts/m6_12s/report_original.json)、[原统计摘要](artifacts/m6_12s/summary_original.md)、[修正摘要](artifacts/m6_12s/summary_corrected.md)、[机器可读修正结果](artifacts/m6_12s/summary_corrected.json)、[真实时程图](artifacts/m6_12s/history.png)、[稳定周期窗口图](artifacts/m6_12s/cycles.png)、[校验清单](artifacts/m6_12s/manifest.json)。两张修正图已实际打开检查，标清稳定窗口和总体FAIL；图表来自保存的1200步真实数据。归档JSON仅移除本机绝对路径/工作树文件列表，CSV仅统一LF换行，数值不变。

本次计算已在12 s停止。当前IQN路径具有稳定周期证据，但主频精度不合格；两路线基准、独立空间/时间收敛与四算法公平成本仍待验证，M6保持未完成。下一步先分别检查网格和时间离散对主频的影响，不能直接用当前配置形成算法成本排名。

下一阶段已[准备两个单变量配置](../development/m6_frequency_accuracy_plan.md)：固定h=0.02把dt减半，以及固定dt=0.01把h减半；均使用12秒共同观察终点、原物理门槛和修正统计方法，配置校验通过，尚未启动额外求解。两档诊断不冒充完整三网格/三步长收敛研究。

最终检查：`git diff --check`、Markdown本地链接、原理页HTML结构/唯一id/202个链接目标和同页导航、归档文件哈希通过；原理页的响应式CSS经静态检查，当前没有可用浏览器，未做宽窄屏截图。新包按当前摘要正式校验旧断点时如预期拒绝，冻结求解包则正常加载，身份守卫没有绕过。对比源码证明其他52个Python文件、以及`fsi2.py`中周期统计以外的全部内容未变；原10秒输入和12秒求解报告SHA256再次核对未变。

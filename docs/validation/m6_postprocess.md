# M6 后处理与 T26 研究准备验证

日期：2026-10-12。单一执行者实现与自查，不声称独立复核；未提交、未推送。[执行前计划](../development/m6_postprocess_plan.md)。求解源码基线为 `1614dc9` 加未提交的IQN、完整断点与单场绝对残差修复；本任务没有修改 `src/fsi_ice` 或正在运行的配置。

## 本轮结论

已交付两个独立脚本：只读FSI2断点后处理，以及只生成配置/命令的T26研究准备。相关 **98项测试通过 / 27.90 s / exit0**，其中新增后处理与研究隔离测试26项。长时IQN算例继续运行；完整周期、独立收敛和算法成本排名仍未完成，M6保持进行中。

## 后处理的实际行为

[`scripts/fsi2_analyze.py`](../../scripts/fsi2_analyze.py) 一次性读取原子发布的NPZ字节快照，使用现有断点加载器校验完整历史、数组和账本，然后复用 `analyze_fsi2` 的周期算法。记录快照、求解源码、当前分析包/脚本及网格摘要；读取旧源码产生的数据只用于分析，不构成跨版本续算许可。不读取正在改写的monitor.csv，不加载live求解根，不修改实验输入。输出目录与求解目录及其父子目录分开。

生成 `summary.json`、`summary.md`、`cycles.csv`、`history.png`；出现三个可提取梁尖周期时另生成 `cycles.png`。图来自真实已接受监测行，标题写截止时间和总体状态。CSV分别列tip_y/drag/lift各自最后三个完整周期的窗口、频率、半峰峰振幅、时间加权均值及三种变化量；阻力参考均值采用梁尖周期窗口。

目标时间未完成为INCOMPLETE；未稳定时给出的参考误差只作暂定估计，不能作为PASS证据。保存了求解失败记录或已接受账本不满足物理门控时为FAIL；失败前的已接受数据仍保留。报告工作量只统计已接受步的求解/迭代计数与求解耗时，明确排除setup、写出及失败尝试，不用并发计时形成算法排名。退出码0/1/2/3分别为PASS/FAIL/输入输出错误/INCOMPLETE，与求解CLI的科学状态一致。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/fsi2_analyze.py \
  results/m6/long_run/iqn_h002_r2 --output results/m6/postprocess/iqn_prefix
```

实际读取560步、5.60/20 s的快照，物理门控通过，但没有足够周期，返回INCOMPLETE/exit3；[报告](../../results/m6/postprocess/iqn_prefix/summary.md)、[JSON](../../results/m6/postprocess/iqn_prefix/summary.json)、[实测时程图](../../results/m6/postprocess/iqn_prefix/history.png)已生成并实看，图中文字、单位、门槛和采样截止时间可读。这里的5.60 s仅为这一张报告的采样截止时间，不是当前实时进度。

对原运行在702步被拒绝后保留的701步断点同样执行后处理，返回FAIL/exit1，周期仍INCOMPLETE，未冒称失败被解决；[失败前缀报告](../../results/m6/postprocess/failed_prefix/summary.md)。当前修复后的运行在独立 `iqn_h002_r2` 目录，不覆盖原失败证据。

## 三网格、三步长与算法矩阵

[`scripts/m6_prepare_study.py`](../../scripts/m6_prepare_study.py) 生成17份候选配置：空间/时间两组三档共享最细交点，共5个收敛算例；四算法×3次独立重复，共12个成本算例。空间组h=0.02/0.01/0.005、固定dt=0.0025；时间组dt=0.01/0.005/0.0025、固定h=0.005；所有算例BDF2、20 s、相同物理/停止/验收配置。没有使用不同配置的旧断点，组内只变一个离散变量。当前h=0.02、dt=0.01的长算例是额外基线，不能代替上述两组的交点。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/m6_prepare_study.py \
  --output results/m6/study_prepared
bash -n results/m6/study_prepared/convergence_commands.sh
```

实际返回PREPARED_NOT_RUN/exit0；[manifest.json](../../results/m6/study_prepared/manifest.json)保存argv、线程、输出与依赖；没有启动任何额外FSI2求解。现有非空目录被拒绝，避免覆盖证据。生成的收敛命令脚本通过Bash语法检查，科学FAIL/INCOMPLETE保留证据并继续，运行错误则停止。算法配置是待收敛/共同精度验证后确认的候选，没有生成会自动运行算法实验的批处理。

研究门槛已在计划中先行冻结：各组有三个完整稳定结果；最细两档三指标差值均≤各参考尺度的2%，细化差值递减，最细档同时通过既有基准门槛。20 s不足以稳定时，组内一致延长时长。公平成本另需同硬件/环境/线程、无其他重负载、冷启动重复和完整墙钟证据，不能用本轮并发测试期间的计时替代。

## 网格准备实测

在独立 `results/m6/study_preflight/` 目录调用项目 `fsi1_mesh`、`P2Space.from_mesh`、`build_matching_interface`、`build_block_layout`，完成三档真实Gmsh网格、匹配界面与相同物理块布局检查；没有组装/求解瞬态方程。这些是几何准备证据，不能称为FSI2精度或收敛验证。

| h (m) | 流体单元 | 固体单元 | 流体P2节点 | 总耦合未知量 | 最小流体网格质量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.02 | 2091 | 148 | 4369 | 12905 | 0.728468 |
| 0.01 | 8078 | 734 | 16526 | 48086 | 0.761762 |
| 0.005 | 31341 | 2730 | 63417 | 182174 | 0.754804 |

[网格摘要与文件SHA256](../../results/m6/study_preflight/mesh_summary.json)。最细档未知量约当前14.1倍，最细时间步需8000步；单步成本尚未测量，不能按未知量线性比例声称整体法或完整矩阵会在某时限内完成。首次统计误用了不存在的 `BlockLayout.size`，改为真实各块尺寸之和后重新运行三档并核验随加密递增。

## 本地正确性验证

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_m6_postprocess.py tests/test_fsi2.py tests/test_fsi2_checkpoint.py -q
```

[测试源码](../../tests/test_m6_postprocess.py)，日志 `results/m6/long_run/logs/postprocess_related.log`。已知正弦信号验证2 Hz、0.0816 m、时间加权阻力均值215.06 N/m及参考误差；这些仅是分析器测试信号，不是FSI2计算结果。覆盖稳定但未完成、增长未稳定、频率/振幅FAIL、已有求解失败、五类物理门控异常、非法/缺失账本、损坏断点、输出目录保护、真实小型FE断点字节不变、两类PNG输出、三档单变量变化、交点复用、12次算法配置同条件及准备时禁止启动进程。第一轮1项失败来自小型测试驱动没有FSI2监测回调，补入真实位移/载荷监测后复测全部通过，没有改动求解器或门槛。

此前求解器修复全量651 passed/2 skipped证据保持在[长时记录](m6_long_run.md)；本轮脚本完成98项相关回归，没有重新声称全量677项已通过。分析时完整源码摘要仍与活动断点一致，续算进度持续增长。

## 文档检查

原理页、README和进度文档同步。HTML标签闭合、14个唯一id、导航锚点与165条HTML本地链接检查通过；主要Markdown本地链接也已检查。使用Orca内置浏览器在1280×900和390×844视口实看新增章节，两种宽度clientWidth与scrollWidth分别均为1265、375，无整页横向溢出；图像文字与证据链接可读。检查日志及截图在 `results/m6/postprocess/logs/`；依赖检查无冲突，`git diff --check`通过。本轮不发布远端，不合并远端注释更新，原理页保持离线静态文档。

## 本次运行退出后的自动分析

本地运行产物 `results/m6/postprocess/watch_finish.py` 使用Linux进程描述符等待本次求解PID退出（不会因PID重复使用而绑定新进程），然后执行已验证的分析CLI，输出到独立 `results/m6/postprocess/iqn_after_exit/`。状态在 `watch_status.json`，日志在 `logs/watch_finish.log`；当前已经验证启动并处于WAITING_FOR_SOLVER_EXIT，尚未触发最终分析。等待本身不进行额外数值求解，分析仍沿用0/1/2/3状态，不把进程退出等同于20 s或周期通过。该等待进程是本次执行的本地产物，不是包内求解功能、持久服务或重启后仍保证生效的任务；若进程被外部结束，可重新手动运行本文的分析命令。

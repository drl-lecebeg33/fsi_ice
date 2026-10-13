# M6/T24：断点续算与长时 IQN 实验

日期：2026-10-12。单一执行者实现与自查，不声称独立复核。代码为 `1614dc9` 加上一轮 IQN 修正及本轮未提交改动；不提交、不推送。[执行前计划](../development/m6_long_run_plan.md)。

## 当前结论

用户将本轮定位为试水，并明确停止后续推进。本轮探索到此结束，保留已有实现、测试、接受断点、监测量与失败诊断，不再恢复20秒求解或运行收敛/算法矩阵。停止探索不改变已有科学状态：长时运行FAIL、周期INCOMPLETE，M6未通过。后续只按用户请求用现有数据制作展示，不扩展数值实验。

完整 AcceptedState 断点保存/恢复、显式CLI续算与分段截止已实现，并通过小型真实FE及真实FSI2网格的连续/分段一致性检查。真实网格 h=0.03、dt=0.01、t=0.04 的连续4步与2+2分段结果 **全部末态数组逐位相同**，T24逐变量比较裕度比为0。

h=0.02、dt=0.01、BDF2、IQN、目标20 s的修复后运行已接受824步至8.24 s，在第825步因候选网格体积比不足被拒绝并停止（详情见下文）。总体FAIL，周期仍INCOMPLETE，未完成20 s，不能勾选T24或M6。三网格/三步长研究和公平重复计时仍未完成。

## 断点行为与校验

每个全场事务publish后，单个 `checkpoint.npz` 保存固体位移/速度历史、流体速度历史、网格位移历史、压力、当前网格/速度、界面载荷、预测器、接受时间/步数/epoch/三场generation及全部监测行。BDF1保存1层，BDF2启动前1层、之后2层。NPZ内嵌JSON元数据，加载禁用pickle；每个数组有形状/类型/内容校验值。同目录临时文件写完并fsync后用 `os.replace` 发布，替换失败保持原断点，清理临时文件。

恢复显式指定 `--resume`，不是检测旧文件后自动续跑；旧 `solution.npz` 缺少历史，被明确拒绝。配置只允许延长目标end_time，其他物理、离散、算法与停止参数必须相同；核对网格文件和全部包内Python源码SHA256，文档/测试文件不影响源码摘要。核对有限值、字段形状、时间/代次/层数与监测顺序，并检查实际驱动网格坐标与界面尺寸，全部通过才调用现有 `StateCoordinator.load`。运行中不更改源代码；跨源码版本断点拒绝恢复。

`--max-steps` 只控制本次调用接受多少步，不改变配置目标时间或BDF步长。未到目标时报告INCOMPLETE/退出码3，即使部分区间已可提取周期；报告另存该部分的 `periodic_status`。每次恢复从完整监测账本重建monitor.csv，无重复接受步。物理失败保留最后已接受断点；断点写出失败保留上一份已成功写出的断点，不把未落盘的步称为已保存。最终 `solution.npz` 仍只用于末态证据。

源码：[fsi2_checkpoint.py](../../src/fsi_ice/validation/fsi2_checkpoint.py)、[fsi2.py](../../src/fsi_ice/validation/fsi2.py)、[cli.py](../../src/fsi_ice/cli.py)。复用的BDF、物理残差、牵引与事务实现未改变；测试：[test_fsi2_checkpoint.py](../../tests/test_fsi2_checkpoint.py)、[test_fsi2.py](../../tests/test_fsi2.py)。

## 实际验证与环境

项目 `.venv`、Python3.12.3、NumPy2.5.3、SciPy1.18.1、pytest9.1.1、Gmsh4.12，BLAS/OMP单线程。

| 实际执行 | 结果 |
| --- | --- |
| `pytest tests/test_fsi2_checkpoint.py tests/test_fsi2.py -q` | 71 passed / 16.88 s / exit0 |
| 断点、FSI2、事务、BDF、IQN、CLI相关回归 | 165 passed / 26.09 s / exit0 |
| 真实网格连续4步与2+2恢复 | 两种运行均INCOMPLETE/exit3；所有末态数组逐位相同，比较passed/exit0 |
| 全量回归 | 648 passed、2 skipped / 244.89 s / exit0 |

初版验证曾有3项失败：BDF1错误要求2层历史，以及重签数组形状未与原状态形状清单对照。修复后71项全部通过，未放宽物理门槛。新增断点测试包含四组合BDF1/BDF2×IQN/整体的四步逐位一致、配置/网格/源码/历史/账本/数组篡改拒绝、原子替换失败、物理失败后显式重试、旧末态拒绝及非法分段参数拒绝。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_fsi2_checkpoint.py tests/test_fsi2.py tests/test_transaction.py \
  tests/test_time.py tests/test_iqn.py tests/test_cli.py -q

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m fsi_ice fsi2 \
  --config cases/fsi2.json --output results/m6/long_run/restart_short \
  --mesh-size 0.03 --end-time 0.04 --method iqn --max-steps 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m fsi_ice fsi2 \
  --config cases/fsi2.json --output results/m6/long_run/restart_short \
  --mesh-size 0.03 --end-time 0.04 --method iqn --max-steps 2 \
  --resume results/m6/long_run/restart_short/checkpoint.npz
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m fsi_ice fsi2 \
  --config cases/fsi2.json --output results/m6/long_run/continuous_short \
  --mesh-size 0.03 --end-time 0.04 --method iqn
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/t24_compare.py \
  results/m6/long_run/restart_short results/m6/long_run/continuous_short
```

证据：`results/m6/long_run/logs/{checkpoint_tests,related,full}.log`、[真实短程比较](../../results/m6/long_run/restart_vs_continuous.json)。部分测试与算例同时运行，墙钟仅为诊断，不属于公平计时。

## 长时实验与性能观察

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m fsi_ice fsi2 \
  --config cases/fsi2.json --output results/m6/long_run/iqn_h002 \
  --method iqn --max-steps 200
# 其后显式 --resume results/m6/long_run/iqn_h002/checkpoint.npz，每次最多200步
```

完整配置目标20 s，周期门槛沿用T24：至少三个完整稳定周期，周期/振幅/均值变化≤2%；y振幅误差≤10%、主频≤5%、阻力均值≤10%。当前每步统一全残差≤1e-7、连续性≤1e-10、质量≥0.1、正Jacobian均不变。通过时间推进不代表通过周期精度。

修改前对h=0.02三步启动执行cProfile，文件 `results/m6/long_run/startup_profile.pstats`；总计22.60 s（含profiling和setup），流体步求解14.16 s、einsum7.04 s、约束线性求解5.58 s。各项为包含子调用的累计时间，不能相加；这是性能定位，不是四算法排名。本轮尚未据此修改物理内核或放宽容差。

## 已完成的启动段

首段实际接受200步，到2.0000000000000013 s，目标仍为20 s；report.status=INCOMPLETE，target_time_complete=false，周期数据不足。最大全残差2.983e-9、最大弱连续性2.186e-18、最小网格质量0.727154；IQN每步9–11次评估。已保存完整BDF2历史并显式恢复第二段。

首段报告单独归档为 [chunk01_report.json](../../results/m6/long_run/chunk01_report.json)，日志 `results/m6/long_run/logs/iqn_h002_chunk01.log`。实时接受进度以 `iqn_h002/checkpoint.npz` 和当前分段日志为准；图 [accepted_prefix.png](../../results/m6/long_run/accepted_prefix.png) 为已接受前缀的实测图，标题注明采样截止时间，不是周期合格解。

## 20 s 推进中的全残差门控失败、诊断与修复（2026-10-12 续）

**故障。** 继续分段（每段200步）推进：第3段（401–600步）正常；第4段接受601–701步后在**第702步（t=7.02 s）被物理门控拒绝**：`residual=1.019e-07 > 1e-7`，同时连续性1.973e-18、最小网格质量7.266e-1、界面残差≤6.1e-15——网格、连续性与界面均正常，失败发生在全方程残差检查，`failure.json`（stage=solve）已保存，运行以退出码2停止，断点保留在已接受的第701步；首段至本次失败的chunk报告与日志全部保留（`chunk0{1,2,3}_report.json`、`logs/iqn_h002_chunk0{1,2,3}*.log`）。

**诊断（仅副本、不改动真实轨迹）。** 在 `results/m6/long_run/diag_702_blocks.py` 中复现该步并输出六块缩放残差：`u_f`=1.019e-7（唯一超限块），`v_s`=1.39e-9，`p_f`/`d_s`/`d_m`/`g`≤4e-17；即失败是**流体动量块**的绝对残差。`results/m6/long_run/diag_702_solvers.py` 进一步测出流体Newton最后一步归一化残差3.5e-11（≤容差1e-10即停），但方程RHS量级scale=5762，原始残差≈2e-7——**单场相对判据与验收绝对门控不匹配**。振幅增长时RHS量级（含惯性历史项）随之增大，同一相对容差留下的绝对残差继续放大，因此冻结配置在渐近周期段必然超限；而单场共享容差无法整体收紧到1e-11，因为固体在第1步存在1.61e-11的归一化地板（T24记录）。

**修复。** 流体与固体Newton求解器新增可选参数 `absolute_tolerance`（默认`None`，行为不变）：仅当相对判据与**原始自由残差二范数≤给定绝对目标**同时满足才停止；瞬态驱动按当前块尺度传入门控的10%（`0.1×residual_max×S_block`，S=(1,1000,.41,1,.41,215.06)；2-范数界蕴含门控的最大分量界，留一档裕度），不新增配置字段。简单例子：同一失败步现在被接受，残差1.392e-9（12次耦合迭代、Newton总迭代49→59），见 `results/m6/long_run/diag_702_fixed.py`。

**回归。** 全量 **651 passed、2 skipped**（198.22 s，原648+新增3项；日志 `results/m6/long_run/logs/full_after_fix.log`）；真实网格 h=0.03、t=0.03 四方法三步一致性重跑，逐变量最不利裕度比与T24记录一致（IQN 0.026369、Aitken 0.007213、fixed 0.006413，全部passed/exit0；证据 `results/m6/postfix/`）——修复对小幅值轨迹零扰动，只在大振幅下增加一次流体Newton。所有物理与周期门槛未改动。

**修复后的20 s运行。** 重新从第0步运行（输出目录 `iqn_h002_r2/`，与原失败目录分开），分段命令与门槛不变；最终周期、振幅、主频、阻力均值与误差结果见下节。

## 修复后运行的第二类失败：大位移网格压缩

2026-10-12，`iqn_h002_r2` 已越过原第702步全残差失败点，最终接受824步到8.239999999999869 s（41.2%）。第825步的第二次耦合评估被拒绝：单元1651的候选体积比0.195847703311，小于ALE原有0.2门槛。求解进程已退出，没有后台继续运行；完整断点保留在824步。该已接受步残差1.3196e-8、连续性1.8896e-18、最小质量0.388415、最小体积比0.260565，仍符合接受条件。失败候选未进入BDF历史或监测账本，未放宽任何门槛。

已配置的退出后分析实际触发：[报告](../../results/m6/postprocess/iqn_after_exit/summary.md)、[JSON](../../results/m6/postprocess/iqn_after_exit/summary.json)、[实测时程图](../../results/m6/postprocess/iqn_after_exit/history.png)。总体FAIL，周期INCOMPLETE；可提取的梁尖最近三周期振幅变化94.88%，显著不稳定，不能将暂定振幅0.0155847 m、主频2.18918 Hz、阻力均值135.275 N/m当作最终基准指标。源内完整摘要仍与断点一致。

只读几何诊断定位同一单元的参考重心约(0.599862, 0.214763) m，在梁尖上方。最后几个已接受步体积比依次约0.774、0.634、0.500、0.374、0.261，已持续接近门槛。以两层网格历史线性外推和当前BDF网格速度预测下一步，得到0.144733和0.151878；二者均不是已求解/已接受状态，只说明已接受运动趋势也会挤压该区域，不能证明收敛后的第825步必然失败。证据：[mesh_limit_diagnostic.json](../../results/m6/postprocess/mesh_limit_diagnostic.json)，读取前后原断点字节完全相同。

下一步需区分耦合候选过冲与调和动网格在大位移下的能力限制，设计并验证保持原质量门槛的处理方案。没有修复证据前不重复启动相同失败调用，不宣称20秒或周期通过；新的源版本也不能静默绕过断点源码守卫。

## 文档检查

断点功能交付时，HTML结构、13个唯一id、导航锚点、154条HTML本地链接与136条主要Markdown本地链接检查通过；Orca浏览器1280×900和390×844截图实看通过，宽屏clientWidth=scrollWidth=1265、窄屏均375，无整页横向溢出。检查与截图在 `results/m6/long_run/logs/{html_check.json,layout_wide.json,layout_narrow.json,principles_wide.png,principles_narrow.png}`。后续后处理维护的结构/链接/布局检查见 [后处理记录](m6_postprocess.md)。本次失败状态维护只做只读数值诊断与HTML结构/链接及diff检查，没有改动求解源码，没有重跑数值回归或重做浏览器布局检查。M6保持进行中。

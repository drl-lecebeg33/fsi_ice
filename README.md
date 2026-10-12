# fsi_ice

基于 Python 的自研流固耦合研究平台，面向分区与整体耦合算法比较及 FSI1/FSI2 基准验证。M0–M5 已完成：数值基础、单场、ALE、界面、事务、分区算法（固定松弛/Aitken）、稳态 FSI1，以及整体耦合（统一残差、完整耦合 Jacobian、整体 Newton、伪时间定常推进）与同一离散配置下的一致性/成本比较均有本地验证证据。M6/T24 与 T25 进行中：FSI2 参数、两路线瞬态驱动、事务衔接与短程一致性已验证；IQN-ILS 控制器已在 T18 线性有限元同离散上验收（fixed=27 / Aitken=7 / IQN-ILS=11 次评估）；稳定周期、网格/时间收敛与四算法公平比较尚未完成。

## 项目文档

- [开发规格与规范](spec.md)
- [开发目标与里程碑](gola.md)
- [任务分工与交付流程](agent.md)
- [代理协作规则](AGENTS.md)
- [原理说明（初学者，浏览器打开）](docs/principles/index.html)
- [原理文档维护约定](docs/principles/维护说明.md)
- [Python 环境与验证记录](docs/environment.md)
- [M0 数值基础验收记录](docs/validation/m0.md)
- [固体与流体单场验证](docs/validation/single_fields.md)
- [M3 ALE、界面与全场事务验收](docs/validation/m3.md)
- [T18 固定松弛设计与验收约定](docs/development/t18_plan.md)
- [T18 本地验证记录](docs/validation/t18.md)
- [T19 Aitken 验收记录](docs/validation/t19.md)
- [T20 FSI1 稳态基准计划](docs/development/fsi1_plan.md)
- [T20 FSI1 稳态验收记录](docs/validation/t20.md)
- [M6/T24 瞬态计划与冻结](docs/development/m6_t24_plan.md)
- [T24 FSI2 瞬态与短程验证](docs/validation/t24.md)
- [T25 IQN-ILS 实现计划](docs/development/t25_plan.md)
- [T25 IQN-ILS 验收记录](docs/validation/t25.md)

T20 已提供正式 FSI1 双域网格入口与稳态分区求解。三档网格最细结果的梁尖 y 位移误差 4.076%、阻力误差 0.394%，满足 5% 门槛；完整场、迭代和失败证据可由配置重复生成。

运行 FSI1：

```bash
.venv/bin/python -m fsi_ice fsi1 --config cases/fsi1.json --output results/t20/fsi1
```

## 本地环境

采用 Python 3.12、NumPy、SciPy、Matplotlib 和 pytest。完整版本固定在 `requirements.txt`，使用项目内 `.venv` 隔离依赖。

在项目根目录安装依赖和项目包：

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
```

检查环境：

```bash
.venv/bin/python scripts/check_environment.py
```

环境重建步骤及本机缺少 `ensurepip` 时的替代方式见 [环境说明](docs/environment.md)。

## 运行与测试

运行全部本地测试：

```bash
.venv/bin/python -m fsi_ice test
```

运行 Poisson 三网格收敛验证并生成结果图：

```bash
.venv/bin/python -m fsi_ice poisson --config cases/poisson_sine.json --output results/poisson
```

输出包括验收报告、误差表、最细网格解和图像；`results/` 默认不纳入 Git。使用 `--no-plot` 可仅保存数值数据。无配置时使用内置的 8、16、32 三档网格；配置格式见 [算例文件](cases/poisson_sine.json)。

包安装后，`python -m fsi_ice` 或控制台命令 `fsi-ice` 均可从其他目录运行；此时配置和输出路径按当前工作目录解释。`test` 为开发入口，需要保留本地项目及测试目录。

最新全量回归 **600 项通过、2 项跳过**（M6/T24+T25，本地216.35 s）；T25 新增16项 IQN-ILS 控制器、重启与配置检查，T24 此前新增27项瞬态、事务、配置/输出与统计检查。M5冻结状态为557通过、2跳过。证据见 [T24记录](docs/validation/t24.md) 与 [T25记录](docs/validation/t25.md)；历史证据见 [M0 记录](docs/validation/m0.md)、[T19 验收](docs/validation/t19.md)、[T20 验收](docs/validation/t20.md) 与 [M5 矩阵](docs/validation/m5.md)。

## 固体与流体验证

运行悬臂梁、CSM1/2、Poiseuille、解析 Stokes/Navier–Stokes 空间收敛、CFD1/2 和独立 BDF1/BDF2 时间收敛：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m fsi_ice single-fields \
  --config cases/single_fields.json --output results/single_fields
```

可将 `single-fields` 换成 `solid` 或 `fluid` 只运行其中一个模块。静力/稳态验收已通过，数值与定义见 [单场记录](docs/validation/single_fields.md)。

动态基准分别运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m fsi_ice csm3 \
  --config cases/csm3.json --output results/csm3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m fsi_ice cfd3 \
  --config cases/cfd3.json --output results/cfd3_refinement
```

CSM3/CFD3 最终细化结果均为 PASS，证据见 [单场验证记录](docs/validation/single_fields.md)。动态命令分别运行三套网格和三个时间步长，保存每步监测值、残差、周期统计及报告；耗时明显长于单元测试。是否通过以报告中的 `status` 和 `checks` 为准，不以命令能运行为准。

已有完整结果时，在动态命令后添加 `--postprocess-only`，可从每一步的监测数据重新计算周期、残差和守恒检查，生成报告及网格/时间对比图。该入口不推进时间步，不支持从监测数据恢复完整物理状态；缺少数据、采样不完整或网格/步长不一致会报错。报告记录输入文件的摘要校验值及已有来源信息。

CFD 基准网格生成使用外部 **Gmsh CLI**（本机 4.12）；无 Gmsh 时 Poisson、矩形解析算例和固体网格仍可运行。未使用现成 FEM/流体/固体求解器。单线程设置用于控制本机 BLAS 线程开销，数学离散相同。

代码功能必须在本地运行、测试通过且没有错误后，才能上传到 GitHub。

## 分区耦合验证：固定松弛、Aitken、IQN-ILS

运行 T18 小型有限元验证；`--method` 可覆盖配置中的算法选择：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m fsi_ice partitioned-check \
  --config cases/partitioned_check.json --output results/t18/check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m fsi_ice partitioned-check \
  --config cases/iqn_check.json --output results/t25/iqn --method iqn
```

验证使用固定参考几何上的 BDF1 单步、非定常 Stokes 和线弹性动力学，比较分区解与相同离散方程的块直接解。保存 `report.json`、`iterations.csv` 和 `solution.npz`；配置读取、解析或校验失败以及求解或写出失败都会清除旧报告和旧解并保存 `failure.json`（输入失败保留配置路径与故障阶段，不声称已加载无效配置）。正式 FSI1 已在 T20 独立验收；三种方法的同离散对比见 [T25 验收记录](docs/validation/t25.md)。

`fsi_ice.coupling` 提供 `DirichletNeumannMap` 和 `solve_dirichlet_neumann`：依次调用流体求解、载荷传递、固体求解及界面提取回调，逐分量尺度必须显式提供。以未松弛界面残差判定收敛；收敛结果保留最后评估的输入与对应单场状态。回调应检查物理残差并在失败时抛出异常，且始终使用同一步已接受历史；成功后的统一时间接受由外层全场事务负责。`CouplingOptions.method` 支持 `fixed`（默认）、`aitken` 与 `iqn`；IQN-ILS 的初始松弛、系数上限、历史长度与重启容差分别由 `iqn_initial_relaxation`、`iqn_alpha_max`、`iqn_history_size`、`iqn_restart_tolerance` 控制。接口、尺度和退出条件见 T18/T25 计划。

## 整体耦合验证（M5）

整体耦合（统一残差、完整耦合 Jacobian、矩阵自由 JVP、整体 Newton）位于 `fsi_ice.coupling`：求值入口 `monolithic_residual` / `monolithic_jacobian` / `monolithic_jvp` / `evaluate_monolithic`（模式 `residual|assembled|jvp`），Newton 与定常推进 `monolithic_newton.solve_monolithic_step` / `march_to_steady`（伪时间 BDF 推进至定常固定点）。接口、门槛与实测见 [T21 记录](docs/validation/t21.md)、[T21 独立复核](docs/validation/t21_review.md) 与 [T22 记录](docs/validation/t22.md)。

运行 T23 一致性与成本比较（h=0.02 同一网格，同机单线程；整体推进逐档热启动，单档约 3–15 分钟）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/m5_compare.py prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/m5_compare.py partitioned \
  --methods fixed,aitken --levels 1e-6,1e-8,1e-10 --repeats 3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/m5_compare.py monolithic \
  --level 1e-6 --dt 1.0 --order 2 --chunk-steps 5 --max-steps 40 --wall-seconds 2400
# 再依次对 --level 1e-8、1e-10 重复（逐档热启动），最后：
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/m5_compare.py compare --levels 1e-6,1e-8,1e-10
```

结果写入 `results/m5/t23/`（配置快照、环境与线程记录、各方法输出、`comparison/consistency_per_variable.csv`、`comparison/cost_per_item.csv`、`comparison/summary.json`）。1e-10 档三种方法逐变量一致（最不利裕度比 0.984）；成本表包含组装/线性求解/Newton/耦合/失败步等项。验收矩阵与全部条目状态见 [M5 矩阵](docs/validation/m5.md)，完整命令、失败修正与记录项见 [T23 记录](docs/validation/t23.md)。

## FSI2 瞬态（M6/T24，进行中）

FSI2 的来源参数、余弦入口启动、BDF1/BDF2 与三种当前可用求解路径（fixed、aitken、monolithic）已接入同一全场事务。稳定周期与加密研究尚未完成，不能将短程结果称为基准精度合格解。短程真实网格两路线的三步一致性与失败重试已验证，见 [T24 记录](docs/validation/t24.md)。

运行启动区间：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m fsi_ice fsi2 \
  --config cases/fsi2.json --output results/m6/t24/aitken \
  --mesh-size 0.03 --end-time 0.03 --method aitken
# 将 --method 改为 monolithic，并用不同 --output，可运行同离散整体法。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/t24_compare.py \
  results/m6/t24/aitken results/m6/t24/monolithic
```

短程命令返回 `INCOMPLETE`、退出码3（正常完成启动区间但缺少周期证据）；数值或配置失败为退出码2，稳定周期参考门槛失败为1，全部通过才返回0。比较脚本退出码0仅表示短程末态一致，不改变周期结论。输出为 report.json、逐接受步 monitor.csv 与末态 solution.npz；失败清除旧报告与末态，保留已接受监测行并写 failure.json。没有隐式续算或从末态恢复 BDF 历史的功能。

省略短程覆盖项时，配置运行 h=0.02、dt=0.01、t=20 s；这项完整周期实验尚未执行，整体组装路径长时耗时较高。T25 已交付并验收 IQN-ILS 控制器（T18 线性有限元同离散），把四个算法跑到同一道 FSI2 题上的公平成本表与图表仍属后续 T26。

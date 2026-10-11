# fsi_ice

基于 Python 的自研流固耦合研究平台，面向分区与整体耦合算法比较及 FSI1/FSI2 基准验证。M0–M4 已完成：数值基础、单场、ALE、界面、事务、固定/Aitken 分区算法及稳态 FSI1 均有本地验证证据。当前停在 M5 整体耦合入口；FSI2 尚未实现。

## 项目文档

- [开发规格与规范](spec.md)
- [开发目标与里程碑](gola.md)
- [任务分工与交付流程](agent.md)
- [代理协作规则](AGENTS.md)
- [Python 环境与验证记录](docs/environment.md)
- [M0 数值基础验收记录](docs/validation/m0.md)
- [固体与流体单场验证](docs/validation/single_fields.md)
- [M3 ALE、界面与全场事务验收](docs/validation/m3.md)
- [T18 固定松弛设计与验收约定](docs/development/t18_plan.md)
- [T18 本地验证记录](docs/validation/t18.md)
- [T19 Aitken 验收记录](docs/validation/t19.md)
- [T20 FSI1 稳态基准计划](docs/development/fsi1_plan.md)
- [T20 FSI1 稳态验收记录](docs/validation/t20.md)

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

最新全量回归 **432 项通过**；包含 T18/T19 分区耦合及 T20 FSI1 网格、稳态求解、重复性和失败处理检查。历史证据见 [M0 记录](docs/validation/m0.md)、[T19 验收](docs/validation/t19.md) 和 [T20 验收](docs/validation/t20.md)。

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

## 固定松弛分区耦合验证

运行 T18 小型有限元验证：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m fsi_ice partitioned-check \
  --config cases/partitioned_check.json --output results/t18/check
```

验证使用固定参考几何上的 BDF1 单步、非定常 Stokes 和线弹性动力学，比较分区解与相同离散方程的块直接解。保存 `report.json`、`iterations.csv` 和 `solution.npz`；配置读取、解析或校验失败以及求解或写出失败都会清除旧报告和旧解并保存 `failure.json`（输入失败保留配置路径与故障阶段，不声称已加载无效配置）。正式 FSI1 将在 T19/Aitken 及 T20 完成后独立验收。

`fsi_ice.coupling` 提供 `DirichletNeumannMap` 和 `solve_dirichlet_neumann`：依次调用流体求解、载荷传递、固体求解及界面提取回调，逐分量尺度必须显式提供。以未松弛界面残差判定收敛；收敛结果保留最后评估的输入与对应单场状态。回调应检查物理残差并在失败时抛出异常，且始终使用同一步已接受历史；成功后的统一时间接受由外层全场事务负责。接口、尺度和退出条件见 T18 计划。

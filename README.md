# fsi_ice

基于 Python 的自研流固耦合研究平台，面向分区与整体耦合算法比较及 FSI1/FSI2 基准验证。已实现并验证数值基础、线弹性/StVK 固体、Stokes/Navier–Stokes 流体和 BDF 时间推进，CSM1/2/3 与 CFD1/2/3 单场验收已通过。本轮按用户要求停在单场验证与结果评估，完整 FSI 尚未实现。

## 项目文档

- [开发规格与规范](spec.md)
- [开发目标与里程碑](gola.md)
- [任务分工与交付流程](agent.md)
- [代理协作规则](AGENTS.md)
- [Python 环境与验证记录](docs/environment.md)
- [M0 数值基础验收记录](docs/validation/m0.md)
- [固体与流体单场验证](docs/validation/single_fields.md)

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

当前回归：93 项测试通过。M0 阶段原有 44 项，Poisson 最后一次加密的 L2 收敛阶约 1.994、H1 半范阶约 0.997；历史证据见 [M0 记录](docs/validation/m0.md)，新增单场证据见下文。

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

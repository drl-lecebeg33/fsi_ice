# T19/A5 实现与验证记录

日期：2026-10-11。结论：A5 指定工作完成，待 A0 集成；不代表完整 T19 CLI 集成、全量回归或 FSI1 验收。没有提交、推送或派发子代理。基线 `2b5f713`，现有 T18 未提交工作保留；验收定义先写入 [T19 计划](../development/t19_plan.md)。

## 改动与接口

- `src/fsi_ice/coupling/partitioned.py`：在现有控制器上增加 `method='fixed'/'aitken'`，默认 fixed 的系数和求解结果兼容；新增冻结的三个参数并校验（固定方法也校验新增参数合法性，只有 Aitken 要求初始值落入 bounds）。既有数据类位置构造仍可用。
- `CouplingIteration.relaxation_reason` 在既有字段后追加默认 `fixed`；事件包含 `initial`、`aitken`、`clipped`、`denominator_degenerate`、`coefficient_nonfinite`、`converged`、`max_iterations`。非终止行的 relaxation 是实际用于下一次更新的系数，终止行保留最近系数，不产生未评估更新。
- Aitken 使用 q=r/S，omega=-omega_prev*(q_prev·delta_q)/(delta_q·delta_q)。omega_prev 是上次实际应用的值（包括截断）；相对退化判据为 delta_q·delta_q <= tolerance*max(q_prev·q_prev,q·q)，沿用上一系数。没有绝对 1 的地板，残差变小不会自动冻结加速。
- 内积/系数非有限计算明确终止为 `CouplingFailure(stage='relaxation')`，当前真实残差与最后已评估输入可用于诊断；更新溢出也转换为诊断失败。仍使用 T18 的未松弛真实残差、原阈值和最大评估次数；局部 Aitken 状态不跨 solve 保留，控制器不接受时间历史。
- 新增 `tests/test_aitken.py`。`coupling/__init__.py` 原有导出已经覆盖所需 API，无须修改。既有 `tests/test_partitioned.py`、CLI、共享验证入口、case 和物理模块未修改。

## 实际执行命令与环境

全部使用项目 `.venv`，Python 3.12.3；NumPy 2.5.3、SciPy 1.18.1、Matplotlib 3.11.2、pytest 9.1.1；解释器和依赖实测值保存在 `results/t19/worker_a5/verification.json`。

| 实际命令 | 退出码 | 结果/证据 |
| --- | --- | --- |
| `.venv/bin/python --version` | 0 | Python 3.12.3 |
| `.venv/bin/python -m pytest tests/test_aitken.py tests/test_partitioned.py -q`（首轮） | 1 | 109 passed / 2 failed，保留 `pytest_initial.log` 和 `pytest_initial.exitcode` |
| 同命令（修正测试设置后） | 0 | 112 passed / 2.47 s |
| 同命令（补充实际已截断系数递推、低残差尺度与动态后退化覆盖后，最终） | 0 | **115 passed / 2.33 s**，`pytest.log` 和 `pytest.exitcode` |
| `.venv/bin/python results/t19/worker_a5/verify.py` | 0 | PASS，`verify.log`、`verify.exitcode`、`verification.json` 和两种方法迭代 CSV |
| `git diff --check` | 0 | 无输出；补充人工检查新增文件布局与指定写入范围 |

首轮失败属于两处测试设置：上限截断标量测试沿用默认停止容差，解误差 1.49e-8 大于测试门槛，因此收紧求解 atol/rtol 后按更严的 <=1e-9 解误差验收；单位变换测试用十进制倍率时，接近零残差的相减舍入导致后期系数差最大 5.76e-8，因此改用可精确表示的 1024 与 1/1024 倍率，仍以 <=1e-12 验收整个系数/残差序列和解。没有放宽 T18 或本计划的数值门槛。单位不变性是数学和归一化算法性质，接近机器精度时不能要求任意十进制单位变换的系数逐位相同。

## 数值结果

标量 G(x)=a*x+0.75，初值 0.1、S=0.2、初始 omega=0.5、显式 maximum=2：

| a | 实际 omega 序列（含终止行） | 评估次数 | 解析解归一化误差 |
| --- | --- | --- | --- |
| -0.5 | 0.5, 0.6666666666666666, 0.6666666666666666 | 3 | 0 |
| 0.25 | 0.5, 1.3333333333333333, 1.3333333333333333 | 3 | 0 |

两者均经一次动态更新达到直接解，事件为 initial/aitken/converged。多分量测试直接解误差 <=1e-9，系数受配置 bounds 限制；精确单位变换测试整个系数、归一化残差序列和解均满足 <=1e-12。

真实 FE 沿用 T18 冻结配置：两个域均 3×2 划分、P1 固体 24 自由度、P2/P1 流体 82 自由度，BDF1 dt=0.1，固定参考几何，物理、尺度、atol=1e-11、rtol=1e-9 与 acceptance 不变。以下值由复用的 `LinearCoupledProblem.metrics` 实测：

| 量 | fixed (omega=0.5) | aitken (0.05<=omega<=1) | 冻结门槛 |
| --- | --- | --- | --- |
| 组合评估次数 | **27** | **7** | 记录，不承诺普遍加速 |
| 位移归一化差 | 3.4109e-13 | 1.8169e-13 | <=1e-8 |
| 速度归一化差 | 6.7610e-12 | 3.2040e-12 | <=1e-8 |
| 压力归一化差 | 4.1610e-12 | 1.9241e-12 | <=1e-8 |
| 载荷归一化差 | 1.2142e-12 | 6.0367e-13 | <=1e-8 |
| 组合方程真残差 | 7.1021e-12 | 2.5025e-12 | <=1e-9 |
| 界面速度误差 | 7.1021e-13 | 2.5025e-13 | <=1e-9 |
| 连续性 | 2.4856e-18 | 9.9149e-18 | <=1e-10 |
| 最终真实界面残差 | 7.1021e-12 | 2.5025e-12 | <=atol+rtol*magnitude |

Aitken FE 系数序列与事件见 `aitken_iterations.csv`；默认 bounds 下本例没有截断或退化，额外事件通过针对性测试实际触发。

## 失败、兼容与剩余工作

实际覆盖零/近退化分母及保留动态更新后的系数、正/负系数截断、用实际已截断系数参与下一次递推、低残差尺度下正常加速、非有限 Aitken 内积计算、过松弛更新溢出、非法新增参数、极小更新但真实残差大、初值已收敛、最后评估不更新、回调异常和旧位置构造。

真实 FE 限制 1/3 次评估分别在真实残差 2.7059e-3 / 3.1738e-5 时明确失败；测试还在真实流体求解之后注入异常，诊断阶段为 fluid。全部失败后 BDF 数组和 generation 保持，重试与冷启动的界面输入、固体位移、流体速度/压力、载荷逐数组相同，重复运行也逐数组相同。默认 fixed 与显式 fixed 一致，既有 T18 测试全部通过。

没有 A5 未解决阻碍；剩余为协调者负责的 CLI/配置及比较报告集成、根文档进度更新和集成后全量回归。本轮按任务要求没有运行耗时全量回归，未将小型固定几何 FE 验证宣称为完整 FSI1/瞬态 FSI 验收。

持久证据：`results/t19/worker_a5/verify.py`（可复现脚本）、`verification.json`（环境、配置、完整数值和检查）、`fixed_iterations.csv`、`aitken_iterations.csv`、`verify.log`、`verify.exitcode`、`pytest.log`、`pytest.exitcode`、`pytest_initial.log`、`pytest_initial.exitcode`。本工作修改文件仅为控制器、新测试、本计划、本报告及该证据目录。

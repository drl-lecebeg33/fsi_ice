# M5/T23 方法学：分区/整体一致性比较与成本统计（A7 设计）

日期：2026-10-11。状态：**设计文档（计划）；T23 未执行**。r1（2026-10-11，A7）：按 [m5_freeze.md](m5_freeze.md) 裁决修正 M-1/M-3/M-6/M-7/M-8/M-9/M-11；T21 已交付并完成独立验证（见 [t21_review.md](../validation/t21_review.md)），T22 实现/集成进行中，T23 仍未执行（**不运行整体或分区求解，不修改求解代码，不写 `results/`**）。本文件不替代实现细节，也不把任何计划项写成已验证。机器可读骨架见 [m5_validation.json](../../cases/m5_validation.json)，矩阵正文见 [m5.md](../validation/m5.md)。

## 1. 背景与目标

[gola.md](../../gola.md) M5 要求：建立统一残差与完整耦合 Jacobian、完成整体 Newton、在相同离散配置下比较整体法与分区法的 FSI1 解，并比较物理误差、耗时和求解工作量。退出条件为“两种方法收紧容差后得到一致离散解，差异与测试容差均有记录”。本文件解决 T23 的两个问题：

1. **一致性**：如何证明同一离散配置下，整体法、分区固定松弛、分区 Aitken 收敛到同一离散解，且差异判据可追溯、可复核；
2. **成本**：如何公平记录并比较求解成本，使不同方法的数字可比、可复现。

[T20 验收记录](../validation/t20.md) 已给出分区 Aitken 稳态 FSI1 的三网格结果与重复性证据；T23 的比较对象包含整体法（T21/T22 产物）与分区法（复用 [cases/fsi1.json](../../cases/fsi1.json) 配置）。

## 2. 比较方法学

### 2.1 比较对象与次序

- 方法 M1：整体法（monolithic Newton，T22 产物）；T21 已交付 `src/fsi_ice/coupling/monolithic.py`、`src/fsi_ice/validation/monolithic.py`、`tests/test_monolithic_residual.py`（独立复核 `tests/test_monolithic_review.py`）；T22 拟名 `src/fsi_ice/coupling/monolithic_newton.py`、`tests/test_monolithic_newton.py`（M-9）；
- 方法 M2：分区固定松弛（Dirichlet–Neumann，`method="fixed"`）；
- 方法 M3：分区 Aitken（`method="aitken"`，T20 配置）。

一致性检查先比较 M1 与 M3（T20 已有基线），再补 M2 以覆盖 [spec.md](../../spec.md) §1“两种耦合方法”及分区内部方法差异。三种方法必须调用同一套物理残差、单元组装、界面映射与时间事务代码（T21 冻结后确认）。

### 2.2 同一离散配置清单（固定项）

依据 [spec.md](../../spec.md) §3.3“分区法与整体法共享相同的物理离散、材料、边界条件、时间步及网格”与 §5.3 第 1 条：

| 项目 | 固定要求 | 来源 |
| --- | --- | --- |
| 物理模型与材料 | 稳态不可压 Navier–Stokes（P2/P1 Taylor–Hood）+ 总拉格朗日 P1 StVK；`rho_f=1000 kg/m³`、`mu_f=1 Pa·s`、`Ubar=0.2 m/s`；`rho_s=1000 kg/m³`、`nu_s=0.4`、`E=1.4 MPa`、单位厚度 | spec §2/§3.2；cases/fsi1.json |
| 网格 | 同一 Gmsh MSH2 文件拆分出的流体/固体网格（同一 h 档）；参考网格只读，候选几何另建 | spec §3.1/§3.3；T20 记录 |
| 时间积分 | 稳态 FSI1 无物理时间推进；**整体法定常解由伪时间 BDF 推进至固定点取得**：固定点处 BDF 时间项精确抵消（`Σa_j=0`），是稳态离散方程的解（T22 实现推进驱动与停机判据，以尺度化时间项控制；成本口径记录步数、Newton 次数与时间项收敛值，见 §4.1）。若补充瞬态一致性小算例，三种方法同为 BDF1/BDF2、同启动方式（BDF2 启动步按 BDF1） | spec §3.2；m3_freeze.md；m5_freeze.md（M-11） |
| 边界与压力规范 | 同入口抛物线剖面、同无滑移/夹持边界；一致性比较**两侧均使用 `pressure_pin=None`**（出口自然边界定压，与 T20 分区实现一致）；`PressurePin` 保持可选，若未来某算例改用 pin，两侧必须使用同一 pin 定义并在报告记录；比较前用同一约定移除压力零模 | spec §3.2；cases/fsi1.json；m5_freeze.md（M-3） |
| 界面约定 | 同一 P1→P2 位移映射 H、同一 Hᵀ 对偶载荷、厚度只计一次、参考匹配一次 | interface_plan.md §9；agent.md §8 |
| 收敛状态 | 每种方法达到声明的归一化残差容差，并保存完整迭代史；未达到容差的运行不进入比较 | spec §5.3 |
| 输出设置 | 相同监测量（tip A 位移、drag/lift、残差、工作量、耗时）与相同落盘结构 | spec §5.3 |

### 2.3 容差收紧策略（拟议，执行前由 A0 冻结）

统一记归一化相对残差容差为 `tau`（定义随方法与文件记录，不得混用）：

- 分区法：[`coupling.absolute_tolerance`/`relative_tolerance`](../../cases/fsi1.json) 与 `solver.fluid_tolerance`/`solver.solid_tolerance` 同步收紧；界面残差采用 [partitioned.py](../../src/fsi_ice/coupling/partitioned.py) 的 `||(G(x)-x)/scales||inf <= atol + rtol*max(...)` 定义，`scales` 保持 `1e-3 m`（接口位移特征尺度）不变。
- 整体法：整体 Newton 的相对残差容差同步收紧（具体到 T22 实现时由 A6/A0 冻结）。
- 拟议序列：`tau in {1e-6, 1e-8, 1e-10}` 三档，逐档收紧。
- 取最紧档作为一致性比较解；若相邻两档的逐变量差未进入平台（见 §3），停止比较并报告为失败/待查，**不选取“看起来通过”的档位**。
- 各档不得低于线性求解可认证下限（现有流体线性求解真残差检查 ≤1e-10、GMRES `rtol=1e-11/atol=1e-12`，见 [linear_solver.py](../../src/fsi_ice/fluid/linear_solver.py)）；为通过而放宽门槛违反 [spec.md](../../spec.md) §5.2 与 [m3_freeze.md](m3_freeze.md) §6。
- **残差定义不得混用（M-1）**：分区法与整体法的残差定义不同（分区为界面不动点残差 `||(G(x)-x)/scales||inf`，整体为归一化方程残差），**不同定义的残差数字不得直接比较**；本计划只在各方法自身的收敛判据中使用其定义值，并在报告记录定义、归一化尺度与最终数值。

**执行 1e-10 档前的裕度验证清单（M-8，执行期）** —— 下列各项必须在正式运行 `tau=1e-10` 档之前实测并写入报告；任一项裕度不足则停止并报告为待查/失败，不得放宽门槛：

1. 流体单场 Newton：`solver.fluid_tolerance` 收紧到对应档后能达到且**未触及迭代上限**（记录容差、实际迭代数与上限余量）；
2. 固体单场 Newton（`solve_static_continuation`）：同上记录容差与迭代余量；
3. 流体线性求解可认证下限：真残差检查与 GMRES 参数（`rtol=1e-11/atol=1e-12`）在 1e-10 档下仍可达到（记录实测真残差）；
4. 分区界面残差容差与 `scales`（`1e-3 m`）不因档位改变；耦合迭代上限余量记录；
5. 整体法（T22）：归一化残差控制量、Newton 迭代上限与线搜索策略在 1e-10 档的实测余量；
6. 上述各值与命令写入报告；未通过项不得静默跳过。

### 2.4 证明次序与不变量

证明次序（前一步未通过，不进入下一步）：

1. T21：整体残差复用单场物理核，逐块与全局方向导数、几何依赖达标；
2. T22：整体 Newton 收敛、失败检测与恢复达标；
3. 单方法自稳定：每种方法在收紧序列下相邻档逐变量差落入比较容差（平台存在）；
4. 跨方法一致：整体/固定/Aitken 两两比较，逐变量判据全部通过；
5. 成本记录：在步骤 3–4 的同一次运行中以统一口径采集（§4）。

不变量（比较期间必须成立）：

- **同一离散恒等**：同一状态向量下，整体残差与分区路径的单场残差来自同一函数，逐项一致（21-01）；
- **参考只读**：候选求解不修改参考网格与已接受历史；失败候选回滚，不进入比较集（复用 M3/T17 事务机制）；
- **压力规范固定**：FSI1 两侧均 `pressure_pin=None`（出口自然边界定压）；若改用 pin，两侧使用同一 pin 定义；比较前用同一规范移除零模，不比较绝对压力常数（M-3）；
- **离散不被收紧改变**：收紧只改变终止阈值，网格、材料、边界、残差/矩阵公式不变；
- **未标注近似禁止**：完整整体 Jacobian 中任何遗漏或近似必须显式标注并单独验证（21-04），不得把近似当作完整耦合切线。

## 3. 逐变量一致性判据

### 3.1 误差定义

记 `a_M`、`a_P` 分别为方法 M 与 P 在同一比较状态下的对应自由度向量（同一 DOF 编号、同一拓扑、同一空间；节点值的比较顺序固定）：

- `Delta a = a_M - a_P`，报告 `max|Delta a|` 与相对量；
- 判据：`max|Delta a| <= rtol_a * max(||a_M||inf, ||a_P||inf) + atol_a`；
- 小量规则：接近零的量不得直接算不可靠的相对误差，由 `atol_a` 兜底（spec §5.2 末段）。

### 3.2 变量、尺度和拟议容差表

来源：spec §5.2“FSI1 两种耦合方法：收紧求解容差后得到一致离散解；采用逐变量尺度和明确的绝对/相对容差”，以及 §5.3“残差按变量尺度归一化；记录其定义”。具体数值为拟议值，执行前由 A0 冻结；任何调整须记录理由，且不得放宽 spec 固定门槛（方向导数 ≤1e-5、FSI1 相对误差 ≤5%、tip_x 绝对 ≤1e-5 m、lift 绝对 ≤0.1 N/m）。

| 变量 | 记号 | 单位 | 显式尺度 `S*` | `rtol_a`（拟议） | `atol_a`（拟议） | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| 固体位移 | `d_s` | m | `L*`（拟 0.41 m，通道高；执行前冻结，全表一致） | 1e-8 | `1e-10*L*` | 全场 P1 节点值 + tip A 同时比较 |
| 界面位移值 | `x_G = H d_s` | m | `L*` | 1e-8 | `1e-10*L*` | 独立检查 H 映射后的界面向量 |
| 流体速度 | `u` | m/s | `U*`（拟 0.2 m/s，Ubar） | 1e-8 | `1e-10*U*` | 同一 P2 空间的节点值 |
| 压力 | `p` | Pa | `P* = max(rho*U*^2, mu*U*/L*)`（拟 40 Pa） | 1e-8 | `1e-10*P*` | 比较前移除压力零模；FSI1 两侧均 `pressure_pin=None`（M-3），若用 pin 须两侧同一定义 |
| 网格位移 | `d_m` | m | `L*` | 1e-8 | `1e-10*L*` | 流体当前几何减参考坐标；同时检查最小体积比/质量 |
| 固体速度 | `v_s` | m/s | `U*`（拟 0.2 m/s） | 1e-8 | `1e-10*U*` | 固体 P1 节点速度（整体法未知量；分区侧取自身固体速度节点值）；与 `d_s` 满足 BDF 相容关系，不替代逐变量判据 |
| 界面双力（整体） | `g` | N/m | `F*`（拟 14.29426 N/m，参考阻力） | 1e-8 | `1e-10*F*` | 整体法未知量：流体 P2 界面迹对偶节点力（厚度只计一次，符号流体对固体，顺序同 `assemble_interface_load`）；与分区侧 `lambda` 为同一对象，按同一节点顺序比较（23-07） |
| 界面载荷（分区） | `lambda`（Hᵀ 对偶节点力） | N/m | `F*`（拟 14.29426 N/m，参考阻力） | 1e-8 | `1e-10*F*` | 分区侧 Hᵀ 对偶节点力；与整体侧 `g` 为同一对象（23-07）；与功/力一致性、符号约定一并检查 |

理由与边界：三种方法求解同一离散方程，当各自终止容差收紧到 `tau=1e-10` 后，返回值差异主要来自终止准则与舍入。比较容差取 `rtol=1e-8`（约比 `tau` 宽两个数量级，覆盖条件数与舍入）并以 `atol=1e-10*S*` 作为接近零量的下限；该值不是 spec 固定值，必须在正式运行前写入配置并冻结（见 [m5_validation.json](../../cases/m5_validation.json) 中各条目 `thresholds` 字段与 `notes`）。若实测平台高于该值，应分析原因（条件数、压力零模、线性求解下限）并报告，不得为通过而放宽。

### 3.3 物理误差差的操作化（23-08，M-7）

23-08 的“两方法误差差 ≤0.005”按**相对/绝对分别**操作化（拟议，执行前由 A0 冻结）：

- **相对量**（`tip_y`、`drag`，本身为相对误差）：`|err_rel(M) - err_rel(P)| ≤ 0.005`（无量纲，即 0.5 个百分点）；
- **绝对量**（`tip_x`，m）：`|err_abs(M) - err_abs(P)| ≤ 0.005 m`；`lift`（N/m）：`|err_abs(M) - err_abs(P)| ≤ 0.005 N/m`；
- 三方法必须**各自**满足 spec 冻结门槛（tip_y ≤5%、drag ≤5%、tip_x ≤1e-5 m、lift ≤0.1 N/m），误差差判据不替代冻结门槛；
- 任一项超限须写入差异分析报告（原因、量级、是否掩盖耦合误差），不得只报告通过项。

## 4. 成本统计方法学

### 4.1 必录项（spec §5.3 第 4 条）

| 成本项 | 字段建议 | 定义与来源 |
| --- | --- | --- |
| 总耗时 | `total_seconds` | 每方法、每网格档的墙钟时间；拆分“不含网格生成”和“端到端”两个口径 |
| 组装时间 | `assembly_seconds` | 流体 Stokes/Newton 矩阵与残差、固体切线/残差、界面载荷、调和延拓的组装累计 |
| 线性求解时间 | `linear_solve_seconds` | 流体 GMRES+LU 与固体直接求解的累计墙钟 |
| Newton 迭代次数 | `newton_iterations` | 整体 Newton；分区内流体/固体 Newton 分列 |
| 整体法定常推进 | `pseudo_time_steps`、每步 `newton_iterations`、`time_term_norm` | 伪时间 BDF 推进步数、每步 Newton 次数与停机时尺度化时间项收敛值（固定点处 `Σa_j=0`；M-11） |
| 线性求解次数/迭代数 | `linear_solves`, `linear_iterations` | 流体 LU 分解次数与 GMRES 迭代数；固体直接求解次数；真残差检查次数 |
| 耦合迭代次数 | `coupling_iterations` | 分区固定/Aitken 的评估次数（现有 `len(result.history)`） |
| 失败步数 | `failed_steps` / `failed_candidates` | 稳态记为失败候选评估次数；瞬态记为失败时间步（M6 扩展） |
| 物理误差 | `tip_x/tip_y errors`, `drag/lift errors` | 与一致性检查同一状态的输出 |

### 4.2 计时口径

- 计时器：`time.perf_counter()`（单调墙钟）；CPU 时间可选，不作主口径；
- 预热：计时区间内不做额外预热调用；导入/首次编译在计时开始前完成；若执行预热运行，单独记录并说明是否计入；
- 绘图：所有计时区间**不含绘图**；图表在计时结束后从落盘数据重生成；
- 网格生成：单独记为 `mesh_seconds`；比较表同时给出“含/不含网格生成”两列，默认比较口径为不含（三种方法共享同一网格文件时），并在报告中声明；
- 线程：统一 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`（与 M3/T20 运行一致），全部线程环境变量写入报告；不得在比较中途改变；
- 重复：每方法每档 ≥3 次独立运行，报告 median 与 min/max；被丢弃的运行（预热、异常）单独记录；
- 硬件与版本：CPU 型号/核数、内存、OS、Python 与 NumPy/SciPy 版本、代码指纹；同一比较必须同一机器同一会话完成（spec §5.3 第 5 条）。

### 4.3 公平性要求（spec §5.3 第 1–2 条）

- 物理模型、网格、时间积分、时间步、模拟时长、启动条件、输出设置完全固定；
- 各方法达到相近的物理误差；单场容差不能掩盖耦合误差——由 §3 的收紧容差一致性检查保证；
- 残差按变量尺度归一化并记录定义；不同定义的残差数字不得直接比较；
- 三种方法在同一网格档、同一环境、同一输出结构下运行，运行顺序（交错或随机化）记录在报告中。

## 5. 现有 T20 分区结果字段盘点（只读）

数据目录 [results/t20/fsi1/](../../results/t20/fsi1/) 为 Aitken 稳态 FSI1 的结果；本轮只读取字段，不修改。生成代码见 [validation/fsi1.py](../../src/fsi_ice/validation/fsi1.py)。

### 5.1 可用文件与字段

| 文件 | 字段（实测） | 用途 |
| --- | --- | --- |
| [report.json](../../results/t20/fsi1/report.json) | `status`、`checks`（11 项）、`config`（fluid/solid/coupling/solver/reference/acceptance）、`results.levels[]`、`code`（commit/worktree/changes）、`environment`（python/executable/packages）、`definitions` | 配置与验收记录；每档结果 |
| [levels.csv](../../results/t20/fsi1/levels.csv) | `mesh_size`、`fluid_cells`、`solid_cells`、`fluid_dofs`、`solid_dofs`、`interface_edges`、`tip_x`、`tip_y`、`drag`、`lift`、误差列、`inlet_flux`、`outlet_flux`、`mass_balance`、`continuity`、`fluid_residual`、`solid_residual`、`interface_residual`、`coupling_iterations`、`fluid_newton_iterations`、网格质量列、`seconds` | 物理误差、残差、部分工作量、每档总耗时 |
| [iterations_h_*.csv](../../results/t20/fsi1/iterations_h_0.0075.csv) | `iteration`、`residual_norm`、`threshold`、`relaxation`、`relaxation_reason` | 分区耦合迭代史（每档 7–8 行） |
| [solution_h_*.npz](../../results/t20/fsi1/solution_h_0.0075.npz) | `solid_points`、`solid_cells`、`solid_displacement`、`fluid_reference_points`、`fluid_current_points`、`fluid_cells`、`velocity_points`、`fluid_velocity`、`fluid_pressure`、`interface_values`、`interface_residuals` | 全场向量，是逐变量一致性比较的现成分区侧数据（Aitken） |
| `meshes/h_*/` | `fsi1.geo`、`fsi1.msh`、`gmsh.log` | 网格文件与生成日志（可用于哈希与配置指纹） |

### 5.2 成本项可提取性与缺口

| 成本项 | 可从何处提取 | 现状 |
| --- | --- | --- |
| 总耗时 | `levels.csv`/`report.json` 的 `seconds` | **部分可用**：计时起点在 `SteadyFSI1Problem` 构造前，含网格生成与初始化，未拆分工序口径 |
| 组装时间 | — | **缺口**：无任何组装计时 |
| 线性求解时间 | — | **缺口**：无线性求解计时 |
| 耦合迭代次数 | `levels.csv.coupling_iterations`、`iterations_h_*.csv` 行数、`report.levels[].coupling_iterations` | **可用** |
| 流体 Newton 次数 | `levels.csv.fluid_newton_iterations` | **部分可用**：只统计最终流体候选的 Newton 迭代数（`len(nonlinear_history)-1`），非全过程累计 |
| 固体 Newton 次数 | — | **缺口**：`solve_static_continuation` 返回 `histories` 但未落盘，仅记录最终 `solid_residual`（见 [stvk.py](../../src/fsi_ice/solid/stvk.py)） |
| 线性求解次数/迭代数 | — | **缺口**：`ReusedLUSolver.factorizations`/`.iterations` 已计数但未导出（见 [linear_solver.py](../../src/fsi_ice/fluid/linear_solver.py)）；固体直接求解无计数 |
| 失败步数 | `failure.json`（仅失败运行时写入 stage/reason/completed_levels） | **部分可用**：成功运行无失败计数字段；失败候选数不统计 |
| 物理误差 | `levels.csv`、`report.levels[]` | **可用**（tip/drag/lift 及规格误差） |
| 残差 | `levels.csv`、`report.levels[]`、npz `interface_residuals` | **可用**，但整体侧还没有任何数据 |
| 压力规范记录 | — | **缺口**：FSI1 配置无 `pressure_pin` 字段（实现为出口自然边界），报告未显式声明规范 |
| 线程/硬件 | `report.environment` 仅 python/executable/packages | **缺口**：无 CPU、内存、`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS` 记录 |
| 重复次数 | 单次正式运行；重复性证据在 [test_fsi1.py](../../tests/test_fsi1.py)（两次运行除耗时外逐字段一致） | **部分可用**：无多次计时聚合（median/min/max） |
| 固定松弛 FSI1 结果 | — | **缺口**：`results/t20/fsi1` 只有 Aitken；T19 的小型 fixed/Aitken 对比不能替代 FSI1 档比较 |
| 整体法结果 | — | **缺口**：T20 运行时整体法未实现；T21 已交付（见 §7），整体 FSI1 解待 T22 补齐 |

### 5.3 缺口处理建议（T23 执行前）

1. 在 T22/T23 实现中增加计时与计数导出：组装、线性求解、Newton（含固体全过程）、LU/GMRES 计数、失败候选数；
2. 报告增加线程/硬件字段，并在 T23 运行脚本中统一设置线程环境变量；
3. T23 至少补跑分区固定松弛（与 Aitken 同配置、同网格），并生成整体法结果；
4. 以 `results/t20/fsi1` 作为分区 Aitken 侧的只读交叉核对（若 T23 重跑配置不同，以 T23 同次运行数据为比较用，T20 仅作历史基线）。

## 6. 复现脚本与报告模板设计

### 6.1 计划入口（未实现，名称待 A0/T23 冻结）

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m fsi_ice m5-compare --config cases/m5_t23.json --output results/m5/t23
```

备选：扩展现有 `fsi1` 子命令（例如 `--method monolithic|fixed|aitken`）加一个 `scripts/m5_compare.py` 后处理脚本。无论采用哪种入口，必须：统一配置模式、运行开始清理旧输出、失败写 `failure.json`（沿用 T20 约定）、禁止手工交互。

计划配置草案（字段待冻结）：

```json
{
  "case": "m5_t23",
  "mesh_sizes": [0.0075],
  "methods": ["partitioned-fixed", "partitioned-aitken", "monolithic"],
  "tolerance_ladder": [1e-6, 1e-8, 1e-10],
  "repeats": 3,
  "threads": {"openblas": 1, "omp": 1},
  "timing": {"include_mesh_generation": false, "include_plotting": false, "warmup": false},
  "acceptance": {
    "consistency": {"rtol": 1e-8, "atol_scale": 1e-10},
    "fsi1": {"tip_y_relative_error_max": 0.05, "drag_relative_error_max": 0.05,
             "tip_x_absolute_error_max": 1e-5, "lift_absolute_error_max": 0.1}
  }
}
```

### 6.2 数据落盘约定

```text
results/m5/t23/
  config_snapshot.json          # 实际运行配置副本 + 文件哈希
  environment.json             # Python/依赖/线程/硬件/代码指纹
  partitioned-fixed/h_<h>/     # report.json, levels.csv, iterations_*.csv, timing.json, solution.npz
  partitioned-aitken/h_<h>/    # 同上
  monolithic/h_<h>/            # 同上
  comparison/
    consistency_per_variable.csv   # 变量、max|Delta|、尺度、判据、通过与否、容差档
    cost_per_item.csv              # 成本项 x 方法 x 重复（median/min/max）
    summary.json                   # 结论、门槛、代码/配置指纹
```

`results/` 目录按仓库策略不纳入源码提交；报告中必须保留可重生成的命令、配置与指纹（[spec.md](../../spec.md) §6.2）。

### 6.3 报告模板字段

- 元信息：case、方法、状态（PASS/FAIL）、时间戳、命令、代码 commit + 工作区状态、配置哈希；
- 离散标识：网格文件哈希/h、材料、边界、压力规范、启动方式；
- 收敛：容差档、归一化残差定义、迭代史、最终残差（单场/界面/整体）；
- 物理量：tip A 位移、drag、lift 及对照误差；
- 一致性表：逐变量 `max|Delta|`、`max(|a|,|b|)`、判据、容差、通过与否；
- 成本表：§4.1 每一项 × 方法，median/min/max 与口径说明；
- 备注：预热/绘图/网格生成是否计入、失败候选、警告与已知限制。

## 7. 计划与已执行的区分

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| 本文档 | 已交付（2026-10-11） | 方法学与验收骨架，仅文档 |
| [m5.md](../validation/m5.md)、[m5_validation.json](../../cases/m5_validation.json) | 已交付并更新至 r1（2026-10-11） | T21 五条 `passed`（独立验证），其余 15 条 `not_executed` |
| T21 整体残差/Jacobian | 已交付并独立验证通过（2026-10-11） | `src/fsi_ice/coupling/monolithic.py`、`src/fsi_ice/validation/monolithic.py`、`tests/test_monolithic_residual.py`；独立复核 [t21_review.md](../validation/t21_review.md) |
| T22 整体 Newton | 实现/集成进行中（2026-10-11，A6 并行） | 依赖 T21（已交付）；拟名 `src/fsi_ice/coupling/monolithic_newton.py`、`tests/test_monolithic_newton.py` |
| T23 一致性比较 | 未执行 | 依赖 T22；本轮不运行任何求解 |
| 成本统计运行 | 未执行 | 缺口见 §5.2，待 T23 实现补数；整体法成本口径按 M-11 记录步数/Newton 次数/时间项收敛值 |
| 本文档 r1 的实际工作 | 仅文档修正（M-1/M-3/M-6/M-7/M-8/M-9/M-11） | 未运行整体/分区求解；T21 独立验证的实测见 [t21_review.md](../validation/t21_review.md) |

本文档 r1 修正仅文档编辑，未运行整体/分区求解，未改动 `results/` 与他人文件；T23 相关求解结论保持“计划/未执行”状态（T21 的独立验证结论以 [t21_review.md](../validation/t21_review.md) 为准）。

## 8. 参考资料

- [spec.md](../../spec.md) §5.2（数值验收）、§5.3（算法比较）、§3.3（耦合约束）；
- [gola.md](../../gola.md) M5 阶段目标、[agent.md](../../agent.md) §3 M5 任务表；
- [T20 验收记录](../validation/t20.md)、[T20 计划](fsi1_plan.md)、[T20 配置](../../cases/fsi1.json)；
- [m3_freeze.md](m3_freeze.md) §6（方向导数 1e-5、重试-基线 1e-10 等既有冻结门槛）、[m3.md](../validation/m3.md)（尺度与证据格式）、[interface_plan.md](interface_plan.md) §9；
- [validation_plan.md](validation_plan.md)（验证方法学）、[environment.md](../environment.md)（环境记录）、[原理说明维护约定](../principles/维护说明.md)（资料链接要求）。

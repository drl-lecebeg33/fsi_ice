# M5 接口冻结与协调裁决（A0）

日期：2026-10-11。适用范围：M5/T21–T23 的实现、集成与验收。本文冻结 T21 实际交付的接口与 M5 推进中的协调裁决；与各任务计划文档冲突时，以本文为准，变更须记录理由。

状态：T21 实现已交付并自测通过（见 [t21.md](../validation/t21.md)）；独立验证与 T22 集成进行中；T23 尚未执行。

## 1. 冻结接口（T21 实际交付）

文件：`src/fsi_ice/coupling/monolithic.py`、`src/fsi_ice/validation/monolithic.py`、`tests/test_monolithic_residual.py`。

- 未知量顺序：`x = [u_f, p_f, d_s, v_s, d_m, g]`；二维节点量节点优先交错；`g` 为流体 P2 界面迹对偶节点力（顺序同 `assemble_interface_load`，厚度只计一次，符号为流体对固体）。
- 入口：
  - `monolithic_residual(x, *, context) -> np.ndarray`
  - `monolithic_jacobian(x, *, context) -> scipy.sparse.csr_matrix`
  - `monolithic_jvp(x, directions, *, context) -> np.ndarray`
  - `evaluate_monolithic(x, *, context, mode="residual"|"assembled"|"jvp", directions=None) -> MonolithicEvaluation`
  - **没有 `mode="full"`**；`MonolithicEvaluation` 只读暴露 `residual`、可选 `jacobian`/`jvp_values`、`candidate_id`、逐块范数、诊断与 `timings`（`total_seconds` 为整次求值累计，M-4 已修复）。
- `BlockLayout`、`MonolithicContext`、`PressurePin`（可选，`pressure_pin=None` 合法）在 `monolithic.py`；`R(x)` 是 `x` 的纯函数（Scheme A：候选几何/界面几何每次由 `x` 现场构造）。
- 拒绝：stale/mixed `candidate_id`、`stencil.step_id` 与 base generation 不一致、翻转/质量/拓扑不合法、混合版本缓存均拒绝；输出只读、输入不可变、重复求值逐位可重复（可调用体力中心差分除外）。
- 边界取值：流体 Dirichlet 速度在候选 P2 坐标取值；网格与固体 Dirichlet 值在参考坐标取值（调用方与单场算例一致）。
- 性能参考（真实 FSI1 h=0.02，n=12905）：残差 0.27 s/次；JVP 0.69 s/次；组装 Jacobian 19.43 s/次。最细网格组装成本更高。

## 2. 协调裁决（交叉复核 M-1…M-11 的处置）

| 项 | 裁决 | 归属 |
| --- | --- | --- |
| M-3 压力规范 | FSI1 一致性比较两侧均用 `pressure_pin=None`（与 T20 分区一致，出口自然边界定压）；`PressurePin` 保持可选；若未来某算例改用 pin，两侧必须使用同一 pin 定义并在报告记录 | A0 冻结；A7 更正 T23 文档 |
| M-11 稳态离散 | T20 分区为稳态单场；T21 残差为 BDF 步格式。整体法定常解由**伪时间 BDF 推进至固定点**取得：固定点处 BDF 时间项精确抵消（Σa_j=0），是稳态离散方程的解。T22 实现推进驱动与停机判据（以尺度化时间项 `‖a₀Δx‖_S/…` 控制），T23 §2.2 按此更正；成本口径记录步数、Newton 次数与时间项收敛值 | A0 裁决；A6 实现；A7 更正文档 |
| M-2 block_norms | 仅作逐块监视量；Newton 收敛判据由 T22 自算 `max_i |R_i|/S_i ≤ atol + rtol·‖R0‖_S` | A6 |
| M-4 计时字段 | 已修复（`total_seconds` 整次累计，数值未变） | 完成 |
| M-5 适配器 | `residual` 模式 `jacobian=None` 属正常；Newton 在需要切线的接受点调用 `assembled`（或后续 Krylov 用 `jvp`）；算法选择与耗时计入成本，不允许静默切换 | A6 |
| M-1 tau 定义 | 各方法残差定义不同（分区界面残差 vs 整体归一化残差），记录定义、不直接比较不同定义的数字 | A7 |
| M-6 T23 变量表 | 补 `v_s`、`g` 行 | A7 |
| M-7 23-08 误差差 | `≤0.005` 按相对/绝对分别操作化并写明 | A7 |
| M-8 收紧裕度 | 执行 1e-10 档前先验证单场容差与迭代上限裕度，报告实测 | A7（执行期） |
| M-9 文件命名 | 统一为 `tests/test_monolithic_residual.py`；文档中 `monolithic/` 改为 `monolithic.py` | A7 |
| M-10 spec 核对 | 无冲突，保持 | — |

T22/T23 测试文件拟名：`tests/test_monolithic_newton.py`、`tests/test_m5_consistency.py`；结果目录 `results/m5/t21|t22|t23/`（不入库）。

## 3. 本轮文件归属（并行执行）

| 执行者 | 可写文件 |
| --- | --- |
| T21/T22 实现（A6，同一终端） | `src/fsi_ice/coupling/monolithic_newton.py`（拟）、`src/fsi_ice/coupling/newton.py`（仅向后兼容扩展）、`coupling/__init__.py`（仅追加导出）、`tests/test_monolithic_newton.py`、`tests/test_newton.py`（仅追加）、`docs/validation/t22.md`、`results/m5/t22/`。**T21 已交付文件（`monolithic.py`、`validation/monolithic.py`、`tests/test_monolithic_residual.py`、`docs/validation/t21.md`）在独立验证期间冻结**，如需改动须先经 A0 批准并说明理由 |
| A7 独立验证与文档（新终端） | `tests/test_monolithic_review.py`、`docs/validation/t21_review.md`、`docs/validation/m5.md`、`cases/m5_validation.json`、`docs/development/m5_t23_plan.md` |
| A0（协调者） | `docs/development/m5_freeze.md`（本文）、`agent.md`、`gola.md`、`spec.md`、`README.md`、CLI、`cases/fsi1.json` 等公共文件 |

## 4. 验收与一致性口径要点

- T21 门槛：非退化方向最佳相对误差 ≤1e-5（实测 7.39e-11，混合六块）；代数恒等 ≤1e-12·S；可调用体力近似显式标注。
- T22 门槛：小问题 Newton 收敛与失败诊断；定常推进固定点与 dt 无关；工作量计数可导出。
- T23 门槛：分区/整体在收紧容差后得到一致离散解（逐变量拟议 `rtol=1e-8`、`atol=1e-10·S*`，执行前由 A0 冻结）；成本统计完整且计时口径固定（`.venv`、`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`）。
- 所有阈值不得为通过而放宽；未执行条目保持 `not_executed`，证据只在本地实际运行后更新。

## 5. 更新记录

- 2026-10-11 r0：建立冻结（T21 交付事实 + M-1…M-11 裁决 + 文件归属）。
- 2026-10-12 r1：M5 执行结果确认与容差冻结。**T23 比较容差正式冻结**为 `rtol=1e-8`、`atol=1e-10·S*`（与已执行运行一致，未调整）；收紧阶梯 {1e-6, 1e-8, 1e-10}；成本口径按 [m5_t23_plan.md](m5_t23_plan.md) §4.2。**23-08** 在比较网格 h=0.02 的字面门槛下记 `failed`（tip_y/lift 为网格分辨率限制，三方法一致且与 T20 h=0.02 基线逐位一致；FSI1 最细档物理门槛以 T20 h=0.0075 证据为准），门槛值未放宽。**22-04** 记 `not_executed`（驱动层不变量已验证，`StepTransaction` 衔接未实现）。T21/T23 其余条目均已本地验证。整体法定常推进采用 `dt=1.0`、BDF2、`jacobian_mode="assembled"`；`jvp` 路由因无预条件 GMRES 不收敛而未用于成本比较（未静默切换）。

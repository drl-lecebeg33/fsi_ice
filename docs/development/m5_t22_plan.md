# M5/T22 前置：整体 Newton 设计与通用阻尼 Newton 原型

日期：2026-10-11（同日完成 T21 契约对齐与 T21/T22/T23 交叉复核）。执行者：A6（单一执行者，
同时承担设计与原型实现自测；不声称独立复核）。
前置：T21（整体残差与完整耦合 Jacobian）由另一执行者并行开发；本任务不导入、不修改
`src/fsi_ice/coupling/monolithic.py`，只按冻结契约书写并在 T22 集成时对接。
状态：设计完成；自包含原型 `src/fsi_ice/coupling/newton.py` 已实现并通过玩具系统测试；
**T22 集成、真实评估器适配、事务衔接、CLI 与 FSI 数值验收均未开始**。
契约对齐（2026-10-11）：§1/§2/§3/§6/§7 已按协调者冻结的 T21 要点更新（`x` 顺序、
入口 API 与 `mode` 命名、只读语义、`candidate_id`、压力 pin 与 Dirichlet 行约定）；
`newton.py` 与 `tests/test_newton.py` 保持已交付状态未改动。交叉复核结论见 §10。

## 0. 范围与「设计 vs 已实现原型」

| 内容 | 本任务状态 | 留待 T22 |
| --- | --- | --- |
| 纯评估协议（duck typing `evaluate(x) -> Evaluation`） | 已实现并测试 | 绑定真实 `evaluate_monolithic`（见 §1） |
| 逐分量归一化 ∞-范数收敛判据 | 已实现并测试 | 用配置尺度实例化；T21 `block_norms` 仅作监视 |
| 阻尼/回溯线搜索、非有限试探拒绝 | 已实现并测试 | 真实系统上的参数整定 |
| 直接法 `splu`、奇异/非有限方向检测 | 已实现并测试 | 规模评估、Krylov/预条件 |
| 失败异常 `NewtonFailure`（阶段/迭代/范数/诊断） | 已实现并测试 | 写入事务失败审计 |
| 只读结果、无副作用、可重复 | 已实现并测试 | 真实评估器不变量检查 |
| 块稀疏组装入口 | 仅设计 | T21 以 `mode="assembled"` 输出 CSR，Newton 不自行组装 |
| 与 `time/transaction.py` 的候选/发布衔接 | 仅设计 | A0 决策接口后实现 |
| 计时桶与 FSI 统计 | 仅计数（评估/分解/试探） | `perf_counter`、Krylov 迭代数、字段名对齐 T23 |
| 压力规范 / Dirichlet 行约定 | 仅设计（约定归 T21） | 冻结：`PressurePin` + 残差行/单位行（见 §7） |
| FSI1 整体解、与分区一致性、成本比较 | 不在本任务 | T22/T23 验收 |

## 1. 与 T21 的评估契约（2026-10-11 冻结要点）

未知量顺序（冻结）：`x = [u_f, p_f, d_s, v_s, d_m, g]`

- `u_f`：流体 P2 速度 `(n_P2, 2)`；`p_f`：流体 P1 压力 `(n_P1,)`；
- `d_s`/`v_s`：固体参考构形 P1 位移/速度 `(n_solid, 2)`；
- `d_m`：流体参考网格 P1 位移 `(n_f1, 2)`；
- `g`：流体 P2 界面迹对偶节点力 `(n_fG, 2)`，厚度只计一次，符号为流体作用于固体。
- 二维节点块按 `vector_dofs` 的 `(node-x, node-y, …)` 交错顺序；块布局/切片由
  `BlockLayout`（`BLOCK_NAMES` 顺序、`slice_of/gather/unpack`）提供。

入口 API（冻结）：

```python
monolithic_residual(x, *, context)                  # -> 只读 residual
monolithic_jacobian(x, *, context)                  # -> 冻结 CSR
monolithic_jvp(x, directions, *, context)           # -> 方向导数（(n,) 或 (batch, n)）
evaluate_monolithic(x, *, context, mode="residual"|"assembled"|"jvp", directions=None)
    -> MonolithicEvaluation
```

- `mode="residual"`：残差 + 候选/物理量；`mode="assembled"`：残差 + 完整耦合 Jacobian；
  `mode="jvp"`：残差 + 指定方向的矩阵自由方向导数（`directions` 仅此模式可用）。
- `MonolithicEvaluation` 只读暴露：`residual`；可选 `jacobian`（CSR，`data/indices/indptr`
  冻结）或 `jvp_values`；`candidate_id`；`block_norms`（逐块监视）；`geometry_diagnostics`；
  `timings`。评估不修改 `x`、context、已接受历史、参考网格或拓扑，重复调用可逐位重现。
- `block_norms` 的定义是 `||R_b||₂ / max(||x_b||₂, 1)`（状态相对监视量），**不是** T22 的
  收敛判据，也不能替代逐变量物理尺度归一化；T22 用 `context.layout` 的切片与配置尺度自行
  计算 `max_i |R_i|/S_i`（见 §3）。
- `context: MonolithicContext` 由调用方（T22）构造，冻结布局、已接受 `MonolithicBase`、
  BDF `stencil`、拓扑、调和延拓、`candidate_id`、厚度与压力约束。`candidate_id` 必须可哈希，
  并随候选几何/界面几何/载荷传递；`require_candidate` 拒绝陈旧或混用版本；
  `base.generation` 必须等于 `stencil.step_id`，否则 context 构造即拒绝（陈旧候选防护）。
- 纯评估不变量：同一 `x` 重复调用给出相同残差/Jacobian；允许内部缓存，但不得改变可观察值。
  原型在每次调用时复制所有读到的数组，评估器可复用内部缓冲区。
- 原型协议只要求 `evaluate(x) -> Evaluation`（`residual` + `jacobian` 属性）；T22 用闭包或
  `functools.partial` 绑定 `context` 与 `mode`。**注意**：`solve_newton` 只在接受点读取
  `.jacobian`，但该对象必须存在；线搜索用 `mode="residual"` 时，接受点需要在适配器内补一次
  `mode="assembled"` 评估（或返回惰性属性的包装对象），详见 §9 第 1 条。

## 2. 整体 Newton 迭代框架

以 `r(x) = 0` 为目标，消费上述入口（T22 绑定 `mode`）：

```text
x ← x0（T22：上一步 predictor 或上一 Newton 解）
r0 ← evaluate(x, mode="assembled").residual（同时取得 J）
阈值 τ ← atol + rtol · ‖r0‖_S
记录 iteration 0
若 ‖r0‖_S ≤ τ：直接收敛返回
循环 k = 1 … max_iterations：
    d ← -J⁻¹ r（J 来自最近一个接受点的 assembled 评估；直接法或 Krylov）
    α ← 回溯线搜索（试探用 mode="residual"，非有限立即拒绝）
    x ← x + α d；记录阻尼、试探范数与统计
    若 ‖r(x)‖_S ≤ τ：返回成功
    J ← 接受点的 assembled 评估（读后立即复制）
超过 max_iterations 仍未收敛：抛 NewtonFailure(phase="convergence")
```

每次 Newton 迭代只组装/求值一次 Jacobian；线搜索只消费残差。原型
`solve_newton(evaluate, x0, *, options=None, scales=None)` 完整实现了该流程，真实系统中
`evaluate` 由 T22 适配；初值预测、几何重建与失败重试由 T22 的事务层负责。

## 3. 逐块残差归一化与收敛判据

- 尺度：调用者提供与 `x` 等长、严格正、有限的逐分量物理尺度 `S`（位移 m、速度 m/s、
  压力 Pa、网格位移 m、力/动量残差按对应单位）。原型默认 `S = 1`。
- 收敛范数：`‖r‖_S = max_i |r_i| / S_i`，判据 `‖r‖_S ≤ atol + rtol · ‖r0‖_S`；
  `r0` 是本次 Newton 起点的残差。`atol` 主导零残差/退化分量，`rtol` 主导相对下降；
  两者无量纲、非负且不得同时为零（`NewtonOptions` 校验）。
- 记录：每次接受迭代同时记录归一化 ∞-范数、原始 ∞-范数；每个试探记录归一化 ∞-范数
  与原始 2-范数（见 §5）。
- 分块层级（设计，T22）：用 `context.layout` 的块切片对 `residual` 计算逐块归一化范数
  `max_i |R_b,i| / S_b` 作为诊断；T21 的 `block_norms`（`||R_b||₂/max(||x_b||₂,1)`）只作
  状态相对监视，不与收敛判据混用。块级指标只报告不独立判停，避免单一物理块提前满足而
  整体未平衡。物理不变量（界面运动连续、压力规范、正 Jacobian）由 T21/T22 的检查项判定，
  不属于通用 Newton 层。
- 线搜索 merit 与收敛范数**分开**（原型已实现并在测试中固定）：Armijo 用**原始残差
  2-范数**，因为它对 Newton 方向在任意尺度比下都保持小步下降；若改用逐分量归一化
  ∞-范数，当尺度相差多个量级时，主分量二次下降而小量级分量穿越零点会污染 merit，
  导致步长被误拒并缓慢爬行（回归：`test_scales_preserve_nonlinear_solution`，改前
  50 次迭代未收敛，改后与未缩放问题收敛到同一解）。

## 4. 线性求解

- 组装入口：Jacobian 由 T21 统一组装（`mode="assembled"`）为 CSR 块稀疏矩阵，W1 的
  `monolithic.py` 当前实现满足；Newton 层只做尺寸/有限性检查、CSR 拷贝与分解，不重复
  组装物理块，也不修改传入矩阵。
- 默认直接法 `splu`：二维问题在 Python 原型规模（约 10⁴–10⁵ DOF）下稳定、可预测；
  T22 可评估符号分解复用（稀疏结构不变时），但 Newton 每步数值不同，复用收益需实测。
- Krylov（设计，T22）：问题规模或内存超限时改用 GMRES/BiCGStab，记录迭代数与真实
  残差；预条件建议块对角/场分裂（单场 Jacobian 的 `splu` 或物理近似），避免无预条件
  的原始 Krylov。切换条件是配置项，不允许静默改变算法。
- 回退策略（设计）：Krylov 不收敛 → 记录诊断 → 回退直接法；直接法奇异 → 失败，不静默
  用拟牛顿或伪逆替代。原型只实现直接法；选择/回退策略留待 T22 实测后冻结。
- 检测（原型已实现）：`J.data` 非有限、`splu` 报“exactly singular”、方向非有限 →
  `NewtonFailure(phase="linear_solve")`，诊断含 `solver`、`nnz`、错误类型/信息。
  T22 追加：逐行/逐列非零计数以定位缺失约束行；与 T21 声明的约束元数据比对。
- 缺失约束/奇异来源（设计）：压力 gauge 未固定、Dirichlet 行未消元或未置单位行、
  界面约束重复/冲突、网格退化。约定见 §7；Newton 层只检测并报告，不代为修正。

## 5. 全局化与失败处理

- 阻尼回溯：`α = 1` 起，Armijo 条件 `φ(x+αd) ≤ (1 − c·α) φ(x)`，`c = 1e-4`，
  `α ← α/2`；`φ` 为原始残差 2-范数。原型参数：`max_backtracks=30`、
  `backtrack_factor=0.5`、`minimum_damping=2⁻²⁴`；`line_search=False` 时只试全步。
- 试探拒绝：试探点/残差非有限（NaN/Inf）→ 立即拒绝并记录 `inf`，继续缩小 `α`；
  全部试探失败 → `NewtonFailure(phase="line_search")`，诊断含每个试探的归一化范数与
  merit。原型对试探评估异常采取硬失败（避免掩盖真实错误）；T22 若确认评估器在域外
  抛错属正常，可把异常降级为“拒绝并记录”，此策略需 A0 冻结。
- 最大迭代：`max_iterations` 用尽未达阈值 → 失败（原型：`phase="convergence"`），
  携带最后接受点、完整迭代历史与阈值。不静默返回“近似解”。
- 已知限制：若迭代收敛到 `‖r‖` 的局部极值而 Jacobian 趋于零，回溯最终失败并如实报告
  （回归测试 `test_stationary_point_stalls_with_line_search_failure`）；重启动/信赖域不在
  第一版默认路径。

## 6. 与 `time/transaction.py` 的衔接（只读研究结论与所需接口点）

`StepTransaction` 的生命周期是 `begin → evaluate_* → validate → prepare → publish | abort`：

- 单写者：`StateCoordinator.begin(plan)` 授予唯一写权限并分配进程内唯一 `attempt_id`；
  失败记录追加在 root 之外，`abort` 后必须走新的 `begin`（新 `attempt_id`），旧 context/
  snapshot 不可重放（anti-ABA）。
- `prepare` 内完成全部可失败操作（三个私有 BDF accept、监测行冻结、构造新 root）；
  `publish` 单次替换 live root。候选阶段永远不能触碰已接受历史。
- 候选评估的 plan 回调收到的 `StepView` 是只读副本；回调抛出的异常进入
  `_fail(stage, field, reason, getattr(error, "diagnostics", None))`，其中 `diagnostics`
  会被冻结写入 `FailureRecord`。原型的 `NewtonFailure.diagnostics` 已是普通映射
  （已实测可被 `time.transaction._freeze_diagnostics` 接受），因此 T22 可直接从 plan
  回调抛出该异常而无需包装。

衔接约定（设计）：

1. **candidate_id**：每个 Newton 试探属于该 attempt 的候选，记
   `candidate_id = (attempt_id, iteration, trial)`，并要求可哈希。T22 用它在每个试探重建
   `MonolithicContext`（布局/已接受 base/BDF stencil/拓扑/调和延拓/厚度/压力约束不变，
   仅 `candidate_id` 与几何状态随试探变化）；该 id 随候选几何、界面几何与载荷传递，
   `require_candidate` 拒绝陈旧或混用版本。它与事务内的几何 revision `(attempt_id, 递增
   计数)` 同源但维度不同；不写入已接受状态。
2. **绝不 accept/publish**：Newton 循环期间不得调用 `history.accept`、`prepare`、
   `publish` 或 `StateCoordinator` 的写入口；只使用事务工作区内的候选数组、几何与
   BDF context。收敛后由事务一次性 `validate → prepare → publish`。
3. **失败恢复**：任一 `NewtonFailure` 使 plan 回调失败 → 事务 `abort` 并记录
   `FailureRecord`（stage 建议 `"monolithic"`，field 用块名或 `None`）。时间步失败不
   发布，重试必须新 attempt、从相同已接受根重建全部候选；跨时间步恢复使用
   `StateCoordinator.snapshot/restore`（快照过期即拒绝）。
4. **收敛后一次性提交**：Newton 收敛解作为该 attempt 唯一候选；`validate` 复核有限性并
   运行物理检查；`prepare` 私有 accept + 冻结新 root；`publish` 单次替换。输出写出在
   发布之后幂等执行（`write_output`/`flush_output`）。
5. **所需接口点与假设**（T22 需 A0 冻结，本任务不改 `transaction.py`）：
   - `StepPlan` 目前没有“整体耦合求解”阶段。推荐 T22 扩展一个耦合评估阶段（例如
     `solve_coupled`/`evaluate_monolithic`，返回几何、固体、流体与界面候选），Newton
     循环位于该回调内部；或经 A0 同意后把 Newton 放进现有 `solve_*` 组合。两种方案都
     必须保持“候选只读、一次提交”的事务语义。
   - 需要读取：`attempt_id`、`mesh_context`/`solid_context`/`fluid_context`、`base`
     的只读形状与已接受数组、`StepView` 的只读候选几何接口。构造 `MonolithicBase` 时：
     `solid_states` 取 `AcceptedState.solid_history`（`(2,n,2)` 的 `(d,v)` 对），
     `fluid_states` 取 `fluid_history`，`generation` 取 `solid_context.stencil.step_id`
     （T21 要求 `base.generation == stencil.step_id`），`accepted_geometries` 由 T22 用
     同一 ALE 构建入口对每层已接受位移重建。
   - 需要 T21 提供：纯评估入口与 `mode` 语义、CSR Jacobian、`BlockLayout` 块切片、
     `block_norms`/`geometry_diagnostics`/`timings` 诊断、`PressurePin` 与约束行约定。
   - 假设：候选几何可在同一 attempt 内按 Newton 试探重复构建且不落盘；重复构建失败
     按 NewtonFailure 记录。该假设需 T22 用真实 ALE/界面构建路径验证。

## 7. 诊断、计时与统计；压力规范与 Dirichlet 行约定

失败诊断字段（原型 `NewtonFailure` 已实现）：

- `phase`：`evaluate` | `residual` | `jacobian` | `linear_solve` | `line_search` |
  `convergence`；`iteration`；`reason`；`residual_norm`（归一化）/`raw_residual_norm`；
  `x`（最后接受点只读副本）；`history`（已接受迭代记录）；`diagnostics`（普通映射，
  含试探范数/merit、solver/nnz、错误类型与信息等）。
- T22 追加诊断：`attempt_id`、`candidate_id`、逐块归一化范数、线性求解器类型与
  Krylov 迭代数、分解/回代耗时、几何质量（最小面积比/翻转）、约束行检查结果。

计时与统计（设计；原型只提供计数）：

- 计时桶：`t_evaluate`（残差/Jacobian 组装）、`t_linear`（分解+回代或 Krylov）、
  `t_line_search`、`t_newton`（整体）、`t_step`（含事务）；失败步单独计数。
- 计数器：Newton 迭代数、评估次数、Jacobian 分解次数、线性迭代数、回溯试探数、
  被拒绝试探数、失败步数。原型 `NewtonResult` 已给出 `evaluations`、`linear_solves`、
  `line_search_trials`；每步试探数/阻尼在 `NewtonIterationRecord` 中。
- 报告口径与 `spec.md` §5.3 一致：残差定义与尺度必须随数据记录，不同定义的残差数字
  不可直接比较。

压力规范与 Dirichlet 行（2026-10-11 冻结要点，Newton 层只依赖结果）：

- 压力 gauge（首版）：`PressurePin(node, value)` 替换一个 P1 压力方程，残差行
  `p_node − value`、Jacobian 行为单位行 `e_i`；Newton 层不对 `p` 做投影或再规范化，否则
  破坏残差判据。与分区法比较前必须先对齐规范（见 §10 复核项 M-3）。
- Dirichlet 行（含固体夹持）：冻结为“残差行保持已知值 + Jacobian 行置单位行”；湿界面行
  优先于 Dirichlet 行（`BlockLayout` 构建时自动剔除重叠节点并报告）；固体位移/速度夹持分别
  用 `solid_displacement_dirichlet_dofs`/`solid_velocity_dirichlet_dofs` 单位行。
- Newton 不重复施加边界条件，只检查（非有限、零行/零列、约束行残差在 x0 是否已满足）；
  约束行必须在每个试探点可满足，否则线搜索会因约束残差无法下降而失败——列为 T22 预检查项。

## 8. 原型实现与验证记录（本任务实测）

实现：`src/fsi_ice/coupling/newton.py`（自包含，`import` 中不含 `fsi_ice.coupling.monolithic`）。
公开 API：`Evaluation`/`Evaluator` 协议、`NewtonOptions`、`NewtonIterationRecord`、
`NewtonResult`、`NewtonFailure`、`solve_newton(evaluate, x0, *, options=None, scales=None)`。
行为：默认逐分量尺度 → ∞-范数判据 `atol + rtol·‖r0‖`；Armijo 回溯（原始 2-范数 merit）；
非有限试探立即拒绝；`splu` 直接法与奇异/非有限检测；结果只读、`x0` 与评估器缓冲区不被
修改；重复调用逐位一致。

运行命令与环境：

```bash
cd /home/iceberg/桌面/tmp/fsi_ice
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_newton.py -q
```

实测结果（2026-10-11，Python 3.12.3、NumPy 2.5.3、SciPy 1.18.1，项目 `.venv`）：

| 项目 | 实测 |
| --- | --- |
| 测试数量/耗时 | **44 passed in 0.35–0.38 s**（多次重复运行一致；无警告） |
| 二维非线性方程组 | 从 (0.2, 0.7) 5 次迭代收敛到根 (0, 1)，归一化范数 3.10e-1 → 1.75e-13 |
| 超线性/二次趋势 | 尾部满足 `r_{k+1} ≤ 10·r_k²`（最近两步），总范数下降 ≥ 1e-10 |
| Rosenbrock 梯度系统 | 从 (-1.2, 1.0) 收敛到 (1, 1)（≤500 次上限，实测约 149 次迭代） |
| 发散方向 + 线搜索 | arctan 系统 x0=1.5：全步被拒（首个试探范数 1.037 > 初始 0.983），α=0.5 接受，4 次迭代收敛 |
| 未阻尼循环 | x³-2x+2 全步在 0↔1 循环，10 次迭代后 `phase="convergence"` 失败 |
| 奇异 Jacobian | 秩亏 J → `phase="linear_solve"`，诊断 `solver="splu"`、`"singular"` |
| 非有限残差 | 初始 NaN → `phase="residual"`、`residual_norm=inf`；试探 NaN 被拒后缩小步长成功 |
| 尺度只改判据 | 线性系统：阈值 1 vs 1000，解逐位相同；非线性系统：接受试探序列与 merit 逐位相同，解与未缩放一致 |
| 只读/无副作用/重复 | 冻结结果与只读数组、回调输入改动不影响求解器、缓冲复用评估器两次运行逐位一致 |

未做的检查（如实声明）：未执行任何 FSI 物理或整体耦合求解；未与 `monolithic.py`、
`time/transaction.py`、CLI、配置或参考基准集成；原型数值仅代表玩具系统。

附加检查：导入 `fsi_ice.coupling.newton` 后 `sys.modules` 不含
`fsi_ice.coupling.monolithic`；相关回归 `tests/test_partitioned.py` +
`tests/test_transaction.py` 共 **114 passed in 8.73 s**，确认新增模块对既有分区耦合与
事务测试没有影响。全量回归未在本任务内运行：工作区同时存在 T21 的未提交改动，避免把
并行工作区的中间状态计入本任务结论。

## 9. 留待 T22 的集成工作

1. **真实评估器适配**：把 `evaluate_monolithic(x, *, context=..., mode=...)` 绑定为
   `evaluate(x)`。原型只在初始点（未立即收敛时）与接受点读取 `.jacobian`，但该对象必须
   存在；`mode="residual"` 的评估 `jacobian=None`，直接传入会以 `phase="jacobian"` 失败。
   适配器二选一：(a) 每次试探都用 `mode="assembled"`（简单，被拒试探也付组装费）；
   (b) 惰性包装：试探用 `mode="residual"`，`.jacobian` 被访问时对同一 `x` 补一次
   `mode="assembled"`（接受点两次评估、被拒一次），并保持纯评估与拷贝语义。该取舍计入
   23-09 成本口径，建议 T22 实测后冻结（见 §10 M-5）。
2. **契约对齐状态**：§1/§2/§6/§7 已按 2026-10-11 冻结要点更新；仍需在 T21 交付版上复核
   实际字段/异常类型是否与文字一致（W1 编辑中，复核快照见 §10 开头）。
3. **事务衔接**：与 A0 冻结 `StepPlan` 的整体耦合阶段与 `stage/field` 命名；把
   `NewtonFailure.diagnostics` 写入 `FailureRecord`；确认“候选绝不 accept/publish，
   收敛后一次 prepare/publish”以及失败重试必须新 attempt。
4. **线性求解扩展**：块对角/场分裂预条件 Krylov、直接法/迭代法回退、符号分解复用评估
   与实测选择；把求解器信息写入诊断。
5. **诊断与计时落地**：逐块归一化范数（`max|R_b|/S_b`）、candidate_id、几何质量、失败候选
   计数、`perf_counter` 计时桶；字段名与 T23 §4.1 对齐（`total_seconds`、`assembly_seconds`、
   `linear_solve_seconds`、`newton_iterations`、`linear_solves`、`linear_iterations`、
   `coupling_iterations`、`failed_steps`/`failed_candidates`），并接入 CLI/配置与验证报告。
   注意 T21 `timings` 提供的是评估级分项（`candidate/physics/residual/jacobian_seconds`），
   需要 T22/T23 冻结“组装时间”的聚合口径（见 §10 M-4）。
6. **数值验收（T22/T23）**：整体残差收敛、与分区法收紧容差后的一致解、非有限/奇异/最大
   迭代/事务恢复的故障注入、FSI1 成本比较；未验证前不得标记完成。

## 10. T21/T22/T23 契约交叉复核（2026-10-11，只读）

复核快照：`src/fsi_ice/coupling/monolithic.py`（起始 sha256 `0fda3d60…`；复核结束时已更新为
`51b99e10…`，M-4 在该版本仍复现，其余条目以行为为准）、
`src/fsi_ice/validation/monolithic.py`（sha256 `4aa76eca…`，232 行）、W3 的
`m5_t23_plan.md` / `m5.md` / `m5_validation.json`（20 条全部 `not_executed`）、
`coupling/partitioned.py`、`cases/fsi1.json`、`time/bdf.py`、`fluid/ale.py`、
`validation/fsi1.py`、`spec.md` §5.2/§5.3。本复核不修改被复核文件；W1 的代码在该快照
之后仍可能变化。

### 10.1 复核结论（对应四个问题）

1. **一致性判据可提供、可复核（有条件）**：T21 只读 `residual`/`jacobian`、`BlockLayout`
   切片、`candidate_id`、`geometry_diagnostics` 与 T22 的 `scales` 判据足以支撑 T23 §3.1
   的逐变量判据；缺口见 M-2/M-3/M-6/M-11。
2. **成本统计可对齐（需冻结字段名）**：T21 评估级 `timings` + T22 计数 +
   事务 `FailureRecord` 可覆盖 T23 §4.1 必录项；需统一“组装时间”的聚合口径（M-4）、
   `linear_iterations`（splu 直解为 0）与 `failed_candidates` 的定义（M-5 同时影响成本口径）。
3. **收紧序列条件可行**：{1e-6, 1e-8, 1e-10} 对分区法可表达（`CouplingOptions` 无下限），
   但需同步收紧单场容差、提高 `max_iterations`，并先验证最后一档与线性认证下限（≤1e-10）
   的裕度（M-8）。
4. **spec 冲突**：无直接冲突；需修正“跨方法同一 tau”措辞（M-1）与 23-08 拟议值的操作化
   （M-7）；M-3 与 M-11 为 T23/T22 执行前必须冻结的方法学项。

### 10.2 发现清单

| 编号 | 严重度 | 发现与建议 | 建议处理者 |
| --- | --- | --- | --- |
| M-1 | 建议 | 收紧序列 tau 的跨方法语义不同：分区判据是 `atol + rtol·max(‖x/S‖∞, ‖G(x)/S‖∞)`（`partitioned.py`），T22 判据是 `atol + rtol·‖r0‖_S`（初始残差）。T23 应按方法记录定义、r0、最终归一化残差与阈值，tau 档位按方法解释；不要在文字上直接比较两法的 tau 数字（spec §5.3(3) 要求记录定义）。一致性结论依赖解差与平台检查，不依赖 tau 数字相等。 | T23/A7；T22/A6 提供定义 |
| M-2 | 建议 | T21 `block_norms = ‖R_b‖₂/max(‖x_b‖₂,1)` 是状态相对监视量，不是收敛判据，也不等于物理尺度归一化。T22 需用 `context.layout` 切片 + 配置尺度（建议按块 L*=0.41 m、U*=0.2 m/s、P*=40 Pa、F*=14.29426 N/m）自行计算 `max_i abs(R_i)/S_i`；`validation.monolithic.block_scale_vector` 已提供按块生成 full-length 尺度向量的工具，建议复用。 | T22/A6；A0 冻结尺度 |
| M-3 | **阻塞（T23 执行前）** | 压力规范不对接：T21 首版要求 `PressurePin(node,value)`（替换一条 P1 方程），而 T20 分区稳态 FSI1 用 `pressure_pin=None`（自然出口，`validation/fsi1.py:145`）且配置无 pin 字段；T23 §3.2 只写“移除压力零模”。执行前必须冻结唯一操作：推荐分区侧改用与整体相同的 `PressurePin`（`solve_navier_stokes` 支持 `pressure_pin`/`pressure_value`，默认 `pressure_pin=0`，T20 显式传 `None`），或在两侧定义同一移除零模操作（如按 P1 压力均值对齐）并记录；同时把硬编码的 `scales=1e-3 m`（`fsi1.py:173`）移入配置/报告。 | A0；T23/A7、T21/A6 |
| M-4 | 建议 | T21 `timings["total_seconds"]` 语义错误：`evaluate_monolithic` 内 `started` 按阶段重置，末尾返回的是最后阶段耗时（`residual` 模式≈残差耗时，`assembled`≈Jacobian 耗时），不是评估总耗时，直接影响 22-05/23-09 的组装计时口径；复核结束时点（sha256 `51b99e10…`）仍未修复。建议：入口保存 `total_started`，末尾用它计算；修后在 `validation/monolithic.py` 增加 `total ≥ Σ分项` 自检。 | T21/W1 |
| M-5 | 建议 | `solve_newton` 与 `mode="residual"` 的适配约束：原型在初始（未收敛时）与接受点读取 `.jacobian`，而 residual 模式的 `MonolithicEvaluation.jacobian=None`，直接套用会以 `phase="jacobian"` 失败。适配器二选一：(a) 全部试探用 `mode="assembled"`（被拒试探也付组装费）；(b) 惰性包装对象，`.jacobian` 访问时对同一 `x` 补一次 assembled 评估（接受点 2 次、被拒 1 次）。两种口径都计入 23-09 的评估次数与组装时间，建议 T22 实测后冻结。 | T22/A6 |
| M-6 | 建议 | T23 §3.2 变量表缺 `v_s` 与 `g`：冻结未知量为 `[u_f,p_f,d_s,v_s,d_m,g]`，表中只有 d_s、Hd_s、u、p、d_m、λ。要么补 `v_s`（尺度 U*；稳态不动点期望→0）与 `g`（尺度 F*；说明与 λ=Hᵀg 的映射），要么在方法论中显式说明不比较的理由与等价关系。 | T23/A7；T21/A6 提供块布局 |
| M-7 | 建议 | 23-08 的“两方法相对误差差 ≤0.005”只是拟议操作化，适用于相对误差指标（tip_y/drag）；tip_x/lift 用的是绝对误差，应给绝对差上限（如 tip_x 差 ≤1e-5 m、lift 差 ≤0.1 N/m）或分别报告；该值非 spec 固定值，须 A0 冻结并注明。 | T23/A7 |
| M-8 | 建议（执行前验证） | 收紧序列最后一档的可行性：当前 `max_iterations=40`、`fluid_tolerance=1e-8`、`solid_tolerance=1e-9`；tau=1e-10 需同步收紧单场容差并提高耦合迭代上限（T20 在 1e-6 档已用 27 次评估）。建议先做单方法前缀实验（相邻档迭代数、最终归一化残差、平台差），再冻结档位；若 1e-10 不可达，按 T23 既有规则报告失败，不得为通过而放宽。 | T23/A7；A5 提供分区运行 |
| M-9 | 建议 | 文档路径一致性：T23 与 m5.md 把未来实现写为 `src/fsi_ice/coupling/monolithic/`（目录），实际为模块 `monolithic.py` 加 `validation/monolithic.py`；按 m5.md 说明保持编号不变、执行前更新路径。 | T23/A7 |
| M-10 | 信息（无冲突） | 与 spec 门槛核对一致：21-02/21-03 的 1e-5、21-01/21-05 的 1e-12·S、22-03 的 1e-10、23-08 的 FSI1 门槛均与 spec §5.2/§5.3 和 m3_freeze §6 一致；`m5_validation.json` 20 条 id 与 m5.md 一一对应且全部 `not_executed`。T21 已提供 `JACOBIAN_BLOCK_SOURCES`/`APPROXIMATIONS`（21-04）与 `geometry_diagnostics`（21-03/22-02）；`require_candidate` 与 `base.generation==stencil.step_id` 满足版本化候选与“绝不 accept/publish”。 | — |
| M-11 | **阻塞（T22/T23 执行前冻结）** | 稳态离散一致性：T20 分区稳态 FSI1 用稳态 NS + 静态弹性（`solve_navier_stokes`/`solve_static_continuation`），而 T21 整体残差是 **BDF 瞬态步**（流体 ALE 的 `a0 M u` + 历史质量载荷；固体 `D_BDF` 与 v_s 未知量；`bdf.py` 强制标准 BDF 权重，不能取 a0=0）。仅在收敛不动点（历史=当前解、v_s=0、网格速度 0）两者方程一致。执行前须冻结整体法稳态解的取得方式：推荐“自一致历史不动点”（外层用当前解更新 base 后重解 BDF 步，直至 base 与 x 一致）或“伪时间推进至稳态”，并规定 dt/order/初始历史/求解次数与成本口径；T23 §2.2 的“稳态 FSI1 无时间推进”表述与 T21/T22 实现不符，需更正。 | A0；T22/A6 提方案、T23/A7 更新方法学 |


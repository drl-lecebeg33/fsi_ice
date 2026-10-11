# T19/A5：Aitken 分区加速

日期：2026-10-11。状态：实现、CLI 集成与本地验收完成。A5 定向回归 115 项通过；协调者集成后再次运行 T19/T18/独立复核/CLI 回归 147 项通过；同一有限元配置的 fixed/Aitken 评估次数为 27/7，完整结果见 [T19 验收记录](../validation/t19.md) 和 [A5 验证记录](../validation/t19_worker.md)。T18 未提交工作原样保留，本工作不提交或推送。

## 范围与公共接口

只修改 `coupling/partitioned.py`、必要时 `coupling/__init__.py`、新增 `tests/test_aitken.py`、本计划、`docs/validation/t19_worker.md` 和 `results/t19/worker_a5/`。复用现有 Dirichlet–Neumann 回调与物理实现。

`CouplingOptions` 在既有字段后增加 `method='fixed'`、`minimum_relaxation=0.05`、`maximum_relaxation=1.0`、`aitken_denominator_tolerance=1e-14`；既有位置构造兼容。所有新增数值须为有限实数，`0<minimum<=maximum`、分母容差严格为正；Aitken 额外要求 `minimum<=relaxation<=maximum`。初始 relaxation 仍须在 (0,1]，maximum>1 必须显式配置。`CouplingIteration` 最后增加带默认值的 `relaxation_reason` 字符串。

## 算法与事件定义

真实残差为 r=G(x)-x，尺度为 S，Aitken 内积使用 q=r/S。首次非收敛评估使用初始 omega；随后 delta=q-q_prev，omega=-omega_prev*(q_prev·delta)/(delta·delta)，omega_prev 是上一次实际更新使用的系数。

分母相对退化定义：delta·delta <= tolerance*max(q_prev·q_prev,q·q)。退化时保留上一系数并记录 `denominator_degenerate`；此相对定义避免残差整体减小导致过早冻结。系数非有限或内积溢出时记录 `coefficient_nonfinite` 并抛出 relaxation 阶段失败，禁止静默传播 NaN/Inf。有效系数按上下限截断并记录 `clipped`，无截断记录 `aitken`；首次记录 `initial`。固定方法记录 `fixed`，收敛行记录 `converged`，耗尽行记录 `max_iterations`；终止行不计算或应用新系数，其 relaxation 保留最近系数（首行则为初始值）。更新算术失败也必须带历史和最后已评估输入诊断。

收敛范数、容差和最大评估次数完全沿用 T18。所有成功字段和失败输入属于最后实际评估的 x；Aitken 状态仅在本次 solve 局部保存，任何时间历史接受由外部事务负责。

## 验收矩阵（不得事后放宽门槛）

1. 标量仿射映射经一至两次动态系数更新达到解析直接解，归一化误差 <=1e-9；检查实际系数、评估次数与返回输入。
2. 多分量仿射映射采用有界系数，解析直接解归一化误差 <=1e-9；含显式 maximum>1 的标量验证。
3. 覆盖正/负系数截断、零和近退化分母、非有限计算、错误新增选项、真实残差大而更新很小、最后一次不更新、回调异常。
4. 分量分别改变单位并相应改变 S 后，系数/归一化残差序列和解一致（浮点容差 <=1e-12）；输入与尺度不被修改。
5. 失败后重启不继承旧残差/系数；真实 FE 失败重试与冷启动逐数组相同，BDF 历史和 generation 保持。
6. 默认 fixed 与显式 fixed 逐数组和系数历史一致，且现有 `tests/test_partitioned.py` 全部通过。
7. 真实 FE 使用 `LinearCoupledProblem.solve(options=CouplingOptions(method='aitken',...))`：沿用 T18 冻结配置和门槛，d/u/p/load 归一化差 <=1e-8，组合方程残差 <=1e-9，运动误差 <=1e-9，连续性 <=1e-10；记录与 fixed 27 次基线的实际次数，不保证所有问题加速。

实际执行项目 `.venv/bin/python -m pytest tests/test_aitken.py tests/test_partitioned.py`，保存退出码、日志、环境、数值和迭代历史。代理并行阶段不运行耗时全量回归；本计划通过证据后只更新自身状态，根目标进度由 A0 更新。

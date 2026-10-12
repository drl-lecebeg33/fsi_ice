# T25：IQN-ILS 分区加速

日期：2026-10-12。单一执行者承担 A5 实现、A0 集成自查与 A7 复核测试；不派发并行执行者，不声称独立复核。前置：T18 固定松弛、T19 Aitken 已经本地验证，T24 短程瞬态接线已就绪。本轮仅完成 IQN-ILS 控制器及验收，**不**负责长时 FSI2 周期、网格/时间收敛或四算法比较。

## 范围与文件

- 修改 `src/fsi_ice/coupling/partitioned.py`：扩展 `CouplingOptions` 增加 `method='iqn'` 与对应字段，新增 IQN-ILS 主循环。
- 修改 `src/fsi_ice/coupling/__init__.py`：必要时重新导出新增类型。
- 修改 `src/fsi_ice/validation/partitioned.py`：把新增 IQN 字段加入 `validate_config` 的可选集合，并把 `method` 取值扩展到 `iqn`。
- 修改 `src/fsi_ice/cli.py`：`partitioned-check --method` 接受 `iqn`。
- 新增 `cases/iqn_check.json`：默认 IQN-ILS 冻结配置。
- 新增 `tests/test_iqn.py`：解析映射、收敛次数、退化重置、系数截断、失败重试、实测 FE 与 T18 直接解的一致性。
- 新增 `docs/validation/t25.md`：验收记录；`docs/development/t25_plan.md` 为本计划。

不修改 `solve_dirichlet_neumann` 的接口与现有签名；不动物理残差、积分、H/Hᵀ、时间和事务模块。

## 算法：Interface Quasi-Newton Inverse Least Squares（IQN-ILS）

记 r_k = G(x_k) − x_k，s_k = x_{k+1} − x_k，y_k = r_{k+1} − r_k。控制参数为正标量 ω_0。

- 第 1 步（无历史）：x_2 = x_1 + ω_0·r_1，记录 `reason='initial'`。
- 第 k ≥ 2 步：构造 V_k = [v_1 … v_{k-1}] ∈ ℝ^{n×(k-1)}，其中 v_j = s_j + ω_0·y_j。
  - 解最小二乘 V_k·α = −r_k（条件不足时为 LS residual）。
  - 更新 x_{k+1} = x_k − V_k·α。
  - 记录 α̂ = clip（α₀ ）、|α|∞ ≤ α_max；LS 解 ∃
  - 新增 (s_k, y_k) 到历史；历史长度限制为 `min(max_iterations-1, n)`。
  - 失败条件之一：
    * `solve` 返回的相对残差 ‖V_k·α + r_k‖∞ > `iqn_restart_tolerance`·max(‖x_k/S‖∞, ‖r_k/S‖∞)；
    * 任何 |α|∞ > `iqn_alpha_max`；
    * V_k 数值秩不足（`lstsq` 报告舍入秩 < 现有历史长度）。
  - 任意一条触发即清空历史、改用 ω_0·r_k 单步更新，记录 `reason='iqn_restart'`，失败保留已接受 BDF 历史与最后一次已评估输入。

范数统一为 max|·/S|；atol+rtol·max(max|x/S|, max|G(x)/S|) 收敛门槛与 T18/T19 保持一致；`max_iterations` 指 G 的总评估次数；最后一次已评估输入和对应单场状态返回。

## 验收矩阵（实现后实际执行；不得事后放宽）

1. 标量仿射映射（a·x + b，|a| ≤ 0.9）：相对误差 ≤1e-9，迭代次数与 ω_0 接近 1；不同 ω_0 验证初始推力生效。
2. 多分量仿射（2D/5D）：相对误差 ≤1e-9；迭代次数比 fixed/aitken 显著减少，且不大于分量数 + 1。
3. 强非线性 + 步长较大仍能收敛或退化为 fixed 步进（不破坏 BDF 历史）。
4. 重置条件：人为构造接近秩亏的 y 序列触发 `iqn_restart`，并验证清空后下一步使用 `initial` 标签。
5. α 截断：人为构造需要 |α| > α_max 的问题，验证触发 `iqn_restart` 而不是用极大 α 跳跃。
6. 失败重试：与 Aitken 同样的 restart-after-failure 不变量——冷启动重复运行逐数组一致。
7. 与 fixed/aitken 兼容：默认 `CouplingOptions()` 仍是 fixed；T18/T19 frozen 回归不失败。
8. 实测 FE：复用 `LinearCoupledProblem`，使用 IQN-ILS 配置 `coupling.method='iqn'`，归一化位移/速度/压力/载荷差 ≤1e-8，组合残差 ≤1e-9，运动误差 ≤1e-9，连续性 ≤1e-10；迭代次数 < fixed=27 次的基线，5e-12 实测地板。
9. CLI `partitioned-check --method iqn` 退出码 0、生成报告，与 `fixed`/`cached` 逐数组一致。
10. 配置校验拒绝 IQN 字段中非有限、非正、非整型的输入。

执行命令：

- `.venv/bin/python -m pytest tests/test_iqn.py tests/test_aitken.py tests/test_partitioned.py tests/test_coupling_review.py tests/test_partitioned_integration.py tests/test_cli.py -q`
- `.venv/bin/python -m fsi_ice partitioned-check --config cases/iqn_check.json --method iqn --output results/t25/final/iqn`
- 全量回归：`.venv/bin/python -m pytest tests -q`（运行前停止其他可接受命令，记录退出码与耗时）。
- 写日志到 `results/t25/logs/`。

## 不在本轮范围

- FSI2 长时周期、网格/时间独立性、四算法比较表、IQN-ILS 在 FSI2 上的实测成本（属 T26）。
- 重新实现 BDF、AMALE/网格、整体残差。
- 不动 `solve_dirichlet_neumann` 既有签名；所有 T18/T19 验收记录保持不变。
- 不修改 `spec.md` 中的算法/门槛描述，本轮计划作为临时记录。

所有本轮改动在本地跑通前不提交、不推送；按 AGENTS.md 等待用户指示。
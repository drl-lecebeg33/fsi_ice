# M6 分区停止条件与全场残差一致性修复验证

日期：2026-10-13。方案见[单独工作包](../development/m6_physical_gate_plan.md)，原因见[第 923 步只读诊断](m6_step923_diagnosis.md)。实现与复核均由单一执行者完成，属于自查。本轮提交上传已获用户授权；所有通过结论均来自下述本地验证。

## 实现与门槛

`solve_dirichlet_neumann` 增加可选 `convergence_check`：界面残差达标后，调用方仍可要求候选满足另一停止条件。检查拒绝则沿同一步已有 fixed/Aitken/IQN 历史继续更新；异常、非法返回、最大迭代和浮点精度停滞明确失败，不发布候选。未使用该钩子的旧调用保持原行为。

FSI2 的分区路径在界面达标时用现有 `evaluate_monolithic(..., mode="residual")` 重算同一六块残差，要求 `max_i |R_i/S_i| ≤ 1e-7`；事务末端仍独立重算全残差，并检查连续性 `≤1e-10`、网格质量 `≥0.1`。本轮未放宽物理/网格门槛，未修改模型、网格权重或 BDF。旧断点身份守卫仍有效。

## 实际本地验证

环境：项目 `.venv/bin/python`，Python 3.12.3、NumPy 2.5.3、SciPy 1.18.1、Matplotlib 3.11.2、pytest 9.1.1；求解运行使用 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。本轮修改后包源码摘要 `7590751f…`，原第 922 步断点摘要 `1e7df094…`。

| 检查 | 实测 |
| --- | --- |
| 控制器与瞬态、IQN、断点等相关测试 | `pytest tests/test_partitioned.py tests/test_aitken.py tests/test_iqn.py tests/test_fsi2.py tests/test_fsi2_checkpoint.py tests/test_m6_repair.py -q`：228 passed / 22.63 s |
| 最终源码全量回归 | `pytest tests -q`：707 passed、2 skipped / 185.76 s；跳过仍为可选 T23 历史产物与真实 FSI1 整体冒烟 |
| 真实加权网格短程 | `cases/fsi2_weighted_fixed.json`、`cases/fsi2_weighted_aitken.json`、`cases/fsi2_weighted.json` 及其 `monolithic` 覆盖，各在 h=0.03、dt=0.01、t=0.03 接受 3 步；四份报告均为 INCOMPLETE，符合尚无周期证据的状态 |
| 小型真实 FE 的物理检查拒绝注入 | 首次界面达标时人为拒绝物理检查，后续继续迭代并通过末端真实全场检查；终态与无注入基线按逐变量容差一致，见 `tests/test_fsi2.py` |

环境冒烟 `scripts/check_environment.py` 为 PASS；`git diff --check` 通过。原理页 HTML 标签闭合、16 个 `id` 唯一、189 个 `href/src` 的本地目标与同页锚点检查通过；宽窄屏依赖的弹性导航、900px 收缩容器、长文本换行和代码块横向滚动规则经静态检查。当前环境无可用浏览器窗口或无头浏览器程序，未做实际宽窄屏截图检查。

第 923 步只在临时目录重放，先校验重建网格 SHA256 与旧断点相同，再按旧断点记录的源码摘要读取完整状态。当前源码摘要不同，**没有**通过正式 `--resume`，也未写出任何新断点。保持原 IQN 配置与全部物理门槛：

- 将耦合上限限制为 12 次时，明确以 `physical convergence check did not pass` 失败；事务后的已接受状态仍为第 922 步。
- 使用配置原上限 160 次时，第 923 步在 **13 次**耦合评估后通过；全残差 `1.9702e-8`、网格质量 `0.510987`、IQN 重启 0，均满足原门槛。第 923 步只存在于临时诊断进程中，不是正式轨迹的一部分。

## 结论边界

本轮证明：原第 923 步失败由分区停止条件与全场动量门槛不一致造成；新增检查让该步继续迭代后通过原物理门槛，相关故障路径和短程回归通过。原 2.905× 计时属于此前源码，本轮新增残差检查后的吞吐量尚未做公平重复计时。新源码的正式长时轨迹必须从零建立，不能把旧 922 步断点或临时第 923 步候选接入正式续算。稳定周期、参考精度、三网格/三步长和 T26 四算法公平成本仍未验证，M6 保持进行中。

后续正式验证（2026-10-13）：当前源码已按用户选定终点从零接受1000步/10 s，连续越过第923步；全程物理门槛通过，周期仍INCOMPLETE。详见[10秒重跑记录](m6_10s_run.md)。

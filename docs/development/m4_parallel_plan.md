# M4 并行推进计划

状态更新（2026-10-11）：三个 Orca worker 终端均因额度耗尽退出，未产生可接受的 `worker_done`。共享工作区已落盘的 A5 实现、A7 复核测试和 T20 网格证据由协调者逐项检查并在本地重跑；T19 集成/针对性验收通过，D1/D2 修复已通过全量回归。终端退出与本地验收边界见 [T19 验收记录](../validation/t19.md)。FSI1 参考配置与 T20 计划已补齐；正式稳态 FSI 求解仍待实现。

2026-10-11，用户明确要求使用并行规划技能。协调者使用 [Orca orchestration 技能](/home/iceberg/.agents/skills/orchestration/SKILL.md)，实际 Run 为 `run_13e46e07bd4f`。三个 Codex 执行者在同一工作区按文件独占；不覆盖已有 T18 工作，不提交或推送。

## 第一波：独立工作包

| 工作包 | Task / Dispatch | 独占文件 | 交付与门槛 |
| --- | --- | --- | --- |
| A5 / T19 Aitken | `task_a8f859d9e749` / `ctx_18d342c8f77f` | `coupling/partitioned.py`、`coupling/__init__.py`、`tests/test_aitken.py`、`docs/development/t19_plan.md`、`docs/validation/t19_worker.md` | 在现有固定点控制器上实现缩放内积 Aitken、退化与截断；保持 fixed 默认行为；解析与真实 FE 验证 |
| A4/A7 / T13-FSI1 与 T20 准备 | `task_eb026be9703e` / `ctx_ff1b29e8e2d5` | `docs/development/fsi1_plan.md`、`cases/fsi1_reference.json` | 实际核对原始参考，冻结量纲/本构/力范围；设计共享曲线匹配网格，并尽可能运行临时原型 |
| A7 / T18 独立复核与 T19 验收准备 | `task_04bc832f6e5d` / `ctx_f66363d54d05` | `docs/validation/partitioned_review.md`、`tests/test_coupling_review.py` | 检查直接解公平性、残差、历史与失败行为；新增必要独立检查；T19 实现交付后再复核 |

每个执行者仅使用自己的 `results/t19/worker_a5/`、`results/t20/reference/` 或 `results/t19/reviewer/` 保存本地证据；源码修改前阅读项目规范，以项目 `.venv` 实际运行测试。已接受历史不能在迭代中提前更新，验收门槛不得为通过而放宽。

## 第二波与依赖

第一波完成并通过协调者检查后：

1. A5 交付 → 协调者集成 `validation/partitioned.py`、CLI、`cases/aitken_check.json` 和独占 `tests/test_partitioned_integration.py`；用相同 FE 配置对比 fixed/Aitken 和块直接解。
2. A7 第一波复核交付且 A5 交付 → 复用复核者，派发新的 T19 独立复核 Dispatch。发现实现缺陷回派 A5 修复，不由复核者修改实现。
3. T19 本地验收、参考核验和网格设计通过 → 才可安排 T20 正式 FSI1 网格/稳态耦合实施；临时网格原型不等于正式 FSI1 完成。

协调者独占根文档、CLI、`validation/partitioned.py`、`cases/aitken_check.json`、`tests/test_partitioned_integration.py`、本计划和最终 `docs/validation/t19.md`，负责统一接口、执行最终全量回归并更新进度。验收输出逐项注明完成/未完成，不把设计文档或参考数值当作实际物理解。

## T19 公共接口冻结

- `CouplingOptions.method`：`fixed`（默认）或 `aitken`。
- 初始 `relaxation` 仍遵守 `(0,1]`，沿用既有 fixed 行为。
- `minimum_relaxation=0.05`、`maximum_relaxation=1.0`、`aitken_denominator_tolerance=1e-14`；Aitken 要求 `0<min≤initial≤max`，更大的上限需显式配置。
- 用 `r/S` 的内积计算动态系数；收敛仍检查未松弛界面残差，返回最后已评估输入。
- `CouplingIteration.relaxation_reason` 以默认字段扩展，记录初始化、动态更新、截断、分母退化和收敛等实际事件。
- 验收沿用 T18 的直接解差、逐行真残差、连续性、历史保持与重试门槛；不要求所有问题都加速，记录实测迭代数。

## 监督与交付

协调者处理 Orca 的 question/escalation/worker_done；收到与当前 Dispatch 匹配的结算后，明确复用或释放终端再确认消息。每个 Task 需附实际命令、退出码、数值及文件范围。所有预期 Dispatch 结算并处理终端归属后，才结束本轮监督。

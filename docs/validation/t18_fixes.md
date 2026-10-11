# T18/T19 输出失败管理修复记录（D1/D2）

日期：2026-10-11。执行者：协调者/A0 单一执行者实现与自查，不声称独立复核；修复 A7 复核登记的 D1、D2 缺陷。基线 `2b5f713` 加本轮未提交工作区。缺陷定义与最小复现见 [A7 复核报告](partitioned_review.md) 的缺陷表。

## 修复范围（先于实现冻结）

- **D1**：`partitioned-check` CLI 原先先加载配置再运行；坏配置重跑退出 2，但旧 `report.json`（PASS）/`solution.npz` 留存且没有本次 `failure.json`，输出清理从未执行。
  修复条件：每次已指明输出目录的运行有自己的失败状态；清除旧成功产物；保存输入故障阶段与原因；保留原输入路径；不伪称读取了无效配置。
- **D2**：`run_partitioned_check` 的 setup、metrics 及成功数据写出不在失败记录范围内；构造未完成时不能引用不存在的 `problem` 或伪造 `histories_unchanged`；写出异常需要明确诊断。

## 实现改动

- `src/fsi_ice/validation/partitioned.py`
  - 新增 `_clear_outputs` 与 `_write_failure`；`problem`/`snapshots` 不可用时 `histories_unchanged` 写 `null`，不伪造布尔值。
  - `run_partitioned_check`：失败记录范围覆盖 setup、coupling、direct_reference、repeat、metrics、checks 与输出写出；失败时清除 `report.json`/`solution.npz` 并写 `failure.json`，无历史时 `iteration` 为 `null`。
  - 新增 `run_partitioned_check_cli(config_path, output, *, method=None)`：先清理输出，再读取、解析、校验配置；输入失败写 `failure.json`（stage=`config_read`/`config_parse`/`config_validate`，`config=null`，保留 `config_path`）。
- `src/fsi_ice/cli.py`：`partitioned-check` 改调上述入口；返回码语义不变（0=数值 PASS，1=数值验收 FAIL，2=输入/求解异常）。
- `tests/test_coupling_review.py`：D1 的 4 项与 D2 的 2 项 strict xfail 转为正常断言，测试逻辑未改。

## 实际执行与证据

环境：项目 `.venv`，Python 3.12.3；NumPy 2.5.3、SciPy 1.18.1、Matplotlib 3.11.2、pytest 9.1.1；`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。修复涉及文件指纹见 [source_sha256.json](../../results/t19/fix_d1d2/source_sha256.json)。

| 实际命令 | 结果 | 证据 |
| --- | --- | --- |
| `.venv/bin/python -m pytest tests/test_coupling_review.py tests/test_partitioned.py tests/test_partitioned_integration.py tests/test_cli.py -q -rxX` | **85 passed，8.20 s，exit 0**（含转入正常断言的 6 项） | `results/t19/fix_d1d2/pytest_related.log`、`pytest_related.exitcode` |
| `.venv/bin/python -m fsi_ice test` | **425 passed，128.86 s，exit 0**，无 xfail 残留 | `results/t19/fix_d1d2/full_suite.log`、`full_suite.exitcode` |
| CLI 手工复现：合法配置成功 → 同一输出目录分别用 dt=0、缺失文件、损坏 JSON 重跑 | 依次退出 0/2/2/2；失败后目录内仅 `failure.json`，无旧 PASS/解 | `results/t19/fix_d1d2/repro/`（`run*.log`、`failure_*.json`） |

手工复现的失败记录对照（`config` 均为 `null`，保留原 `config_path`，`histories_unchanged` 均为 `null`）：

| 输入 | stage | reason（摘要） |
| --- | --- | --- |
| dt=0 | `config_validate` | `dt must be finite and positive.` |
| 缺失文件 | `config_read` | `[Errno 2] No such file or directory ...` |
| 损坏 JSON | `config_parse` | `Expecting property name ...` |

D2 的 setup/metrics 注入用例，以及 `stage=convergence`/`direct_reference` 时 `histories_unchanged=true` 的既有断言，均在上述 85 项与全量回归中实际通过。

## 边界与剩余工作

- 本记录为修复者自查；A7 独立复核（建议新 Dispatch：重跑 `tests/test_aitken.py`、`tests/test_partitioned.py`、独立测试与 CLI 检查并重建数值证据）尚未执行，T19 完成勾选继续等待独立验收。
- 非 `partitioned-check` 命令（`csm3`/`cfd3`/`single-fields`/`poisson`）的输入失败仍沿用既有行为，不在 D1/D2 修复范围内。
- 本轮未提交、未推送；T20/正式 FSI1 尚未开始。

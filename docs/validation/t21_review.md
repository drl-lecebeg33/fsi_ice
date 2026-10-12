# T21 独立验证记录（A7）：整体残差、完整耦合 Jacobian 与矩阵自由 JVP

日期：2026-10-11。结论：**passed**（T21 范围）。

本记录是 A7 对 M5/T21 交付（实现者 A6，见 [t21.md](t21.md)）的独立验证：实际复跑实现者测试，并新建独立复核脚本 [`tests/test_monolithic_review.py`](../../tests/test_monolithic_review.py)（9 项），在**未修改任何实现文件**的前提下完成方向导数扫描、恒等式与拒绝条件抽查、证据数字抽样，并记录被验证文件的 sha256 指纹。验证范围不含 T22（整体 Newton，另一 worker 并行进行）与 T23（FSI1 一致性），这两项在本记录中不作任何通过声明。

## 1. 验证对象与文件指纹

| 文件 | sha256（2026-10-11 本地） | 角色 |
| --- | --- | --- |
| `src/fsi_ice/coupling/monolithic.py` | `a708a859760dc449939c851d8414921d33b215def7a0f382529c54497f2e7f3d` | 被验证实现 |
| `src/fsi_ice/validation/monolithic.py` | `2dfcab0401255ccd7a5dd2f9b68d2ab7d4bedf7e737d86f9f82a106aec620d46` | 被验证扫描工具（独立复核未使用其辅助函数） |
| `tests/test_monolithic_residual.py` | `28bd3c6c127cd8f8abf02bb5ae2299da15c8ca380a1e617cdb5690b857f0cb9c` | 实现者自测（被验证） |
| `results/m5/t21/scan_evidence.json` | `eeb3aaa2b13124ca9e9ce94f93e95e2b1c72ac922ef5fbc7718fdc050b793778` | 实现者数值证据（只读引用） |
| `results/m5/t21/fsi1_smoke.json` | `47e7040a1d9ad9c458b3aac880fd838b0683eae75fa4fe165ea153eac378bb96` | 冒烟记录快照（每次运行冒烟测试会被重写，时间字段随运行漂移，见 §5） |
| `tests/test_monolithic_review.py` | `1a15db8265cd4e24401cd3714cb192ba38fb746e64544b4dd9d682aa287baf6e` | 本次独立复核脚本（新建） |

代码状态：`git rev-parse HEAD = 9647efa0aa25aee72c621289021e872e87363d3d`；验证前后三个被验证文件指纹一致、mtime 未变（2026-10-11 21:03–21:44 冻结），未被本次验证修改。

## 2. 实际运行命令与环境

环境：Python 3.12.3（项目 `.venv/bin/python`），NumPy 2.5.3、SciPy 1.18.1、pytest 9.1.1；Linux-7.0.0-38-generic x86_64；`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。

| 命令 | 结果 | 耗时 |
| --- | --- | --- |
| `.venv/bin/python -m pytest tests/test_monolithic_residual.py -q` | **21 passed** | 26.47 s（wall 26.68 s） |
| `.venv/bin/python -m pytest tests/test_monolithic_review.py -q -s` | **9 passed** | 2.18 s（wall 2.39 s） |
| `.venv/bin/python -m pytest tests/test_monolithic_residual.py tests/test_monolithic_review.py -q` | **30 passed** | 29.32 s（结束前复跑 28.61 s，wall 28.82 s） |

独立复核脚本耗时 2.2 s，满足快速复核要求；实现者自测耗时与其报告（28.10 s）同量级。

## 3. 独立复核方法与实测数值

方法说明：问题构造（两域匹配网格、已接受 BDF 历史、候选状态与上下文）以只读方式复用 `tests/test_monolithic_residual.py` 的 `build_problem` 等助手；**方向、扰动、随机种子、中心差分实现、断言与容差均在本任务独立编写**，未调用 `src/fsi_ice/validation/monolithic.py` 的扫描辅助。逐变量尺度沿用 t21.md §4 的冻结约定（`u:U*=0.05`、`p:P*=0.01`、`d_s/d_m:L*=1`、`v_s:U*`、`g:F*=0.01`；逐块残差尺度 `R_u:P*`、`R_p:U*L*`、`R_sd:U*`、`R_sv:P*`、`R_m:L*`、`R_g:F*`）。相对误差 = `||(fd−解析)/S_R|| / ||解析/S_R||`，门槛沿用冻结值 `≤1e-5`；恒等式门槛沿用 `≤1e-12·S`。

### 3.1 非退化方向扫描（`h = 1e-2 … 1e-7`）

方向 1：**确定性光滑方向**（六块都有分量，非随机）；方向 2：**随机混合方向**（seed=20261019）；方向 3：**界面链方向**（仅湿界面 `d_s`/`v_s` 与 `g`，seed=20261021）。三个方向都满足界面相容 `d_m|Γ = d_s|Γ`，扫描同时与矩阵自由 JVP 和**装配 Jacobian** 对照：

| h | 光滑方向 rel | 随机混合 rel | 界面链 rel |
| ---: | ---: | ---: | ---: |
| 1e-2 | 1.147e-05 | 5.821e-03 | 1.869e-03 |
| 1e-3 | 1.147e-07 | 5.821e-05 | 1.869e-05 |
| 1e-4 | 1.147e-09 | 5.821e-07 | 1.869e-07 |
| 1e-5 | **2.406e-11** | 5.821e-09 | 1.869e-09 |
| 1e-6 | 2.681e-10 | 5.740e-11 | **2.295e-11** |
| 1e-7 | 3.367e-09 | **5.564e-11** | 2.267e-10 |
| 最佳 / h | 2.406e-11 @1e-5 | 5.564e-11 @1e-7 | 2.295e-11 @1e-6 |
| 趋势 | decreasing_then_increasing | not_resolved（1e-7 仍在下降） | decreasing_then_increasing |

三个方向的最佳相对误差（2.41e-11、5.56e-11、2.30e-11）均 **≤1e-5**，且装配 Jacobian 的同一扫描逐点相同（装配 `J@q` 与解析方向导数在相同 h 下的相对误差与上表一致到所列位数）。`J@q` 与 `monolithic_jvp` 的相对差：光滑 1.90e-16、随机 1.46e-16、界面链 1.93e-16，均 ≤1e-12。h=1e-2 单点相对误差可略高于 1e-5（截断未消），符合冻结口径“全扫描最小相对误差 ≤1e-5”。

### 3.2 零响应方向（绝对判据）

取界面 owner 单元之外的网格节点 12（坐标 (0.5, 2/3)，方向 `d_m = (L*, -0.5L*)`）：`g` 行解析响应**逐位为 0**；中心差分在全部 6 个步长下 `max|(fd−解析)/S_g| = 0.000e+00`（自设判据 ≤1e-9，实测为零；实现者报告 §4.6 为 ≤1e-10）。

### 3.3 恒等式抽查（数值级结论）

| 恒等式 | 独立实测 | 判据 | 结论 |
| --- | ---: | ---: | --- |
| `H`：`ΔR_u,Γ = −H Δv_s`（只扰动 `v_s` 迹） | 6.938894e-18 | ≤1e-12·U* | PASS |
| `Hᵀ`：`ΔR_sv = −S_s Hᵀ Δg`（自由行） | 2.775558e-17 | ≤1e-12·F* | PASS |
| `R_g` 行对 `g` 的导数为单位映射 `ΔR_g = Δg` | ≤1e-15（绝对） | 数值舍入 | PASS |
| 装配湿流体迹行 `[u_f|Γ] = I`、`[v_s|Γ] = −H` | 0.0、0.0 | ≤1e-14 | PASS |
| 装配网格迹行 `[d_m|Γ] = I`、`[d_s|Γ] = −I` | 0.0、0.0 | ≤1e-14 | PASS |
| BDF 相容行：`v_s = a₀d_s+Σa_j d_j` ⇒ `R_sd = 0`（自由行） | 0.0 | ≤1e-12·U* | PASS |
| 装配 `J[d_s,d_s] = a₀I`、`J[d_s,v_s] = −I`（自由行） | 0.0、0.0 | ≤1e-14 | PASS |
| 厚度只计一次：`t=4.0` 与 `4×t=1.0`；`t=0.25` 与 `0.25×t=1.0` | 0.0、0.0（相对） | ≤1e-12 | PASS |
| `R_g = g − g_f_gamma`（与独立重建候选的界面载荷一致） | 0.0 | ≤1e-12·F* | PASS |
| 压力 pin 行：`R_p,3 = p_3 − (−0.7P*)`、行导数为 1、装配行为单位行 | 0.0（三项） | ≤1e-12·P* / ≤1e-14 | PASS |
| `pin=None` 时保留原始连续性行（与独立重建的单场残差一致） | 0.0 | ≤1e-12·U*L* | PASS |

### 3.4 拒绝条件抽查

| 条件 | 独立构造 | 实测拒绝信息 | 结论 |
| --- | --- | --- | --- |
| 真翻转（`det(J_x) ≤ 0`） | `d_m = (−3x, 0)` ⇒ 变形梯度 `diag(−2,1)` | `cell 8 has det(J_x)=−0.1667 … (orientation flip)`；`ValueError` | PASS |
| 质量不合格 | `d_m = −0.98(X−c)` ⇒ 体积比 4e-4 | `minimum volume ratio 0.0004 < 0.2`；`ValueError` | PASS |
| 非法 `x` | 长度 263 的向量 | `x must have shape (264,)`；`ValueError` | PASS |
| stale stencil | 2 步接受态（generation 2 ≠ 基态 1） | `stale candidate: stencil.step_id=2 …`；`ValueError` | PASS |
| stale 候选几何 | 不同 `candidate_id` 的几何 | `stale candidate geometry 'review-old-candidate'`；`ValueError` | PASS |
| 混合拓扑 | 另一拓扑（nx=3）的界面几何 | `interface geometry belongs to a different topology (mixed version)`；`ValueError` | PASS |
| 混合载荷版本 | 同一 `candidate_id`、不同相容状态的载荷 | `interface load was assembled on a different interface geometry (mixed version)`；`ValueError` | PASS |

以上全部由 `tests/test_monolithic_review.py` 在本次运行中实际触发并通过（消息为运行输出摘录）。

## 4. 实现者报告数字抽样对照

抽样 30 项（脚本另含趋势与几何列批量一致性 3 项附加断言），`t21.md` 报告值与 `results/m5/t21/scan_evidence.json`（含其 `smoke` 快照）逐项一致；下列检查由独立脚本 `test_review_scan_evidence_matches_report` 在本次运行中程序化通过（报告值按报告有效位数设置容差）：

| t21.md 位置 | 报告值 | 证据 JSON 实测值 | 结论 |
| --- | ---: | ---: | --- |
| §4.3 混合方向最佳相对误差 | 7.39e-11 | 7.385765587812849e-11 | 一致 |
| §4.3 最佳步长 / 趋势 | 1e-6 / 先降后升 | 1e-6 / decreasing_then_increasing | 一致 |
| §4.3 `J@q` vs JVP | 1.6e-16 | 1.5988163275793417e-16 | 一致 |
| §4.2 随机 `d_s` | 5.67e-11 | 5.670870399705216e-11 | 一致 |
| §4.2 随机 `d_m` | 2.94e-11 | 2.9375831555039694e-11 | 一致 |
| §4.2 随机 `u_f` | 1.94e-16 | 1.9388038855442818e-16 | 一致 |
| §4.1 `u_f` 内部 | 2.96e-16 | 2.958187289453368e-16 | 一致 |
| §4.1 `d_s` 内部 | 1.87e-11 | 1.8699065247798755e-11 | 一致 |
| §4.1 `d_m` 内部 | 1.04e-10 | 1.0439305442978104e-10 | 一致 |
| §4.1 `g` 迹分量 | 1.57e-16 | 1.570092458683775e-16 | 一致 |
| §4.4 仅内部 `d_m` | 4.40e-11 @1e-6 | 4.395017770562414e-11 @1e-6 | 一致 |
| §4.4 仅湿界面 `d_m`+相容 `d_s` | 9.73e-11 @1e-7 | 9.725048982255435e-11 @1e-7 | 一致 |
| §4.4 相容混合 `d_m`/`d_s` | 3.93e-11 @1e-7 | 3.929452791239411e-11 @1e-7 | 一致 |
| §4.5 BDF1 / BDF2 | 5.64e-11 / 5.33e-11 | 5.6370452321248074e-11 / 5.326531290703482e-11 | 一致 |
| §5 `J_um/J_pm` 贡献 | 1.417e+01 | 14.16974423356489 | 一致 |
| §5 `S_s Hᵀ` 贡献 | 2.705e+02 | 270.5359912171539 | 一致 |
| §5 删块 `J_um/J_pm` 相对误差 | 1.23e-03 | 0.0012260474811709744 | 一致 |
| §4.7 载荷两积分路径（载荷/导数） | 4.3e-19 / 1.7e-16 | 4.336808689942018e-19 / 1.6653345369377348e-16 | 一致 |
| §6 自由度 / Jacobian nnz | 12905 / 377096 | 12905 / 377096 | 一致 |
| §6 `J@q` vs JVP | 1.79e-16 | 1.791432727409034e-16 | 一致 |
| §6 残差范数 | 7.595 | 7.595211436834509 | 一致 |
| §6 最小 `det(J_x)` / 质量 | 4.73e-5 / 0.728 | 4.7279447904506554e-05 / 0.7284679028009242 | 一致 |
| §6 最小体积比 | ≈1.0 | 0.9999999999999996（在线冒烟记录） | 一致 |
| §6 残差/JVP/组装耗时 | 0.27 / 0.69 / 19.43 s | 0.272912 / 0.692937 / 19.430672 s（scan 快照） | 一致（计时允许运行波动） |
| §4.8 几何列批量 vs 逐自由度 | 逐位相同（0.0） | 0.0 / 0.0 | 一致 |

## 5. 差异、说明与观察

1. **冒烟 JSON 会在运行时被重写**：`tests/test_monolithic_residual.py::test_real_fsi1_mesh_smoke` 自身会把结果写入 `results/m5/t21/fsi1_smoke.json`，因此复跑实现者测试会刷新该文件（本次 21 项复跑与 30 项合并复跑均触发）。确定性字段（DOF、nnz、几何量、`J@q` 一致性）与 t21.md/scan 快照一致；`mode_seconds` 属计时量，多次复跑在 0.25–0.28 / 0.67–0.69 / 17.9–19.5 s 量级波动（scan 快照 0.273/0.693/19.43 s），不构成数值差异。任务要求的“只读”指不修改实现与他人的源码/报告，此处的重写是冒烟测试的既定行为，特此披露。
2. **证据 JSON 键名笔误（不影响数值）**：`scan_evidence.json` 中删块相对误差的键名为 `"d dropped_best_relative_error"`（多出前缀 `d `），实现者报告 §5 表格数值与该键一致；建议后续修正键名（本任务不修改实现方文件）。
3. **实现者“镜像翻转”用例的实际触发路径**：`test_context_and_candidate_rejections` 中 `d_m = −2(X−c)` 在二维是保向刚体旋转（`det(J_x) > 0`），实际拒绝来自 GCL 扫掠路径退化（消息 `min det(F(s))=… <= 0; the straight interpolation degenerates`），而非 `det(J_x) <= 0`。该用例仍验证了“非法候选被拒绝”，且我的独立用例已单独覆盖真正的 `det(J_x) ≤ 0` 路径；建议后续把该用例注释改为“扫掠路径退化”（不改结论）。
4. **趋势判定**：我的随机混合方向在 `1e-7` 仍未回升（趋势 `not_resolved`），属该 seed 的舍入拐点更晚；门槛按“全扫描最佳相对误差 ≤1e-5”判定（5.56e-11），与实现者“先降后升”的 seed 差异无矛盾。
5. 独立复核脚本不依赖实现者的扫描辅助函数，也不复制其测试断言；§3–§4 的全部数字均由本次实际运行产生。

## 6. 结论

**结论：passed（T21 范围）**。理由：

- 实现者自测 21 项实际复跑全部通过（26.47 s）；
- 独立脚本 9 项实际通过（2.18 s）：三个自选非退化方向（含界面链）最佳相对误差 2.30e-11–5.56e-11 ≤1e-5，装配 Jacobian 与矩阵自由 JVP 一致（≤1.9e-16），零响应方向绝对误差为 0；
- 恒等式抽查（H、Hᵀ、BDF 相容行、厚度只计一次、压力 pin、`pin=None` 原始行）全部在 0–2.8e-17 内成立；
- 拒绝条件抽查（真翻转、质量、非法 `x`、stale stencil、stale 几何、混合拓扑、混合载荷）全部按预期抛出并定位；
- 实现者报告 30 个抽样数字（另 3 项附加断言）与 `scan_evidence.json` 逐项一致，未发现数值性差异；被验证文件 sha256 在验证前后一致。

未决（交下游，与本记录无关的既有事项）：T22 Newton/失败恢复与 T23 一致性仍按计划推进，本记录不改变其状态；`cases/m5_validation.json` 中 22-01…22-05、23-01…23-10 保持 `not_executed`。

## 7. 证据位置

- 独立脚本：[`tests/test_monolithic_review.py`](../../tests/test_monolithic_review.py)（命令 `.venv/bin/python -m pytest tests/test_monolithic_review.py -q -s`，本次输出含 §3 全部表格；抽样断言见 `test_review_scan_evidence_matches_report`）。
- 实现者报告：[`t21.md`](t21.md)；实现者证据：[`scan_evidence.json`](../../results/m5/t21/scan_evidence.json)、[`fsi1_smoke.json`](../../results/m5/t21/fsi1_smoke.json)。
- 文件指纹：见 §1（`sha256sum` 于 2026-10-11 验证前后各执行一次，一致）。
- 验证结论已在 [m5.md](m5.md) 与 [m5_validation.json](../../cases/m5_validation.json) 中把 21-01…21-05 更新为 `passed`。

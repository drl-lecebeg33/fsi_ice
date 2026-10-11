# A7 独立复核：T18 与 T19 验收准备

日期：2026-10-11。任务 `task_04bc832f6e5d`，Dispatch `ctx_f66363d54d05`。

**结论：T18 固定松弛数值、同方程线性 FE 直接解、候选历史与重试检查通过；输出失败管理存在 D1/D2，尚未通过该部分验收。T19 这里只制定独立验收计划，未验收其实现，不更新完成状态。**

## 角色、范围及代码状态

A7 独立于本次 A5 控制器实现和 A0 集成，未修改任何实现、既有测试、算例或根文档，未提交、推送或派生子代理。本轮可写文件仅本报告、[独立测试](../../tests/test_coupling_review.py) 和 `results/t19/reviewer/`。按要求先读取 AGENTS、spec、gola、agent、[T18 记录](t18.md)、控制器、验证入口及其既有测试；既有报告中的 338 项全量通过是原执行者证据，本复核没有重跑或声称独立执行该全量。

基线提交 `2b5f713`，Python 3.12.3；项目 `.venv/bin/python`，NumPy 2.5.3、SciPy 1.18.1、pytest 9.1.1。设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`。详细版本见 [environment.json](../../results/t19/reviewer/environment.json)。所有测试临时产物指定到允许的 reviewer 目录，并关闭 pytest 缓存。

并行期间 A0 为 T19 扩展了 CLI 与验证配置入口；数值审计时这两个文件与开始时指纹不同，控制器、T18 配置和原 T18 测试仍相同。因而本报告区分开始时指纹 [baseline_sha256.json](../../results/t19/reviewer/baseline_sha256.json) 与交付时 [review_source_sha256.json](../../results/t19/reviewer/review_source_sha256.json)，不把后续入口变化误记为 A7 改动或整个工作区冻结。T19 的实现交付须在新 Dispatch 下重新取证。

## 实际命令、退出码与证据

以下命令从项目根目录执行，均加上述三个环境变量；真实完整参数与退出码保存在对应 `*_run.json`。

```bash
.venv/bin/python results/t19/reviewer/numeric_audit.py
.venv/bin/python -m pytest tests/test_coupling_review.py tests/test_partitioned.py \
  -q -rxX -p no:cacheprovider --basetemp=results/t19/reviewer/pytest_tmp_final
.venv/bin/python -m pytest tests/test_coupling_review.py -q --runxfail \
  -k 'invalid_config_cli or setup_and_postsolve' -p no:cacheprovider \
  --basetemp=results/t19/reviewer/defect_tmp_final
```

| 实际检查 | 结果 | 原始证据 |
| --- | --- | --- |
| 独立数值审计脚本，内部运行 FE check 并重建物理方程 | 退出 0，FE 报告 PASS | [命令](../../results/t19/reviewer/numeric_audit_run.json)、[日志](../../results/t19/reviewer/numeric_audit.log)、[数值](../../results/t19/reviewer/numeric_evidence.json)、[FE 报告](../../results/t19/reviewer/fe_check/report.json) |
| 原 T18 测试与 A7 独立检查 | 退出 0，64 passed、6 xfailed，4.47 s；其中原 T18 53 项、新数值/事务 11 项 | [命令](../../results/t19/reviewer/pytest_t18_review_final_run.json)、[日志](../../results/t19/reviewer/pytest_t18_review_final.log) |
| 去掉 xfail 遮罩单独执行 6 个缺陷用例 | **pytest 退出 1，6 failed、11 deselected，2.05 s** | [命令](../../results/t19/reviewer/defects_unmasked_final_run.json)、[完整失败栈](../../results/t19/reviewer/defects_unmasked_final.log) |
| 初始 CLI 最小复现：成功后非法 dt 重跑 | 成功退出 0，重跑退出 2，旧 PASS/solution 留存 | [复现数据与两次命令](../../results/t19/reviewer/stale_marker_reproduction.json) |

6 个 `strict=True` xfail 是明确登记的未修复缺陷，**不是通过**；修复后必须去除对应标记并重新实际执行。第一次尚未扩展输入类型时，结果为 64 passed/3 xfailed、单独运行 3 failed，旧日志也保留供时间顺序核对。主 shell 的日志收集脚本退出 0 不代表内部 pytest 都成功；以上列出的子命令退出码是实际判据。

## T18 数值复核

### 范数、尺度与最终状态

控制器定义 `r=G(x)-x`，`N=max_i |r_i|/S_i`，`T=atol+rtol*max(max_i |x_i|/S_i,max_i |G_i(x)|/S_i)`。S 是每个界面分量的正物理尺度，atol/rtol 无量纲；收敛采用未松弛残差。实际独立检查用 `[1e6,1e-8]` 目标和 `[1e6,1e-9]` 尺度，第二分量虽绝对值小仍得到 N=10 并失败，不能被另一分量单位掩盖。既有单位变换、极小 omega 和非法/溢出输入回归实际执行通过。

返回的是最后已评估 x，固体 trace 为 G(x)，两者允许在容差内不相等；不能在成功时将输入替换成未重新求过流体的 G(x)。A7 使用最后返回 x 重新完成 fluid/load/solid 求解，位移、速度、压力和载荷逐数组完全相同。默认 omega=.5 的输入与目标最大差为 `7.1020626039e-14 m`，保留了真实非零残差。

opaque 单场对象和“回调不接受历史”属于回调责任边界：控制器无法自动证明任意用户回调解过物理方程或没有改外部可变状态。本复核验证的是实际 FE 回调与原有失败路径。

### 固定点残差与物理残差

从原单场矩阵独立计算：

```text
R_s = (K + alpha² M_s)d - b_s - scatter(f_s_gamma)
R_f = (A_stokes + alpha M_f)[u,p] - b_f
R_gamma = u_gamma - H v_s_gamma
```

只对相应自由方程行检查平衡，夹持/速度 Dirichlet 行为反力，不要求原平衡残差为零。在零历史的本算例中流体以 x 求得 `u_gamma=alpha H x`，固体速度为 `alpha G(x)`，故 `R_gamma=-alpha H r`。R_s/R_f 为真实单场求解精度；耦合未闭合的物理缺陷集中于运动约束。

刻意采用较松容差 `atol=2e-4, rtol=0`，5 次评估时 N=`1.23817775675e-4`，运动缺陷 `1.23817775675e-5 m/s`；上述预测的最大误差仅 `2.0328790734e-20 m/s`。固体自由行残差 `1.7347234760e-18 N`、流体自由行最大缺陷 `3.4694469520e-17`。

因此界面残差与全部物理残差不是任意问题下的同义指标。即使仿射映射，误差也受 `(I-G')` 条件数影响，不能把一次固定点残差小当作直接解误差保证；FE 验收同时保留直接差、物理行残差、运动与连续性门槛。

默认冻结尺度为 d=.01 m、u=.1 m/s、p=1 Pa、load=1 N；组合自由行残差按力=load、运动=u、积分连续性=u×1 m 分别归一化。本例 alpha=10/s，alpha×S_d/S_u=1，且匹配 H 保持节点/中点插值，因此组合运动残差数值恰与界面范数接近；这依赖本配置，不能普遍推广。

### 实际 FE 与直接块系统

两域各 12 个三角形，固体 24 DOF、流体 82 DOF，P1 平面应变线弹性/BDF1 与 P2/P1 非定常 Stokes，固定参考几何、零历史、dt=.1 s。采用既有冻结配置，不重新调门槛。

| omega | 评估数 | N / 末次门槛 | d/u/p/load 最大归一化差 | 组合归一化残差 | 运动缺陷 m/s | 连续性 |
| --- | --- | --- | --- | --- | --- | --- |
| .25 | 63 | 1.1880e-11 / 1.2516e-11 | 6.3461e-13 / 1.1245e-11 / 7.4401e-12 / 2.2131e-12 | 1.1880e-11 | 1.1880e-12 | 1.8045e-17 |
| .5 | 27 | 7.1021e-12 / 1.2516e-11 | 3.4109e-13 / 6.7610e-12 / 4.1610e-12 / 1.2142e-12 | 7.1021e-12 | 7.1021e-13 | 2.4856e-18 |
| 1 | 9 | 9.9994e-12 / 1.2516e-11 | 8.3761e-13 / 1.5788e-11 / 8.7103e-12 / 2.7395e-12 | 9.9994e-12 | 9.9994e-13 | 1.0553e-17 |

门槛仍为四场差 ≤1e-8、组合残差 ≤1e-9、运动 ≤1e-9 m/s、连续性 ≤1e-10，均通过。详细全精度数值、最后门槛、单场残差及每次残差向量见 reviewer 数值 JSON 和 `fe_check/solution.npz`、`iterations.csv`。

直接系统不是独立重写的物理求解器：固体/流体对角块使用相同 FE 矩阵，运动块使用相同 alpha/H；载荷块由已有解析 JVP 构建，而分区路径直接积分实际 Cauchy 应力。A7 对 3 个固定种子随机、未收敛的全场向量比较块的载荷作用与真实应力积分，最大力行误差 `2.2737367544e-13 N`。这比只对最后收敛状态比较更能排除偶然抵消。直接解在原物理自由行的固体/流体/运动缺陷分别为 `5.6898930012e-16 N`、`3.4802889737e-17`、`1.6263032587e-19 m/s`。确认它求解的是本线性验证所定义的同一方程，仍不代表 T21 的非线性整体 FSI。

### 边界、压力和历史

流体外边界与界面重合 DOF 为 `[0,1]`，即左下端点两速度分量；对应固体界面与夹持重合 `[16,17]`。该点固体夹持、流体左侧静止，两条条件一致。默认零初值与各次更新保持端点为零，直接解也满足；对该点注入非零初值时实际返回 fluid 阶段冲突失败。直接系统在该点外速度约束取代运动行有效，是因为固体对应点也夹持；若后续改边界，必须重新验证这一条件。

outer_dofs 中压力 pin 数为 0。右侧自然出口包含自由速度方程，常数压力不是自由系统的零模。对实际解压力统一增加 1 Pa，自由流体行最大缺陷为 `.2666666666666663`，证明本算例无压力任意常数；不能把“未 pin”误判成漏约束，也不能推广到全速度 Dirichlet 算例。

独立执行真实 Stokes 求解后载荷阶段失败、迭代耗尽，再从同一已接受历史重试；接口输入、d/u/p/load 与成功基线逐数组一致，solid/fluid BDF 的时间、generation、全部状态数组保持不变，参考网格坐标完全不变。实际成功后耗尽重跑会清除旧报告/解并保存 FAIL，原 direct_reference 故障回归也通过。控制器不接受时间历史；本例不是完整瞬态 FSI 事务验收。

## 已确认缺陷及修复归属

| 编号 | 定位与最小复现 | 实际失败及影响 | 修复归属与复核条件 |
| --- | --- | --- | --- |
| D1 | 初始 `src/fsi_ice/cli.py:32` 的 load 调用先于 run；当前 T19 CLI 同一位置在第 33 行。先合法运行到同一 output，再用 dt=0、poisson=.5、损坏 JSON 或缺失文件运行 | CLI 均退出 2，但旧 report.json 仍为 PASS，solution.npz 仍在，没有本次 failure.json；清理从未进入 | A0/CLI 与验证入口；让每次已指明输出目录的运行有自己的失败状态，清除或明确隔离旧成功产物，保存输入故障阶段/原因，保留原输入路径，不能伪称读取了无效配置 |
| D2 | 初始 `validation/partitioned.py:251` 构造及 `:279` metrics 在 try 外；当前为 `:255` 与 `:283`，记录 try 是 `:260–282`。成功后注入 assemble_stokes 或 metrics 抛 RuntimeError | 旧成功会清掉，但两者均没有 failure.json；阶段、原因、环境及可获得历史诊断未持久化 | A0/验证入口；将 setup/metrics 纳入失败证据范围，构造未完成时不能引用不存在的 problem 或伪造 histories_unchanged；成功数据写出异常也需要明确诊断 |

D1/D2 已分别通过 Orca escalation `msg_e146165a37c5` / `msg_25780e4e1597` 报告，协调者已确认接收，并明确后续新修复 Dispatch 处理。A7 未自行修复原实现。新增测试用例的行号和失败栈保存在未遮罩日志中，修复时必须改为正常断言并实际重跑；本轮验收准备任务完成并不等于这些实现缺陷已修好。

## T19/Aitken 独立验收计划（待执行）

协调冻结接口为 `CouplingOptions(method='fixed'/'aitken', minimum_relaxation=.05, maximum_relaxation=1., aitken_denominator_tolerance=1e-14)`，保留原 fixed 默认、既有位置字段顺序与接口。已读取 [A5 计划](../development/t19_plan.md)，以下门槛与之协调；仍不把计划作为执行证据。

必要算法：使用 q_k=r_k/S、delta=q_k-q_(k-1)，以**上一实际应用**系数计算 `omega_raw=-omega_prev*(q_prev·delta)/(delta·delta)`；首次非收敛用初始 relaxation，有效系数夹到显式上下界。相对退化采用 `delta·delta <= tolerance*max(q_prev·q_prev,q·q)`，保留上一有限系数；不得用固定绝对分母阈值使小残差迭代过早冻结。收敛/耗尽行先终止，不算或应用新系数。

| 独立验收项 | 门槛与可证伪检查 | 状态 |
| --- | --- | --- |
| 标量解析解与索引 | 用 G(x)=1-.5x、S=1、x0=0、omega0=.5：前两次真实更新应为 .5 与 2/3，输入序列 0→.5→2/3，第三评估收敛到2/3；解归一化误差 ≤1e-9，系数 ≤1e-12 浮点误差 | 未执行 |
| 超过默认上界的显式配置 | G(x)=.5x+1、显式 maximum=2，初始 .5：应允许动态系数2并收敛到2；默认 maximum=1 时应截断并记录 clipped，不能暗中超界 | 未执行 |
| 多分量内积与单位 | 非对角仿射映射和逐分量异单位；同时改变 x/r/S 的单位后实际 omega 与归一化残差序列误差 ≤1e-12，最终解析解误差 ≤1e-9；至少一组能区分 r 内积与 r/S 内积 | 未执行 |
| 分母零与近退化 | G(x)=x+c 产生不收敛且相邻残差相同，记录 denominator_degenerate 并保留上一实际 omega；构造 delta²/reference 在1e-14两侧的序列，核验分支。r=0 应先收敛而不除零 | 未执行 |
| 系数截断与退化原因 | 分别构造原始负值、低于.05、高于1；更新使用夹界后的值，下一步公式也用夹界后的上一值。区分 initial、aitken、clipped、denominator_degenerate | 未执行 |
| 非有限/算术失败 | 有限 q 但内积溢出、非有限系数、更新溢出必须 relaxation 阶段失败，保存最后已评估输入及之前历史，不留下 NaN/Inf 的成功/日志系数；原因 coefficient_nonfinite。极小 q 检查退化策略与浮点可表达范围 | 未执行 |
| 真实残差与终止 | 小实际更新但大未松弛残差不成功；max_iterations=1、已收敛初值仅评估一次；converged/max_iterations 行无实际新更新且原因明确，不能用终止行系数推断执行过更新 | 未执行 |
| 历史记录与重放 | 每个非终止行记录该行**实际用于下一输入**的 omega/原因，按 `x_next=x+omega*r` 重放输入序列 ≤1e-12；CSV 与返回历史一致，失败有原因。终止行保留最近 omega（首行初始值），明确无更新 | 未执行 |
| 新选项拒绝 | method非枚举、bool/数组/NaN/Inf系数、minimum≤0、minimum>maximum、非正分母容差拒绝；Aitken 初始值在界内；fixed 默认行为不被新边界收窄 | 未执行 |
| fixed兼容 | 默认 fixed 与显式 fixed、旧配置均复现本报告 `.5/27次` 全场解及残差序列；原 T18 53 项再次实际通过。新增原因字段有默认值，不能破坏既有构造 | 未执行 |
| 历史与失败重启 | Aitken 失败后新 solve 不继承 q_prev/omega；真实 FE 回调失败及耗尽后的重试和冷启动逐数组相同，BDF time/generation/状态与网格不变 | 未执行 |
| FE 同方程解与独立物理残差 | 使用本冻结配置和同一直接解：d/u/p/load差各≤1e-8，组合≤1e-9，运动≤1e-9 m/s，连续性≤1e-10；另外重建原自由平衡行及 `-alpha H r`，不只读报告PASS | 未执行 |
| 输出失败回归 | D1/D2正常断言通过后，fixed/aitken成功、数值验收FAIL退出1、输入/求解异常退出2分别验证，新运行不能冒用旧PASS | 未执行 |

T19 须记录完整配置、源指纹、环境、每次实际 omega/原因/残差/门槛、终止输入和全场结果，比较同容差下 fixed 与 Aitken 的实测次数。本报告固定方法的 27 次仅为 `.5` 基线；omega=1 实测9次，不能只挑27次据此声称全局加速，不要求每个问题都更快，也不依据这次短运行做性能结论。

后续建议在**新 Dispatch**下运行项目 `.venv` 的 `tests/test_aitken.py`、`tests/test_partitioned.py`、本独立测试及必要 CLI 检查，把缺陷 xfail 转成正常检查，并重建上述独立数值证据。FSI1 来源、非线性 ALE、整体 Newton 与正式瞬态集成不属于本轮验收。

# A7 / T17 与 M3 验收设计

日期：2026-10-11。职责：A7 单场证据复核、T14–T17 验收设计；T17 实现负责人仍为 A0，相关接口由 A2/A3/A4 复核。本轮只编辑本文件，未修改代码、其他文档或里程碑勾选，未提交或上传，未启动其他 worker。**设计交付不等于功能完成：M3/T14–T17 均保持未实现、未验收。**

依据：[协作规范](../../AGENTS.md)、[规格](../../spec.md)、[阶段目标](../../gola.md)、[任务分工](../../agent.md)、[环境记录](../environment.md)。本轮复核是读取本地实测证据及执行已有快速测试，不重新核验外部原始网站，不宣称对原实现做过独立重算。

## 1. 单场依赖与实测证据复核

### 1.1 证据来源与结论边界

读取 [单场验证记录](../validation/single_fields.md)、[单场配置](../../cases/single_fields.json)、[CSM3 配置](../../cases/csm3.json)、[CFD3 配置](../../cases/cfd3.json) 及以下实际 JSON，而非只根据进度勾选判断：

| 本地报告 | 实际状态与复核内容 | 对下游的含义 |
| --- | --- | --- |
| [单场汇总](../../results/single_fields/report.json) | `PASS`，`solid`/`fluid` 均为 true | 静力、稳态及制造解证据入口存在 |
| [固体单场](../../results/single_fields/solid/report.json) | `PASS`，梁误差降低、反力、CSM 最细参考、正 Jacobian、力平衡、残差、BDF1/2 时间阶检查均为 true | 支持 M1 的静力和时间验证依赖 |
| [流体单场](../../results/single_fields/fluid/report.json) | `PASS`，速度/压力空间阶、Poiseuille、CFD 最细参考、质量/连续性、残差、BDF1/2 时间阶均为 true | 支持 M2 的稳态和时间验证依赖 |
| [CSM3](../../results/csm3/report.json) | `PASS`，2026-10-11 00:30:35 +08:00，13 项 checks 均为 true | 支持 T08 动态验收 |
| [CFD3 加密研究](../../results/cfd3_refinement/report.json) | `PASS`，2026-10-11 00:30:34 +08:00，10 项 checks 均为 true | 支持 T12 的最终细化解验收 |
| [首轮 CFD3](../../results/cfd3_validation/report.json)、[原始报告备份](../../results/cfd3_validation/report_original.json) | 两文件实际存在且均为 `FAIL` | 保留首轮未达到振幅门槛的事实 |

上述动态报告基于 `dd6232b829c976a0307135200138ced8c00fbc00` 的 modified 工作区，Python 3.12.3，NumPy 2.5.3、SciPy 1.18.1、Matplotlib 3.11.2。报告执行方式为对完整已保存轨迹后处理；监测 CSV 不是全场重启文件，不能据此恢复压力、速度、固体或网格状态。`results/` 是默认忽略的本地产物，链接只保证本工作区存在；交付时应另行保留报告及数据摘要，不能假设远端仓库拥有它们。

CSM3 采用平面应变 P1 StVK、从静止突然施加重力，运行至 5 s。CFD3 为固定圆柱和梁、P2/P1，ρ=1000 kg/m³、μ=1 Pa·s、入口平均速度 2 m/s、前 2 s 余弦启动、自然零牵引出口，运行至 10 s。两者均为固定步长 BDF2、一阶系统的 BDF1 启动。周期诊断使用最后三个完整周期、三点抛物线峰时；均值为极值中点，振幅为半极差，不能改称时间平均或 RMS。CFD 阻力统计使用最后一个升力周期。

| 最终解 | 实测值 | 与报告参考比较 | 既定门槛 |
| --- | --- | --- | --- |
| CSM3，280×16，dt=0.005 s | y 频率 1.104788845 Hz，振幅 0.063790670 m，均值 -0.062504687 m | 频率 0.4810%，振幅 2.1015%，均值 1.7330% | 5%、10%、10% |
| 同一 CSM3 的 x | 频率 1.103501794 Hz，均值 -0.013732865 m，振幅 0.013731338 m | 0.3640%、约 4.00%、约 4.01% | 5%、10%、10% |
| CFD3，14915 单元，dt=0.005 s | 升力频率 4.417049692 Hz，振幅 436.155399 N/m，均值 -11.536134 N/m | 频率 0.4880%，振幅 0.3779%，均值绝对误差 0.356866 N/m | 5%、10%、20 N/m |
| 同一 CFD3 的阻力 | 均值 435.607849 N/m，振幅 5.443403 N/m | 0.8743%、3.1130% | 10%、10% |

CSM3 最终采样窗约 1.36–4.08 s，CFD3 约 9.07–9.75 s；CSM3 y 周期/振幅变化约 0.433%/0.097%，CFD3 约 0.007%/0.045%，均低于 5%。CSM3 最终最大残差 6.055e-9、最小 det(F)=0.976504；CFD3 最大残差约 9.745e-9、最大质量误差约 1.354e-15。动态配置残差/质量门槛分别为 1e-8/1e-10。

### 1.2 粗网格与大步长失败必须保留

| 研究点 | 实际表现 | 判定 |
| --- | --- | --- |
| CSM3，70×4，dt=0.01 | 频率 1.252930 Hz、y 振幅 0.049260 m | 虽解析出周期，频率约 13.96%、振幅约 24.40% 的偏差超门槛；粗网格偏硬 |
| CFD3，h=0.02，dt=0.01，1015 单元 | `resolved=false`，`cycles=0`，原因为完整可解析周期不足 | 禁止填入虚假频率或将运行成功解释为精度通过 |
| CFD3，h=0.01，dt=0.01，3782 单元 | 升力振幅 363.537873、阻力振幅 4.208692 N/m | 分别约 16.96%、25.09% 误差；未通过振幅门槛 |
| 首轮 h=0.01，dt=0.005 | 升力振幅 370.1852、阻力振幅 4.35578 N/m | 原记录约 15.45%、22.47% 误差，报告 `FAIL` |
| CFD3，h=0.005，dt=0.02 | 升力振幅 123.201543 N/m，三个周期振幅变化 29.217% | 不满足稳定性及振幅门槛 |
| CFD3，h=0.005，dt=0.01 | 阻力振幅 4.979778 N/m | 约 11.365% 误差，仍超 10% |

动态报告对全部研究点检查残差、质量和正 Jacobian，但参考精度及周期稳定性 checks 针对最终细化点。因此 `PASS` 不是“所有粗网格也通过”。空间研究固定 dt=0.01，时间研究固定最细网格，各有三档；这些结果支持当前 M1/M2 的单场完成结论，不能当作完整 FSI、ALE 几何守恒或全场事务已验证的证据，也不能从周期频率的非单调误差直接宣称 BDF2 收敛阶。时间阶另由制造解测试证明。

### 1.3 本轮实际执行与输出

读取及校验脚本在项目 `.venv` 内运行，使用现有 [read_complete_trajectory](../../src/fsi_ice/validation/transient.py) 对报告中的每个唯一目录重查完整采样、mesh/dt/cell 数，重新提取统计与残差/质量/Jacobian；没有调用 `run_trajectory` 或生成 Gmsh 网格。CSM3 的 5 个唯一目录共 10 个文件 SHA-256、CFD3 的 5 个唯一目录共 20 个文件 SHA-256 全部匹配（mesh.msh/mesh.geo 在目录的 `mesh/` 下）；两个复用 CFD 目录的 `origin.original_report_sha256` 也匹配。报告中重复引用 mesh_2/dt=0.01 只计一次。

复核脚本的核心操作如下，`assert` 实际全部通过，未写入报告或数据：

```python
import hashlib
import json
from pathlib import Path
from fsi_ice.validation.transient import read_complete_trajectory

for name in ("csm3", "cfd3_refinement"):
    report = json.loads((Path("results") / name / "report.json").read_text())
    seen = set()
    for rows in report["results"].values():
        for row in rows:
            folder = Path(row["saved_evidence"]["folder"])
            if folder in seen:
                continue
            seen.add(folder)
            actual = read_complete_trajectory(
                report["case"], row["mesh"], row["dt"],
                report["config"]["end_time"], folder)
            for key in ("statistics", "maximum_residual", "maximum_mass_error",
                        "minimum_jacobian"):
                assert actual[key] == row[key]
            for filename, digest in row["saved_evidence"]["files_sha256"].items():
                path = folder / filename
                if not path.exists():
                    path = folder / "mesh" / filename
                assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            origin = row["saved_evidence"].get("origin")
            if origin:
                path = Path(origin["original_report"])
                assert hashlib.sha256(path.read_bytes()).hexdigest() == origin["original_report_sha256"]
```

运行环境前缀：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python`。实际输出摘要：

```text
csm3: 5 unique complete trajectories rechecked; statistics, residual/mass/J and provenance hashes match; no physical solve
cfd3_refinement: 5 unique complete trajectories rechecked; statistics, residual/mass/J and provenance hashes match; no physical solve
results/cfd3_validation/report.json exists= True status= FAIL
results/cfd3_validation/report_original.json exists= True status= FAIL
```

已有快速测试实际命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -p no:cacheprovider \
  tests/test_stvk_time.py tests/test_single_fields.py \
  tests/test_transient.py tests/test_flow_linear_solver.py
```

实际输出：Python 3.12.3、pytest 9.1.1、pluggy 1.6.0；`collected 34 items`；各文件分别 6/10/15/3 项；**`34 passed in 2.68s`，退出码 0**。该次运行覆盖 StVK 导数/失败、ODE 及真实单场 BDF 阶、单场候选失败保持历史、周期统计、保存轨迹异常、失败步诊断及 LU 与直接求解一致性。单场记录中的“93 passed in 5.30s”是此前证据，本轮没有重跑全部 93 项，更没有执行下面尚未实现的 M3 测试。

当前真实测试入口为 [CLI](../../src/fsi_ice/cli.py) 的 `.venv/bin/python -m fsi_ice test`，它从任意工作目录定位仓库并运行 pytest；动态命令 `csm3`/`cfd3 --postprocess-only` 会重写报告/图，本轮选择直接调用只读读取函数。异常采样、失败标记及禁止虚假频率的入口见 [transient 测试](../../tests/test_transient.py)；单场历史入口见 [BDF/固体测试](../../tests/test_stvk_time.py)、[单场测试](../../tests/test_single_fields.py)，缓存入口见 [线性求解测试](../../tests/test_flow_linear_solver.py)。

## 2. T17 的真实代码约束与缺口

以下是已存在行为；后文 API 名称和文件名均为拟议设计，不能据此声称功能可运行。

| 现有入口 | 已实现行为 | M3 需要补充的约束 |
| --- | --- | --- |
| [time/bdf.py](../../src/fsi_ice/time/bdf.py) | `frozen_copy` 拷贝且检查有限值；`states` 返回防写副本；BDF2 启动用 BDF1；`context` 带 owner 身份、generation、时间；`accept` 检查 owner/generation/时间并复制候选 | 只能保证一个历史对象的接受；没有全场接受协调器 |
| 同一 BDF 模块 | `snapshot()` 仅返回 time/generation/states；`restore()` 校验有限值与形状，但不含 owner、dt/order，也把 generation 恢复为旧值 | 同形状其他历史快照可能被恢复；恢复后原同 owner/generation 的 context 可重新有效；不能把它当全场持久化/鉴权快照 |
| [solid/dynamics.py](../../src/fsi_ice/solid/dynamics.py) | 状态为 `(2,n_s,2)` 的 d/v；`solve_solid_step` 从 context 消去速度，`v=alpha*d+hd`，质量矩阵在固定参考网格上组装，返回候选与诊断 | 不调用 accept；候选 d/v 都需要事务持有，外载、约束及 initial 必须从同一尝试输入构造 |
| [fluid/dynamics.py](../../src/fsi_ice/fluid/dynamics.py) | 历史仅为 `(n_P2,2)` 速度；流体候选通过 `solve_navier_stokes` 返回 u/p | 压力虽然无 BDF 历史，仍必须被保存/接受/恢复；参数 `velocity` 是速度边界数据，不是已接受速度历史 |
| [StVK](../../src/fsi_ice/solid/stvk.py)、[Taylor–Hood](../../src/fsi_ice/fluid/taylor_hood.py) | 求解器复制 initial，单场残差/切线已有方向导数验证；FlowSolution u/p 为底层解向量的切片/视图 | dataclass frozen 不会冻结内部 ndarray；接受时必须深复制，禁止用 result 或 initial 的共享视图作为接受缓冲 |
| [transient.py](../../src/fsi_ice/validation/transient.py) | CSM/CFD 先求解再单独 accept；CFD 的 u/p initial、边界监测器和 LU 求解器在历史外持有；单场异常保留成功监测行 | 候选后诊断、网格、界面或另一场失败不在当前单场回滚范围内；CSV 只存监测值 |
| [TriMesh](../../src/fsi_ice/mesh/triangular.py) | 构造时复制并置只读，拒绝负方向/退化；提供质量指标 | 暴露数组置只读不是所有权隔离的完整保证；ALE 禁止原地改已接受 points，需构造新候选网格，不能换节点排序以隐藏翻转 |

`BDFHistory.dt/order/time` 当前是可赋值属性；事务层必须封装，不允许活动尝试期间修改。BDFContext 的 history_term 虽只读，仍需避免向外暴露可重启写权限的内部拥有缓冲。候选求解函数也不会验证 context 属于哪个全场事务，这项必须由适配层完成，不能只依赖数组形状。

## 3. 全场 transaction、接受与失败回滚设计

### 3.1 状态、数组所有权及标识

建议第一版采用“已接受状态根对象 + 私有尝试工作区 + 新根一次发布”，单进程单写者；多线程读写若启用须用同一锁保护发布和取根，不以多个属性逐项赋值充当原子提交。

| 状态对象 | 必须包含的字段 | owner 与共享规则 |
| --- | --- | --- |
| `AcceptedState`（拟议） | 全局 t、固定 dt/order、接受步号、accepted_epoch；固体 d/v 与最近两层一阶历史；流体 u/p 与速度历史；网格位移 a、当前 x=X+a、网格速度 w 与 a 历史；界面运动/载荷；已接受预测初值与监测行 | A0 是唯一写入方；内部拥有独立深复制的有限 ndarray；公开读入口返回副本或不暴露底层拥有缓冲；已接受历史永不原地修改 |
| 参考与拓扑 | 固体 X、流体参考 X、cells、边界标签、P1/P2 DOF 及界面关联、材料/BC 配置指纹 | A1/A4 提供；只读长期共享且不向候选代码暴露可写别名；M3 固定拓扑，不支持重网格；阶次不同必须显式映射 |
| `StepTransaction`（拟议） | base_epoch、独立单调 attempt_id、owner token、status；三个私有 BDFHistory；共同 target_time 与系数；candidate d/v/u/p/a/x/w；局部界面载荷、预测器、求解诊断与监测行；候选几何版本 | A0 生命周期管理；A2/A3/A4 只收到所属场 context 和工作区副本，单场求解器不写全局 history；尝试关闭后禁止重用其状态/context |
| 可恢复快照 | transaction owner、base_epoch、配置/拓扑指纹、t/dt/order、历史层数及完整物理数组 | 只接受本协调器生成的快照，不能接受调用者任意拼接的 BDF tuple；不同 owner/config/shape 的恢复请求在任何写入前拒绝 |

数组验收要检查初始化输入、候选返回值、接受缓冲、两层历史、snapshot 和公开读取值之间的 `np.shares_memory`。工作区和快照不与接受根共享可变存储；接受后再修改候选或重新开启外部数组 writeable 不能改变根。只读拓扑的内部可信共享可保留，但所有对外可修改视图必须隔离。稀疏矩阵的 data/indices/indptr 也不能被工作区原地覆盖。

压力不加伪造的 BDF 时间项；p 是代数量，仍与 u 同步发布。w_kin 从与固体相容的 BDF 网格位移求导获得；界面 `a=H_vertex*d`、`w_kin=H*v` 的条件是 H 固定于参考拓扑且历史、初始值、alpha 和目标时间一致，并且 `v=D_BDF d`。未收敛整体候选若 d/v 独立，应检查运动残差，不事后覆盖 w_kin。本文件状态表中的 w 均指 w_kin；ALE 计划中的扫掠通量 Q_BDF/有效输运速度 w_gcl 是另外的几何输运数据，不能作为物理界面速度或功的输入。

### 3.2 owner/generation 与启动历史

每次 begin 从接受根复制私有 BDF 历史，保留该历史 dt/order/time 和 1 或 2 层状态，但创建新的 history 身份 token。复制时可由 A0 的私有适配器使用现有 snapshot/restore；裸 tuple 不作为公开恢复协议，严禁在 live 根上调用 restore。候选 context 只从这些私有历史生成，所有迭代复用同一已接受历史，不调用 accept。三个历史须同 t/dt/order/层数，BDF1 启动 alpha=1/dt，成熟 BDF2 alpha=1.5/dt。各场 history owner 不同是正常情况，不要求它们相等；共同 transaction 标识负责跨场关联。

若冻结 [ALE 计划](ale_plan.md) 的历史质量/扫掠输运路线，根还须保存与速度历史一一对应的最近两份接受几何（可从完整 a 历史和固定 X 重建）、质量算子版本和公共 `BDFStencil` 的显式权重/时刻/步号。BDFContext 只有合并 history_term，不足以计算不同构形的历史质量时间项；A0 唯一提供并校验 stencil，T15 不自己猜系数。候选 Q_BDF 及 g_BDF 随几何版本产生，失败即丢弃。

现有 per-history owner/generation 校验继续保留，同时协调器校验 transaction owner、attempt_id、base_epoch、status、字段名/DOF 布局和 target_time；不能将全局 epoch 和局部 BDF generation 混为一谈。accept 之前要求全局根仍为 base_epoch。失败或关闭的 attempt_id 永不复用；新 begin 创建新 owner，故失败尝试的 context 不能用于重试。成功发布后旧根 context、旧候选、重复 accept 也必须拒绝。被拒绝的验证不得修改根。

回滚要求物理历史、t、历史层数和原有 BDF generation 与开始时一致；**attempt_id/生命周期审计计数允许且必须单调增加**，避免恢复旧 generation 造成 ABA 问题。接受根封装自己的 BDFHistory，外部不可拿裸对象调用 accept；旧 BDF 本身允许 restore 后重用 context 的事实不能被包装层当作许可。

M3 只验收相同 dt 的失败重试。当前 BDF 是固定步长，不能失败后直接 `history.dt /= 2` 并继续套固定步长 BDF2；变步长 BDF2、重启降阶及最终时间对齐属于另行定义的扩展。非法 dt 变更在 begin 时失败且不改变状态。

### 3.3 生命周期与原子接受

建议状态流为 `open -> prepared -> committed` 或 `open/prepared -> aborted`，终态不可重入。每一尝试按以下顺序执行：

1. **begin**：获取唯一写权限，读取一个接受根；校验三个历史同步、初值/配置有限且兼容；深复制工作区并生成 contexts。失败不创建可继续使用的事务。
2. **evaluate**：几何、固体、流体及界面可反复求候选；每次几何变化使用独立版本并建立候选算子。所有求解均由同一 base 历史产生；不得每个迭代推进一次 history。总状态有任何 NaN/Inf、det(F)≤0、网格翻转、线性/非线性失败即 abort。
3. **validate**：检查各场收敛、有限值/形状、BC/压力约束、正几何、质量、界面运动和载荷/功门槛、owner/epoch/时间一致性；先计算完整诊断和可序列化监测行。禁止在先接受固体后才发现流体质量或界面失败。
4. **prepare**：预分配、深复制并冻结下一根的全部数据。在三个**私有**历史上分别 `accept`；每个单场 `accept` 都可能抛错，但任何部分完成只存在于私有根。确认新根的各场时刻均为 target_time、历史层数一致、generation 各增 1、全局 accepted_epoch/步号各增 1。全部可失败运算在发布前完成。
5. **publish**：锁内再查 base_epoch 和 transaction 状态，仅一次替换全局 `accepted_state` 引用；读取者取得完整新根。无物理求解、数组分配、诊断或多个单场接受发生在该操作之后。事务标记 committed，重入必须显式拒绝。
6. **abort/retry**：发布前任何错误丢弃整个私有根与工作区，不修改 live 接受根；清除候选几何/求解缓存并记录失败阶段、场、步号/t/dt、attempt_id/base_epoch、残差/单元号/原因。用新的事务和同一接受根、相同 dt/初值重试，不继承失败的预测器或迭代历史。

不采用“依次修改 live 固体/流体/网格，再依赖补偿恢复”的接受路径；测试仍在私有 accept 之间注入异常，证明用户观察不到部分接受。全场一致性优先于复用现有轨迹循环的独立 accept 调用。T17 最小集成使用规定运动/小型独立场的组合，不要求实现 T18 分区固定点或 T21 整体 Newton；后续两种耦合方法都调用同一事务和单场残差。

成功步监测数据在 prepare 时构造，并跟随新根发布；失败尝试日志单独保存，不附加伪成功监测行。发布后外部文件 I/O 失败应报告 `committed_output_pending`，禁止回滚物理步或重新推进同一步；以 accepted_epoch/步号幂等补写。M3 不承诺跨进程崩溃恢复，若将来需要，另行设计完整状态 checkpoint；目前的 CSV、summary.json 和 `--postprocess-only` 均不具备这一能力。

### 3.4 网格、算子与求解器缓存

| 实际或拟议缓存 | 风险 | M3 最小安全策略及验收 |
| --- | --- | --- |
| TriMesh / P2Space 当前坐标、物理梯度、面积/法向 | 旧几何数据与新 x 混用 | 每次候选几何构造新网格/P2 坐标；固定 cells/DOF 标识；geometry_id=(attempt_id,candidate_revision)，即使坐标回到原值也不复用失效版本 |
| FlowOperators.matrix/mass/divergence/gradients 与 cached `convection_tensor` | [assemble_stokes](../../src/fsi_ice/fluid/taylor_hood.py) 的系数和 tensor 绑定组装时几何 | 默认为新候选几何重建整个算子；拓扑/参考形函数可共享；几何、材料或边界配置变化使相关缓存失效；T15 自己提供 ALE 算子，不把固定网格 convection 当 ALE |
| [BoundaryMonitor](../../src/fsi_ice/fluid/forces.py) | 预存长度、法向和梯度只适用于固定网格 | 候选几何下重新创建；界面力、通量和参考牵引不能使用接受根的旧监测器 |
| 固体 mass / 材料 / 切线 | 总拉格朗日 mass 可基于固定 X；切线随 d 改变 | mass 只在 reference/material/厚度指纹一致时共享；候选切线与线搜索缓存不得作为接受物理状态 |
| [ReusedLUSolver](../../src/fsi_ice/fluid/linear_solver.py) 的 `_factor/_free/_uses/_time_coefficient` | 求解过程中可变；当前 prepare 只按时间系数失效，不能证明几何/配置一致 | 每个事务创建新求解器，在该尝试内复用；候选几何/DOF/约束/材料/时间系数变更时重新创建或显式失效；abort 丢弃全部 opaque LU，不深复制 SciPy factor，不共享 live 求解器 |
| relaxation/Aitken/IQN 或 Newton 预测器（后续） | 失败迭代污染下一尝试 | M3 用小型替身验证 owner 生命周期；未来算法历史与 attempt 绑定，提交或丢弃规则统一，不在 T17 提前实现这些算法 |

LU 回退继续检查**当前物理矩阵**的真残差，沿用 ≤1e-10 的现有检查；缓存的预条件残差不能替代物理残差。第一版可牺牲跨步 factor 复用以保证状态安全。失败尝试中的 factorization/iteration 计数作为审计成本保留并单列，不能伪称回滚为零；重试后的物理解/历史应与冷启动一致，但不用强制迭代次数或运行耗时逐位相同。

## 4. 误差定义与统一尺度

以下是 **M3 拟议门槛**，须由 A0/A2/A3/A4 在实现前冻结到配置。它们不改变单场现有门槛，也不代表已经测得。

小型矩形验证采用 L*=1 m、U*=1 m/s、T*=L*/U*=1 s、ρ*=1 kg/m³、μ*=0.01 Pa·s，压力 P*=max(ρ*U*²,μ*U*/L*)=1 Pa；节点载荷包含厚度 b*=1 m，力 F*=P*L*b*=1 N、功 W*=F*L*=1 N·m。现有流体 monitor 返回 N/m，须乘 b 后比较节点总力；牵引为 Pa。时间测试固定空间误差远低于最小时间误差。Turek 派生诊断使用本配置的几何长度、入口速度、ρ/μ/厚度重算尺度，不直接套小型算例单位。

数组误差默认 `max(abs(a-b))/S`，S 使用对应物理尺度而非接近零的参考解；物理残差逐块按动量/力、连续性、几何和界面各自尺度归一化。参考为零也用绝对量除固定尺度。虚功残差分母取 `max(abs(Wf),abs(Ws),W*)`，总力残差取 `max(norm(Ff),norm(Fs),F*)`；不得用很大的总功掩盖小界面项，测试同时报告两侧有符号功。

| 类型 | 门槛/定义 |
| --- | --- |
| 数组/历史回滚 | `np.array_equal`，包括两层数组和压力、初值、界面/网格；t/dt/order/层数/接受计数/原 history generation 精确相等；公开 owner 和 attempt 新旧关系符合第 3 节 |
| 同条件重试 vs 无失败基线 | 各场归一化 max 差 ≤1e-10；接下来至少两步也通过；迭代前 base 快照必须精确一致 |
| 解析线性/常量及匹配映射 | 归一化误差 ≤1e-12；小型调和线性求解自由残差 ≤1e-10；几何浮点误差用 L* 归一化 |
| BDF 时间阶 | 三个 dt=0.1/0.05/0.025 s，BDF1 两次减半阶均 ≥0.9，BDF2 均 ≥1.8，单独检查启动阶与两层历史 |
| Jacobian 方向导数 | 扫 eps=1e-2…1e-7 的尺度化非退化方向，中心差分最小相对误差 ≤1e-5；分母用有量纲缩放后的 `max(norm(Jv),R*)`，另确保 Jv 非近零；非线性非二次残差应显示截断误差下降后舍入上升趋势 |
| 界面运动/功/力 | d/a、v/u/w 映射归一化 max 差 ≤1e-12；代数转置映射功/力误差 ≤1e-12；解析牵引积分/构形转换归一化误差 ≤1e-10 |
| ALE 常量场与质量 | 100 步最大速度漂移/U* ≤1e-9，固定压力规范下压力漂移/P* ≤1e-8；自由物理残差 ≤1e-8，当前离散连续性及相容物理边界通量归一化误差 ≤1e-9；同时保留现有 continuity_residual 诊断 |
| ALE 几何守恒 | 单元 `abs(D_BDF j-Div_X Q_BDF)/max(sum(abs(a_j*j_j)),1/T*)` ≤1e-11，并报告相对 1/T* 的缺陷；共享边扫掠通量相消/U*L* ≤1e-12，全域几何收支/U*L* ≤1e-10；独立几何/边界积分两侧都记录 |
| 几何健康 | 每步 det(J)>TriMesh 的尺度退化阈值；det(F)>0；规定平滑运动验收还要求 min(det(J_current)/det(J_reference))≥0.2、min(quality)≥0.1，低于阈值拒绝该候选并给出单元号；边界自交/候选路径失效亦拒绝 |

质量诊断对完整边界积分，尺度 `max(U*L*,abs(Qin))`，关闭域不能除零 inflow；弱式连续性与净通量均需报告，二者不能互相替代。变体积域区分瞬时运动与离散扫掠：`∮u·n=0`，`∮(u-w_kin)·n=-∫div(w_kin) dx` 是瞬时关系；若采用 ALE 计划的离散扫掠方案，`∮Q_BDF·N dS_X=D_BDF(volume)`，对应有效速度 w_gcl，而一般不等于 `∮w_kin·n`。不得要求相对净通量恒为零，也不得错误地要求有限步长的 BDF 位移速度给出精确 BDF 体积差；需同时报告扫掠 GCL 缺陷和 w_gcl/w_kin 差的时间收敛，不能通过放宽误差掩盖不相容。

实现前冻结项：A0/A3/A4 确认 ALE 守恒时间项/扫掠通量或另一明确离散路线、显式 BDFStencil 与历史几何、GCL 的尺度和观测量、运动路径/自交检查、MMS 与其右端、导数格式、质量配置及所有数值门槛。第 4–5 节给出具体候选值，以便实现和复核，并不将 ALE 推导当作已验证方案；若选择不同路线，先更新测试定义和理由，不能混用历史质量项与未经验证的网格速度。

## 5. T14–T17 具体测试矩阵

拟议测试文件和配置入口只作为归属清单列在第 6 节，目前不存在，不伪造可点击运行链接。

### T14：调和位移、网格速度及几何更新

| 编号 | 输入与过程 | 检查与门槛 |
| --- | --- | --- |
| 14-01 | 4×4/8×8/16×16 单位矩形，全边界施加 a=(0.02,-0.01) 及仿射 a=(0.03X_y,0.02X_x) | 调和扩展再现解析仿射场；误差≤1e-12 L*，边界强制值正确，自由线性残差≤1e-10；参考坐标完全不变 |
| 14-02 | 解析调和场 a_x=0.02 sin(πx)sinh(πy)/sinh(π)、a_y=0，零体源、全边界给解析值，三套网格 | P1 位移 L2 阶≥1.8、H1 阶≥0.9，误差两次加密均下降；不增加非调和物理源项，也不用离散算子生成自身真值 |
| 14-03 | a(t)=b(1-exp(-t/T*)) 的仿射空间场，dt=0.05/0.025/0.0125，分别 BDF1/2；同时做刚性平移和恒速运动 | 对解析 w_kin 检查 BDF1≥0.9/BDF2≥1.8；成熟 BDF2 导数阶另测，启动步单独按 BDF1 检查，不把其误差混进最坏时间点；多次候选不得推进历史 |
| 14-04 | 有限平移/剪切 100 步，加边界顺序打乱但相同拓扑标签 | P2 中边点由当前 P1 顶点一致生成；全程几何健康，固定连接关系；不会用累计增量产生坐标漂移 |
| 14-05 | 单元翻转、压扁、NaN/Inf、错形状/未知标签/冲突角点 BC；极端位移使质量跌破阈值 | 写入根前失败，错误定位标签/单元/尝试；禁止交换节点来修复移动后的负 det；原几何和历史精确不变 |

### T15：ALE 流体残差、GCL 与导数

| 编号 | 输入与过程 | 检查与门槛 |
| --- | --- | --- |
| 15-01 | w=0、a=0；同 mesh/material/BC/history/state 的现有瞬态流体与 ALE 残差/切线 | 逐项归一化差≤1e-12；同求解容差解差≤1e-10，确保复用已有物理离散而非另抄一套 |
| 15-02 | u=(0.4,-0.2)、p 固定规范；规定域平移与体积不变仿射剪切，边界给相容 u，自然外载按解析场生成；100 步，BDF1/2 | 启动和成熟步常量保持、连续性、净流量及 GCL 满足第 4 节；包含 u≠w，不能只测零相对对流 |
| 15-03 | 变体积仿射域、同常量 u 与相容开放边界；独立计算面积、瞬时 w_kin 通量、Q_BDF 扫掠及 BDF 体积变化 | 分开检查瞬时控制域关系与离散 GCL，并检查 w_gcl/w_kin 差随 dt 收敛；避免以关闭不可压缩域配体积增长这一不相容问题作假失败 |
| 15-04 | ALE 制造解：参考平滑散度零速度用 Piola `u=J^-1 F u_hat` 映射到当前域，p 为映射后的解析标量；右端/边界用解析时空导数，具体表达式运行前冻结 | 8/16/32 三网格固定小 dt：P2 速度 L2 阶≥2.5、P1 压力≥1.8；固定空间误差小的网格上三 dt：BDF1≥0.9、BDF2≥1.8；两类误差分开研究 |
| 15-05 | 固定 history，分别扰动 u/p、a 与 w；eps 扫描；冻结 reference/topology，仅重建被扰动几何 | 每个已实现导数块≤1e-5；说明 w 对 a 的 BDF 链式系数；几何变化影响 gradients/mass/divergence/法向，不能只验对流 u 块；M3 不要求完整 FSI Newton |
| 15-06 | 相同时间系数下先移网格再移回，改变 viscosity/BC/pressure pin；故意取旧 geometry_id 算子/监测器/LU | 旧缓存被拒绝或重建；当前力/通量与逐边积分差≤1e-10；真线性残差≤1e-10，和冷组装/直接求解解差≤1e-10 |

### T16：界面拓扑、运动与载荷

| 编号 | 输入与过程 | 检查与门槛 |
| --- | --- | --- |
| 16-01 | 同一分段界面，节点编号打乱、边顺序反向；P1 固体端点到 P2 流体端点/中点 | H 保持常量和仿射值，P1→P2 中点为端点插值，定位每个 DOF；相同编号不是判定匹配的唯一条件；误差≤1e-12 |
| 16-02 | d/v 与 a/u/w_kin 在启动和成熟 BDF2 步使用同一 context 时间；对零、平移、仿射及随机界面值 | a=H_vertex d、u=H v、w_kin=H v；同 H 及两层初始历史且 v=D_BDF d 满足差≤1e-12；故意异步/缺历史/旧 owner 在求解前拒绝 |
| 16-03 | 水平、竖直、斜边上的恒压及线性剪应力；按单元 CCW 计算 n_fluid；反向边标签不改变物理符号 | 流体对固体载荷 `-σ n_fluid`，解析总力及逐边/监测积分≤1e-10；同时检查流体反力，作用反作用符号明确 |
| 16-04 | 定义 g_f_gamma 为以流体 P2 迹基积分的、作用于固体的 Cauchy 应力迹节点力，`f_s_gamma=H.T@g_f_gamma`；固定种子非零 δd/v，多组界面数据 | `δd.T f_s_gamma=(H δd).T g_f_gamma`、功率同理，误差≤1e-12；H 保常量保证总力，保仿射保证当前力矩；流体约束残差反力仅做独立诊断，不混作另一载荷离散 |
| 16-05 | 已知有限拉伸/剪切和旋转；当前 σ、固体参考 F_s/N_s，Nanson：`n_s ds=J_s F_s^-T N_s dS`；厚度 b=0.2/1/2 m | 当前 t_s=-σ n_f=σ n_s 与参考 `T_s0=J_s σ F_s^-T N_s` 积分相符≤1e-10，长度、法向、厚度只计一次；包含非单位 J_s，不用 ALE 体积 F 代替固体 F_s |
| 16-06 | 重复/遗漏节点、重复边、非流形/非外边、法向不一致、零长边、阶次不符或坐标不匹配 | 明确拒绝并定位；M3 不悄悄启用最近邻非匹配映射；失败后映射/界面缓存和接受状态不变 |

H 是参考拓扑上的运动映射，接口需区分一致节点力与牵引样本。如果由牵引样本积分成 g_f_gamma，边界质量/积分权重已经包含在其中，不再错误乘一次边界 mass；只用逐节点复制载荷不能保证功一致。完整紧凑向量、gather/scatter、厚度和参考/当前转换沿用 [界面计划](interface_plan.md)；T16 不把现有 obstacle 的全部圆柱/梁边都当可动梁界面。

### T17：故障注入、部分准备失败与重试

每种故障均在 BDF1 启动（1 层历史）和成熟 BDF2（2 层历史）执行；正常候选至少包含非零 d/v/u/p/a/w 与载荷，避免全零状态掩盖遗漏。基线用小型真实固体/流体残差加规定动网格/界面测试，不进入 FSI 实现；生命周期异常另用替身精确定位注入位置。

| 编号/注入点 | 注入方式 | 验收断言 |
| --- | --- | --- |
| 17-01，begin 与上下文 | 外来同形状 snapshot/context、错字段 owner、过期 epoch、dt/order/层数不同、非法 dt，故意输入可写别名 | 在任何 live 写入前拒绝；复制输入后修改原数组不污染事务/根；同 generation/time 的异 owner 也拒绝 |
| 17-02，几何与固体 | 移动网格后触发翻转/低质量；固体求解成功后触发 det(F) 非正或 Newton/线搜索失败 | 全场回滚，包括 x/a/w、压力/预测器、载荷和两层历史；无成功行；诊断含阶段/单元号或迭代原因 |
| 17-03，流体与缓存 | 固体候选完成后强制奇异线性矩阵/GMRES 回退失败/max_iterations=1；已建新 BoundaryMonitor 和 LU 后注入异常 | 不接受固体；失败缓存不在下一尝试出现；根精确一致；记录真残差和失败原因 |
| 17-04，末端校验 | 最后一个场或界面载荷中注入 NaN/Inf、错 shape、运动/功失配、质量超限 | 即使所有单场 solve 返回成功，validate 失败也必须丢弃整个候选，所有接受计数不变 |
| 17-05，prepare | 分别在固体私有 accept 后、流体私有 accept 后、网格私有 accept 后、冻结新根/监测行序列化时抛 RuntimeError | live 根及所有历史完全不变；读取者只能见旧根；不存在时间不同步或仅部分 generation 增加的根 |
| 17-06，publish 及重入 | publish 前模拟 base_epoch 冲突；成功后重复 commit、abort、accept 旧候选/context | 冲突候选无写入；成功恰好推进一次，全场 target_time/epoch 同步；终态调用明确拒绝，已提交后 abort 不撤回物理步 |
| 17-07，重复评估 | 一个事务内对相同输入至少调用 3 次单场 solve，再改变/恢复候选初值 | base 及 contexts 历史保持精确不变；同输入候选满足重试差阈值；仅最终 commit 增历史 |
| 17-08，失败重试 | 每种 17-02…17-05 故障失败后去掉注入，相同 dt/输入重新 begin，之后继续至少 2 步；连败 3 次再成功 | 旧 attempt/context 全部失效；所有数组历史与无失败冷启动路径差≤1e-10；接受步数/t 完全一致，失败计数和日志单列 |
| 17-09，所有权 | 对 candidate/result/snapshot/public states 重新设 writeable 并修改；流体 u/p 共底层 view 的情况单独覆盖 | 不改变接受根；深复制接受和快照，两层历史无可写共享；数组字节/shape/dtype 均保持不变量 |
| 17-10，输出故障 | prepare 序列化失败；publish 后注入磁盘写失败，再执行幂等补写 | 前者 abort 无推进；后者保留已提交物理根并标记 output_pending，不重新 solve/accept；同 accepted_epoch 仅有一条成功记录 |

精确回滚检查通过状态遍历器覆盖所有字段，而不是只检查 `history.time`；缓存遍历器记录 topology/config/geometry/owner/version，确保候选缓存不可跨 attempt 使用。失败审计记录允许追加，物理解和接受监测行不允许变化。每个注入点必须给出已触发故障的证据，不能依赖 solver 偶然不收敛。

## 6. 文件归属、可并行工作与集成顺序

本表仅为将来实现的建议归属，须由 A0 派发时冻结；本轮未授权启动这些实现或 worker。未存在路径使用代码文字，避免伪造现有入口。

| 负责人/工作包 | 拟议独占实现与测试文件 | 并行边界 |
| --- | --- | --- |
| A0 / T17 | `src/fsi_ice/time/transaction.py`、`tests/test_transaction.py`；唯一修改现有 `time/bdf.py`、time 导出及跨场适配协议的人 | 先冻结根/contexts/owner 生命周期；其他人提接口需求，不共同编辑 BDF |
| A4a / T14 | `src/fsi_ice/ale/{harmonic,geometry,kinematics,gcl}.py`、`tests/test_ale_mesh.py`，文件细分以 ALE 计划冻结为准 | 只读 A1 的 TriMesh/P2 接口；mesh/fem 公共修改由 A1 单独承担 |
| A3 / T15 | `src/fsi_ice/fluid/ale.py`、`tests/test_ale_fluid.py`；独占现有 fluid/dynamics、taylor_hood、forces、linear_solver 的必要扩展 | A4 不同时改 fluid 文件；等 T14 产出真实几何和网格速度后集成，早期可用规定几何输入测试残差 |
| A4b / T16 | `src/fsi_ice/interface/{topology,mapping,geometry,traction,checks}.py`、`tests/test_interface.py`，细分以界面计划冻结为准 | 可与 T15 在不同文件开展；A2/A3 复核符号、DOF 和力接口；interface 导出文件由 A4b 唯一维护 |
| A2 / 固体适配 | 独占 `solid/dynamics.py` 的事务适配需求及 `tests/test_solid_transaction.py` | 不改 time 或 fluid；保持 StVK 总拉格朗日参考几何，提交输入/输出深复制及 d/v 一致性证据 |
| A7 / 验收 | `docs/development/validation_plan.md`；将来 `cases/m3_validation.json`、`src/fsi_ice/validation/m3.py`、`tests/test_m3_validation.py`、`docs/validation/m3.md` | 配置/验证文件派发时逐一归属；A0 独占 CLI、根 README/spec/gola/agent 及集成测试总入口 |

T14 与 T16 若要同时交给两个执行者，A0 必须明确拆成“网格负责人”和“界面负责人”，两者不共享 `__init__.py` 或公共状态文件。上述角色相同不等于应同文件并行编辑。统一测试套件入口由现有 pytest 自动发现测试，无需每人修改 CLI。

集成闸门按真实依赖推进：

1. **协议先冻结**：A0+A2/A3/A4 确认 root 字段、pressure/initial 所有权、owner/epoch、固定 dt/启动步、符号/DOF 与缓存失效；先用替身完成 T17 生命周期单元设计/实现，但不提前标记 T17 完成。
2. **T14 独立闸门**：14-01…05 的真实本地输出全部满足门槛，A1 复核拓扑/P2 坐标；提交几何字段和网格 BDF context。
3. **T15/T16 闸门**：T15 在 T14 真实几何上通过 15-01…06；T16 在真实几何/场 DOF 上通过 16-01…06。可在文件隔离后并行实现，但各自按依赖完成验证。
4. **T17 真实集成**：依赖 T08/T15/T16 的可验证实现，组合非零小型场，覆盖全部 17-01…10、BDF1/2 启动及成熟历史；不以替身测试代替物理集成。
5. **M3 验收**：A0 实际运行受影响单场回归及 M3 数值矩阵，A7 复核误差/质量/导数/故障证据；没有未处理失败或缺失矩阵项，才能更新 M3/T14–T17。之后才可申请进入 T18/FSI。

现有快速命令可重复使用第 1.3 节命令。实现后拟议 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_ale_mesh.py tests/test_ale_fluid.py tests/test_interface.py tests/test_transaction.py tests/test_solid_transaction.py tests/test_m3_validation.py`，以及现有 `.venv/bin/python -m fsi_ice test` 作为完整回归；这些拟议文件目前未实现，不能把命令列出当作已经执行。报告入口是否新增 `m3` 子命令由 A0 集成时决定。

## 7. M3 最终证据要求与本轮文档验收

每项测试/算例报告需保存：实际命令与退出码、配置和代码状态指纹、Python/依赖、mesh/cell/DOF、dt/order/启动状态、变量尺度与容差、实测误差/阶/最小 det/quality、物理与线性真残差、绝对及相对通量/GCL、两侧有符号力/功、history owner/generation 与 global epoch/attempt_id 生命周期、触发的注入点、失败前后及重试数组差、缓存失效证据。每个研究包含至少三档网格/时间步的分离误差，失败点照样保存，不能只给 PASS 总表或示意图。

M3 完成门槛是矩阵全部实现并实际验证通过、完整全场精确恢复与可重复重试、无静默翻转、共享物理残差及所需导数可访问、证据可重放。不能以已有 34/93 项单场测试、本文设计或单个常量场测试替代 M3 验收。A0 负责在获得实际证据后同步进度，本文本轮不改变完成状态。

本轮验收：单场实际报告/10 条轨迹/30 个唯一文件 SHA-256 与来源复核通过；现有 34 项相关快速测试通过，未重跑耗时 CFD。已读取协调者更新的 agent.md 第 8 节，并对齐 ALE/界面计划的 H/Hᵀ、参考/当前构形、厚度、w_kin 与 Q_BDF 区别、显式历史几何/时间模板及文件归属。本轮只产生设计文档，无 ALE、界面、事务或 FSI 新实现。

本文件实际文档检查输出：`34 local links checked; missing=[]`；`Markdown whitespace/fences/EOF: PASS`。`git diff --check` 退出码 0、无输出；因 docs/ 当前未跟踪，另执行 `git diff --no-index --check /dev/null docs/development/validation_plan.md`，无空白错误输出，退出码 1 表示文件与 /dev/null 存在差异；同时显式检查无行尾空白、无 tab、围栏成对、末尾换行。未用 git add 改变索引，也未把其他执行者的文档变化记为本任务修改。

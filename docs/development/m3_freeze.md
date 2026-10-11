# M3 第一波接口冻结（A0）

日期：2026-10-11。性质：对 [ALE 计划](ale_plan.md)、[界面计划](interface_plan.md)、[验收与回滚计划](validation_plan.md) 的裁决与冻结，用于 T14、T16 映射基础与 A7 验收配置的第一波实现。本文与设计文档冲突时以本文为准。所有功能均未实现、未验收；本文不更新任何完成勾选。

## 1. 本轮范围与文件归属

| 执行者 | 工作包 | 唯一可写文件 | 其他文件 |
| --- | --- | --- | --- |
| A4a | T14 ALE 网格基础（调和扩展、候选几何、BDF 网格速度、扫掠通量原语） | `src/fsi_ice/ale/*.py`、`tests/test_ale_mesh.py` | 只读 |
| A4b | T16 界面拓扑、运动映射 H 与对偶载荷映射基础（不含真实牵引集成） | `src/fsi_ice/interface/*.py`、`tests/test_interface.py` | 只读 |
| A7 | M3 验收配置与矩阵（实施前冻结，不含运行证据） | `cases/m3_validation.json`、`docs/validation/m3.md` | 只读 |
| A0 | 公共时间模板、集成复核、进度同步 | `src/fsi_ice/time/*.py`、`docs/development/m3_freeze.md`、`agent.md`、`gola.md` | — |

公共基础（`mesh/`、`fem/`、`solid/`、`fluid/`、CLI、既有测试、三份设计文档）本波只读；需要改动时先报告 A0。任何人不得修改他人文件；同一文件只由归属者编辑。

## 2. 离散与命名裁决

1. T15 采用 [ALE 计划](ale_plan.md) 第 3 节路线：**历史质量保守时间差分 + 满足离散 GCL 的扫掠通量**；不采用“仅把对流速度换成 `u-w_kin`”的非守恒形式。该裁决可在实现评审中经 A0/A3/A4 书面更新，不得由单个执行者自行更改。
2. `w_kin`（BDF 位移/界面运动速度）、`Q_BDF`（参考域扫掠通量）、`g_BDF`（几何体积差分源）、`w_gcl`（扫掠有效输运速度）分别命名、分别验证；有限 dt 下不得要求 `w_gcl` 与 `w_kin` 相等，也不得用 `w_gcl` 覆盖界面 Dirichlet 值。
3. 几何约定：参考网格只读；候选当前网格由 `TriMesh(X+d_m)` 新建；固定拓扑、P1 仿射几何、P2 中边点取端点平均。记 `F=J_x J_X^{-1}`、`j=det F`；`det(J_x)>0` 必须。
4. 界面约定：厚度只计一次；`n_s=-n_f`；`t_s=-σ_f n_f` 为流体对固体牵引；对已积分对偶节点力使用 `Hᵀ`；固定参考拓扑一次匹配，不在变形后重新匹配。
5. 压力为代数量，不进入 BDF 历史，但必须与速度同步保存/接受/恢复（T17）。候选求解不得调用 `BDFHistory.accept`；候选计算不得修改已接受数组。
6. 单位：长度 m、时间 s、速度 m/s、力 N、功 N·m；验证尺度 L*=1 m、T*=1 s、U*=1 m/s、ρ*=1 kg/m³、μ*=0.01 Pa·s、b*=1 m、F*=1 N、W*=1 N·m；其它尺度必须写入配置。

## 3. 公共时间模板冻结（A0 提供）

`src/fsi_ice/time/bdf.py` 新增只读 `BDFStencil`，`BDFHistory.stencil()` 产生，`BDFContext` 增加 `stencil` 字段：

```python
@dataclass(frozen=True)
class BDFStencil:
    order: int                 # 本步实际阶次：BDF1 启动为 1，成熟 BDF2 为 2
    dt: float
    step_id: int               # 生成时的 BDFHistory generation
    coefficients: np.ndarray   # (order+1,) 只读，已除以 dt；0=候选，1=最近接受，2=更早接受
    times: np.ndarray          # (order+1,) 只读绝对时间，严格递减 [t(n+1), t(n), (t(n-1))]
```

- 取值：BDF1 `[1,-1]/dt`、`times=[t+dt,t]`；成熟 BDF2 `[1.5,-2,0.5]/dt`、`times=[t+dt,t,t-dt]`；BDF2 启动步 `order=1`。
- `BDFStencil.interval_weights`：相邻一步构形区间的无量纲权重，BDF1 `(1.0,)`、BDF2 `(1.5,-0.5)`；满足
  `Q_BDF = Σ_k interval_weights[k]·Q[区间 k]`、`Div_X Q_BDF = Σ_j coefficients[j]·j_j = g_BDF`。
- 校验：系数与标准 BDF 权重一致（rtol 1e-12）；`times` 间距为 dt；数组只读；`context.stencil` 与 `history.stencil()` 一致，且对任意候选满足 `coefficients[0]*candidate + Σ coefficients[1:]*states == context.derivative(candidate)`。
- 现有单场回归不得被破坏。

## 4. ALE 冻结 API（T14）

模块与符号（字段名冻结；内部实现细节由 A4a 决定）：

```python
# fsi_ice/ale/harmonic.py
build_harmonic_extension(reference_mesh, *, diffusion=1.0) -> HarmonicExtension
HarmonicExtension.extend(boundary_nodes, displacement_values) -> MeshDisplacement
#   boundary_nodes: 唯一 P1 顶点编号 (Nb,)；displacement_values: (Nb,2) 绝对参考位移 d_m
#   必须覆盖参考网格全部外部边界节点；未知节点/重复/冲突/非有限值拒绝；
#   分量复用 assemble_poisson(reference_mesh, 0.0, diffusion) 与 solve_dirichlet。
# MeshDisplacement.values (N1,2) 只读；component_residuals (2,)；boundary_error；boundary_nodes。

# fsi_ice/ale/geometry.py
build_ale_geometry(reference_mesh, displacement, *, quality_limits=None, candidate_id) -> ALEGeometry
#   quality_limits 缺省 {"min_volume_ratio": 0.2, "min_quality": 0.1}；
#   字段: reference_mesh, current_mesh, reference_space, current_space,
#         displacement(只读), deformation_gradients (E,2,2), volume_ratios (E,),
#         quality (E,), minimum_volume_ratio, minimum_quality, quality_limits, candidate_id。
#   违规 raise ValueError，消息含 candidate_id、单元号与最小值；失败不得修改参考/接受数组。

# fsi_ice/ale/kinematics.py
evaluate_mesh_velocity(geometry, mesh_context) -> MeshVelocity
#   values_p1 = mesh_context.derivative(geometry.displacement)；values_p2 由当前 P2 空间端点平均；
#   不推进、不修改任何历史。

# fsi_ice/ale/gcl.py
build_gcl_transport(reference_mesh, candidate_geometry, accepted_geometries, stencil) -> GCLTransport
#   accepted_geometries: [最近接受, 更早接受]，长度匹配 stencil.order；几何须同参考网格。
#   Q[a,b] 由共因子多项式沿直线路径在 s=0.5 精确积分（二维 P1 精确）；
#   flux (E,Q,2) 在 triangle_quadrature(5) 点求值；divergence=g_BDF (E,)；
#   edge_fluxes 每条唯一全局边一个标量：正向法向 = rot_cw(排序端点对 min→max 切向)，
#     内部边按各单元外法向相加（严格相消），外部边求和 = 参考域 BDF 体积收支（T14 实现约定，已验收）；
#   w_gcl = F_x Q_BDF / j 在当前 quadrature 点；cell_defects、edge_defect、budget_defect 诊断。
#   直线路径健康检查：各单元 det(F(s))（对 s 为二次）必须在端点和区间内部严格为正且满足面积比界限；
#     端点有效但路径中段退化（如大转动直线插值）时拒绝该步并定位单元（T15 实现）。
#   输出全部只读；不调用 accept。
```

### 4.1 T15 冻结 API（ALE 流体，第二波）

```python
# fsi_ice/fluid/ale.py
@dataclass(frozen=True)
class ALEFlowContext:
    geometry: ALEGeometry            # 候选几何
    mesh_velocity: MeshVelocity      # w_kin（界面运动速度，不用于几何守恒项）
    transport: GCLTransport          # Q_BDF/g_BDF 与已接受几何
    stencil: BDFStencil              # 与本步网格位移/流体速度历史同阶同时间
    history_velocities: tuple        # 已接受 P2 速度，最新在前，长度 = stencil.order，只读

build_ale_flow_context(operators, geometry, transport, history_velocities) -> ALEFlowContext
#   operators 必须组装在 geometry.current_mesh 上；历史为参考基系数（拉回表示）。

ale_geometry_jvp(operators, values, *, ale_context, delta_d_m)
#   未消元残差对网格位移方向的导数；历史固定；callable 体力/边界无空间导数协议时
#   允许中心差分 JVP 但必须标注；覆盖 ale_plan §3.4 列出的全部几何依赖块。

# fsi_ice/fluid/taylor_hood.py（只做必要扩展）
navier_stokes_residual(operators, values, *, time_coefficient=0.0, time_rhs=None,
                       with_jacobian=True, ale_context=None)
#   ale_context=None 时必须与现行为逐位一致（固定网格回归）；
#   ale_context 非空时禁止同时传 time_coefficient/time_rhs；残差与切线公式见 ale_plan §3.3/§3.4。
```

T15 的实现文件归属：`fluid/ale.py`、`tests/test_ale_fluid.py` 新增；`fluid/taylor_hood.py`、`dynamics.py`、`forces.py`、`linear_solver.py` 由 T15 负责人独占修改；`ale/gcl.py` 允许增加路径健康检查（保持 T14 测试通过）。`ale/__init__.py` 导出 §4 的公共符号；数值门槛见第 6 节。

### 4.2 T16c 与 T17 冻结符号（第二波）

```python
# fsi_ice/interface/geometry.py（T16c 新增）
evaluate_interface_geometry(topology, current fluid mesh/space, solid displacement, geometry 版本) -> InterfaceGeometry
#   InterfaceGeometry：当前积分点/长度/法向、固体 F_s/J_s、j_Gamma 两条路径、匹配误差；不更新历史。

# fsi_ice/interface/traction.py（T16c 新增）
assemble_interface_load(topology, geometry, operators, solution, *, thickness) -> InterfaceLoad
#   g_f_gamma 为流体 P2 迹对偶节点力（含 b）；f_s_gamma=Hᵀg；T_s0=j_Gamma·t_s；总力/总矩与 geometry_id。
pullback_traction(...)   # 参考路径转换检查

# fsi_ice/fluid/forces.py（T16c 抽取公共核；不得改变现有数值结果）
boundary_cauchy_stress(operators, solution, 边界 owner 局部点)  # σ=-pI+μ(∇u+∇uᵀ)

# fsi_ice/time/transaction.py（T17 新增）
class AcceptedState      # 全场已接受根：t/dt/order、步号/epoch、固体/流体/网格/界面状态与历史
class StepTransaction    # begin -> evaluate -> validate -> prepare -> publish | abort；attempt_id 单调
```

详细字段与流程见 [interface_plan](interface_plan.md) §4–§7 与 [validation_plan](validation_plan.md) §2–§3；冲突时以本冻结文档与本轮任务卡为准。

## 5. 界面冻结 API（T16a/b）

```python
# fsi_ice/interface/topology.py
build_matching_interface(solid_mesh, fluid_mesh, fluid_space, *, solid_tag, fluid_tag, atol, rtol) -> InterfaceTopology
#   参考构形一次匹配；记录 edge_pairs（固端/流端/流体中边/owner/方向）、solid_nodes、
#   fluid_vertex_nodes、fluid_p2_nodes、参考长度与固体参考外法向 N_s、标量 h、交错 H（稀疏）、
#   gather/scatter 索引、topology_id；重复/歧义/非外部边/分段不匹配/零长/非有限拒绝并定位。

# fsi_ice/interface/mapping.py
map_motion(topology, solid_values) -> InterfaceMotion
#   输入全固体位移/速度 (n_s_nodes,2)；输出 vertex_values (n_sG,2) 与 p2_values (n_fG,2)；
#   p2_values = H·（接口交错向量），中边=端点平均；不接受历史。
map_load(topology, fluid_loads) -> (2*n_sG,)      # fluid_loads: (2*n_fG,) 交错已积分对偶节点力
scatter_solid_load(topology, compact) -> (2*n_s_nodes,)   # 由约定索引 scatter 到全固体向量

# fsi_ice/interface/checks.py
interface_diagnostics(topology, *, motion=None, loads=None, moment_origin=..., scales=...) -> dict
#   至少包含: 总力一致性、当前总矩一致性、虚功恒等式、运动残差 v_s-D_BDF d_s、
#   n_s+n_f、零/小量尺度处理与明确 pass/fail；不得静默忽略 NaN/Inf。
```

测试至少覆盖 [界面计划](interface_plan.md) 第 9 节的拓扑与方向、运动阶次、时间一致性、常牵引解析力、虚功/总力/当前总矩、异常与诊断条目；本波不集成真实流体应力迹（T16c 等 T14 验收后进行）。常数检验使用合成对偶载荷并明确标注为合成。

## 6. 实施前冻结的数值门槛

以下数值为第一波及其后续 M3 验收的冻结值；由 A7 汇总进 `cases/m3_validation.json`（含尺度与失败矩阵）。失败时分析原因，不得放宽。

| 项目 | 冻结值 |
| --- | --- |
| 几何健康 | `min(j)>=0.2`、`min(quality)>=0.1`；`det(J_x)>0`；拒绝后参考/接受数组逐位不变 |
| 调和仿射 patch | `max|d_h-d|/L*<=1e-11`；自由残差 `<=1e-10`；约束误差 `<=100·eps·L*` |
| 调和一般解收敛 | P1 L2 阶 `>=1.8`、H1 半范阶 `>=0.9`（三网格，误差递减） |
| BDF 网格速度 | 线性轨迹（含启动步）`<=1e-12·U*`；正弦轨迹 BDF1/BDF2 阶 `>=0.9/1.8`，启动步另按 BDF1 检查 |
| 离散 GCL | 逐单元 `|g_BDF−Div_X Q_BDF|/S_g <=1e-11`，`S_g=max(Σ|a_j·j_j|,1/T*)` |
| 扫掠体积/共享边 | 逐单元 `<=1e-11`（同 S_g）；内部边相消 `<=1e-12·U*L*`；全域收支 `<=1e-10·U*L*` |
| 界面代数恒等式 | 运动/功/力/矩 `<=1e-12·S`，S 取显式尺度与相关贡献和的较大者 |
| 常牵引解析力 | 流体 P2 节点 `b·l·t·(1/6,1/6,2/3)`；固体端点 `b·l·t/2`；厚度只用一次 |
| 法向/表面比 | `n_s+n_f` 与两种 `j_Gamma` 差 `<=1e-12`（尺度化） |
| 固定网格退化 | ALE 与既有 BDF 残差/切线相对差 `<=1e-12`（T15 冻结） |
| 导数扫描 | 非退化方向中心差分最小相对误差 `<=1e-5`（T15/T21 冻结） |
| T17 回滚 | 数组与历史 `np.array_equal`；重试-基线差 `<=1e-10`；17-01…17-10 全部注入用例通过 |

### 6.1 计划差异的 A0 裁决（2026-10-11）

A7 在汇总三份计划与本冻结文档时报告 D1–D3 三处数值差异，裁决如下；`cases/m3_validation.json` 与 `docs/validation/m3.md` 按此标注，门槛不变更：

- **D1 调和仿射 patch 上限**：维持 **1e-11·L***。该项是线性求解级检查（含消元与舍入余量），与 16-01 等纯代数/匹配恒等式的 1e-12·S 属不同类别；[validation_plan](validation_plan.md) 14-01 的 1e-12 不作为本项生效门槛。
- **D2 重试-基线容差**：T15-ROLLBACK 与 T17 的完整重试-基线比较维持 **1e-10**；[ale_plan](ale_plan.md) 的 1e-12 仅作为 T14/T15 同输入重复候选评估的观察目标，不作为拒绝门槛。数组与历史仍要求精确相等。
- **D3 界面解析牵引/构形转换容差**：小型确定性用例采用更严的 **1e-12·S**（16-03、16-05）；[validation_plan](validation_plan.md) 的 1e-10 保留为参考值。若实现给出明确条件数限制证据，经 A0/A7 复核后方可凭证据调整，不得静默放宽。

## 7. 证据与纪律

- 每个执行者只运行项目 `.venv`，先跑自己负责的测试文件（必须全绿），再运行 `.venv/bin/python -m fsi_ice test`；若回归失败仅来自其他并行工作包尚未完成的文件，必须在交付说明中逐条列出并标注为集成阶段处理。不得跳过测试、不得声称未运行过的测试通过。
- 交付必须报告：实际命令与退出码、实际数值（阶、误差、拒绝用例）、修改文件清单、发现的阻碍或设计文档不一致之处。
- 本波不提交、不推送 GitHub；不使用 `BDFHistory.accept` 于候选迭代；不修改他人文件。
- M3/T14–T17 完成状态由 A0 在收到全部实际证据后统一更新；设计或本冻结文档不构成完成。

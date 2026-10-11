# A4a / T14–T15：ALE 开发设计

日期：2026-10-11。性质：基于当前工作区源码的开发方案，未实现 ALE，未执行 ALE 数值测试，不能据此将 T14、T15 或 M3 标记完成。本次唯一修改文件为本文；公共规格、任务状态和进度由协调者维护。

遵循 [AGENTS.md](../../AGENTS.md)、[开发规格](../../spec.md)、[阶段目标](../../gola.md) 和 [任务分工](../../agent.md)。本方案限定二维、固定拓扑、P1 仿射几何、P2/P1 流体、固定步长 BDF1/BDF2、调和网格位移；不引入重网格、曲边几何或新的耦合求解器。分区与整体法调用同一网格构形生成器和流体残差。

## 1. 源码基线与必须改变的位置

以下为实际存在的文件和符号；后文带“拟新增”的接口尚不存在。

| 文件与现有符号 | 当前行为 | ALE 需要的改变 |
| --- | --- | --- |
| [mesh/triangular.py](../../src/fsi_ice/mesh/triangular.py)：`TriMesh.__post_init__`、`jacobians`、`determinants`、`quality`、`boundary_nodes` | 复制并冻结单份坐标和连通性；拒绝负方向、数值退化及非有限值；无参考/当前构形或几何历史 | 保留验证语义；ALE 层保存参考网格并用新坐标构造新的 `TriMesh`，禁止原地修改坐标或调换节点来掩盖翻转；增加相对参考面积、质量和运动路径检查 |
| [fem/p1.py](../../src/fsi_ice/fem/p1.py)：`physical_gradients`、`map_points` | 从传入网格计算物理梯度与积分点 | 调和问题传参考网格，流体组装传当前网格；两者不混用 |
| [fem/poisson.py](../../src/fsi_ice/fem/poisson.py)：`assemble_poisson`、`solve_dirichlet` | 支持正扩散系数、非零 Dirichlet 消元 | 复用为两个位移分量的调和扩展；不改变 Poisson 数学实现 |
| [fem/p2.py](../../src/fsi_ice/fem/p2.py)：`P2Space.from_mesh`、`p2_physical_gradients` | P2 中边点来自 P1 端点平均；唯一边排序决定全局编号；梯度依赖当前 Jacobian | 几何仍为 P1；在相同连通性上重建 P2 坐标并检查编号不变，P1 网格速度在 P2 中边点取端点平均 |
| [fluid/taylor_hood.py](../../src/fsi_ice/fluid/taylor_hood.py)：`assemble_stokes`、`FlowOperators.convection_tensor` | 一次组装黏性、压力、散度、质量、体力；`cached_property` 保存带几何的对流张量 | 每个不同候选几何生成新的 `FlowOperators`；只能缓存拓扑、参考基函数和积分规则，不能复用旧梯度、质量或张量 |
| 同文件：`convection`、`navier_stokes_residual` | 对流为 `rho (u·grad)u`；瞬态项为 `alpha M u - time_rhs` | 保留固定网格入口，增加统一 ALE 上下文：当前物理对流、几何守恒的网格输运、不同历史构形上的质量作用及其导数；不能只增加 `u-w` 参数 |
| 同文件：`flow_constraints`、`solve_navier_stokes` | 在 `operators.space.points` 取边界值；Newton 及线搜索重复调用残差 | 在当前 P2 坐标上取值；界面速度由离散位移导数指定；Newton 与线搜索必须使用同一个候选几何/时间上下文；新增残差参数须传到所有调用点 |
| 同文件：`unpack_flow`、`flow_error_norms` | 当前散度残差与 L2 积分由算子中的网格定义 | 用当前构形计算；保留现有监测，同时补充有尺度的全边界流量与 GCL 诊断 |
| [fluid/dynamics.py](../../src/fsi_ice/fluid/dynamics.py)：`solve_flow_step` | `history_rhs = -(operators.mass @ context.history_term.ravel())`，默认所有历史都使用同一质量矩阵 | ALE 分支逐历史使用对应 `M_j @ u_j`；从显式模板取系数，不能用合并的 `history_term` 恢复不同历史质量作用；候选求解仍不接受历史 |
| [fluid/forces.py](../../src/fsi_ice/fluid/forces.py)：`BoundaryMonitor.__init__`、`force`、`flux`；`boundary_force`、`boundary_flux` | monitor 预计算固定几何的边长、法向、梯度；独立函数按传入算子取几何 | monitor 每个候选几何重建或显式校验几何键；载荷和物理流量用当前边长/法向；另提供 BDF 扫掠体积通量，不能混名 |
| [fluid/linear_solver.py](../../src/fsi_ice/fluid/linear_solver.py)：`ReusedLUSolver.prepare` | 仅时间系数变化使 LU 失效；GMRES 仍使用当前矩阵并核验真残差 | 增加几何/约束签名失效策略；首版每次几何变化重置预条件器，后续才评估复用成本，不能把旧 LU 当新矩阵的直接解 |
| [time/bdf.py](../../src/fsi_ice/time/bdf.py)：`BDFContext.derivative`、`BDFHistory.context/accept/snapshot/restore` | 固定步长；BDF2 首步用 BDF1；上下文含合并历史项、owner 和 generation | 复用网格位移导数；A0 提供显式 BDF 模板和多场原子接受契约，保证速度、位移、几何历史同阶同时间 |
| [validation/transient.py](../../src/fsi_ice/validation/transient.py)：`run_trajectory` | CFD 网格、算子、monitor 只在时间循环前创建；每次单场求解成功就接受速度历史 | 保持现有固定网格 CFD 基准入口；另建规定网格运动验证入口，逐候选重建几何；完整 FSI 的接受步骤交给 T17，不能照搬单场逐场提交 |

[mesh/turek.py](../../src/fsi_ice/mesh/turek.py) 的 `cfd_mesh` 将圆柱和旗杆整段边界都标为 `obstacle`；当前文件不能区分固定圆柱与运动梁边界。T14 的独立矩形验证不依赖此修改；FSI 接入前由 A1/A0 拆分标签并定义接合点归属。`read_gmsh2` 的导入方向归一化策略只适用于初始导入，动网格更新不能复用它来修复翻转。

## 2. T14：参考几何、调和扩展与网格速度

### 2.1 构形与自由度约定

设参考顶点为 `X (N1,2)`，固定连通性为 `cells (E,3)`；绝对网格位移为 `d_m (N1,2)`，当前坐标为 `x = X + d_m`，单位依次为 m、m、m。候选迭代总是由 `X + d_m_candidate` 构造，禁止在上一候选 `x` 上继续累加位移。

参考单元 Jacobian 为 `J_X`，当前单元为 `J_x`，ALE 变形梯度 `F = J_x @ inv(J_X)`，体积比 `j = det(F) = det(J_x)/det(J_X)`。这里 `J_X/J_x` 是参考三角形到物理单元的映射；`j` 无量纲，不可把它与 `TriMesh.determinants` 混用。

流体速度、压力历史是同一参考节点/形函数上的系数：`u_j (N2,2)`、`p_j (N1,)`，每一份 `u_j` 表示对应构形上的场拉回参考网格后的系数。固定拓扑下无需空间重插值；不同构形下直接按相同数组索引访问是拉回，不是在当前物理位置对旧场求值。若未来重网格，这一约定失效，必须另设计守恒转移。

`P2Space.from_mesh` 使用顶点与唯一边排序，固定 `cells` 保证中边编号不变；每次重建检查 `space.cells`、`edge_nodes` 与参考空间一致。几何位移和网格速度都是 P1，以线性插值嵌入 P2 节点；不能将 P2 流体速度空间误称为 P2 几何。

### 2.2 调和问题及输入输出

参考域求解每个分量 `-Div_X(kappa grad_X d_m) = 0`。第一版 `kappa=1` 为无量纲正常数；按参考位置定义的正标量系数可沿用 `assemble_poisson`，不在候选构形重算刚度。只有改变扩散模型时才重组装。

拟新增 `ale/harmonic.py`：

```python
build_harmonic_extension(reference_mesh, *, diffusion=1.0) -> HarmonicExtension
HarmonicExtension.extend(boundary_nodes, displacement_values) -> MeshDisplacement
```

`boundary_nodes` 是唯一 P1 顶点编号 `(Nb,)`，`displacement_values` 是参考构形上的绝对位移 `(Nb,2)`。输出冻结的 `d_m (N1,2)` 及两分量自由节点残差、Dirichlet 误差。内部复用 `assemble_poisson(reference_mesh, 0.0, diffusion)` 和 `solve_dirichlet`，不复制组装代码；首版可分别调用两次求解，之后再缓存固定自由节点子矩阵的分解。

外边界、固定障碍物明确给零位移；运动界面给映射后的结构位移。所有外边界顶点默认必须指定；不能把遗漏标签默认为自然边界。共享角点先合并：相容位移只保留一个自由度，冲突拒绝；接合点固定约束与界面运动不得静默覆盖。每个网格连通分量须有充分 Dirichlet 约束，孤立自由分量应报错；空约束、未知节点、非有限值、尺寸错或重复冲突均在求解前拒绝。

### 2.3 候选几何与质量失败

拟新增 `ale/geometry.py`：

```python
build_ale_geometry(reference_mesh, displacement, *, quality_limits,
                   candidate_id) -> ALEGeometry
```

输出包含参考网格、当前 `TriMesh`、参考/当前 P2 空间、冻结的位移、`F (E,2,2)`、`j (E,)`、最小质量/面积比及不可复用的候选键。检查 `x` 有限、连通性/标签/编号不变、`det(J_x)>0`。初始矩形验证配置建议 `min(j)>=0.2`、`min(quality)>=0.1`；实际 FSI 网格须在运行前配置门槛，至少高于现有 `32 eps max(edge_length²)` 数值退化界限。这些质量门槛是新验证配置建议，不是已冻结的项目验收值。

失败输出应携带时间、候选迭代、单元号、最小 `det(J_x)`、`j` 和质量。不得裁剪节点、取绝对行列式或调换单元方向；拒绝整个候选，不提交历史。局部正方向不保证无边界自交或全局重叠：T14 应补充非邻接边界段相交检查，复杂拓扑的全局重叠检测列为接入 FSI 前风险。

### 2.4 BDF 网格速度与显式模板

网格位移由 `BDFHistory(initial_d_m, dt, order=...)` 保存。每步只调用 `mesh_context.derivative(d_m_candidate)` 得到 P1 运动速度 `w_kin`，再在 P2 中边点取端点平均得到 `(N2,2)`；单位 m/s。它是界面运动约束使用的网格速度。

- BDF1：`w_kin = (d_m^(n+1)-d_m^n)/dt`。
- BDF2：`w_kin = (3 d_m^(n+1)-4 d_m^n+d_m^(n-1))/(2 dt)`。
- 首步严格跟随 `BDFHistory.context().order == 1`；流体与固体首步同样启动，不能某场擅自使用 BDF2。

A0 拟在公共时间模块提供 `BDFStencil(dt, order, coefficients, times, step_id)`：`coefficients` 已除以 dt，依次对应候选、最新接受、再前接受，分别为 `[1,-1]/dt` 或 `[1.5,-2,0.5]/dt`；必须由公共时间上下文产生并校验，不让 ALE、流体各自猜测阶次。现有 `BDFContext` 只有合并的 `history_term`，无法表达不同历史质量矩阵，因此 T15 不能仅依赖它。

拟新增 `ale/kinematics.py` 的 `evaluate_mesh_velocity(geometry, mesh_context) -> MeshVelocity`，以及 `ale/gcl.py` 的 `build_gcl_transport(reference_mesh, candidate_geometry, accepted_geometries, stencil) -> GCLTransport`。后者依赖 T14 的完整几何历史，详见下一节。所有候选计算只读接受状态。

## 3. T15：离散几何守恒和流体残差

### 3.1 明确采用的离散策略

连续 ALE 动量含 `rho [∂_t u|X + ((u-w)·grad_x)u]`，同时要求 `∂_t j = Div_X(j F^-1 w)`。仅将 `convection` 中的速度换为 `u-w_kin`，仍遗漏变化的几何算子、历史质量、边界积分和几何守恒；即使此种非守恒形式恰好保持常量，也不能据此证明几何通量正确。

本方案建议 T15 使用**历史质量的保守时间差分 + 满足离散 GCL 的网格输运**，物理流体对流继续保留现有 advective Galerkin 形式。下面公式是本项目拟定的二维仿射离散方案及自包含推导，不宣称已有能量稳定性证明。GCL 只保证几何一致性/常量保持，不能代替质量、能量、界面功或一般流场精度验证。

不能直接假定 `D_BDF j = j^(n+1) div_x w_kin^(n+1)`：行列式不是坐标的线性函数，这个等式对一般有限步长变形不成立。为避免以事后抵消常量残差冒充守恒，先从真实单元扫掠建立几何通量。

### 3.2 扫掠通量的具体构造

对相邻构形 a、b，以同一参考坐标参数化直线路径 `x(s)=x_b+s(x_a-x_b)`，`s∈[0,1]`，`F(s)=grad_X x(s)`，`delta_x=x_a-x_b`。定义参考域通量：

```text
Q[a,b](X) = (1/dt) integral_0^1 det(F(s)) inv(F(s)) delta_x(X) ds
Div_X Q[a,b] = (j_a-j_b)/dt
```

实现时直接用 `det(F) inv(F)` 的二维余因子多项式，避免在路径积分中数值求逆；二维 P1 单元中它对 s 为一次多项式，`delta_x` 对 X 为一次，s 中点积分即可精确积分，建议仍写成显式平均余因子便于求导。`Q` 单位 m/s，在每个参考单元上为 P1 向量。共享边的法向扫掠通量相容；按全局固定边方向记录一次，两侧符号相反，验证内部通量抵消。

依据本步实际模板设置：

```text
BDF1: Q_BDF = Q[n+1,n]
BDF2: Q_BDF = 1.5 Q[n+1,n] - 0.5 Q[n,n-1]
g_BDF = sum_j a_j j_j
Div_X Q_BDF = g_BDF
```

其中 `a_j` 已除以 dt。BDF2 几何通量需要两份已接受构形；第二条路径完全由已接受数据决定，不能随耦合迭代改变。BDF1 启动不访问不存在的第二历史。

每条直线路径还需检查中途几何：各单元 `det(F(s))` 是二次多项式，在端点和区间内极值处检查严格正值及配置的面积比界限；不能仅检查端点。有效端点间的直线插值也可能退化，例如较大的刚体转动；首版拒绝该步并报告路径失效，后续才设计不同运动路径。

`GCLTransport` 返回冻结的 `Q_BDF (E,Q,2)`、`g_BDF (E,)`、边扫掠通量 `(Nedge,)`、模板与候选键、逐单元 GCL 缺陷和共享边缺陷。这里 `Q` 表示空间积分点个数时仅用于数组维度，通量符号 `Q_BDF` 不表示阶次。为便于诊断，还返回当前构形上的有效输运速度 `w_gcl=F^(n+1) Q_BDF/j^(n+1)`。

`w_gcl` 与 `w_kin` 的用途不同：前者保证扫掠通量与 BDF 体积差分一致，后者保证 BDF 位移/界面速度关系。有限 dt 时二者一般不相等，不能用 `w_gcl` 覆盖界面 Dirichlet 值。验证应检查二者差异随 dt 的一阶/二阶衰减；共享边只要求 `Q_BDF` 的参考法向通量连续，不要求有效速度切向连续。

### 3.3 残差、输入输出与常量抵消

拟新增 `ALEFlowContext`，包含 `ALEGeometry`、`MeshVelocity`、`GCLTransport`、显式 `BDFStencil`、对应接受构形与拉回速度历史。`navier_stokes_residual(operators, values, *, ale_context=None, ...)` 保持现有返回 `(residual (2N2+N1,), jacobian CSR or None)`。若使用 `ale_context`，禁止同时传旧的 `time_rhs/time_coefficient` 来形成重复时间项；上下文必须匹配当前 `operators` 的几何键、流体拓扑和模板。

令参考域速度测试函数为 `v`；候选及各历史构形上的一致质量矩阵为 `M_j = rho ∫_Omega0 j_j N_a N_b dX ⊗ I`。动量残差采用：

```text
R_u = A_current u - B_current.T p - f_current
      + sum_j a_j M_j u_j
      + rho integral_Omega0 j_current v [(u·grad_x)u] dX
      - rho integral_Omega0 v [(Q_BDF·grad_X)u + g_BDF u] dX
R_p = -B_current u
```

`A_current/B_current/f_current` 来自当前 `assemble_stokes`；压力、黏性、体力均在当前构形积分。`grad_X` 使用参考物理网格梯度，非参考三角形梯度；积分测度为 `det(J_X)*weights`。物理对流项可复用现有 `convection`，网格输运为独立线性算子 `K_mesh(Q_BDF,g_BDF)`，只组装一次后加入同一残差。参考 P2 梯度一次、`Q_BDF` 为 P1、速度/测试为 P2，空间被积函数最高五次，现有 [五次积分规则](../../src/fsi_ice/fem/quadrature.py) 足够用于仿射几何与多项式项；非多项式体力的积分精度另作验证。

网格项等价于 `-rho ∫ v Div_X(Q_BDF u) dX`。若改用分部积分实现，必须保留外边界项 `-rho ∫_boundary v u (Q_BDF·N) dS_X`，不能只写体积分 `+rho ∫ u Q_BDF·grad_X v`；内部边通量按共享边符号抵消。首版直接使用上式的强散度体积分，边扫掠通量用作独立守恒检查。物理对流继续使用现有 advective 形式，因此不宣称弱散度非零时获得严格的全局动量守恒；不能把物理对流改为保守形式而不更新固定网格离散及回归定义。

对于 `u_j=U`（所有速度节点同一常向量）、`p=0`、`f=0`，且所有外边界给 `U`：

- 黏性、物理对流和散度项为零；`grad_X u=0`。
- `sum_j a_j M_j U = rho ∫ v g_BDF U dX`，与网格项中的 `-rho ∫ v g_BDF U` 在相同积分规则下逐自由度抵消。
- 这同时要求真正的 `Div_X Q_BDF = g_BDF` 和历史质量正确；不能只令某矩阵行和匹配常量后跳过单元/边通量检查。

固定网格时 `j_j=1`、`Q_BDF=g_BDF=0`，所有 `M_j` 相同，严格退化为当前 BDF 残差；这应作为兼容回归的数值要求。稳态动网格 FSI1 只更新候选几何、令物理时间导数和网格输运为零；耦合迭代的位移变化不是物理网格速度，不能把迭代次数当作 dt。

### 3.4 Jacobian 和几何缓存契约

在给定 ALE 几何下，`J_uu = A_current + J_convection + a_0 M_current + K_mesh`，`J_up=-B_current.T`、`J_pu=-B_current`。`K_mesh` 含负号，与上述残差一致。相对输运只对 u 求导时网格场固定；现有 `convection` 的 advecting 和 advected 两部分导数都保留，不能把 u 依赖也冻结。

拟新增 `ale_flow_geometry_jvp(operators, values, ale_context, delta_d_m)`：输出未消元残差对网格位移方向的导数，接受历史固定。必须覆盖当前 `J_x/F/j`、梯度、质量、散度/压力、黏性、体力空间求值、物理对流、扫掠余因子及 `g_BDF`；`delta_w_kin=a_0 delta_d_m`。任意 callable 体力或边界值如果没有空间导数协议，首版允许标明中心差分 JVP 并扫描扰动误差，不能标为完整解析耦合切线。T15 至少验证流体块与几何方向导数；T21 再补完整界面约束、载荷和全局耦合块，但不能丢失几何访问接口。

每个候选几何创建新的算子和 monitor，是首版安全缓存策略。可缓存 `P2Space` 拓扑、参考基函数、参考调和刚度；不可缓存当前法向、边长、体力值、质量、梯度或对流张量跨几何使用。缓存键至少包含参考拓扑版本、候选几何唯一标识、时间模板和边界约束签名；不要只用时间或 `BDFHistory.generation`，同一步不同候选共用 generation。

## 4. 与 T16 界面及 T17 回滚的契约

### 4.1 界面运动和载荷交接

T16 提供固定拓扑上的交错向量运动映射 `H (2NfGamma,2NsGamma)`，附流/固节点顺序、gather/scatter 索引与固定接合点策略；`H_vertex` 是 H 的 P1 顶点行子矩阵，等价于本方案的几何位移映射。对 P1 固体匹配界面，P2 中边位移/速度为端点线性插值；若固体界面非线性或网格不匹配，不能凭节点坐标近似替代映射。

输入 `d_s_candidate` 和同模板的固体位移历史，输出 `d_m|Gamma = H_vertex d_s_gamma`；界面流体速度 `u|Gamma = H v_s_gamma`。在固体单场消元求解或收敛的一阶系统中 `v_s = D_BDF d_s`，此时 `w_kin|Gamma` 嵌入 P2 后等于 `H D_BDF d_s_gamma`；固定线性映射应满足 `D_BDF(H d_s_gamma)=H D_BDF(d_s_gamma)`。未收敛的整体候选中若位移/速度独立，须报告 `v_s-D_BDF d_s`，不能强行声称二者一致。现有 [solid/dynamics.py](../../src/fsi_ice/solid/dynamics.py) 的 `solve_solid_step` 已按 `alpha*d+hd` 给出速度，是对齐对象；不得另对固体位移做差分。

载荷接口输入当前 `FlowSolution`、当前几何及界面顺序，输出流体对固体的力/结点载荷和构形说明。沿用 `forces.py` 的符号：`n_f` 为当前流体域外法向，流体对固体牵引为 `-sigma_f n_f`，单位 Pa；现有总力积分为单位厚度 N/m。T16 的节点力先乘实际固体厚度 b，输出 N；参考固体上的弱载荷使用 `ds_x/ds_X`（或 Nanson 转换），不能把当前牵引直接当参考面积密度。T16 统一以 Cauchy 应力迹组装 P2 对偶节点力 `g_f_gamma`，以 `f_s_gamma=H.T g_f_gamma` 传递并验证总力/虚功；转置前必须已做积分，不能转置映射裸牵引系数，也不能重复乘厚度。T15 提供当前应力/边界积分数据和未消元残差，后者仅供反力诊断，不混用为另一套耦合载荷。

界面功使用物理 `v_s/u/w_kin` 与当前牵引。几何扫掠诊断使用 `Q_BDF`；其有限步长有效速度不能替代物理界面速度来计算功。扫掠与瞬时运动通量差异需随 dt 收敛，FSI 能量稳定性仍需额外研究。

### 4.2 原子候选、接受和恢复

T14/T15 自身均不调用 `BDFHistory.accept`。A0/T17 的 `StepSnapshot` 拟包含：固体位移/速度历史、流体速度历史及当前压力、网格位移历史和两份对应接受几何、时间/dt/阶次/步号、初始迭代猜测、界面映射版本、监测接受记录位置；缓存可销毁重建，不作为权威状态。各个 `BDFHistory` 的 owner 不同是正常情况，校验各自 owner，另外用共同 `step_id/time/dt/order` 对齐，不能要求跨场 owner 相同。

同一步开始建立不可变上下文，耦合候选反复从已接受历史计算；T15 使用的历史质量/扫掠路径不随候选覆盖。成功后统一预检查全部上下文与候选，再原子接受；如果某场 accept 失败，恢复全快照，不得留下半提交状态。输出日志区分候选诊断和接受监测，失败候选不追加有效时间轨迹。

现有 `BDFHistory.restore` 恢复 generation：恢复到原 generation 后，失败前的某些上下文可能再次通过 owner/generation 检查。因此 T17 必须用新的 attempt token 使所有旧候选失效，并销毁旧几何/求解缓存；不能声称当前时间模块已有完整多场事务防护。相同物理步重试使用恢复后的接受状态和全新 token，测试两次得到相同结果。

当前 BDF 是固定步长，snapshot 不含 dt/order。T17 首版禁止自动缩 dt；需要缩步时须由 A0 定义变步长模板、历史和快照扩展或明确回退重启策略，不能只修改 `history.dt` 后继续 BDF2。T14 路径失效可以报出缩步建议，但不自行实现缩步重试。

## 5. 文件归属与可执行开发顺序

本文是归属建议，新增路径均尚未实现，不使用指向不存在文件的 Markdown 链接。跨模块修改先由 A0 指定单一维护者。

| 文件或范围 | 建议维护者 | 交付内容 |
| --- | --- | --- |
| 拟新增 `src/fsi_ice/ale/{__init__,harmonic,geometry,kinematics,gcl}.py`，`tests/test_ale_mesh.py` | A4a，A1 复核几何/P2 | 调和扩展、构形、质量/路径、BDF 运动速度、扫掠通量原语 |
| `mesh/triangular.py`、`mesh/turek.py`、`fem/p2.py`、相关基础测试（只在需要时修改） | A1，A0 排期 | 稳定拓扑接口、FSI 标签分割；优先复用已有符号 |
| `time/bdf.py`、时间测试，拟新增公共 step/stencil 契约 | A0，A2/A3/A4 复核 | 显式模板、多场对齐、attempt token；不由 A4a 同时改公共时间文件 |
| `fluid/taylor_hood.py`、`fluid/dynamics.py`、`fluid/forces.py`、`fluid/linear_solver.py`；拟新增 `tests/test_ale_fluid.py` | A3，A4a 共同冻结接口后单人编辑 | ALE 残差、各块导数、当前几何监测、缓存失效和固定网格回归 |
| 拟新增 `validation/ale.py`、`cases/ale_prescribed.json`，验证报告 | A4a 实现，A7 定义算例/复核证据，A0 接运行入口 | 规定运动、GCL、自由流、MMS、误差记录；不直接修改既有 CFD 结论 |
| 拟新增 `interface/`、全场状态与回滚模块 | T16/T17 负责人 | 映射、法向/功、原子接受恢复，消费本方案接口 |
| `spec.md`、`gola.md`、`agent.md`、README、公共环境/验证总表 | A0 | 冻结离散选择、任务状态及实际验收证据 |

实施次序和门禁：

1. A0/A1/A3/A4 冻结坐标/自由度/时间模板与质量配置，保留固定网格基准；不先创建大量空接口。
2. T14a：两个调和分量、非零边界、冲突/空约束诊断及仿射 patch；输出绝对位移。
3. T14b：候选几何、P2 编号不变、质量/边界自交/路径拒绝；参考坐标和接受状态不受失败影响。
4. T14c：BDF 运动速度、历史几何和二维扫掠原语，核对面积差及内部边抵消。扫掠原语作为 T14→T15 交接扩展，完整流体 GCL 验收仍属于 T15。
5. **T14 数值验收后才能进入 T15 的集成实现/验收**：T15 的历史质量依赖真实接受几何；网格输运依赖有效路径；界面运动依赖 BDF 速度和固定编号。仅有节点位移数组或静态 `TriMesh` 不满足依赖。T15 的公式推导/接口讨论可以提前进行。
6. T15a：新几何组装、历史质量时间项、GCL 网格项与固定网格退化；先检查常量残差，再运行自由流求解。
7. T15b：流体块和几何 JVP、当前边界监测、质量/扫掠诊断；规定运动 MMS 时间/空间收敛与失败重试。
8. A7 汇总真实数据后验收 T15；T16 可在 T14 后开发映射，但完整界面残差消费 T15；T17 等 T15/T16 都通过后集成全场事务。不得跳到 FSI1 来掩盖独立 ALE 验证缺口。

## 6. 拟定数值验证和误差尺度（均未执行）

所有实现后运行使用 [项目 `.venv`](../environment.md) 的 Python 3.12。下面门槛是运行前应冻结的建议，失败时分析误差，不为通过而放宽。统一设置 `L*=1 m`、`T*=1 s`、`U*=L*/T*`、`rho*=1 kg/m³`、`mu*=0.01 Pa·s` 的二维单位深度验证；其它尺度必须明确写入配置。机器精度 `eps=np.finfo(float).eps`。

| 验证 | 配置与量 | 建议判据 |
| --- | --- | --- |
| 调和仿射 patch | `rectangle_mesh(4/8/16)`，全外边界给 `d=A X+b`，含纯平移和非零剪切 | `max|d_h-d|/L* <= 1e-11`；自由节点残差相对 `max(||K d||,||lifted_rhs||,1)` 不超过 `1e-10`；约束误差 `<=100 eps L*` |
| 调和一般解收敛 | 两分量选 `exp(X/L*) cos(Y/L*)` 等解析调和函数，三网格固定边界 | P1 位移 L2/H1 半范数误差均降低，观测阶下限分别 1.8/0.9；记录实际 h，不靠节点数推断 |
| 构形/质量拒绝 | 小尺度合法网格、仿射伸缩/剪切、节点翻转、退化、NaN、未知/冲突边界；端点有效而路径退化的大转动 | 正常更新 `x-X-d=0` 至 `100 eps L*`；失败定位单元/边且参考/接受数组逐项相同；P2 拓扑完全一致，不仅近似坐标相同 |
| BDF 运动速度 | `d(X,t)=a(X)t` 的平移/仿射系数；再用 `a(X)sin(t/T*)`，dt=0.05/0.025/0.0125 s | 线性轨迹含启动步达到 `1e-12 U*`；正弦误差 BDF1/BDF2 观测阶至少 0.9/1.8；BDF2 阶次单独测启动后的导数，启动步另按 BDF1 检查 |
| 离散 GCL | 全域伸缩 `x=(1+0.05 sin t)X`，再用固定边界内部运动 `d=0.03 L* sin(pi X/L*)sin(pi Y/L*)[sin t,cos t]` 的 P1 插值；BDF1/BDF2 | 逐单元 `|D_BDF j-Div_X Q|/S_g <= 1e-11`；`S_g=max(sum|a_j j_j|,1/T*)`；同时报告有量纲缺陷及相对 `1/T*` 的缺陷，避免 1/dt 分母掩盖问题 |
| 扫掠体积/共享边 | 每单元 `D_BDF area = integral_boundary Q·N dS_X`，全域只剩外边通量 | 用 `S_e=max(sum|a_j area_j|,area_ref/T*)` 归一化的单元缺陷 `<=1e-11`；内部边正反通量相加，以 `L*U*` 归一化 `<=1e-12`；全域差以 `L*U*` 归一化 `<=1e-10` |
| 常量残差代入 | 所有历史 `u=U*[1,0.37]`、`p=0`、体力零，所有外边界 Dirichlet 同 U；分别全域伸缩和内部非仿射运动 | 消元前动量残差和全压力残差分别检查；用速度测试行的 `rho* integral |v| sum|a_j j_j| |U| dX` 加 `rho*U*²*L*` 作尺度，最大行缺陷 `<=1e-11`；不能只检查自由行 |
| 常量保持求解 | 上述运动 1 s，dt=0.05/0.025/0.0125；初始常量，首步和 BDF2 历史齐全；保留压力 pin | `max|u-U|/U* <=1e-9`、`max|p|/(rho*U*²)<=1e-8`；逐步记录，不只测终点；常量边界允许流入流出，不用扩张闭域的零壁速制造不相容不可压边界 |
| 固定网格退化 | `d=0`，相同速度/压力状态、BDF 模板及约束 | ALE/既有残差与 Jacobian 按项尺度相对差 `<=1e-12`；相关固定网格流体与时间回归无失败 |
| ALE 质量 | 当前全边界 `integral u·n_f ds` 与 `B_current u`；另测几何扫掠 `D_BDF volume` | `|physical_flux|/(U*L*)<=1e-9`，散度向量逐行用 `U* integral |grad Np| dx` 加小尺度下限归一化 `<=1e-9`；同时给绝对量和现有 `continuity_residual`，压力 pin 行也检查 |
| 移动控制域收支 | 瞬时 `integral (u-w_kin)·n ds + integral div(w_kin) dx`；BDF 几何收支单独用 Q | 各自归一化 `U*L*` 后 `<=1e-9/1e-10`；不能要求一般瞬时 w 的边通量与有限步长 BDF 体积差完全相同，更不能要求相对通量自身为零 |
| 导数扫描 | 非退化随机 u/p/d，几何在质量门槛内；扫描无量纲扰动 `1e-2…1e-7`，方向按 U*/rho*U*²/L* 分别缩放 | 流体块及网格 JVP 的中心差分相对方向误差最佳值 `<=1e-5`，并展示截断/舍入趋势；对零响应方向改用绝对尺度，不除以零 |
| 当前边界监测 | 常压力、仿射速度的变形矩形，解析当前法向/边长 | `BoundaryMonitor` 与独立积分、解析力/流量相对尺度 `<=1e-11`；旧几何 monitor 应拒绝新候选或明确重建 |
| 重复候选和回滚 | 同一时间步重复求解；注入翻转/路径失效/Newton 失败；恢复后同配置重试 | 已接受历史与快照逐项精确相等，旧 token 拒绝；重试解按 `1e-12` 数值尺度一致；全场接受与回滚的完整验收归 T17 |

为验证非恒定流场的 ALE 精度，拟用光滑无散度制造解：令 `psi=U*L* sin²(pi x/L*) sin²(pi y/L*)`、`u=(∂y psi,-∂x psi) exp(-t/T*)`、`p=rho*U*² sin(pi x/L*)sin(pi y/L*) exp(-t/T*)`，体力按 Eulerian Navier–Stokes 解析导数构造，再在当前积分点取值；边界给同一解析速度，压力 pin 给该当前点的精确压力。用固定外边界的内部规定运动，排除不相容边界通量，独立确认解析导数而非由离散残差反算体力。

时间验证至少三 dt，固定充分细网格，终点速度 L2 误差 BDF1/BDF2 观测阶建议下限 0.9/1.8；须报告空间误差平台与启动影响。空间验证至少三网格 `8/16/32`，固定足够小 dt，P2 速度 L2 建议下限 2.5、P1 压力 L2 下限 1.8；若压力约束或求积影响阶次，应说明并加密后判断。当前仿射几何在固定矩形内的内部运动不改变外边界形状，适合隔离流体阶次；曲边 FSI 几何误差另作研究。常量保持与 GCL 的精度级容差不替代此收敛验证。

拟新增验证入口命令由 A0 接 CLI 后冻结；本次未执行以下命令：

```bash
.venv/bin/python -m pytest tests/test_ale_mesh.py tests/test_ale_fluid.py
.venv/bin/python -m pytest tests/test_mesh.py tests/test_fem.py tests/test_fluid.py tests/test_single_fields.py tests/test_stvk_time.py tests/test_flow_linear_solver.py tests/test_transient.py
```

未来证据必须记录实际解释器/依赖、源码状态、网格/时间步/运动幅度/模板/质量门槛、求解容差、逐步质量/GCL/流量/常量误差、导数扫描、收敛率、失败原因与接受步数。A7/A0 才据这些数据更新 [进度](../../gola.md) 和 [验证总表](../validation/single_fields.md)。

## 7. 风险、未决问题和协调者决策项

1. **离散方案待冻结。** 本文建议扫掠通量方案，它满足所给代数 GCL，但 `w_gcl` 与 BDF 运动速度不同；A3/A4/A0 应在实现前确认这一离散选择。若改用非守恒 `M_current D_BDF u + C(u-w_kin)`，必须明确其常量保持与几何收支检查、质量定义和稳定性依据，不能混用本方案的历史质量项。
2. **能量与界面功。** 常量/GCL 不保证能量稳定，扫掠通量与物理界面速度在有限 dt 下的差异可能影响 FSI 功；T15 需测时间收敛，完整功/稳定性由 T16 及后续耦合验证补充，不能提前声称 FSI2 稳定。
3. **调和扩展质量。** `kappa=1` 可能在细小单元或大梁位移附近导致低质量；首版明确拒绝，无自动补救。按参考单元尺寸加权、弹性网格或重网格需另设任务与验证。
4. **几何路径和全局有效性。** 两端正方向不足以保证路径有效或域无交叠；路径检查、边界自交与复杂障碍物邻近碰撞需区分。单元正方向只是一项必要条件。
5. **界面网格不一定匹配。** 当前 `csm_mesh` 与 `cfd_mesh` 独立生成，不能假定固体/流体节点与边段相同；T16 须先验证匹配或统一网格生成，非匹配守恒映射不应悄悄纳入第一版。
6. **公共时间接口和事务。** 显式历史权重、跨场模板对齐、attempt token、全场快照尚未实现；T15 需要时间模板接口，完整事务依赖 T17。固定 dt 限制必须保留。
7. **导数覆盖范围。** 几何 JVP 的 callable 体力/边界导数与约束消元导数、载荷构形导数需明确协议；只验证速度 Jacobian 不能算完成几何依赖。所有有限差分近似需标明。
8. **性能与诊断尺度。** 每次重建算子/monitor/LU 可能增加成本，先保证正确性再计时；现有 `max(norm,1)` 型残差不能替代物理量尺度，新增诊断同时报告绝对值和尺度归一化值。

## 8. 本次文档检查证据

本节仅记录文档检查，不是实现验收。实际检查日期为 2026-10-11；仓库已有大量未跟踪项目文件以及 `.gitignore`、README 等既存改动，公共规划文档还由协调者同步。本 worker 仅创建本文，未改代码、未运行 ALE 数值测试、未更新完成勾选、未提交或上传。

- 阅读实际 `AGENTS.md/spec.md/gola.md/agent.md` 及第 1 节源码，检查参考/当前构形、固定拓扑、BDF 启动、历史质量与现有缓存行为；随后读取协调者更新的 `agent.md` 第 8 节并遵守两波实现依赖。
- 使用 `.venv/bin/python`（实际 Python 3.12.3）运行内联只读检查：剔除代码块/行内代码后，20 处本地 Markdown 引用、19 个唯一目标全部存在；检查覆盖本文的所有真实文件链接，不将代码中的 `Q[a,b](X)` 当链接。首次简单正则曾误报该公式，修正解析后重检，未发现失效链接。
- 同一 `.venv` 内联 AST 检查：第 1 节及相关契约引用的 28 个现有类/顶层函数在 12 个源码文件中均存在；方法实现另经源码阅读核对。此检查不导入或执行数值模块。
- 代码围栏 12 个，成对闭合；检查无行末空白、无 tab、文件末有换行。人工检查现有/拟新增接口区分、T14 前置依赖、常量场抵消、误差尺度及实现前冻结项。
- 实际执行 `git diff --check`：退出码 0，无格式诊断。本文位于未跟踪目录，另执行 `git diff --no-index --check -- /dev/null docs/development/ale_plan.md`：无空白错误诊断，退出码 1 表示新增文件与空文件有差异；补充逐行空白检查避免将未跟踪文件漏检。
- 阅读同轮界面计划后，对齐 `H/H_vertex/H.T`、当前 Cauchy 应力迹载荷、厚度只乘一次及未收敛运动残差契约；未修改该 worker 的文件。上述门槛及未来测试均保持“拟定/未执行”。

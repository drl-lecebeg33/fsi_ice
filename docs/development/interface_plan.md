# A4b / T16 界面设计与交接计划

日期：2026-10-11（Asia/Shanghai）。负责人：A4b，向 A0 交接；状态：设计文档待协调者冻结。本文只提出接口和后续验收，不代表 T16 已实现或数值验证通过。

本轮唯一修改范围是本文；不修改代码、公共进度文件，不提交或上传，不启动其他 worker。依据已阅读的 [AGENTS.md](../../AGENTS.md)、[spec.md](../../spec.md)、[gola.md](../../gola.md)、[agent.md](../../agent.md)。公共规格、分工和里程碑由 A0 维护；本文所列未来文件均尚未创建。

## 1. 现有实现边界与真实符号

| 已有文件与真实符号 | 本轮核对到的行为 | T16 使用约定 |
| --- | --- | --- |
| [mesh/triangular.py](../../src/fsi_ice/mesh/triangular.py)：`TriMesh`、`points`、`cells`、`boundaries`、`jacobians`、`determinants`、`boundary_nodes` | 坐标和连通数组只读；三角形必须 CCW；标签保存端点对，但不保证按所属单元方向排列；内部 `_exterior_edges` 为排序边 | 保留参考网格；候选当前网格另建。法向从唯一 owner 单元的 CCW 局部边恢复，不能直接使用标签端点顺序或排序键算方向 |
| [fem/p2.py](../../src/fsi_ice/fem/p2.py)：`P2Space.from_mesh`、`points`、`cells`、`edge_nodes`、`boundary_nodes`、`p2_values`、`p2_reference_gradients` | 几何仍为仿射 P1；P2 局部顺序 `[v0,v1,v2,edge01,edge12,edge20]`；中边点是端点平均，边键是排序顶点对 | 速度 P2 不等于几何 P2；T16 中边位移必须服从端点平均，不独立弯曲界面 |
| [fem/p1.py](../../src/fsi_ice/fem/p1.py)：`shape_values`、`physical_gradients` | 行梯度按 `grad_ref @ inv(mesh.jacobians)` 转换 | 固体参考梯度用于 `F_s`；流体当前梯度用于 Cauchy 应力，两者不可交换 |
| [solid/linear.py](../../src/fsi_ice/solid/linear.py)：`ElasticMaterial.thickness`、`vector_dofs`、`assemble_elasticity` | 向量交错 `[x0,y0,x1,y1,...]`；体力与边牵引组装已乘厚度 | 不能将已经积分的节点力再送入 `tractions`；线弹性边积分不能充当有限变形转换器 |
| [solid/stvk.py](../../src/fsi_ice/solid/stvk.py)：`stvk_residual`、`stvk_energy`、`solve_stvk`、`assemble_mass` | 总拉格朗日 P1、平面应变；`deformation=I+grad_X d`，`first_piola=F_s S_s`；内力、质量乘参考面积及厚度；`external` 为长度 `2*n_nodes` 的已组装节点力并直接从残差减去 | 外载直接传 `external`，无二次 Jacobian/厚度；现有切线只对固定 `external` 求内力导数，不包含界面随动载荷导数 |
| [fluid/taylor_hood.py](../../src/fsi_ice/fluid/taylor_hood.py)：`FlowOperators.space`、`viscosity`、`FlowSolution.velocity`、`pressure`、`flow_constraints` | 速度 `(n_P2,2)`，压力 `(n_P1,)`，压力为物理压力；混合向量先速度后压力 | 压力不进入 H；界面运动为指定 P2 速度值，沿用既有自由度顺序 |
| [fluid/forces.py](../../src/fsi_ice/fluid/forces.py)：`boundary_force`、`BoundaryMonitor.force`、`BoundaryMonitor.flux` | 三点 Gauss；当前应力 `mu*(grad u+grad u.T)-p I`；输出流体对障碍物总力 `-integral sigma_f n_f ds`，单位厚度；monitor 缓存固定网格梯度、长度和法向 | 复用应力、owner、积分和符号定义；总力函数不能代替节点载荷。几何改变后不可沿用旧 monitor 的几何缓存 |
| [time/bdf.py](../../src/fsi_ice/time/bdf.py)：`BDFContext.derivative`、`BDFHistory.context/accept/snapshot/restore` | `derivative(candidate)=coefficient*candidate+history_term`；BDF2 首步自动 BDF1；已接受历史独立 | H 固定且各场时间上下文相容时，映射与 BDF 微分交换；T16 不自行接受或覆盖历史 |
| [mesh/turek.py](../../src/fsi_ice/mesh/turek.py)：`cfd_mesh` | 当前 `obstacle` 标签包括固定圆柱与固定梁的全部流体障碍边；没有 FSI 可动梁专属标签 | 后续 FSI 建网必须把湿润梁界面、固定圆柱和固体夹持边分开，不能把全部 `obstacle` 载荷送给梁 |

现有文件中尚无可直接调用的界面拓扑、H、P2 节点牵引积分或相容载荷映射。本文不新增这些符号的实现。

## 2. 参考/当前拓扑及几何契约

定义固体参考域 `Omega_s0`、流体参考网格域 `Omega_f0`、湿润界面 `Gamma0`；候选时刻的共同湿润曲线为 `Gamma`。固体夹持边与圆柱固定壁不属于 `Gamma0`；夹持端点若同时属于湿润边，作为共享顶点保留，力积分不删除该顶点贡献。

第一版“匹配”是双方参考端点及边分段一一对应，网格全局编号可以不同、边方向可以相反、字段阶次可以不同；不是要求两边向量长度相等。只端点坐标偶合但分段不同、端点落在另一边内部、曲线阶次不同或分支界面，均超出本版匹配映射范围，须拒绝并指出缺失/歧义边，不能最近邻静默修补。

拟建不可变 `InterfaceTopology`，至少保存：

- `solid_nodes`、`fluid_vertex_nodes`、`fluid_p2_nodes`：去重、有确定顺序的接口全局节点 ID；每个顶点只有一行。
- `edge_pairs`：每条物理湿润边只出现一次，记录固/流全局端点、owner 单元、CCW 局部端点 `(a,b)`、流体 P2 中边节点及端点对应的正/反方向。
- `component_offsets`：允许多条不分支的开/闭曲线；用端点邻接检验重复边、遗漏边、交叉配对及 degree>2 的分支。角点逐边计算法向，不平均法向。
- `X_s_gamma`、`X_f_gamma`、参考边长、参考固体外法向 `N_s`；稳定的 `topology_id` 和网格连通标识。
- 标量稀疏映射 `h`、交错向量映射 `H` 和紧凑/全局 gather/scatter 索引；标签集合及单位、厚度语义。

配对只在参考构形做一次。先按 A0 冻结的绝对/相对几何容差生成候选匹配，必须证明每个参考顶点唯一匹配、每条边具有双方唯一 exterior owner、整段几何相同，再建立确定端点对应。不得在每次变形后重新最近邻匹配。建议无量纲小测试 `atol=1e-12*L`、`rtol=1e-12`，`L` 为显式正长度尺度；生产建网容差由 A0/A1 核定并记录。容差通过仅说明比较通过，不允许隐藏几何缝隙；双方生成器应共享界面参考坐标并审计最大坐标偏差。若 `X_f_gamma=h X_s_gamma` 未满足既定精度，应在建网阶段输出一致参考界面后重建拓扑，T16 不自动移动参考节点。

固体参考 mesh 保持 `X_s`，当前节点 `x_s=X_s+d_s`；流体参考节点 `X_f`，T14 候选 mesh 节点 `x_f=X_f+d_m`。连通性、节点顺序、边 owner 和 P2 边编号必须保持；若重网格则失效并重建拓扑，不能复用旧 H。候选界面顶点满足 `d_m=H_vertex d_s_gamma`，中边坐标必须与流体当前 `P2Space.from_mesh(current_mesh).points` 一致，即端点当前坐标平均。

当前几何对象拟保存 `geometry_id`、共同时间/候选版本、逐边积分点 `x_q`、边长 `l`、法向 `n_s/n_f`、积分权重以及固体 owner 的 `F_s/J_s`。每次候选几何变化都重算长度、法向、流体物理梯度和应力积分；参考拓扑、基函数在局部参数位置的值可复用。几何对象不得修改参考网格或已接受历史。

## 3. P1 到 P2 运动映射 H

令 `n_sG` 为固体接口顶点数，`n_fG` 为流体接口 P2 节点数。紧凑向量按 `vector_dofs` 的交错规则排列，`H` 为 `(2*n_fG,2*n_sG)` 的稀疏矩阵；标量 `h` 为 `(n_fG,n_sG)`，`H=h tensor I_2`。

每条对应参考边，按对应固体端点 `(A,B)` 和流体 P2 节点 `(a,b,m)`，局部标量映射为：

```text
[d_a]   [1    0  ] [d_A]
[d_b] = [0    1  ] [d_B]
[d_m]   [1/2  1/2]       （每个 x/y 分量相同）
```

此矩阵行按端点配对定位；如果流体 owner 边反向，积分参数用 `s_s=1-s_f`，不能把端点列次序当作一致。局部重复顶点行只登记一次，不相加；每个中边节点仅对应一条边。`h*1=1`，且共同参考坐标 `X_f_gamma=h X_s_gamma`。P2 边形函数乘上述 h 后精确等于 P1 边形函数，因此映射在整条直边上精确，不仅在节点成立；其值域约束了 P2 中边运动。固体没有中边自由度，不能截断流体中边载荷或单独指定其运动。

记 `G_s` 从全固体位移提取接口交错向量；`E_f` 把接口量散布到全流体速度向量。则 `d_f_gamma=H G_s d_s`、`v_f_gamma=H G_s v_s`。ALE 只需要 P1 接口顶点位移，从 H 的顶点行取值；P2 网格速度在中边取顶点平均。

对于固体一阶位移/速度系统，在收敛状态 `v_s=D_BDF d_s`。同一 dt、BDF 实际阶次（含启动）、系数及已接受历史下，固定 H 保证 `D_BDF(H d_s)=H D_BDF(d_s)`，流体界面速度和 ALE 网格速度一致。任意未收敛候选中两种速度可能不同，应报告运动残差 `v_s-D_BDF d_s`，不能宣称相等或修改历史消除它。T16 只映射给定候选，不决定 T17 接受步。

## 4. 法向、应力与流体对固体力

对所属 CCW 流体单元的局部边切向 `tau_f=x_b-x_a`，取流体外法向 `n_f=(tau_f_y,-tau_f_x)/|tau_f|`。固体 owner 同样得到 `n_s`；共同界面必须 `n_s=-n_f`。法向是各域的外法向，不能以全局排序边方向代替。参考固体法向 `N_s` 由参考固体 owner 构造。

流体当前 Cauchy 应力和作用在固体上的牵引：

```text
sigma_f = -p I + mu (grad_x u + grad_x u^T)
t_s = -sigma_f n_f = sigma_f n_s          （流体对固体，当前面积上的力）
t_f =  sigma_f n_f = -t_s                （相反侧的力）
```

`p>0,u=0` 时 `t_s=-p n_s`，压向固体内部。沿 y 向上的固体上表面，力的 y 分量为负；流体静压力的方向测试应包含上下/左右四种法向。使用 P1 压力和 P2 当前物理梯度在流体 owner 单元迹上评估；流体不连续梯度从流体这一侧取值，不跨单元平均。

界面 Cauchy 牵引作为本版统一的物理离散载荷：分区法和整体法必须共享同一积分及 Hᵀ，不得一边用应力迹、另一边改用 Dirichlet 残差反力。应力迹积分和离散流体约束反力在有限网格上一般不精确相同；可做诊断，但不把二者数值相等作为 T16 恒等式。若 A0 后续选择残差反力路线，须作为离散模型变更统一冻结并重做验收。

## 5. 当前牵引到参考构形及厚度

固体参考 owner 单元的 P1 变形梯度为 `F_s=I+sum_a d_sa tensor grad_X N_a`，`J_s=det(F_s)>0`。二维平面应变等价于三维 `diag(F_s,1)`，展向厚度 `b=material.thickness` 不随面内变形变化。

Nanson 与 Piola 关系 `n_s ds=J_s F_s^(-T) N_s dS`、`P=J sigma F^(-T)` 采用[deal.II 官方有限变形教程的应力与牵引约定](https://dealii.org/developer/doxygen/deal.II/step_44.html)。在本版直边/仿射几何下：

```text
j_Gamma = ds/dS = l/L0 = J_s ||F_s^(-T) N_s||
n_s = F_s^(-T) N_s / ||F_s^(-T) N_s||
T_s0 = j_Gamma t_s = J_s sigma_f F_s^(-T) N_s
T_s0 dS = t_s ds
```

`T_s0` 是每单位参考面积的外牵引，仍以当前空间 x/y 分量表示，与固体节点位移的分量共轭；不对牵引向量额外乘 `F_s^-1`，不当作第二 Piola 应力。本式中的 `F_s` 是固体变形梯度，不是流体 ALE 体积变形或 `TriMesh.jacobians` 的几何 Jacobian。双方体积变形可不同，只有共同界面位置、切向和表面测度必须相容。

直接 P1 参考弱式外力与当前弱式外力等价：

```text
f_s,A = b integral_Gamma0 N_s,A T_s0 dS
      = b integral_Gamma  N_s,A t_s ds
```

两条合法积分路径只选一条执行：

| 路径 | 输入牵引 | 积分测度 | 可乘的额外系数 |
| --- | --- | --- | --- |
| 默认当前路径 | `t_s` | `l*w_q` | 厚度 b 一次；不再乘 `j_Gamma`、`J_s` |
| 等价参考路径（用于转换检查） | `T_s0=j_Gamma*t_s` | `L0*w_q` | 厚度 b 一次；不再乘 `j_Gamma`、`J_s` |
| 已组装节点力输入 StVK | `external=f_body+G_s.T*f_s_gamma` | 无 | 无；不再积分，不乘任何 Jacobian/厚度 |

流体 `boundary_force` 及 `BoundaryMonitor.force` 当前返回单位厚度力（物理单位 N/m）；乘 b 才能与固体已含厚度的总节点力（N）比较。拟定 T16 节点载荷输出统一为已含 b 的 N，牵引为 Pa，位移 m，功 N·m，总矩 N·m。不得把 `boundary_force` 的总力再次当牵引积分；也不得在 Hᵀ 后再次乘 b。非单位厚度测试必须覆盖 `b=0.2` 和 `b=2`。

## 6. 相容载荷映射与检查量

先在流体 P2 接口迹空间组装“作用在固体上的力”的紧凑节点载荷 `g_f_gamma`（这个名字的 f 表示所用迹基，不表示作用于流体）：

```text
g_f,i = b sum_edges sum_q N_f,i(s_q) t_s(x_q) l_edge w_q
f_s_gamma = H.T g_f_gamma
external = f_body + G_s.T f_s_gamma
```

在直边上应力迹至多一次，P2 测试函数二次，三点 Gauss（精确到五次）与现有 forces.py 一致。每条边的两端和中边贡献 scatter-add 到去重的节点向量；边界 owner 的非边 P2 基函数在边上为零，不混入其他速度或压力自由度。参考转换路径采用同一参数点与基函数，在检查时独立计算表面比。

任何虚位移 `delta d_s_gamma` 对应 `delta d_f_gamma=H delta d_s_gamma`，严格有：

```text
(delta d_f_gamma).T g_f_gamma = (delta d_s_gamma).T f_s_gamma
```

这要求 Hᵀ 作用的是已积分的对偶节点力。若未来接口传的是节点牵引系数 `tau_f`，需先通过界面质量矩阵形成 `g_f=M_f tau_f`；如需固体牵引系数，可解 `M_s tau_s=H.T M_f tau_f`。本版不把系数与节点力混用，也不以逐点插值替代转置映射。

单边常牵引 t、长度 l 时流体 P2 端点/中点等效力分别为 `b*l*t*(1/6,1/6,2/3)`，Hᵀ 后固体端点各为 `b*l*t/2`。这是检测丢失中边力和重复端点行的解析检查。

诊断输出至少同时记录：

- 总力 `F_s=sum_A f_s,A`、`F_g=sum_i g_f,i`、`F_quad=b*sum_eq t_s l w`，三者比较；若采用单位厚度 monitor，则比较 `F_s/b`。
- 作用反作用：迹力 `g_on_fluid=-g_f_gamma` 的总力与 `F_s` 抵消；此处是共同牵引的符号检查，不是流体方程残差恒等式。
- 当前总矩（关于同一原点 o）：`M_s=sum_A (x_s,A-o) cross f_s,A`、`M_g=sum_i (x_f,i-o) cross g_f,i`、`M_quad=b*sum_eq (x_q-o) cross t_s l w`，二维 cross 为 `x f_y-y f_x`。
- 虚功/功率、运动残差、界面位置误差、`n_s+n_f`、两种 `j_Gamma` 的差、最小固体 `J_s` 与当前网格质量；包括约束顶点载荷，约束反力另外统计。

`h*1=1` 保总力；`x_f_gamma=h x_s_gamma` 保当前总矩和刚体旋转虚功。H 固定于参考坐标，但因线性重现与 `d_f=H d_s`，上述当前坐标关系仍成立。不能只检查参考坐标矩，也不能先删掉夹持节点再检查完整界面总力/总矩。

## 7. 未来模块归属、输入输出与导数边界

下表为提议的未来 API 名称和归属，不是现有可调用接口，也不在本轮创建空壳文件。

| 未来文件 / 负责人 | 提议符号 | 输入 | 输出与副作用约定 |
| --- | --- | --- | --- |
| `src/fsi_ice/interface/topology.py` / A4b，A1 复核 | `InterfaceTopology`、`build_matching_interface` | 固/流参考 `TriMesh`，参考 `P2Space`，两侧湿润标签，容差与尺度 | 不可变拓扑、边 owner 与端点配对、紧凑 ID、h/H、索引；异常带边/标签，不修改网格 |
| `src/fsi_ice/interface/mapping.py` / A4b | `map_motion`、`map_load` | 拓扑、全固体 `(n_s,2)` 位移/速度或紧凑 P2 节点力 `(2*n_fG,)` | 接口 P2 `(n_fG,2)` 运动、P1 ALE 顶点数据；载荷输出 `(2*n_sG,)` 并由约定 scatter 成全固体 external；不接受历史 |
| `src/fsi_ice/interface/geometry.py` / A4b，与 T14 对接 | `InterfaceGeometry`、`evaluate_interface_geometry` | 拓扑、T14 候选当前流体 mesh/space、固体位移、候选几何版本 | 当前积分点/长度/法向、固体 F_s/J_s、j_Gamma、匹配误差；拒绝版本不一致/翻转/裂缝，不更新历史 |
| `src/fsi_ice/fluid/forces.py` / A3，A4b 协商修改 | `boundary_cauchy_stress`（提议提取公共评估核） | 当前 mesh/space、velocity、pressure、mu、owner、局部积分点 | 每点 `(2,2)` Cauchy 应力；现有总力 monitor 和未来界面积分共享此核，避免两套符号实现 |
| `src/fsi_ice/interface/traction.py` / A4b | `assemble_interface_load`、`pullback_traction`、`InterfaceLoad` | 相同候选几何和流体状态、拓扑、b；转换检查额外使用固体 F_s/N_s | Pa 的 `t_s/T_s0`、N 的 `g_f_gamma/f_s_gamma`，积分元数据、单位厚度总力、总矩；输出带 geometry_id，不修改状态 |
| `src/fsi_ice/interface/checks.py` / A4b，A7 复核 | `interface_diagnostics` | 同一 topology/geometry/load、候选运动、测试虚位移、尺度和容差 | 各误差及通过/失败原因；不静默忽略零尺度或 NaN/Inf |
| `src/fsi_ice/ale/` / A4a (T14/T15) | 以 A4a 设计冻结为准 | 参考 mesh、顶点位移、时间上下文 | 固定连通候选 mesh/P2 space、相容网格速度、几何版本与质量；本文不替 A4a 发明已存在符号 |
| `tests/test_interface.py`（及必要的细分测试） / A4b | 后续 T16 验收用例 | 小型双域网格、解析场与非法拓扑 | 记录下节误差与拒绝信息；本轮不创建/运行 |

公开函数保持输入/输出数组形状、作用对象、构形和厚度语义明确。紧凑节点力只存一份；全固体外力通过 G_sᵀ scatter 与已有体力合并。接口载荷组装不调用固体 solve；A5/A6 调用共同残差，不复制物理实现。

整体法 T21 需要访问 `g_f(u,p,x_f)` 对速度、压力、当前几何的导数，并沿 T14 几何映射传播到固体位移；H 在固定参考匹配中为常矩阵，`D f_s=H.T D g_f`。几何导数包含应力物理梯度、法向及长度，参考路径还包含表面比。`stvk_residual(...,external)` 返回的现有矩阵不包含这些项。现有 `stvk_energy` 的 `-external dot d` 只适用于冻结 external；不能据此声称已获得随动界面力的势能或完整 Newton 切线。具体导数返回格式等待 A0/A3/A6 冻结，T16 核保持可独立评估，方向导数检查不依赖 solve 黑盒。

## 8. T14 依赖与待冻结决策

依赖保持 [agent.md](../../agent.md) 的 **T16 ← T06、T11、T14**。本轮 T16 设计可与 T14 设计并行，不等于删除 T14 集成依赖；T14 当前几何与时间约定未验收前，T16 不能标记实现/集成完成。

| 必须由 A0 协调冻结的交接 | 需明确的内容 | 未冻结时对集成的影响 |
| --- | --- | --- |
| A1/A4a/A4b 匹配几何 | 湿润边标签与固定圆柱/夹持边分离，共享参考端点及分段、编号稳定性、配对容差、候选几何版本 | 无法建立可复用 H，不能接入完整 FSI |
| T14 当前几何 | 参考坐标只读；当前 mesh 另建；P1 ALE 几何且 P2 中点平均；owner 不变；网格翻转拒绝与质量阈值 | 界面法向、积分及总矩检查没有共同几何 |
| T14/T07/T17 时间契约 | dt、BDF 实际阶次与启动、候选与接受版本、位移/网格速度/固体速度的历史相容，重复迭代不提交 | 不能保证运动与离散功率一致，失败恢复可能污染状态 |
| A2/A3/A4b 力契约 | 使用共同 Cauchy 应力迹；`t_s=-sigma_f n_f`；流体监测单位厚度、T16 节点力含 b；输出给 StVK external | 符号、Jacobian 或厚度可能重复，A5/A6 可能求不同离散问题 |
| A3/A4b 公共应力核 | forces.py 的可复用粒度与动态几何缓存失效方式、operator/solution 与 geometry_id 绑定 | 不能安全复用固定 monitor，容易把旧梯度和新坐标混用 |
| A0/A6 导数接口 | load 对 u/p/几何的块导数、索引与 gather/scatter、T21 完整链式规则 | T16 可做独立映射验证，但不能宣称整体 Jacobian 完成 |
| A7 验收尺度 | 长度、力、功和矩的正参考尺度、几何容差及异常策略；FSI 配置厚度与力积分范围 | 不能以接近零的参考值构造不可靠相对误差 |

上述定义是本轮建议冻结的基线；如协调者已有 T14 设计，A0 应逐项对照字段及命名后冻结，不能以“可并行”略过依赖，也不能把本文未来符号写成已实现模块。

## 9. 后续可执行工作包与具体验收（全部未执行）

在公共约定冻结后按以下增量交付；子包不替代 T16 的总体验收，也不改变 T14 集成依赖。

| 增量 / 负责人 | 修改归属 | 进入条件与输入 | 可交接结果与退出检查 |
| --- | --- | --- | --- |
| T16a 拓扑与运动 / A4b | `interface/topology.py`、`interface/mapping.py`、`tests/test_interface.py` | T06/T11 数据结构及 A0 冻结的参考分段、索引、边界标签；可与 T14 实现并行 | 构建 h/H、去重及方向配对；通过拓扑、整边运动、BDF 交换与非法匹配检查，输出具体误差 |
| T16b 对偶节点力 / A4b | `interface/mapping.py`、`interface/checks.py`、对应测试 | T16a；先使用明确标识的合成节点力与规定共同几何 | Hᵀ、虚功、总力/当前总矩通过解析检查；此阶段不能声称真实流体牵引已接入 |
| T16c 当前牵引及参考转换 / A4b，A3 单独维护公共流体核 | `interface/geometry.py`、`interface/traction.py`；`fluid/forces.py` 由 A0 指定唯一 A3 执行者 | **T14 已验收**；T16a/b；一致当前 mesh/space、BDF 和 geometry_id；公共应力核及厚度输出冻结 | 真实流体应力迹积分、Nanson 等价、符号/厚度/缓存失效与异常检查；提供 external 输入说明及证据 |
| T16d 集成复核 / A0、A2/A3/A7 | 集成验收归属由 A0 分配，公共进度由 A0 维护 | T16c 和 T14 证据齐全 | 固体 residual 直接消费已积分 external，运动和功率关系检查；A7 复核尺度，A0 才可更新 T16 状态；T17 仍等待 T15/T16 |

每个增量本地使用 `.venv` 执行相关功能及测试，按下表预定标准记录命令、最大误差和失败诊断；不能用 T16a/b 的合成载荷检查代替 T16c 的几何/真实牵引验收。


小型确定性用例的代数/积分容差建议统一为 `1e-12*S`；`S` 取显式正物理尺度与相关绝对贡献和的较大者，避免零力/零功除零。例如功取 `max(W_ref,||delta d_f||*||g_f||,||delta d_s||*||f_s||)`，力取 `max(F_ref,sum_eq b*l*w*||t||)`，矩取 `max(M_ref,sum_eq b*l*w*||(x_q-o)||*||t||)`。大规模累加容差应事先依据项数、机器精度和条件数评审，不能仅为消除失败放宽。几何误差用第 2 节 A0 冻结的匹配容差；候选法向单位性/反向关系以及 j_Gamma 的尺度化误差建议不超过 `1e-12`。

| 用例 | 设置与实际应检查量 | 验收要求 |
| --- | --- | --- |
| 拓扑与方向 | 双域直边、多边角点、开链/闭链；随机重排全局编号与标签方向 | 配对边/节点唯一；H 等价于置换前；owner 法向相反；顶点行不重复；所有湿润边覆盖一次 |
| 运动阶次 | 常量平移、一般仿射位移、有限刚体转动 `d=(R-I)X+c`；不同固/流编号 | 整条边的 P2 迹等于 P1 迹；中点平均；`h*1=1`；当前 `x_f=h x_s`；无几何缝隙 |
| 时间一致性 | BDF1、BDF2 的启动步/正常步，一致的已接受线性与二次时间历史 | `D_BDF(Hd)=H D_BDF(d)`、接受解网格/固体界面速度一致；重复候选不修改历史；不一致上下文被拒绝 |
| 常牵引解析力 | 单边长度 l、常向量 t，b 为 1、0.2、2 | P2 力为 `b*l*t*(1/6,1/6,2/3)`，P1 两端各 `b*l*t/2`；总力正确，无中点力丢失及二次厚度 |
| 压力和黏性符号 | `u=0,p>0` 的四种表面；纯剪切 `u=(gamma*y,0),p=0` | 压力为 `-p n_s`；剪切应力为 `mu*[[0,gamma],[gamma,0]]`；与已有 boundary_force 单位厚度结果同符号 |
| 虚功与功率 | 固定种子随机虚位移、随机迹节点力；含零/小量，含夹持节点；映射速度 | `delta d_f.T*g_f=delta d_s.T*f_s`；同样检查速度功率；受约束节点载荷完整保留 |
| 总力与当前总矩 | 常/线性牵引、多边角点；在非零位移与旋转后积分；两个不同原点 | P2、P1 与积分总力一致；三个当前矩一致；作用反作用；平移原点后矩满足 `M(o+a)=M(o)-a cross F` |
| Nanson/参考转换 | `F_s=I`、有限旋转、`diag(2,0.5)`、带剪切的正 J 变形；不同方向边 | `l/L0=J_s*norm(F_s^-T N_s)`；参考/当前节点力、总力与虚功相等；特别检查 J=1 而边长比非 1 的情况，排除用体积 J 代替表面比 |
| 动态几何缓存 | 同一拓扑连续两个非等长候选，压力/速度固定 | 新载荷按新长度/法向/物理梯度变化；旧几何 monitor 不能当新载荷；退回旧几何后可重复 |
| 异常与诊断 | 未知/空标签、重复/缺失边、非 exterior/多个 owner、歧义匹配、分段不匹配、零长度、NaN/Inf、非正 J、错误维度/厚度、版本错配、冲突端点约束 | 明确失败并定位标签/边/候选；不 silently continue，不修改网格或历史；正常空载荷是有效零值，但空界面不是成功拓扑 |
| 下游导数门槛 | T21 前针对同一组装核的 u/p/几何非退化方向，扫描扰动大小 | 按 spec.md 相对方向导数误差不超过 `1e-5` 并有误差趋势；未完成前不称完整整体 Jacobian |

T16 后续验收证据应包含 `.venv` 的 Python/依赖版本、命令、Git 状态、网格/标签、厚度、几何与力尺度、积分规则、实际最大误差以及失败用例信息，写入 A0/A7 约定的验证记录。以上全是计划，不是运行结果；完整 FSI1/FSI2 参数及参考值仍由 T13/A7 核验。

## 10. 本轮实际文档检查证据

代码基线读取：`git rev-parse HEAD` 返回 `dd6232b829c976a0307135200138ced8c00fbc00`。开始时仓库有既存修改及大量未跟踪项目文件；本 worker 只创建本文，不改写这些内容。静态阅读核对了第 1 节的实际文件与符号，并核对任务范围、T16 的 T14 依赖、未来模块与现有实现的区分。

实际检查命令、结果和最终交接见本节下方记录；这里只验证文档，不运行求解器，不执行或声称任何 T16 数值测试。

本轮实际执行并记录：

- 使用 `.venv/bin/python` 运行内联文档检查（`pathlib`、`re`、`ast`）：14 个本地 Markdown 文件链接全部存在；第 1 节引用的 21 个顶层类/函数在对应源文件 AST 中存在；代码围栏闭合；Hᵀ、T14 依赖、未执行标记、力/矩/虚功及厚度主题齐全。此脚本只读文件，不导入或运行数值模块。
- 最终再次使用项目 `.venv/bin/python`（Python 3.12.3）检查上述 14 个本地链接、21 个顶层符号，并增加 Markdown 表格列数一致检查：全部通过，代码围栏闭合，四个增量包及必需主题齐全。
- `git diff --check`：退出码 0，无格式错误。
- 因本文属于未跟踪新文件，额外运行 `git diff --no-index --check /dev/null docs/development/interface_plan.md`：无空白错误输出；退出码 1 表示新文件有差异，是 no-index 的差异状态，不是数值测试失败。
- 阅读协调者关于公共约定的消息，并复读 `agent.md` 第 8 节：确认首波仅做映射/功/力基础，T16 牵引/几何集成仍依赖 T14；已据此写入上节四个增量工作包。

检查结论：本轮完成设计文档及文档自查；尚待 A0 冻结第 8 节接口，再按第 9 节实施。未实现界面或 ALE 功能，未运行数值测试，未提交、未上传；本文也是本轮持久报告。

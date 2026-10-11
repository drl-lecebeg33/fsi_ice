# T20：FSI1 稳态分区基准计划

状态：前置定义、正式双域 Gmsh 导入链和网格拓扑检查已完成；耦合求解尚未实现，不能报告 FSI1 PASS。任务依赖 T19 Aitken 和 T13 FSI1 参数来源核实。机器可读参数与来源见 [FSI1 reference case](../../cases/fsi1_reference.json)，网页原件、访问清单和网格原型证据在 `results/t20/reference/`。

## 已冻结的基准定义

依据 Dortmund FeatFlow 基准的 [Definitions](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_definitions.html)、[FSI tests](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_tests/fsi_fsi_tests.html) 和 [Reference Values](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_reference.html)：

- 通道 `L=2.5 m, H=0.41 m`；圆柱中心 `(0.2,0.2) m`、半径 `0.05 m`；梁长 `0.35 m`、厚 `0.02 m`，右下角 `(0.6,0.19) m`，参考尖端 A=`(0.6,0.2) m`。
- 流体 `rho_f=1000 kg/m³`、运动黏度 `nu_f=0.001 m²/s`（故 `mu_f=1 Pa s`）；平均入口速度 `Ubar=0.2 m/s`，参考 Reynolds 数 20。入口抛物线最大速度为 `1.5 Ubar`；顶部、底部、圆柱及流固界面无滑移；出口使用零均值压力参考的自然边界实现并须验证其积分通量。
- 固体采用可压缩 St. Venant–Kirchhoff：`rho_s=1000 kg/m³`、`nu_s=0.4`、剪切模量 `mu_s=0.5 MPa`。由 `E=2 mu (1+nu)` 得 `E=1.4 MPa`，而非把表格中剪切模量误作 Young 模量。梁根部夹持于固定圆柱。
- 官方 FSI1 为稳态解。官方逐网格结果在 level 5+0 给出 `ux(A)=2.271553e-5 m`、`uy(A)=8.208126e-4 m`、drag `14.29486 N/m`、lift `0.7636992 N/m`；随 level 2+0 至 7+0 的变化应作为网格趋势参考，不能把单档值当作全离散精确解。

来源待决项：Definitions 正文把压力差测点 B 写为 `(0.15,0.2) m`，同页几何表写为 `(0.2,0.2) m`。两者都保留在原始网页快照中；在引用压差作为输出量前先确认权威定义。该歧义不影响上述 tip 和合力表格对照。

## 网格原型现状

已落盘脚本 `results/t20/reference/mesh_prototype.py` 与 Gmsh 4.12.1 输出，对 `h=0.01/0.005/0.0025 m` 三档网格构造同一份共享流固边界。三档均通过 `build_matching_interface`：界面 P1 固体节点数分别 147/291/579，P2 流体节点数 293/581/1157；坐标误差、常数保持误差和相反法向误差为 0，合成虚功与总力误差均为约 `1e-13` 以下。详情见 `mesh_prototype_summary.json` 和各档 `metrics.json`。

该原型已表明匹配拓扑可行，但不是正式求解网格选择：最细档已有 562,645 个混合流体自由度且尚未加入非线性雅可比/网格方程成本。实现前需用实际求解器内存、稳态收敛与参考误差确定可运行的网格层级；必要时增加更粗档，不能仅凭几何原型挑选精度档。

## 正式网格入口（2026-10-11）

`fsi_ice.mesh` 现导出 `FSIMeshPair`、`read_fsi_gmsh2`、`fsi1_geometry` 和 `fsi1_mesh`。导入器从同一个 MSH2 文件按 `fluid`/`solid` 物理面拆出两个 `TriMesh`，保留本地顶点到原 Gmsh 节点号的只读映射，并要求所有冻结边界标签存在。它会拒绝缺失物理域、边界节点不属于对应域、重复/无效节点、非平面或非一阶三角形输入；三角形方向统一为逆时针后仍由 `TriMesh` 验证外边界、流形性和质量。

本地实际执行 `.venv/bin/python -m pytest tests/test_fsi1_mesh.py tests/test_single_fields.py -q`，结果 **13 passed / 2.18 s**。真实 Gmsh 测试生成 `h=0.04 m` 双域网格，再用 `P2Space` 与 `build_matching_interface` 验证共享全局顶点号、参考坐标、常数保持、相反法向、夹持/湿边界交点和尖端 A。

公共网格入口集成后执行 `.venv/bin/python -m fsi_ice test`，结果 **428 passed / 119.75 s，退出码 0**，覆盖原有单场、ALE、界面、事务与分区耦合回归以及新增 3 项 FSI1 网格检查。

协调者还用新的正式导入器重新读取三档原型文件并重建拓扑，得到与原证据相同的结果：`h=0.01/0.005/0.0025 m` 分别为流体单元 `8078/31341/124220`、固体单元 `734/2730/10656`、界面边 `146/290/578`，拓扑 ID 仍为 `31a52dd13a02e770`、`1e7e1ba3d784656d`、`246cf64e86a10d3b`。这证明原型产物可由正式代码路径读取，不代表已完成流固求解。

## T20 实施与验收顺序

1. [已完成] 在 Gmsh 标签到 `TriMesh`/高阶空间的导入链上验证区域方向、界面节点共享、实体法向、圆柱与梁根部约束、P2/P1 离散和网格质量；保存网格摘要与源文件指纹。
2. 先独立验证 CFD1 固定圆柱流场和 CSM1 线性小变形，再启用有限变形 StVK、ALE 网格方程及分区 Dirichlet–Neumann。复用现有单场残差、界面功守恒、事务回滚和 Aitken 控制器。
3. 稳态迭代中检查流体动量/连续性、固体平衡、界面运动与牵引平衡。每轮用同一已接受网格/历史构造候选；任何失败都不得接受结构或网格状态。
4. 在至少三档可承受网格上记录 tip A 位移、官方积分路径上的圆柱加梁合力、网格尺度、单场/界面残差、耦合次数、运行时间和失败原因；做网格趋势及重复运行逐字段确定性检查。力的符号、积分路径和单位按 quantities 页面实现并写回配置。
5. 与官方 FSI1 表比较时注明空间离散/力定义差异，检查逐级趋势并报告相对差；若未达到预先声明的容差，应标为未通过/需解释，不能以耦合残差小替代物理参考误差。

T20 可完成门槛：所有方程与网格接受路径已实现；单场及界面守恒前置验证通过；三档稳态计算达到声明残差；参考量、网格趋势及重复运行检查有可复现文件和本地测试证据。当前仅满足参考参数核验和网格映射原型，不满足 T20 完成门槛。

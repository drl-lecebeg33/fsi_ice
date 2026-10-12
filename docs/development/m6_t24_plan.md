# M6/T24 瞬态接口与验收冻结

日期：2026-10-12。单一执行者承担 A0/A5/A6/A7，自查，不声称独立复核。

## 范围与依赖

M5 已交付；本轮首先实现 T24 的 FSI2 参数、两路线瞬态驱动、全场事务与短程一致性。完整稳定周期、空间/时间独立性、IQN-ILS 和四算法成本比较仍分别等待 T24/T25/T26 的数值证据。不提交、不推送。

## 物理与离散

采用原始 FEATFLOW FSI2 定义：rho_s=10000 kg/m³、nu_s=0.4、mu_s=500000 Pa（E=1400000 Pa）；rho_f=1000 kg/m³、mu_f=1 Pa·s、Ubar=1 m/s。二维平面应变 StVK，单位厚度 1 m；力报告为单位深度 N/m。几何和匹配拓扑复用 FSI1。

入口由静止开始，0≤t<2 s 乘 (1-cos(pi*t/2))/2，此后为完整抛物线。压力两侧均 pressure_pin=None，自然出口定压。固体、网格与界面运动统一由同一 BDF 模板计算；支持 BDF1、BDF2，BDF2 第一步用 BDF1。

整体法复用 T21/T22。分区法逐次使用给定位移计算 v_s=a0*d_s+history_d，调和扩展网格，求 ALE 流体，积分牵引，以 Hᵀ 传给 BDF 固体求解器，再返回位移。全固体位移作为固定点向量，收敛时保留最后评估的输入与其流体/载荷，统一重算 T21 全残差。

两条路线经同一 StepTransaction 执行 evaluate→validate→prepare→publish。所有求解在候选工作区进行；失败不改变已接受根、监测行或任何历史，重试采用新 attempt_id。

## 本轮验收（运行前冻结）

- 小型真实 FE 两域问题：BDF1/BDF2 三步，fixed/Aitken/整体逐变量 max 差 ≤1e-8*max(norm_inf)+1e-10*S；S={速度1、压力1000、长度0.41、载荷215.06}。
- 每个接受步检查同一全残差 max|R/S|≤1e-7、无 NaN/Inf、网格最小质量≥0.1、正固体 Jacobian、连续性≤1e-10。
- 注入求解、validate、prepare 失败，已接受状态逐位不变；撤销故障后重试与基线一致；每步三个历史恰接受一次，BDF2 正确启动。
- 真实 FSI2 双域网格短程运行 h=0.03、dt=0.01、t_end=0.03，验证启动和两条路线；此网格和区间仅验证接线，不判周期精度。
- 配置错误、不收敛及输出失败不得残留旧成功报告；保存失败阶段及已接受步数。

## 后续周期验收（门槛不放宽）

基准配置 h=0.02、dt=0.01、t_end=20 s。分析排除入口启动，最后至少三个完整周期的周期/振幅/均值变化均≤2%；梁尖 y 均值接近零时用振幅尺度。参考目标采用原始表 4+0、字面 Δ=0.0005 行：y 振幅0.0816 m、主频1.93 Hz、阻力均值215.06 N/m。保留原始不同 dt/网格表差异，不将来源时间积分认作本项目 BDF。

符合 spec 的 y 振幅误差≤10%、主频≤5%、阻力均值≤10% 才能判 PASS。短程或未稳定结果为 INCOMPLETE。三网格与三步长分别固定另一变量后研究，不用短程结果替代周期证据。整体 Jacobian 当前组装昂贵，先测真实运行成本，再决定长时实验安排。

来源：

- [FSI tests](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_tests/fsi_fsi_tests.html)
- [Definitions（入口启动）](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_definitions.html)
- [Reference values（圆柱与梁力之和）](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_reference.html)

原始网页在线核对日期为2026-10-12；本轮数值证据见后续 T24 验证记录，不将计划当成通过。

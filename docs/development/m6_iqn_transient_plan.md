# M6：IQN-ILS 瞬态接入与公式复核

日期：2026-10-12。负责人：单一执行者承担 A0/A5/A7，实现与复核均为自查；不提交、不推送。

## 目标与依赖

基线 `1614dc9`，T24 短程瞬态与 T25 小型 FE 最终解一致性已有证据。本轮将 `iqn` 接入共享瞬态驱动、FSI2 配置与 CLI，并复核控制器数学更新。旧 T25 的最终解一致性不证明 IQN 公式正确，历史记录保留，由本轮证据更正其算法结论。

## 数学修正

设 r=G(x)-x，相邻已评估输入差 s=Δx，残差差 y=Δr。标准 IQN-ILS 以 V=[y] 拟合 Vα=-r，再由求解器输出差 W=[ΔG]=[s+y] 更新 x_new=G(x)+Wα。所有最小二乘行使用调用方物理尺度 S：拟合 (V/S)α=-r/S。初始步及重启使用 x_new=x+ω₀r。与 preCICE 官方实现对照（仅核验公式，不引入依赖）：

- [IQNILSAcceleration.cpp](https://github.com/precice/precice/blob/develop/src/acceleration/IQNILSAcceleration.cpp)：残差差最小二乘系数与输出差更新。
- [BaseQNAcceleration.cpp](https://github.com/precice/precice/blob/develop/src/acceleration/BaseQNAcceleration.cpp)：相邻残差差、求解器输出差及候选更新。

保留 α 上限、历史容量、数值秩与相对 LS 残差重启选项；相对 LS 残差采用 ||V_scaled α+r_scaled||₂ / ||r_scaled||₂。默认保守重启可降低加速效果，不以调大物理误差门槛解决。历史仅限当前候选步，跨步、失败重试不复用。

## 文件范围

`coupling/{partitioned,transient}.py`、`validation/fsi2.py`、CLI、FSI2 IQN 配置、相关测试、README/spec/gola/agent、原理页与本轮验证记录。保持物理残差、BDF、网格、牵引与事务实现不变。

## 运行前冻结验收

1. 非收缩标量映射 G(x)=1-2x：不同 ω₀ 下 IQN 在三次评估内到 x=1/3（绝对误差≤1e-12），区别于直接固定点迭代。
2. 二维非收缩仿射映射：允许完整历史，残差投影容差为1；四次评估内到解析解（归一化误差≤1e-10）；单位变换与相应尺度变化不改变候选轨迹。
3. BDF1/BDF2 三步小型真实两域 FE：fixed/Aitken/IQN/整体逐变量差遵守 T24 的 1e-8 相对加 1e-10·S 判据；每步统一残差、连续性与几何门控不变。
4. IQN 求解中途、validate、prepare 失败后已接受根不变；重试与无失败基线逐数组一致。
5. FSI2 配置/CLI 接受 IQN 四个控制项，拒绝未知项及非法值；旧三路线配置兼容。
6. 真实 FSI2 h=0.03、dt=0.01、t=0.03 的 IQN 与同配置 Aitken/整体末态按 T24 判据比较；保留 INCOMPLETE。
7. 相关测试通过后全量回归，保存日志；随后对 h=0.02、dt=0.01 长时运行做实际探索，任何不收敛保留失败证据，完整周期无证据不勾选。

本轮修改必须同步原理、功能状态及验证记录。HTML 结构、导航与本地链接检查，宽/窄屏布局检查情况如实记录。

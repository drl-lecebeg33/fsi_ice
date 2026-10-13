# M6：IQN-ILS 公式修正与 FSI2 瞬态接入

日期：2026-10-12。基线 `1614dc9`，本轮未提交、未推送。单一执行者承担实现及自查，不声称独立复核。[运行前计划与门槛](../development/m6_iqn_transient_plan.md)。

## 结论与验证边界

IQN-ILS 已按残差差/输出差公式修正并接入 FSI2 配置、CLI 与共享瞬态驱动。小型真实 FE 四方法三步（BDF1/BDF2）、失败回滚与重试通过；真实 FSI2 h=0.03、dt=0.01 的 fixed/Aitken/IQN/整体三步末态全部一致。每步仍使用原全耦合残差、连续性、正 Jacobian 与网格质量门控。

真实三步周期状态均为 **INCOMPLETE**，不是周期基准 PASS。完整稳定周期、三网格/三步长研究、四算法公平成本表及 T27 尚未完成；M6 不勾选完成。

## 为什么必须修正旧公式

旧 T25 以 [s+ω₀y] 拟合 -r，再用该矩阵更新输入。若拟合恰好满足右端，则更新退成 x+r，即普通固定点，不能证明实现了逆残差 Jacobian 割线。旧小型 FE 的11次评估只证明最终解一致。

现在定义 r=G(x)-x、s=Δx、y=Δr，构造 V=[y]、W=[s+y]=[ΔG]，拟合 (V/S)α=-r/S，再更新 x_new=G(x)+Wα；S 为调用方显式物理尺度。ω₀只用于初始与重启松弛。公式核对来源为 preCICE 官方 [IQNILSAcceleration.cpp](https://github.com/precice/precice/blob/develop/src/acceleration/IQNILSAcceleration.cpp) 与 [BaseQNAcceleration.cpp](https://github.com/precice/precice/blob/develop/src/acceleration/BaseQNAcceleration.cpp)（核对日期2026-10-12，仅核对算法，不调用其求解器）。

先新增反例并在旧代码上实际运行：`pytest tests/test_iqn.py -q -k 'noncontractive or multisecant'`，**4 failed**。修正后 G(x)=1-2x 在 ω₀=0.25/0.5/1 三种设置下均三次评估到 x=1/3；二维非收缩映射在四次内到解析解，单位缩放1e-3/1e3及相应S变化保持轨迹一致。更新秩亏测试，使它必须实际触发重启；α上限测试改用真实需要超限系数的映射，避免旧宽松断言掩盖算法错误。

最小二乘残差是历史空间未覆盖全部残差方向时的正常投影余量。旧默认0.01会过早重启、阻止历史积累（真实小型 FE 修正后此设置退为27次固定松弛）。因此 `cases/iqn_check.json` 与 `cases/fsi2.json` 显式采用1.0的相对投影余量上限，允许正常投影，同时保留数值秩、α≤10和非有限值守卫。此为加速器策略调整，**所有物理验收门槛未改变**；dataclass 默认0.01保持兼容。相对余量为 ||V_scaled α+r_scaled||₂/||r_scaled||₂。

## 接口、配置与事务

- `TransientFSI.step('iqn')` 与其他分区方法共用物理求解顺序，成功后统一 evaluate→validate→prepare→publish。
- FSI2 的 `--method` 为 fixed/aitken/iqn/monolithic，配置接受四个可选 IQN 控制项；未知字段、NaN/Inf、非法历史容量被拒绝；不带新字段的旧配置仍有效。
- IQN 割线只存在于当前 `solve_dirichlet_neumann` 调用中，跨步不复用，失败重试冷启动。求解第三次流体评估中注入失败，已接受根及历史保持不变，撤销后重试与无失败基线逐数组一致。
- 分区监测新增 `iqn_updates` / `iqn_restarts`；收敛依旧基于未松弛残差，返回最后已评估输入与对应单场状态。
- CLI短程退出码仍为3；参数/求解失败为2；稳定周期门槛失败为1，完整通过才为0。没有新增隐式续算功能。

## 实际环境与命令

项目 `.venv`、Python3.12.3、NumPy2.5.3、SciPy1.18.1、Matplotlib3.11.2、pytest9.1.1；Gmsh4.12；BLAS/OMP单线程。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_iqn.py tests/test_fsi2.py tests/test_aitken.py tests/test_partitioned.py \
  tests/test_partitioned_integration.py tests/test_coupling_review.py tests/test_cli.py -q
# 211 passed / 23.47 s / exit 0，日志 results/m6/iqn_transient/logs/related.log

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m fsi_ice partitioned-check \
  --config cases/iqn_check.json --output results/m6/iqn_transient/linear_iqn
# PASS / exit 0；6次评估，组合残差4.362e-14，逐变量归一化差≤4.311e-14

# 依次把 method 替换为 iqn、aitken、monolithic、fixed，使用各自独立输出目录
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m fsi_ice fsi2 \
  --config cases/fsi2.json --output results/m6/iqn_transient/iqn \
  --mesh-size 0.03 --end-time 0.03 --method iqn
# 四个命令均 INCOMPLETE / exit 3，完成3步，无数值失败

# 将左目录分别替换为 iqn、aitken、fixed
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/t24_compare.py \
  results/m6/iqn_transient/iqn results/m6/iqn_transient/monolithic
# 三组均 passed=true / exit 0；不改变周期结论
```

代码与配置入口：[partitioned.py](../../src/fsi_ice/coupling/partitioned.py)、[transient.py](../../src/fsi_ice/coupling/transient.py)、[fsi2.py](../../src/fsi_ice/validation/fsi2.py)、[cli.py](../../src/fsi_ice/cli.py)、[fsi2.json](../../cases/fsi2.json)。测试：[test_iqn.py](../../tests/test_iqn.py)、[test_fsi2.py](../../tests/test_fsi2.py)。

## 真实三步数据

| 方法 | 每步耦合评估 / Newton校正 | 最大全残差缩放值 | 相对整体法最不利一致性裕度比 |
| --- | --- | ---: | ---: |
| fixed | 41 / 46 / 53（耦合评估） | 1.521e-10 | 0.006413 |
| Aitken | 15 / 15 / 17（耦合评估） | 1.122e-9 | 0.007213 |
| IQN-ILS | 9 / 9 / 9（耦合评估） | 8.466e-10 | 0.026369 |
| monolithic | 2 / 2 / 2（整体Newton校正） | 8.262e-12 | 基线 |

IQN每步有7次真实拟牛顿更新、0次重启；全部结果弱连续性≤1e-10、最小质量≥0.1、正固体Jacobian。三步时程与所有场保存在 `results/m6/iqn_transient/{fixed,aitken,iqn,monolithic}/`；三个比较文件为 `*_vs_monolithic.json`。初次 fixed 比较在运行未结束时尝试，因缺少report报错；等待运行完成后已重新执行并通过。

计数区别：耦合评估包含流体/固体内层求解，不能与整体Newton校正数等价比较。短程solve_seconds仅为诊断；部分测试与fixed运行重叠，没有重复计时，不作为T26公平成本结论。

## 启动区间探索、全量与文档检查

实际运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m fsi_ice fsi2 \
  --config cases/fsi2.json --output results/m6/iqn_transient/probe_h002 \
  --end-time 0.5 --method iqn
# h=0.02、dt=0.01，50步全部接受；INCOMPLETE / exit 3

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests -q
# 621 passed、2 skipped / 241.05 s / exit 0
```

启动探索：[摘要](../../results/m6/iqn_transient/probe_summary.json)、[报告](../../results/m6/iqn_transient/probe_h002/report.json)、[监测](../../results/m6/iqn_transient/probe_h002/monitor.csv)、日志 `results/m6/iqn_transient/logs/probe_h002.log`。

| 检查 | 实测 | 冻结门槛 |
| --- | ---: | ---: |
| 最大缩放全残差 | 2.531e-9 | ≤1e-7 |
| 最大弱连续性 | 2.061e-18 | ≤1e-10 |
| 最小网格质量 | 0.728457 | ≥0.1 |
| 最小面积比 | 0.999236 | >0 |
| 最小固体 Jacobian | 0.999545 | >0 |
| IQN 每步评估 | 9–11 | 每步≤100 |
| 真实拟牛顿更新 / 重启 | 407 / 0 | 记录 |

梁尖 y 的最大绝对位移4.436e-5 m，仍处于0–2 s入口启动区间，不据此判稳定振幅。累计 solve_seconds=387.976 s，与全量回归部分重叠，仅为诊断，不能据此外推稳定周期或用于公平性能排名。

全量日志：[full.log](../../results/m6/iqn_transient/logs/full.log)。两个既有可选跳过：`FSI_ICE_T22_SMOKE` 未启用的真实FSI1整体推进与 `FSI_ICE_T23_RESULTS` 未设置的历史产物验证。本轮真实FSI2四方法三步及50步启动实验已单独执行。新增21项回归（原600→621），覆盖正确割线、单位变化、IQN事务故障、配置与CLI。环境检查PASS、`pip check`无依赖冲突、`git diff --check`通过。

原理页HTML结构、12个id唯一、导航锚点、148条HTML本地链接及136条主要Markdown本地链接检查通过；主要内容修改后通过Orca浏览器实看1280×900与390×844截图，仅代码块局部滚动。最后仅修改证据链接和测试数字后再次测得宽屏clientWidth=scrollWidth=1265、窄屏均375，无整页横向溢出；该次截图复拍因浏览器未绘制帧而超时，未声称完成最终版本截图复核。先前已实看的截图保留。证据在 `results/m6/iqn_transient/logs/` 的 `html_check.json`、`layout_{wide,narrow}.json`、`principles_{wide,narrow}.png`。

本轮交付为公式修正、瞬态接入及启动正确性，不替代完整20 s周期验收。后续先运行统一配置的完整周期，检查稳定性与参考误差，再分别固定dt/网格开展三网格/三步长研究和公平重复计时；长时运行仍无持久化BDF历史续算能力。M6保持进行中；本轮未提交、未推送。

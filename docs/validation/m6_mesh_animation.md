# 试水收尾：已保存动网格的真实快照动画

日期：2026-10-12。用户明确本轮仅为试水，停止后续数值推进，并请求用现有数据制作动画。本任务只读加载数据与渲染，不运行流体、固体或动网格方程，不恢复求解，不改包内源码、物理参数、门槛或实验数据；不提交、不推送。

## 数据覆盖与结果

现有 `iqn_h002_r2/checkpoint.npz` 只保留最近两层BDF2完整历史：823步（8.23 s）、824步（8.24 s）。更早时刻的monitor.csv是监测量，不包含逐步全场网格坐标；其他目录有独立运行/诊断快照，不能混成同一连续时程。因此本次动画只显示这两帧，原间隔0.01 s，每帧播放1 s，约100倍慢放；循环跳回早一帧不表示真实往返振动。

| 已接受步 | 时刻 (s) | 最小面积比 | 最小质量 |
| --- | ---: | ---: | ---: |
| 823 | 8.23 | 0.3741612600 | 0.5448013826 |
| 824 | 8.24 | 0.2605645167 | 0.3884154351 |

脚本使用保存的 `mesh_history` 与 `solid_history` 加参考坐标恢复当前流体网格/固体位置。没有插值、位移放大、拟合梁形或补造中间时刻。先校验NPZ数组/历史/账本、参考网格SHA256与当前坐标，再重算两帧面积比/质量并核对已接受监测行；灰线只作初始参考网格背景。颜色表示当前/参考单元面积比，无单位：0.2为原压缩门槛，≥1.2使用同一最高绿色，图例带上界标记。黄色区域是实际保存的固体梁，单元1651的位置以红点标出。

导出全域、梁周围与梁尖局部三个视图。动画没有第825步的失败候选；运行状态仍是FAIL、周期INCOMPLETE，M6未通过。试水到此收尾，完整周期/收敛/成本实验保持未执行。

- [离线播放页](../../results/m6/mesh_animation/index.html)：单个HTML内嵌GIF和末帧PNG，可显示末帧/重播和查看原始分辨率，离线无需外部资源。
- [GIF动画](../../results/m6/mesh_animation/mesh_motion.gif)：2个实际状态，各1000 ms，循环播放。
- [MP4视频](../../results/m6/mesh_animation/mesh_motion.mp4)：2 s、1680×960、24 fps、48个编码帧；编码帧是两张实际状态图的重复停留，不是48个求解时刻。
- [末帧静态图](../../results/m6/mesh_animation/frame_01.png)、[数据与渲染摘要](../../results/m6/mesh_animation/manifest.json)、[输入文件完整性](../../results/m6/mesh_animation/input_integrity.json)。

## 可复现入口与验证

源码：[fsi2_mesh_animation.py](../../scripts/fsi2_mesh_animation.py)；测试：[test_mesh_animation.py](../../tests/test_mesh_animation.py)。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/fsi2_mesh_animation.py \
  results/m6/long_run/iqn_h002_r2 --output results/m6/mesh_animation
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_mesh_animation.py -q
```

本地 `.venv` Python3.12；NumPy/Matplotlib/Pillow使用既有依赖，MP4使用已安装ffmpeg/libx264，未安装新包。8项测试通过（首次0.92 s，最终0.85 s，日志 `results/m6/mesh_animation/tests.log`），覆盖实际保存顺序、坐标逐值还原、面积比一致性、数组/网格/监测篡改拒绝、历史不足时拒绝伪造动画、失败记录匹配、输出路径保护、GIF帧数/时长、离线内嵌与输入字节不变。单元测试的解析刚体平移仅验证读取/导出，不是FSI2计算结果；真实渲染使用上述823/824步。

真实导出前后求解目录全部文件SHA256完全相同。ffprobe确认MP4尺寸、帧率、帧数与2秒时长；GIF编码检查确认两帧各1000 ms。PNG已实看：中文、真实时刻、位移倍率、图例、参考网格与压缩单元标注可读。仅渲染与相关测试，没有重跑数值求解或全量回归。

离线页面与原理文档HTML结构、15个原理页唯一id/导航和172条原理页本地链接、183条主要Markdown本地链接检查通过。Orca内置浏览器对动画播放页完成1280×1000与390×844实看，文本、数据范围与按钮可读；宽屏clientWidth=scrollWidth=1265，窄屏适应窗口时均390，原始分辨率模式下均375，超宽图像仅在查看器内部滚动（viewerWidth328、scrollWidth1680），没有整页横向溢出。显示末帧/重播按钮和原始分辨率切换实际操作通过，PNG/GIF源切换与按钮文本一致。检查日志/截图在 `results/m6/mesh_animation/qa/`。原理页本次视觉复查未完成：新建的检查页随后返回browser_tab_not_found，实际tab列表只有动画页；没有重复打开页面，保留错误日志并如实记录。此前原理页布局检查见[后处理记录](m6_postprocess.md)。`git diff --check`通过。

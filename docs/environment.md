# Python 本地运行环境

## 验证记录

- 日期：2026-10-10（Asia/Shanghai）。
- 任务：T00，Python 环境核验。
- 平台：本机 Linux，Python 3.12.3。
- 环境：项目内 `.venv`，与系统软件包隔离。
- 代码状态：基于 `dd6232b`，环境脚本及规划调整尚未提交。
- 依赖：NumPy 2.5.3、SciPy 1.18.1、Matplotlib 3.11.2、pytest 9.1.1；pip 26.2.1。
- 完整依赖版本见 [requirements.txt](../requirements.txt)；直接依赖见 [requirements.in](../requirements.in)。

实际运行：

```bash
.venv/bin/python scripts/check_environment.py
.venv/bin/python -m pip check
```

检查结果：

| 检查 | 实测结果 |
| --- | --- |
| 独立解释器和依赖导入 | 通过 |
| NumPy 稠密线性求解 | 通过，残差范数 0 |
| SciPy 稀疏直接求解 | 通过，残差范数 0 |
| SciPy GMRES | 通过，残差范数约 8.88e-16 |
| 无界面绘图 | 通过，生成有效 PNG；临时文件自动清理 |
| pytest 启动 | 通过，版本入口正常 |
| 依赖一致性 | 通过，pip check 未发现冲突 |

以上为 T00 环境冒烟检查，不构成物理求解器验收。后续 T01–T04 已建立项目包、统一测试入口和 Poisson 数值验证，证据另见 [M0 验收记录](validation/m0.md)。

## 重建环境

从项目根目录运行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/check_environment.py
.venv/bin/python -m pip check
```

本机系统 Python 缺少 `ensurepip`。此次使用系统已提供的 pip 引导项目环境，没有修改系统软件包。相同条件下，可使用以下步骤替代第一组环境创建与安装命令：

```bash
python3 -m venv --without-pip .venv
python3 -m pip --python .venv/bin/python install pip==26.2.1
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/check_environment.py
.venv/bin/python -m pip check
```

第二种方式要求系统已有支持 `--python` 的 pip；本机系统 pip 为 24.0。环境目录不纳入 Git，移动项目后应重建环境。依赖锁定记录本次实际安装版本，其他平台仍需重新验证。

依据：[Python venv 文档](https://docs.python.org/3.12/library/venv.html)、[pip 管理其他解释器的官方说明](https://pip.pypa.io/en/stable/topics/python-option/)。

## 日常运行

所有项目验证均显式使用 `.venv/bin/python`，避免误用系统解释器。也可以先激活 `.venv`，但验证记录应保留解释器信息。

当前可运行的入口包括 [环境检查脚本](../scripts/check_environment.py)、`python -m fsi_ice test` 及 `python -m fsi_ice poisson`；完整使用方式见 [README](../README.md)。包和测试入口已通过本地验证，流体、固体与完整 FSI 仍待开发。

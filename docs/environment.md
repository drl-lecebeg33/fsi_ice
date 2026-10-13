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

当前可运行的入口包括 [环境检查脚本](../scripts/check_environment.py)、`python -m fsi_ice test`、`poisson`、`single-fields`/`solid`/`fluid`、`csm3`/`cfd3`、`partitioned-check`、`fsi1` 与 `fsi2`；完整使用方式见 [README](../README.md)。单场、ALE/界面与 FSI 基准的验证证据见 [验证记录目录](validation/)。

## Windows 工作区复验（2026-10-13，辅验环境）

本节记录在 Windows 工作区重建同一代码环境并复跑回归的结果。它提供一个可用于开发与回归的环境，不改变求解器、模型、容差或验收门槛；正式基准证据仍以各里程碑验证记录为准，上文 T00 的 Linux 记录保持原样。

- 平台：Windows（win32）；解释器 Python 3.12.14，来自 uv 管理的 Astral CPython（`...\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe`）。
- 环境：项目内 `.venv`，入口为 `.venv\Scripts\python.exe`；pip 26.2.1。
- 依赖：与 [requirements.txt](../requirements.txt) 一致（NumPy 2.5.3、SciPy 1.18.1、Matplotlib 3.11.2、pytest 9.1.1）。Windows 另需 `tzdata==2026.5`，已按平台标记写入依赖文件；Gmsh CLI 仍为外部工具（见下）。
- 代码状态：重建时 `63ea9af`；随后提交测试可移植性修复 `22d4fb7`，工作区无未提交改动。

实际运行（项目根目录，Windows PowerShell；依赖下载使用清华与阿里 PyPI 镜像，本机默认 PyPI 与 gmsh.info 不可达）：

```powershell
<uv 解释器> -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -i <镜像>
.venv\Scripts\python.exe -m pip install --no-deps -e . -i <镜像>
.venv\Scripts\python.exe scripts/check_environment.py
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m pytest tests -q
```

结果：

| 检查 | 实测结果 |
| --- | --- |
| `check_environment.py` | PASS（Python 3.12.14；稠密残差 0，稀疏直接求解 0，GMRES 8.88e-16；无界面绘图 PASS；pytest 9.1.1） |
| `pip check` | 未发现冲突 |
| 全量回归 | **706 passed, 3 skipped / 176.34 s**（2026-10-13；`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`） |

三个跳过均为设计内的可选门控：T23 历史产物检查（需设置 `FSI_ICE_T23_RESULTS`）、T22 FSI1 整体冒烟（需设置 `FSI_ICE_T22_SMOKE=1`）、T21 本地证据抽样（`results/m5/t21/` 不入库，缺文件时跳过，见 `22d4fb7`）。

### Windows 上的两处环境差异

- **时区数据**：缺少 `tzdata` 时，读取 `ZoneInfo("Asia/Shanghai")` 时间戳的测试全部以 `ZoneInfoNotFoundError` 失败；安装 `tzdata 2026.5` 后该类失败消失。这是 Windows 没有系统时区库所致，与求解器无关，因此将该包按 `sys_platform == "win32"` 记入依赖。
- **Gmsh CLI（外部工具，不入 Python 依赖）**：本机无法访问 gmsh.info，改用 PyPI 的 `gmsh==4.12.2`（与已验证的 4.12 系列一致，仅作三角剖分生成器，未在 Python 中导入其 API）：

  ```powershell
  .venv\Scripts\python.exe -m pip install gmsh==4.12.2 -i <镜像>
  # 运行测试前让脚本目录进入 PATH（或先激活 .venv），shutil.which("gmsh") 才会命中：
  $env:PATH = "$PWD\.venv\Scripts;$env:PATH"
  gmsh --version   # 4.12.2
  ```

  该轮子把 `gmsh.bat` 与同名 Python 脚本装入 `.venv\Scripts`；原始 `gmsh.bat` 调用 PATH 上的 `python`。本工作区将它改为显式调用同目录解释器（`.venv` 不入库，仅为本地环境步骤）：

  ```bat
  @echo off
  "%~dp0python.exe" "%~dp0gmsh" %*
  ```

`pytest` 在 Windows 上还会自动安装平台标记依赖 `colorama`，无需在依赖文件中单列；Windows 实测耗时不能与 Linux 记录的耗时直接比较。本次复验不新增数值基准，也不改变 `spec.md` 中"正式验证环境"的表述；是否将 Windows 列为正式验证环境需另行决定。

"""Reproducible validation artifacts and convergence utilities.

中文说明：本模块提供各验证 driver 共用的可复现产物工具：CSV/JSON 报告写出（含
Asia/Shanghai 时间戳、PASS/FAIL 状态、代码版本与环境信息）、收敛阶估计，以及单场
验证配置的 schema 校验。
"""

import csv
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from zoneinfo import ZoneInfo

import numpy as np

from fsi_ice.validation.poisson import code_metadata


def write_csv(path, rows):
    """写出 CSV：字段名取第一行的键顺序，自动创建父目录，UTF-8 编码。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(output, case, config, results, checks, definitions):
    """写出 report.json 并返回报告字典。

    checks 为 {检查项: 布尔}，全部为真时 status 记为 "PASS"，否则记 "FAIL"；
    时间戳取 Asia/Shanghai 时区；同时记录代码版本（commit/工作树状态）与运行环境
    （Python 版本、解释器路径、依赖包版本）。
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"case": case, "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "config": config, "definitions": definitions, "results": results,
              "code": code_metadata(), "environment": {"python": platform.python_version(),
                  "executable": sys.executable, "packages": {name: importlib.metadata.version(name)
                      for name in ("fsi-ice", "numpy", "scipy", "matplotlib")}}}
    # ensure_ascii=False 保留中文可读性；allow_nan=False 阻止 NaN/Inf 静默写入报告。
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def add_orders(rows, errors, spacing):
    """为每个误差列追加收敛阶 *_order（原地修改并返回 rows）。

    errors 为误差列名，spacing 为网格尺寸或时间步列名。第 i 行阶数按
    log(e_{i-1}/e_i) / log(s_{i-1}/s_i) 计算（误差与间隔同时缩小）；首行无前一行，
    阶数记为 None。
    """
    for index, row in enumerate(rows):
        for error in errors:
            # 相邻两档的对数误差比除以对数间距比，得到经验收敛阶。
            row[error+"_order"] = None if index == 0 else float(
                np.log(rows[index-1][error]/row[error])/np.log(rows[index-1][spacing]/row[spacing]))
    return rows


def load_single_config(path):
    """读取并校验单场验证配置，返回原始字典。

    要求字段恰为 case/beam_meshes/csm_meshes/fluid_meshes/manufactured_meshes/
    time_steps/acceptance，且 case == "single_fields"；网格须严格加密（至少三档），
    流体网格与时间步须严格递减、为正且能整除终止时刻 1；接受阈值须为有限正数。
    """
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"case", "beam_meshes", "csm_meshes", "fluid_meshes", "manufactured_meshes", "time_steps", "acceptance"}
    if not isinstance(config, dict) or set(config) != required or config["case"] != "single_fields":
        raise ValueError("Invalid single-field config schema.")
    for field in ("beam_meshes", "csm_meshes"):
        levels = config[field]
        if not isinstance(levels, list) or len(levels) < 3 or any(
                not isinstance(pair, list) or len(pair) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in pair)
                for pair in levels):
            raise ValueError(f"{field} requires at least three [nx,ny] integer meshes.")
        if any(any(b <= a for a, b in zip(first, second)) for first, second in zip(levels, levels[1:])):
            raise ValueError(f"{field} must strictly refine both directions.")
    levels = config["manufactured_meshes"]
    if not isinstance(levels, list) or len(levels) < 3 or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in levels) or any(b <= a for a, b in zip(levels, levels[1:])):
        raise ValueError("manufactured_meshes must contain at least three increasing integers.")
    for field in ("fluid_meshes", "time_steps"):
        values = config[field]
        if not isinstance(values, list) or len(values) < 3 or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not np.isfinite(x) or x <= 0 for x in values) or any(b >= a for a, b in zip(values, values[1:])):
            raise ValueError(f"{field} requires at least three decreasing positive values.")
    if any(abs(round(1/dt)*dt-1) > 1e-12 for dt in config["time_steps"]):
        raise ValueError("Each temporal step must divide the final time 1 exactly.")
    thresholds = {"beam_relative_error_max", "csm_relative_error_max", "cfd_relative_error_max", "velocity_order_min",
                  "pressure_order_min", "bdf1_order_min", "bdf2_order_min", "residual_max", "mass_error_max"}
    if not isinstance(config["acceptance"], dict) or set(config["acceptance"]) != thresholds or any(
            isinstance(x, bool) or not isinstance(x, (float, int)) or not np.isfinite(x) or x <= 0 for x in config["acceptance"].values()):
        raise ValueError("Invalid acceptance thresholds.")
    return config

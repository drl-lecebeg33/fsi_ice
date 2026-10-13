"""Monolithic steady FSI1 runner and T23 consistency/cost helpers (M5/T23, A7).

This module owns two things requested by the T23 task card:

* the *monolithic steady runner*: it reuses the frozen T21/T22 interfaces to
  build a ``MonolithicContext`` with ``pressure_pin=None`` and an explicit
  initial state (fluid at rest with the inlet Dirichlet data, undeformed solid,
  reference mesh, zero interface load), then drives
  :func:`~fsi_ice.coupling.monolithic_newton.march_to_steady` to the BDF fixed
  point (M-11).  Steady state is obtained when the scaled BDF time term drops
  below the declared tolerance, so the accepted state solves the steady
  discrete equation (``sum_j a_j = 0`` cancels the time term exactly).  No
  partitioned solution is ever used as an initial value; warm starts between
  tolerance levels are monolithic-to-monolithic and recorded as such.

* the *T23 comparison helpers*: the frozen per-variable criterion of
  ``docs/validation/m5.md`` (``max|a_M - a_P| <= rtol*max(||a_M||inf, ||a_P||inf)
  + atol`` with ``rtol=1e-8`` and ``atol=1e-10*S*``), the frozen FSI1 scales
  ``L*=0.41 m, U*=0.2 m/s, P*=40 Pa, F*=14.29426 N/m`` and the cost-table
  aggregation of ``docs/development/m5_t23_plan.md`` section 4.  The comparison
  helpers are pure functions so ``tests/test_m5_consistency.py`` can exercise
  them quickly without running the h=0.02 solves.

The comparison tolerance columns and the scale values are the values frozen by
the T23 task dispatch (A0); they are not spec gates and must never be relaxed
to make a comparison pass.  Residual definitions differ between methods (M-1)
and are recorded separately, never compared numerically.

中文说明：本模块负责 M5/T23 的两件事。其一，在**同一套离散**（相同网格、拓扑与
ALE 几何）上驱动整体（monolithic）稳态 FSI1 求解：以显式初值（流体静止但带入口
Dirichlet 数据、固体未变形、参考网格、零界面载荷、``pressure_pin=None``）调用
``march_to_steady`` 逼近 BDF 不动点，当标度后的 BDF 时间项低于给定容差时即认为达到
稳态；整体与分区共享物理离散与残差实现，因此可直接做逐变量一致性对比与成本统计。
其二，提供 T23 的一致性/成本“纯函数”辅助：按冻结的逐变量判据
（``max|a_M-a_P| <= rtol*max(||a_M||inf,||a_P||inf) + atol``，``rtol=1e-8``、
``atol=1e-10*S*``）与冻结的 FSI1 尺度（``L*=0.41 m, U*=0.2 m/s, P*=40 Pa,
F*=14.29426 N/m``）生成对比表与成本聚合。关键约定：上述容差列与尺度是任务派发冻结
值（A0），不是规范门槛，绝不允许为让对比通过而放宽；不同方法的残差定义不同（M-1），
只分别记录、绝不跨方法数值比较。
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fsi_ice.ale import build_ale_geometry
from fsi_ice.coupling import (BLOCK_NAMES, MonolithicBase, MonolithicContext, build_block_layout,
                              evaluate_monolithic, interface_compatibility_gap)
from fsi_ice.coupling.monolithic_newton import (MonolithicCounters, MonolithicNewtonFailure,
                                                march_to_steady)
from fsi_ice.coupling.newton import NewtonOptions
from fsi_ice.fluid.forces import boundary_force
from fsi_ice.fluid.taylor_hood import assemble_stokes
from fsi_ice.time import BDFHistory
from fsi_ice.validation.common import write_csv
from fsi_ice.validation.fluid import inlet_profile
from fsi_ice.validation.monolithic import block_scale_vector

# ---------------------------------------------------------------------------
# frozen T23 values (A0 dispatch; docs/validation/m5.md)
# ---------------------------------------------------------------------------

# 以下常量均为 T23 任务派发（A0）冻结值：用于对比判据、物理尺度与收敛阶梯。
# 它们不可随求解结果动态调整，也不得为“凑通过”而放宽。

#: Comparison tolerance of the per-variable criterion (frozen proposed value).
# 逐变量一致性判据的相对容差 rtol（冻结值）。
T23_RTOL = 1e-8

#: Absolute floor as a fraction of the per-variable physical scale S*.
# 绝对下限 = 该比例 × 对应变量物理尺度 S*，用于覆盖接近零的量。
T23_ATOL_SCALE = 1e-10

#: Frozen physical scales of the FSI1 consistency study.
# FSI1 一致性研究的冻结物理尺度：长度[m]、速度[m/s]、压力[Pa]、线载荷[N/m]。
T23_SCALES = {"L_star": 0.41, "U_star": 0.2, "P_star": 40.0, "F_star": 14.29426}

#: Per-block scales of the monolithic Newton criterion (same frozen values).
# 整体 Newton 逐块收敛判据使用的块尺度（复用上面同一组冻结值）。
T23_BLOCK_SCALES = {"u_f": T23_SCALES["U_star"], "p_f": T23_SCALES["P_star"],
                    "d_s": T23_SCALES["L_star"], "v_s": T23_SCALES["U_star"],
                    "d_m": T23_SCALES["L_star"], "g": T23_SCALES["F_star"]}

#: Tolerances of the (proposed, dispatch-frozen) tightening ladder.
# 收敛容差的收紧阶梯（由宽到严）：分别对应三个“level”。
T23_TOLERANCE_LADDER = (1e-6, 1e-8, 1e-10)

#: The FSI1 fixed spec gates (cases/fsi1.json); these are never relaxed.
# FSI1 规范固定门槛（来自 cases/fsi1.json）：相对/绝对误差上限，绝不放宽。
FSI1_GATES = {"tip_y_relative_error_max": 0.05, "drag_relative_error_max": 0.05,
              "tip_x_absolute_error_max": 1e-5, "lift_absolute_error_max": 0.1}

#: Method error-gap operationalisation of 23-08 / M-7 (relative vs absolute).
# 23-08/M-7 的“方法间误差差距”操作化门槛：相对量用统一上限，绝对量分变量设限。
METHOD_GAP_RELATIVE_MAX = 0.005
METHOD_GAP_ABSOLUTE_MAX = {"tip_x": 0.005, "lift": 0.005}


def now_iso():
    """当前本地时区的 ISO 8601 时间戳（带时区偏移），用于产物记录。"""
    return datetime.now().astimezone().isoformat()


def format_level(tau):
    """Stable directory label of a tolerance level, e.g. ``1e-06``.

    把容差值格式化为稳定的目录标签（科学计数、两位指数）；用于按 level 建目录。
    """
    return f"{float(tau):.0e}"


# ---------------------------------------------------------------------------
# per-variable consistency comparison (frozen criterion)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonRow:
    """One row of the per-variable consistency table.

    逐变量一致性表的一行（不可变）。字段含义：``size`` 该变量自由度个数；
    ``magnitude`` 两侧无穷范数的较大者；``max_abs_delta`` 两侧最大绝对差；
    ``limit = rtol*magnitude + atol`` 为判据上限；``passed`` 是否通过；
    ``limit_ratio`` = ``max_abs_delta/limit``，>1 即超限。
    """

    name: str
    unit: str
    scale: float
    rtol: float
    atol: float
    size: int
    magnitude: float
    max_abs_delta: float
    limit: float
    passed: bool
    limit_ratio: float
    description: str = ""

    def as_dict(self):
        return {"name": self.name, "unit": self.unit, "scale": self.scale, "rtol": self.rtol,
                "atol": self.atol, "size": self.size, "magnitude": self.magnitude,
                "max_abs_delta": self.max_abs_delta, "limit": self.limit,
                "passed": bool(self.passed), "limit_ratio": self.limit_ratio,
                "description": self.description}


#: ``(name, unit, scale, description)`` of every compared variable of the T23 table.
# T23 对比表中每个被比较变量的 (名称, 单位, 物理尺度, 说明)；尺度用于绝对下限 atol。
CONSISTENCY_SPECS = (
    ("d_s", "m", T23_SCALES["L_star"], "solid P1 displacement, full field"),
    ("tip_A", "m", T23_SCALES["L_star"], "tip A displacement (2 components)"),
    ("u_f", "m/s", T23_SCALES["U_star"], "fluid P2 velocity, same nodes"),
    ("p_f", "Pa", T23_SCALES["P_star"], "fluid P1 pressure (same pin=None convention)"),
    ("d_m", "m", T23_SCALES["L_star"], "fluid mesh displacement (current minus reference)"),
    ("v_s", "m/s", T23_SCALES["U_star"], "solid P1 velocity (partitioned steady value is zero)"),
    ("interface_x", "m", T23_SCALES["L_star"], "interface displacement trace x = d_s|Gamma"),
    ("interface_H", "m", T23_SCALES["L_star"], "H d_s|Gamma mapped to the fluid P2 trace"),
    ("g", "N/m", T23_SCALES["F_star"], "fluid P2 trace dual nodal load g"),
    ("lambda", "N/m", T23_SCALES["F_star"], "H^T g on the compact solid trace"),
)


def compare_values(name, left, right, *, unit, scale, rtol=T23_RTOL, atol_scale=T23_ATOL_SCALE,
                   description=""):
    """Frozen per-variable criterion of one variable.

    ``max|a_M - a_P| <= rtol*max(||a_M||inf, ||a_P||inf) + atol`` with
    ``atol = atol_scale*scale``; near-zero quantities are covered by ``atol``
    and never by an unreliable relative error.  Shapes must match exactly.

    中文：对单个变量做冻结判据比较。输入左右两侧数组（形状必须完全一致、有限、
    非空）与单位/尺度；返回 :class:`ComparisonRow`。``atol=atol_scale*scale``
    保证近零量由绝对下限兜底，不使用不可靠的相对误差。shape 或数值非法时抛
    ``ValueError``。
    """
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"{name}: shapes differ, {a.shape} vs {b.shape}.")
    if a.size == 0:
        raise ValueError(f"{name}: empty vectors cannot be compared.")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError(f"{name}: values must be finite.")
    if not (np.isfinite(scale) and scale > 0):
        raise ValueError(f"{name}: scale must be a finite positive number.")
    if rtol < 0 or atol_scale < 0:
        raise ValueError(f"{name}: tolerances must be nonnegative.")
    delta = np.abs(a-b)
    max_abs_delta = float(np.max(delta))  # 最大绝对差 max|a_M - a_P|
    magnitude = float(max(np.max(np.abs(a)), np.max(np.abs(b))))  # 两侧无穷范数较大者
    atol = float(atol_scale)*float(scale)  # 绝对下限 = 比例 × 物理尺度
    limit = float(rtol)*magnitude+atol  # 判据上限
    return ComparisonRow(name=name, unit=unit, scale=float(scale), rtol=float(rtol), atol=atol,
                         size=int(a.size), magnitude=magnitude, max_abs_delta=max_abs_delta,
                         limit=limit, passed=bool(max_abs_delta <= limit),
                         limit_ratio=max_abs_delta/limit, description=description)


def consistency_rows(bundle_a, bundle_b, *, rtol=T23_RTOL, atol_scale=T23_ATOL_SCALE,
                     label_a="a", label_b="b"):
    """All frozen variable rows of one method pair.

    ``bundle_a``/``bundle_b`` are mappings with a value per :data:`CONSISTENCY_SPECS`
    name (any array-like with matching shape).

    中文：对一对方法（整体 vs 分区）的完整变量集合逐项比较，返回
    :class:`ComparisonRow` 列表；``description`` 末尾追加 ``[a vs b]`` 标注配对标签。
    两个 bundle 都必须含 :data:`CONSISTENCY_SPECS` 的每个变量名，否则抛错。
    """
    rows = []
    for name, unit, scale, description in CONSISTENCY_SPECS:
        if name not in bundle_a or name not in bundle_b:
            raise ValueError(f"Both bundles must provide variable {name!r}.")
        row = compare_values(name, bundle_a[name], bundle_b[name], unit=unit, scale=scale,
                             rtol=rtol, atol_scale=atol_scale, description=description)
        row = ComparisonRow(**{**row.__dict__, "description": f"{description} [{label_a} vs {label_b}]"})
        rows.append(row)
    return rows


def pressure_gauge_diagnostic(bundle_a, bundle_b):
    """Mean/zero-mode shift of two pressure fields (diagnostic only, M-3).

    中文：诊断两个压力场之间的整体零模（常数）偏移，仅用于记录、不参与判定。
    返回：均值差、去均值后的最大残差、以及原始最大差。用于解释压力可差一个常数。
    """
    a = np.asarray(bundle_a["p_f"], dtype=float)
    b = np.asarray(bundle_b["p_f"], dtype=float)
    return {"mean_difference": float(np.mean(a)-np.mean(b)),
            "max_after_mean_shift": float(np.max(np.abs((a-np.mean(a))-(b-np.mean(b))))),
            "max_raw": float(np.max(np.abs(a-b)))}


def physical_error_gaps(method_metrics, *, gates=None, relative_max=METHOD_GAP_RELATIVE_MAX,
                        absolute_max=None):
    """23-08/M-7 operationalisation of the method error gaps.

    ``method_metrics`` maps a method label to the ``metrics`` mapping produced
    by the partitioned ``SteadyFSI1Problem.metrics`` or by
    :func:`monolithic_metrics` (keys ``tip_x_absolute_error``,
    ``tip_y_relative_error``, ``drag_relative_error``, ``lift_absolute_error``).
    Returns ``(gates_check, gaps)``: per-method fixed-gate pass table and every
    method pair's relative/absolute error-gap check.

    中文：把各方法的物理误差指标与固定规范门槛、以及方法两两间的误差差距做检查。
    ``method_metrics`` 为 {方法标签: 指标字典}；返回 (固定门槛通过表, 两两差距检查列表)。
    相对误差差距用 ``relative_max``，绝对误差差距按变量用 ``absolute_max``。
    """
    gates = FSI1_GATES if gates is None else gates
    absolute_max = METHOD_GAP_ABSOLUTE_MAX if absolute_max is None else absolute_max
    fixed = {}
    for label, metric in method_metrics.items():
        fixed[label] = {
            "tip_y_relative_error_max": bool(metric["tip_y_relative_error"] <= gates["tip_y_relative_error_max"]),
            "drag_relative_error_max": bool(metric["drag_relative_error"] <= gates["drag_relative_error_max"]),
            "tip_x_absolute_error_max": bool(metric["tip_x_absolute_error"] <= gates["tip_x_absolute_error_max"]),
            "lift_absolute_error_max": bool(metric["lift_absolute_error"] <= gates["lift_absolute_error_max"]),
        }
    gaps = []
    labels = list(method_metrics)
    for first_index, first in enumerate(labels):
        for second in labels[first_index+1:]:
            a, b = method_metrics[first], method_metrics[second]
            entries = {
                "tip_y_relative": abs(a["tip_y_relative_error"]-b["tip_y_relative_error"]),
                "drag_relative": abs(a["drag_relative_error"]-b["drag_relative_error"]),
                "tip_x_absolute": abs(a["tip_x_absolute_error"]-b["tip_x_absolute_error"]),
                "lift_absolute": abs(a["lift_absolute_error"]-b["lift_absolute_error"]),
            }
            gaps.append({
                "pair": f"{first} vs {second}",
                "tip_y_relative_gap": entries["tip_y_relative"],
                "drag_relative_gap": entries["drag_relative"],
                "tip_x_absolute_gap": entries["tip_x_absolute"],
                "lift_absolute_gap": entries["lift_absolute"],
                "passed": bool(entries["tip_y_relative"] <= relative_max
                               and entries["drag_relative"] <= relative_max
                               and entries["tip_x_absolute"] <= absolute_max["tip_x"]
                               and entries["lift_absolute"] <= absolute_max["lift"]),
            })
    return fixed, gaps


# ---------------------------------------------------------------------------
# cost table aggregation (m5_t23_plan section 4.1)
# ---------------------------------------------------------------------------

#: ``item -> (unit, definition)`` of the required T23 cost items.
# T23 成本表要求的成本项 -> (单位, 定义)；单位 None 表示计数类，无量纲。
COST_ITEMS = {
    "total_seconds": ("s", "wall clock of the solve stage (mesh generation excluded, plotting excluded)"),
    "mesh_seconds": ("s", "mesh generation, recorded separately"),
    "setup_seconds": ("s", "problem/context construction after meshing, recorded separately"),
    "assembly_seconds": ("s", "fluid/solid residual+tangent, interface load, Stokes operators, harmonic extension"),
    "linear_solve_seconds": ("s", "direct fluid/solid linear solves and Krylov time"),
    "newton_seconds": ("s", "whole Newton/march call; overlaps assembly+linear (stated口径)"),
    "newton_iterations": (None, "accepted Newton iterations (all solves)"),
    "coupling_iterations": (None, "partitioned interface evaluations G(x)"),
    "linear_solves": (None, "direct/Krylov linear solves"),
    "linear_iterations": (None, "Krylov inner iterations (0 = direct factorization)"),
    "failed_steps": (None, "failed candidates/steps"),
    "pseudo_time_steps": (None, "monolithic BDF march steps (M-11)"),
    "time_term_norm": ("1/s", "final scaled BDF time term of the monolithic march"),
    "physical_errors": (None, "tip/drag/lift error columns, see comparison tables"),
}


def summarize_values(values):
    """Median/min/max/count of a numeric sequence (NaN-free).

    中文：对数值序列做稳健汇总；自动跳过 ``None`` 与非有限值（NaN/inf）。
    输入可为含缺失项的序列，输出含 ``count/median/min/max``；全空时除 count 外为 None。
    """
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not clean:
        return {"count": 0, "median": None, "min": None, "max": None}
    array = np.asarray(clean, dtype=float)
    return {"count": int(array.size), "median": float(np.median(array)),
            "min": float(np.min(array)), "max": float(np.max(array))}


def cost_rows(entries):
    """Long-format cost table ``method x level x item`` with median/min/max.

    ``entries`` is a sequence of ``{"method", "level", "repeat", "cost"}``
    records (the ``cost`` mapping may omit items that do not apply to the
    method).  Only items present in at least one repeat are emitted.

    中文：把多次重复的成本记录聚合成“方法 × 层级 × 成本项”的长表。``entries``
    为记录序列，每条含 method/level/repeat/cost；同一键的多次重复取中位数与极值。
    仅输出至少出现过一次的成本项；单位与定义取自 :data:`COST_ITEMS`。
    """
    grouped = {}
    for entry in entries:
        method = str(entry["method"])
        level = str(entry["level"])
        for item, value in entry["cost"].items():
            if value is None:
                continue
            grouped.setdefault((method, level, item), []).append(value)
    rows = []
    for (method, level, item), values in sorted(grouped.items()):
        stats = summarize_values(values)
        unit, definition = COST_ITEMS.get(item, (None, ""))
        rows.append({"method": method, "level": level, "item": item, "unit": unit,
                     "repeats": stats["count"], "median": stats["median"], "min": stats["min"],
                     "max": stats["max"], "definition": definition})
    return rows


def write_json(path, payload):
    """中文：将 ``payload`` 以 UTF-8、缩进 2、禁止 NaN 的 JSON 写入 ``path``；
    自动创建父目录，并原样返回 payload 便于链式使用。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return payload


def file_sha256(path):
    """中文：分块（1 MiB）计算文件 SHA-256 十六进制摘要，用于产物可复现性指纹。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_snapshot():
    """Code/environment fingerprint required by T23 section 4.2.

    中文：采集运行环境指纹：Python/可执行文件/平台、CPU 与内存（读 /proc，缺失则
    记 unknown）、关键依赖版本、BLAS 线程数环境变量、git HEAD 与状态、时间戳。
    用于 T23 4.2 要求的可复现代码/环境快照。
    """
    packages = {}
    for name in ("fsi-ice", "numpy", "scipy", "matplotlib", "gmsh"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    memory = "unknown"
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal"):
                memory = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                  check=False).stdout.strip() or None
        git_status = subprocess.run(["git", "status", "--short"], capture_output=True, text=True,
                                    check=False).stdout.strip().splitlines()
    except OSError:
        git_head, git_status = None, []
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu": cpu,
        "memory": memory,
        "packages": packages,
        "threads": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                                             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "git_head": git_head,
        "git_status": git_status,
        "timestamp": now_iso(),
    }


# ---------------------------------------------------------------------------
# monolithic steady runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonolithicCaseSetup:
    """Frozen structural inputs of one monolithic FSI1 case (h level).

    中文：单个整体 FSI1 算例（某一网格尺寸 h）的冻结结构输入：块布局、拓扑、调和
    延拓、材料、流体密度/粘度/平均速度、翼尖节点、参考网格、固体网格、块尺度、
    模板上下文与初值 ``x0``、时间步 ``dt``、候选标识。``__post_init__`` 对
    ``scales``/``x0`` 做防御性拷贝，避免外部修改。
    """

    layout: object
    topology: object
    harmonic: object
    material: object
    fluid_density: float
    fluid_viscosity: float
    mean_velocity: float
    tip_node: int
    reference_mesh: object
    solid_mesh: object
    scales: np.ndarray
    template: MonolithicContext
    x0: np.ndarray
    dt: float
    candidate_id: object

    def __post_init__(self):
        object.__setattr__(self, "scales", np.array(self.scales, dtype=float, copy=True))
        object.__setattr__(self, "x0", np.array(self.x0, dtype=float, copy=True))


def initial_velocity(layout, fluid_space, mean_velocity):
    """Explicit initial P2 velocity: inlet Dirichlet data, zero elsewhere.

    中文：构造显式初始 P2 速度场：仅在流体 Dirichlet 节点（入口）填入抛物线
    :func:`inlet_profile`，其余为零。输入布局、流体速度空间与入口平均速度[m/s]；
    返回 ``(n_points, 2)`` 速度数组[m/s]。
    """
    velocity = np.zeros((len(fluid_space.points), 2))
    velocity[layout.fluid_dirichlet_nodes] = inlet_profile(
        fluid_space.points[layout.fluid_dirichlet_nodes], mean_velocity)
    return velocity


def build_monolithic_case(problem, config, *, dt, candidate_id=("m5-t23", "monolithic")):
    """Build the template context, frozen scales and explicit initial state.

    ``problem`` is a :class:`~fsi_ice.validation.fsi1.SteadyFSI1Problem` (the
    same mesh/topology objects as the partitioned runs), so all three methods
    share one discretisation by construction.  The initial state uses no
    partitioned data: zero displacement/velocity/load, reference geometry and
    the inlet parabolic Dirichlet values (the same boundary callable as the
    partitioned fluid solves).

    中文：为整体求解构建模板上下文、冻结块尺度与显式初值，并打包为
    :class:`MonolithicCaseSetup`。共用分区算例的同一网格/拓扑，保证三种方法离散一致
    （``pressure_pin=None``）。初值完全不含分区数据：固体位移/速度/界面载荷为零、
    参考 ALE 几何、流体入口抛物线 Dirichlet（与分区流体同一边界函数）。
    """
    fluid = problem.fluid_mesh
    solid = problem.solid_mesh
    topology = problem.topology
    layout = build_block_layout(topology,
                                fluid_dirichlet_tags=["inlet", "walls", "cylinder-rigid-wall"],
                                solid_dirichlet_tags=["solid-clamp"],
                                pressure_pin=None)
    scales = block_scale_vector(layout, dict(T23_BLOCK_SCALES))
    zero_solid = np.zeros_like(solid.points)
    zero_fluid = np.zeros((len(topology.fluid_space.points), 2))
    zero_mesh = np.zeros_like(fluid.points)
    geometry = build_ale_geometry(fluid, zero_mesh, candidate_id=(candidate_id, "initial"))
    # 初始 BDF 历史为一层零状态（BDF1 启动），随后取一步 stencil 作为代际号。
    history = BDFHistory(np.zeros((2, len(solid.points), 2)), dt, order=1)
    history.accept(history.context(), np.zeros((2, len(solid.points), 2)))
    stencil = history.stencil()
    base = MonolithicBase(
        reference_mesh=fluid, material=problem.material,
        fluid_density=float(config["fluid"]["density"]),
        fluid_viscosity=float(config["fluid"]["dynamic_viscosity"]),
        solid_states=((zero_solid, zero_solid),), fluid_states=(zero_fluid,),
        accepted_geometries=(geometry,), generation=stencil.step_id,
        fluid_boundary_velocity=lambda points: inlet_profile(
            points, float(config["fluid"]["mean_velocity"])))
    template = MonolithicContext(layout, base, stencil, topology, problem.extension,
                                 candidate_id=(candidate_id, "template"),
                                 thickness=problem.material.thickness, pressure_constraint=None)
    x0 = np.zeros(layout.n_total)  # 全局状态向量；初值仅流体速度块非零
    x0[layout.slice_of("u_f")] = initial_velocity(
        layout, topology.fluid_space, float(config["fluid"]["mean_velocity"])).ravel()
    return MonolithicCaseSetup(
        layout=layout, topology=topology, harmonic=problem.extension, material=problem.material,
        fluid_density=float(config["fluid"]["density"]),
        fluid_viscosity=float(config["fluid"]["dynamic_viscosity"]),
        mean_velocity=float(config["fluid"]["mean_velocity"]), tip_node=problem.tip_node,
        reference_mesh=fluid, solid_mesh=solid, scales=scales, template=template, x0=x0,
        dt=float(dt), candidate_id=tuple(candidate_id))


def _step_record_dict(record, *, global_index, chunk):
    """中文：把单个 march 步记录展开为 JSON 可序列化字典。``global_index`` 为跨分块
    的全局步号，``chunk`` 为所属分块号；展开 BDF 时间项、界面间隙、计数器与计时。"""
    return {
        "index": int(global_index), "chunk": int(chunk), "generation": int(record.generation),
        "order": int(record.order), "dt": float(record.dt),
        "newton_iterations": int(record.newton_iterations),
        "residual_norm": float(record.residual_norm), "threshold": float(record.threshold),
        "time_term": float(record.time_term),
        "time_term_u_f": float(record.time_terms["u_f"]),
        "time_term_d_s": float(record.time_terms["d_s"]),
        "time_term_v_s": float(record.time_terms["v_s"]),
        "interface_gap": float(record.interface_gap),
        "counters": record.counters.as_dict(),
        "timings": {key: float(value) for key, value in dict(record.timings).items()},
    }


@dataclass(frozen=True)
class MarchRecord:
    """Aggregated outcome of one budgeted monolithic march level.

    中文：一个受限（预算内）整体 march 层级的汇总结果：``reason`` 停止原因
    （steady/max_steps/wall_seconds/newton_failure）、是否达到稳态、最终标度后 BDF
    时间项、逐步记录、最终状态 ``x``、计数器与计时、分块数与墙钟时间、失败详情与
    启动预算。``__post_init__`` 将 ``x`` 置为只读并冻结各 dict/tuple。
    """

    level: float
    steady_tolerance: float
    newton_rtol: float
    newton_atol: float
    dt: float
    order: int
    reason: str
    steady_reached: bool
    final_time_term: float
    final_time_terms: dict
    steps: tuple
    x: np.ndarray
    counters: dict
    timings: dict
    chunks: int
    wall_seconds: float
    failure: dict
    start: dict

    def __post_init__(self):
        x = np.array(self.x, dtype=float, copy=True)
        x.setflags(write=False)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "final_time_terms", dict(self.final_time_terms))
        object.__setattr__(self, "counters", dict(self.counters))
        object.__setattr__(self, "timings", dict(self.timings))


def _plain(value):
    """Best-effort JSON-safe conversion of diagnostics values.

    中文：尽力把诊断值转为 JSON 安全类型：dict/list 递归；ndarray 直接丢弃（None，
    避免体积巨大的状态数组进入失败记录）；numpy 标量转原生 bool/int/float；其它转 str。
    """
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _failure_dict(error, *, chunk, completed):
    """中文：从 ``MonolithicNewtonFailure`` 提取失败详情字典（阶段、迭代、原因、残差、
    步号、已完成步数、计数器/计时、可序列化诊断），供 ``failure.json``/summary 记录。"""
    counters = getattr(error, "counters", None)
    timings = getattr(error, "timings", None)
    return {
        "phase": getattr(error, "phase", None), "iteration": getattr(error, "iteration", None),
        "reason": str(getattr(error, "reason", error)),
        "residual_norm": getattr(error, "residual_norm", None),
        "raw_residual_norm": getattr(error, "raw_residual_norm", None),
        "step_index": getattr(error, "step_index", None),
        "chunk": int(chunk),
        "completed_steps": int(completed),
        "counters": None if counters is None else counters.as_dict(),
        "timings": None if timings is None else {key: float(value) for key, value in dict(timings).items()},
        "diagnostics": _plain(dict(getattr(error, "diagnostics", {}) or {})),
    }


def state_context(case, x, *, dt, candidate_id):
    """Continuation/warm-start template whose accepted layer-0 is ``x``.

    ``march_to_steady`` re-anchors its BDF history at ``context.base``; any
    budgeted or warm-started march must therefore hand it the intended accepted
    state, otherwise the march re-integrates the initial transient from a stale
    base.  The fresh history is a one-layer BDF1 startup (the march driver
    always starts with a BDF1 step), which is the correct continuation behaviour.

    中文：构造以 ``x`` 为“已接受第 0 层”的续算/热启动上下文。因为 ``march_to_steady``
    会以 ``context.base`` 重新锚定 BDF 历史，任何受限或热启动的 march 都必须传入
    预期的已接受状态，否则会从过期基态重新积分初始瞬态。新历史为单层 BDF1 启动层
    （驱动器总是先做一步 BDF1），这正是正确的续算行为。
    """
    blocks = case.layout.unpack(x)
    d_s = np.array(blocks["d_s"], copy=True)
    v_s = np.array(blocks["v_s"], copy=True)
    u_f = np.array(blocks["u_f"], copy=True)
    d_m = np.array(blocks["d_m"], copy=True)
    source = case.template.base
    geometry = build_ale_geometry(case.reference_mesh, d_m, quality_limits=source.quality_limits,
                                  candidate_id=(candidate_id, "geometry"))
    history = BDFHistory(np.stack((d_s, v_s)), dt, order=1)
    history.accept(history.context(), np.stack((d_s, v_s)))
    stencil = history.stencil()
    base = MonolithicBase(
        reference_mesh=case.reference_mesh, material=case.material,
        fluid_density=case.fluid_density, fluid_viscosity=case.fluid_viscosity,
        solid_states=((d_s, v_s),), fluid_states=(u_f,), accepted_geometries=(geometry,),
        generation=stencil.step_id, solid_body_force=source.solid_body_force,
        fluid_body_force=source.fluid_body_force,
        fluid_boundary_velocity=source.fluid_boundary_velocity,
        mesh_boundary_displacement=source.mesh_boundary_displacement,
        solid_boundary_displacement=source.solid_boundary_displacement,
        solid_boundary_velocity=source.solid_boundary_velocity,
        quality_limits=source.quality_limits)
    return MonolithicContext(case.layout, base, stencil, case.topology, case.harmonic,
                             candidate_id=(candidate_id, "continuation"),
                             thickness=case.material.thickness, pressure_constraint=None)


def march_level(case, *, level, steady_tolerance, newton_rtol, newton_atol, start=None,
                dt=None, order=2, chunk_steps=6, max_steps=60, wall_seconds=3600.0,
                max_iterations=25, initial_context=None, progress=None):
    """Budgeted chunked ``march_to_steady`` of one tolerance level.

    The charter requires a single march to stop within its wall budget; the
    driver therefore calls ``march_to_steady`` in chunks and stitches the step
    records, counters and timings.  Between chunks the continuation context is
    rebuilt from the current accepted state (:func:`state_context`), so each
    restart performs a genuine BDF1 startup step from the current state instead
    of re-integrating a stale transient; the BDF fixed point is unchanged.  A
    ``MonolithicNewtonFailure`` is captured, recorded and returned as a failed
    level (no exception, no silent continuation).  ``initial_context`` overrides
    the first chunk's history anchor (used for warm starts); the default
    ``case.template`` is the explicit cold-start state.

    中文：对单个容差层级做“分块受限”的 ``march_to_steady``。为满足墙钟预算，把 march
    切成若干 chunk 并拼接步记录/计数器/计时；每个 chunk 之间用当前已接受状态重建续算
    上下文（:func:`state_context`），使每次重启都从当前状态做真实 BDF1 启动，BDF 不动点
    不变。``MonolithicNewtonFailure`` 被捕获并作为“失败层级”返回（不抛异常、不静默续算）。
    ``initial_context`` 覆盖首块的历史锚点（热启动用），默认 ``case.template`` 为冷启动。
    """
    dt = float(case.dt if dt is None else dt)
    state = np.array(case.x0 if start is None else start, dtype=float, copy=True)
    start_norm = float(np.max(np.abs(state)))  # 记录起点的最大绝对值（预算诊断）
    started = time.perf_counter()
    steps = []
    counters = MonolithicCounters()
    buckets = {"evaluate_seconds": 0.0, "linear_seconds": 0.0, "newton_seconds": 0.0,
               "chunk_total_seconds": 0.0}
    final_terms = {}
    final_term = float("inf")
    failure = None
    chunks = 0
    steady_reached = False
    reason = "init"
    context = case.template if initial_context is None else initial_context
    while True:
        elapsed = time.perf_counter()-started
        if steady_reached:
            break
        if len(steps) >= int(max_steps):
            reason = "max_steps"
            break
        if elapsed >= float(wall_seconds):
            reason = "wall_seconds"
            break
        chunk = min(int(chunk_steps), int(max_steps)-len(steps))  # 本块步数，不越过总步数上限
        options = NewtonOptions(max_iterations=int(max_iterations), rtol=float(newton_rtol),
                                atol=float(newton_atol))
        if progress is not None:
            progress(f"level {level:g}: chunk {chunks+1}, steps so far {len(steps)}, "
                     f"elapsed {elapsed:.1f}s", flush=True)
        try:
            result = march_to_steady(context, state, dt=dt, order=int(order), max_steps=chunk,
                                     steady_tolerance=float(steady_tolerance), options=options,
                                     scales=case.scales, jacobian_mode="assembled")
        except MonolithicNewtonFailure as error:
            # 失败：保留已接受的 partial 步与计数器/计时，记录 failure 而非抛出。
            failure = _failure_dict(error, chunk=chunks, completed=len(steps))
            partial = list(getattr(error, "march_steps", ()) or ())
            for record in partial:
                steps.append(_step_record_dict(record, global_index=len(steps)+1, chunk=chunks))
            extra = getattr(error, "counters", None)
            if extra is not None:
                counters = counters+extra
            partial_timings = getattr(error, "timings", None)
            if partial_timings is not None:
                for key in ("evaluate_seconds", "linear_seconds", "newton_seconds"):
                    buckets[key] += float(partial_timings[key])
            reason = "newton_failure"
            break
        chunks += 1
        for record in result.steps:
            steps.append(_step_record_dict(record, global_index=len(steps)+1, chunk=chunks-1))
        counters = counters+result.counters
        for key in ("evaluate_seconds", "linear_seconds", "newton_seconds"):
            buckets[key] += float(result.timings[key])
        buckets["chunk_total_seconds"] += float(result.timings["total_seconds"])
        final_terms = {key: float(value) for key, value in dict(result.final_time_terms).items()}
        final_term = float(result.final_time_term)  # 标度后 BDF 时间项；低于容差即达稳态
        state = np.array(result.x, dtype=float, copy=True)
        if progress is not None:
            progress(f"level {level:g}: chunk {chunks} done, steps {len(steps)}, "
                     f"time term {final_term:.3e} (target {steady_tolerance:.1e}), "
                     f"chunk wall {float(result.timings['total_seconds']):.1f}s", flush=True)
        if result.steady_reached:
            steady_reached = True
            reason = "steady"
            continue
        context = state_context(case, state, dt=dt,
                                candidate_id=(case.candidate_id, "chunk", chunks))
    wall = time.perf_counter()-started
    timings = {"evaluate_seconds": buckets["evaluate_seconds"],
               "linear_seconds": buckets["linear_seconds"],
               "newton_seconds": buckets["newton_seconds"],
               "chunk_total_seconds": buckets["chunk_total_seconds"],
               "total_seconds": wall}
    return MarchRecord(level=float(level), steady_tolerance=float(steady_tolerance),
                       newton_rtol=float(newton_rtol), newton_atol=float(newton_atol), dt=dt,
                       order=int(order), reason=reason, steady_reached=bool(steady_reached),
                       final_time_term=final_term, final_time_terms=final_terms, steps=tuple(steps),
                       x=state, counters=counters.as_dict(), timings=timings, chunks=int(chunks),
                       wall_seconds=wall, failure=failure,
                       start={"max_abs_x0": start_norm, "chunk_steps": int(chunk_steps),
                              "max_steps": int(max_steps), "wall_seconds": float(wall_seconds),
                              "max_iterations": int(max_iterations)})


def steady_diagnostics(case, x, *, dt, candidate_id=("m5-t23", "fixed-point")):
    """Steady residual of ``x`` on a self-consistent BDF2 history.

    Both accepted layers equal ``x``, so the BDF2 time term cancels exactly and
    the residual is the steady discrete equation (T22 section 5).  Returns the
    scaled block maxima, the T21 block norms and the candidate geometry
    diagnostics.

    中文：在“自洽”的 BDF2 历史上评估状态 ``x`` 的稳态残差。因为两个已接受层都等于
    ``x``，BDF2 时间项精确抵消，残差即稳态离散方程（T22 第 5 节）。返回按块尺度归一
    的残差最大值、T21 块范数监控与候选几何诊断（含界面相容间隙）。
    """
    layout = case.layout
    blocks = layout.unpack(x)
    geometry = build_ale_geometry(case.reference_mesh, np.array(blocks["d_m"], copy=True),
                                  candidate_id=(candidate_id, "geometry"))
    state = np.stack((np.asarray(blocks["d_s"]), np.asarray(blocks["v_s"])))
    history = BDFHistory(state, dt, order=2)
    history.accept(history.context(), state)
    stencil = history.stencil()
    d_s = np.asarray(blocks["d_s"])
    v_s = np.asarray(blocks["v_s"])
    u_f = np.asarray(blocks["u_f"])
    base = MonolithicBase(
        reference_mesh=case.reference_mesh, material=case.material,
        fluid_density=case.fluid_density, fluid_viscosity=case.fluid_viscosity,
        # 两层都取 (d_s, v_s)/(u_f, u_f)：时间项恒为零，只留下稳态残差。
        solid_states=((d_s, v_s), (d_s, v_s)), fluid_states=(u_f, u_f),
        accepted_geometries=(geometry, geometry), generation=stencil.step_id,
        fluid_boundary_velocity=lambda points: inlet_profile(points, case.mean_velocity))
    context = MonolithicContext(layout, base, stencil, case.topology, case.harmonic,
                                candidate_id=(candidate_id, "context"),
                                thickness=case.material.thickness, pressure_constraint=None)
    evaluation = evaluate_monolithic(x, context=context, mode="residual")
    residual = evaluation.residual
    # 每个块按自身物理尺度归一后取最大绝对值，再对块取最大，得到稳态残差范数。
    scaled = {name: float(np.max(np.abs(residual[layout.slice_of(name)]/case.scales[layout.slice_of(name)])))
              for name in BLOCK_NAMES}
    return {
        "steady_residual_norm": max(scaled.values()),
        "steady_block_norms": scaled,
        "t21_block_norms": {key: float(value) for key, value in dict(evaluation.block_norms).items()},
        "geometry": {key: float(value) for key, value in dict(evaluation.geometry_diagnostics).items()},
        "interface_gap": interface_compatibility_gap(x, context=context),
        "candidate_id": repr(context.candidate_id),
        "definition": "max_i |R_b,i|/S_b of the step residual with two identical accepted BDF2 "
                      "layers (time term cancels); T21 block norms are 2-norm/max(2-norm,1) monitors",
    }


def previous_level_state(root, mesh_size, level):
    """Final state of the *less tight* level closest to ``level`` (or ``None``).

    Level labels are stopping tolerances, so a less tight level has a
    numerically **larger** value (1e-6 is less tight than 1e-8).  The closest
    less-tight state is therefore the minimum among the values greater than
    ``level``.  Used for monolithic-to-monolithic warm starts only.

    中文：返回比 ``level`` 更宽松（即容差数值更大、如 1e-6 相对 1e-8 更松）的最近层级
    的最终状态；不存在则返回 ``None``。仅用于整体到整体的热启动；按层级目录名解析。
    """
    base = Path(root)/"monolithic"/f"h_{mesh_size:g}"
    if not base.exists():
        return None
    candidates = []
    for path in base.glob("level_*"):
        try:
            value = float(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        state = path/"final_state.npz"
        if value > float(level) and state.exists():  # 只取更宽松且存在最终状态的层级
            candidates.append((value, state))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]  # 更宽松层级中取最接近者


def monolithic_metrics(case, problem, config, x):
    """tip/drag/lift of a monolithic state using the partitioned force recipe.

    The cylinder force is recomputed with the shared ``boundary_force`` kernel
    on the candidate geometry; the interface contribution is the sum of the
    converged ``g`` block (already integrated, thickness counted once).

    中文：用与分区一致的受力公式计算整体状态的翼尖位移与阻力/升力。圆柱面力由共享
    ``boundary_force`` 核在候选几何上重算；界面贡献为已收敛 ``g`` 块之和（已积分、
    厚度只计一次）。输出含 tip/drag/lift 及其相对/绝对误差、圆柱面/界面分项、以及
    界面力与 ``H^T g`` 之和的核对项、网格质量指标。单位：力为 N/m（单位厚度）。
    """
    layout = case.layout
    blocks = layout.unpack(x)
    geometry = build_ale_geometry(case.reference_mesh, np.array(blocks["d_m"], copy=True),
                                  candidate_id=("m5-t23", "metrics"))
    operators = assemble_stokes(geometry.current_mesh, viscosity=case.fluid_viscosity,
                                density=case.fluid_density)
    solution = SimpleNamespace(velocity=np.array(blocks["u_f"], copy=True),
                               pressure=np.array(blocks["p_f"], copy=True))
    cylinder = boundary_force(operators, solution, "cylinder-rigid-wall")
    g = np.asarray(blocks["g"]).ravel()
    interface_force = g.reshape(-1, 2).sum(axis=0)  # 界面节点力求和 -> 合力 (N/m)
    total = cylinder+interface_force
    tip = np.asarray(blocks["d_s"])[case.tip_node]
    reference = config["reference"]
    return {
        "tip_x": float(tip[0]), "tip_y": float(tip[1]),
        "drag": float(total[0]), "lift": float(total[1]),
        "tip_x_absolute_error": float(abs(tip[0]-reference["tip_x"])),
        "tip_y_relative_error": float(abs(tip[1]/reference["tip_y"]-1)),
        "drag_relative_error": float(abs(total[0]/reference["drag"]-1)),
        "lift_absolute_error": float(abs(total[1]-reference["lift"])),
        "cylinder_force_x": float(cylinder[0]), "cylinder_force_y": float(cylinder[1]),
        "interface_force_x": float(interface_force[0]), "interface_force_y": float(interface_force[1]),
        "interface_force_sum_vs_solid": float(np.sum(interface_force
                                                     - (case.topology.H.T @ g).reshape(-1, 2).sum(axis=0))),
        "minimum_mesh_volume_ratio": float(geometry.minimum_volume_ratio),
        "minimum_mesh_quality": float(geometry.minimum_quality),
    }


def monolithic_solution_fields(case, x):
    """T23 comparison bundle of one monolithic state.

    中文：把一个整体状态打包为 T23 逐变量对比所需的字段字典。``interface_x`` 为固体
    界面位移迹 ``d_s|Gamma``（按 ``solid_dofs`` 抽取），``interface_H = H d_s|Gamma``
    为其映射到流体 P2 迹，``lambda = H^T g`` 为映射回固体紧凑迹的对偶载荷。
    """
    layout = case.layout
    blocks = layout.unpack(x)
    d_s = np.asarray(blocks["d_s"])
    u_f = np.asarray(blocks["u_f"])
    g = np.asarray(blocks["g"]).ravel()
    interface_x = d_s.ravel()[case.topology.solid_dofs]
    lambda_ = np.asarray(case.topology.H.T @ g).ravel()  # H^T g：流体迹载荷映射到固体迹
    h_d_s = np.asarray(case.topology.H @ interface_x).ravel()  # H d_s|Gamma：映射到流体 P2 迹
    return {
        "d_s": d_s.ravel().copy(), "tip_A": np.array(d_s[case.tip_node], copy=True),
        "u_f": u_f.ravel().copy(), "p_f": np.asarray(blocks["p_f"]).ravel().copy(),
        "d_m": np.asarray(blocks["d_m"]).ravel().copy(),
        "v_s": np.asarray(blocks["v_s"]).ravel().copy(),
        "interface_x": interface_x.copy(), "interface_H": h_d_s,
        "g": g.copy(), "lambda": lambda_.copy(),
    }


# ---------------------------------------------------------------------------
# output conventions (T20 style: clean slate, failure.json on failure)
# ---------------------------------------------------------------------------

# 整体算例的标准产物文件清单（T20 风格：先清空旧产物，失败时写 failure.json）。
_MONOLITHIC_FILES = ("summary.json", "failure.json", "steps.csv", "solution.npz", "final_state.npz")


def clear_monolithic_outputs(output):
    """中文：清空某层级的旧产物（标准文件与 ``steps_partial*.csv``），避免失败重跑
    残留上次成功结果。返回输出目录 ``Path``。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in _MONOLITHIC_FILES:
        (output/name).unlink(missing_ok=True)
    for path in output.glob("steps_partial*.csv"):
        path.unlink()
    return output


def write_monolithic_failure(output, *, level, stage, reason, extra=None):
    """中文：写 ``failure.json``（status=FAIL）记录失败层级、阶段与原因，并删除可能
    残留的勾选产物（summary/steps/solution），确保失败不会被误读为成功。返回 payload。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "steps.csv", "solution.npz"):
        (output/name).unlink(missing_ok=True)
    payload = {"status": "FAIL", "case": "m5_t23_monolithic", "level": float(level),
               "stage": str(stage), "reason": str(reason), "timestamp": now_iso()}
    if extra:
        payload.update(extra)
    write_json(output/"failure.json", payload)
    return payload


def write_monolithic_level(output, record, *, metrics=None, steady=None, config_snapshot=None,
                           environment=None, setup_seconds=None, mesh_seconds=None,
                           initial_state=None):
    """Write ``steps.csv``, ``summary.json``, ``solution.npz``, ``final_state.npz``.

    中文：写某一层级的结果产物。``status`` 依 ``record.steady_reached`` 判定
    PASS/FAIL；可选字段（网格/装配耗时、物理指标、稳态残差、配置快照、环境指纹、
    初始状态路径）在提供时追加。最终状态存 ``final_state.npz``，摘要存 ``summary.json``。
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if record.steps:
        write_csv(output/"steps.csv", _flatten_rows(record.steps))
    payload = {
        "status": "PASS" if record.steady_reached else "FAIL",
        "case": "m5_t23_monolithic",
        "level": float(record.level),
        "reason": record.reason,
        "steady_reached": bool(record.steady_reached),
        "steady_tolerance": float(record.steady_tolerance),
        "final_time_term": float(record.final_time_term),
        "final_time_terms": record.final_time_terms,
        "newton": {"rtol": record.newton_rtol, "atol": record.newton_atol},
        "dt": record.dt, "order": record.order,
        "steps_total": len(record.steps), "chunks": record.chunks,
        "counters": record.counters, "timings": record.timings,
        "wall_seconds": record.wall_seconds, "budget": record.start,
        "failure": record.failure, "timestamp": now_iso(),
    }
    if mesh_seconds is not None:
        payload["mesh_seconds"] = float(mesh_seconds)
    if setup_seconds is not None:
        payload["setup_seconds"] = float(setup_seconds)
    if metrics is not None:
        payload["metrics"] = metrics
    if steady is not None:
        payload["steady_residual"] = steady
    if config_snapshot is not None:
        payload["config_snapshot"] = config_snapshot
    if environment is not None:
        payload["environment"] = environment
    if initial_state is not None:
        payload["initial_state"] = str(initial_state)
    np.savez_compressed(output/"final_state.npz", x=np.asarray(record.x))
    write_json(output/"summary.json", payload)
    return payload


def _flatten_rows(rows):
    """中文：把嵌套的步记录行展平为 CSV 行：``counters``/``timings`` 展开为
    ``counter_*``/``timing_*`` 前缀列，其余字段原样保留。"""
    flat = []
    for row in rows:
        item = {key: value for key, value in row.items()
                if key not in ("counters", "timings")}
        for key, value in row["counters"].items():
            item[f"counter_{key}"] = value
        for key, value in row["timings"].items():
            item[f"timing_{key}"] = value
        flat.append(item)
    return flat


def flatten_step_rows(rows):
    """Public alias of the step-row flattener used by ``steps.csv``.

    中文：``_flatten_rows`` 的公开别名，供 ``steps.csv`` 生成使用。
    """
    return _flatten_rows(rows)


def write_monolithic_solution(output, case, x):
    """``solution.npz`` with the same field names as the partitioned artifacts.

    中文：写 ``solution.npz``，字段名与分区产物保持一致，便于同一后处理流程读取：
    含固体点/单元与位移/速度、流体参考/当前点与单元、流体速度/压力、网格位移、
    全局状态、翼尖节点号，以及界面位移/映射值/载荷。返回对比字段 bundle。
    """
    layout = case.layout
    blocks = layout.unpack(x)
    d_m = np.asarray(blocks["d_m"])
    geometry = build_ale_geometry(case.reference_mesh, np.array(d_m, copy=True),
                                  candidate_id=("m5-t23", "solution"))
    bundle = monolithic_solution_fields(case, x)
    np.savez_compressed(
        Path(output)/"solution.npz",
        solid_points=case.solid_mesh.points, solid_cells=case.solid_mesh.cells,
        solid_displacement=np.asarray(blocks["d_s"]),
        solid_velocity=np.asarray(blocks["v_s"]),
        fluid_reference_points=case.reference_mesh.points,
        fluid_current_points=geometry.current_mesh.points, fluid_cells=case.reference_mesh.cells,
        velocity_points=case.topology.fluid_space.points,
        fluid_velocity=np.asarray(blocks["u_f"]), fluid_pressure=np.asarray(blocks["p_f"]),
        mesh_displacement=d_m, state=np.asarray(x), tip_node=np.int64(case.tip_node),
        interface_values=bundle["interface_x"], interface_mapped=bundle["interface_H"],
        interface_load=bundle["g"], interface_load_solid=bundle["lambda"])
    return bundle


__all__ = [
    "T23_RTOL", "T23_ATOL_SCALE", "T23_SCALES", "T23_BLOCK_SCALES", "T23_TOLERANCE_LADDER",
    "FSI1_GATES", "METHOD_GAP_RELATIVE_MAX", "METHOD_GAP_ABSOLUTE_MAX",
    "ComparisonRow", "CONSISTENCY_SPECS", "COST_ITEMS", "MarchRecord", "MonolithicCaseSetup",
    "build_monolithic_case", "compare_values", "consistency_rows", "cost_rows",
    "clear_monolithic_outputs", "environment_snapshot", "file_sha256", "flatten_step_rows",
    "format_level", "initial_velocity", "march_level", "monolithic_metrics",
    "monolithic_solution_fields", "physical_error_gaps", "pressure_gauge_diagnostic",
    "previous_level_state",
    "state_context", "steady_diagnostics", "summarize_values", "write_json", "write_monolithic_failure",
    "write_monolithic_level", "write_monolithic_solution",
]

"""Algebraic consistency diagnostics for the fixed interface mapping.

Every performed check reports an explicit ``passed`` boolean, its numerical
value and the effective scale used for the frozen tolerance
``<= 1e-12 * S`` with ``S = max(explicit scale, relevant contribution sum)``.
Checks that lack inputs are listed in ``skipped`` with a reason instead of
being silently dropped.  Non-finite inputs are rejected outright and
non-finite derived values fail the corresponding check.

Passing ``geometry`` (an :class:`InterfaceGeometry`) adds the current
configuration block: both ``j_Gamma`` paths, ``n_s + n_f``, the Nanson normal,
the solid/fluid gap and the minimum Jacobians.  If ``motion`` is omitted the
current motion stored in the geometry is used.  Passing ``load`` (an
:class:`InterfaceLoad`) fills the compact/fluid loads from the assembled
result and adds the node-vs-quadrature traction block.

本模块对固定界面映射做代数自洽性诊断。每次执行的检查都给出显式 ``passed`` 布尔、
数值与所用尺度；冻结容差规则为 ``<= 1e-12 * S``，其中 ``S = max(显式尺度,
相关贡献量之和)``。关键约定：缺少输入的检查列入 ``skipped`` 并说明原因，而非静默
跳过；非有限输入直接拒绝，派生量非有限则对应检查判失败；``configuration`` 为
``"current"``（提供 ``motion``）或 ``"reference"``。传入 ``geometry``（:class:`InterfaceGeometry`）
会增加当前构形检查块：两条 ``j_Gamma`` 路径、``n_s + n_f``、Nanson 法向、固流
间隙与最小雅可比；若省略 ``motion`` 则使用几何中存储的当前运动。传入 ``load``
（:class:`InterfaceLoad`）会从装配结果填充压缩/流体载荷，并增加节点-对积分面力
检查块。
"""

from collections.abc import Mapping

import numpy as np

from fsi_ice.interface.geometry import InterfaceGeometry
from fsi_ice.interface.mapping import InterfaceMotion, scatter_solid_load
from fsi_ice.interface.topology import InterfaceTopology
from fsi_ice.interface.traction import InterfaceLoad

_TOLERANCE_FACTOR = 1e-12
_SCALE_KEYS = ("length", "force", "moment", "work", "velocity")


def _finite_vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array.astype(float, copy=True)


def _finite_matrix(value, rows, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (rows, 2):
        raise ValueError(f"{name} must have shape ({rows}, 2), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array.astype(float, copy=True)


def _explicit_scales(scales):
    if scales is None:
        return {}
    if not isinstance(scales, Mapping):
        raise ValueError("scales must be a mapping with keys length/force/moment/work/velocity.")
    unknown = sorted(set(scales) - set(_SCALE_KEYS))
    if unknown:
        raise ValueError(f"Unknown scale keys {unknown}; allowed keys are {list(_SCALE_KEYS)}.")
    result = {}
    for key, value in scales.items():
        number = float(value)
        if not np.isfinite(number) or number < 0:
            raise ValueError(f"Scale {key!r} must be finite and nonnegative, got {value!r}.")
        result[key] = number
    return result


def _effective_scale(explicit, contributions):
    """Frozen rule ``S = max(explicit scale, sum of relevant contributions)``.

    冻结尺度规则：``S = max(显式尺度, 相关贡献量绝对值之和)``。若 ``S <= 0`` 则回退
    到 1.0（避免零容差），并返回 ``(S, fallback)`` 供调用方记录是否发生回退。
    """
    total = float(sum(abs(float(value)) for value in contributions))
    scale = max(float(explicit or 0.0), total)
    fallback = scale <= 0.0
    return (1.0 if fallback else scale), fallback


def _cross_sum(coordinates, loads):
    # 二维力矩标量：sum(r_x F_y - r_y F_x)，即 r × F 的 z 分量。
    return float(np.sum(coordinates[:, 0] * loads[:, 1] - coordinates[:, 1] * loads[:, 0]))


def _contribution_sum(coordinates, loads):
    # 力矩尺度的贡献量：sum(|r| |F|)，用于冻结容差的尺度估计。
    return float(np.sum(np.linalg.norm(coordinates, axis=1) * np.linalg.norm(loads, axis=1)))


def _max_abs(values):
    return float(np.max(np.abs(values))) if len(values) else 0.0


def interface_diagnostics(topology, *, motion=None, loads=None, moment_origin=(0.0, 0.0), scales=None,
                          virtual_solid=None, virtual_fluid=None, motion_residual=None,
                          analytic_force=None, current_normals=None, geometry=None, load=None):
    """Check force, moment, virtual work, motion and normal consistency.

    ``loads`` is either the compact interleaved solid interface force vector
    ``(2*n_sG,)`` or a mapping with optional ``"compact"`` (already mapped
    solid interface forces) and ``"fluid"`` (already integrated fluid P2 dual
    nodal forces) entries.  ``motion`` is an :class:`InterfaceMotion`; with it
    the checks use current coordinates, otherwise reference coordinates are
    used and ``configuration`` is ``"reference"``.  ``virtual_solid`` is a
    compact virtual displacement, ``motion_residual`` a ``(velocity,
    derivative)`` pair of compact vectors, ``analytic_force`` the expected
    total force and ``current_normals`` a ``(n_s, n_f)`` pair of per-edge
    current outward normals.  ``geometry`` adds the configuration block and,
    when ``motion`` is omitted, supplies the current motion; ``load`` is an
    assembled :class:`InterfaceLoad` whose compact/fluid forces fill ``loads``
    and whose node-vs-quadrature totals are checked.  The result has per-check
    ``passed`` flags plus a global ``pass``; failed checks are also summarised
    in ``failures``.

    检查力、力矩、虚功、运动与法向的自洽性。``loads`` 要么是压缩交错固体界面力向量
    ``(2*n_sG,)``，要么是带可选 ``"compact"``（已映射的固体界面力）与 ``"fluid"``
    （已积分的流体 P2 对偶节点力）键的映射。``motion`` 为 :class:`InterfaceMotion`：
    提供时检查使用当前坐标，否则使用参照坐标且 ``configuration="reference"``。
    ``virtual_solid`` 为压缩虚位移，``motion_residual`` 为 ``(velocity, derivative)``
    压缩向量对，``analytic_force`` 为期望总力，``current_normals`` 为逐棱当前外法向
    对 ``(n_s, n_f)``。``geometry`` 增加当前构形检查块，并在省略 ``motion`` 时提供
    当前运动；``load`` 为已装配的 :class:`InterfaceLoad`，其压缩/流体力填充 ``loads``，
    并检查其节点-对积分总量。``moment_origin`` 为力矩参考点；``scales`` 可给出
    length/force/moment/work/velocity 的显式尺度。返回结果含各检查的 ``passed``、
    全局 ``pass``，失败检查另汇总于 ``failures``。
    """
    if not isinstance(topology, InterfaceTopology):
        raise TypeError("topology must be an InterfaceTopology.")
    geometry_object = None
    if geometry is not None:
        if not isinstance(geometry, InterfaceGeometry):
            raise TypeError("geometry must be an InterfaceGeometry.")
        if geometry.topology_id != topology.topology_id:
            raise ValueError("geometry was evaluated for a different topology.")
        geometry_object = geometry
    load_object = None
    if load is not None:
        if not isinstance(load, InterfaceLoad):
            raise TypeError("load must be an InterfaceLoad.")
        if load.topology_id != topology.topology_id:
            raise ValueError("load was assembled for a different topology.")
        if geometry_object is not None and load.geometry_id != geometry_object.geometry_id:
            raise ValueError("load was assembled for a different geometry version.")
        if loads is not None:
            raise ValueError("provide either loads or load, not both.")
        loads = {"compact": load.f_s_gamma, "fluid": load.g_f_gamma}
        load_object = load
    if motion is None and geometry_object is not None:
        displacement = geometry_object.solid_displacement
        motion = InterfaceMotion(
            topology_id=topology.topology_id, solid_values=displacement,
            compact_values=np.array(displacement.ravel()[topology.solid_dofs], copy=True),
            vertex_values=np.array(geometry_object.vertex_coordinates-topology.X_s_gamma, copy=True),
            p2_values=np.array(geometry_object.fluid_p2_coordinates-topology.X_f_gamma, copy=True))
    if current_normals is None and geometry_object is not None:
        current_normals = (geometry_object.n_s, geometry_object.n_f)
    origin = _finite_vector(moment_origin, 2, "moment_origin")
    explicit = _explicit_scales(scales)
    n_solid = topology.n_solid_interface_nodes
    n_fluid = topology.n_fluid_interface_nodes
    n_edges = topology.n_edges
    skipped = {}
    failures = []

    compact = None
    fluid = None
    if loads is not None:
        if isinstance(loads, Mapping):
            unknown = sorted(set(loads) - {"compact", "fluid"})
            if unknown:
                raise ValueError(f"Unknown loads keys {unknown}; allowed keys are ['compact', 'fluid'].")
            if "compact" in loads:
                compact = _finite_vector(loads["compact"], 2 * n_solid, "loads['compact']")
            if "fluid" in loads:
                fluid = _finite_vector(loads["fluid"], 2 * n_fluid, "loads['fluid']")
        else:
            compact = _finite_vector(loads, 2 * n_solid, "loads")

    vertex_values = None
    p2_values = None
    if motion is not None:
        if not isinstance(motion, InterfaceMotion):
            raise TypeError("motion must be an InterfaceMotion.")
        if motion.topology_id != topology.topology_id:
            raise ValueError("motion was built from a different topology.")
        vertex_values = _finite_matrix(motion.vertex_values, n_solid, "motion.vertex_values")
        p2_values = _finite_matrix(motion.p2_values, n_fluid, "motion.p2_values")
    configuration = "current" if motion is not None else "reference"

    X_s = np.asarray(topology.X_s_gamma, dtype=float)
    X_f = np.asarray(topology.X_f_gamma, dtype=float)
    # 映射自洽：h 每行和应为 1（分区一致性），且 h @ X_s 应重现 X_f（参照坐标匹配）。
    row_sum_error = _max_abs(np.asarray(topology.h.sum(axis=1)).ravel() - 1.0)
    reference_scale = max(float(np.linalg.norm(X_s, axis=1).max()),
                          float(np.linalg.norm(X_f, axis=1).max()))
    reference_tolerance = topology.atol + topology.rtol * reference_scale
    with np.errstate(over="ignore", invalid="ignore"):
        reference_error = float(np.max(np.linalg.norm(X_f - np.asarray(topology.h @ X_s), axis=1)))
    position_error = position_scale = position_tolerance = None
    position_fallback = None
    if motion is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            # 当前构形位置一致性：流体 P2 当前坐标应等于 h 映射后的固体当前坐标。
            current_s = X_s + vertex_values
            predicted_f = np.asarray(topology.h @ current_s)
            current_f = X_f + p2_values
            position_error = float(np.max(np.linalg.norm(current_f - predicted_f, axis=1)))
            position_scale, position_fallback = _effective_scale(
                explicit.get("length"),
                [float(np.linalg.norm(current_f, axis=1).max()),
                 float(np.linalg.norm(predicted_f, axis=1).max())])
            position_tolerance = _TOLERANCE_FACTOR * position_scale
    mapping_passed = (np.isfinite(row_sum_error) and row_sum_error <= _TOLERANCE_FACTOR
                      and np.isfinite(reference_error) and reference_error <= reference_tolerance)
    if motion is not None:
        mapping_passed = mapping_passed and np.isfinite(position_error) and position_error <= position_tolerance
    mapping = {"row_sum_error": row_sum_error, "row_sum_tolerance": _TOLERANCE_FACTOR,
               "reference_match_error": reference_error, "reference_match_tolerance": reference_tolerance,
               "position_error": position_error, "position_scale": position_scale,
               "position_tolerance": position_tolerance, "position_scale_fallback": position_fallback,
               "passed": bool(mapping_passed)}
    if not mapping_passed:
        failures.append("mapping: " + ", ".join(
            f"{name}={mapping[name]!r}" for name in
            ("row_sum_error", "reference_match_error", "position_error") if mapping[name] is not None))

    total_force = None
    full = None
    if compact is None:
        skipped["total_force"] = "no compact solid interface loads provided"
        skipped["total_moment"] = "no compact solid interface loads provided"
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            # 力守恒：压缩固体界面力之和应与散布后的完整向量之和一致；
            # 若给出流体力/解析力，则再与其总和比较。
            loads_s = compact.reshape(-1, 2)
            force_compact = loads_s.sum(axis=0)
            full = scatter_solid_load(topology, compact)
            force_full = full.reshape(-1, 2).sum(axis=0)
            full_error = float(np.max(np.abs(force_compact - force_full)))
            force_fluid = fluid_error = force_analytic = analytic_error = None
            contributions = [np.linalg.norm(force_compact), np.linalg.norm(force_full)]
            if fluid is not None:
                force_fluid = fluid.reshape(-1, 2).sum(axis=0)
                fluid_error = float(np.max(np.abs(force_compact - force_fluid)))
                contributions.append(np.linalg.norm(force_fluid))
            if analytic_force is not None:
                force_analytic = _finite_vector(analytic_force, 2, "analytic_force")
                analytic_error = float(np.max(np.abs(force_compact - force_analytic)))
                contributions.append(np.linalg.norm(force_analytic))
            force_scale, force_fallback = _effective_scale(explicit.get("force"), contributions)
            force_tolerance = _TOLERANCE_FACTOR * force_scale
            force_errors = [full_error] + [error for error in (fluid_error, analytic_error) if error is not None]
            force_passed = (all(np.isfinite(error) for error in force_errors)
                            and all(error <= force_tolerance for error in force_errors))
        total_force = {"compact": force_compact, "full": force_full, "fluid": force_fluid,
                       "analytic": force_analytic, "full_error": full_error, "fluid_error": fluid_error,
                       "analytic_error": analytic_error, "scale": force_scale, "tolerance": force_tolerance,
                       "scale_fallback": force_fallback, "passed": bool(force_passed)}
        if not force_passed:
            failures.append("total_force: " + ", ".join(
                f"{name}={total_force[name]!r}" for name in
                ("full_error", "fluid_error", "analytic_error") if total_force[name] is not None)
                + f", tolerance={force_tolerance!r}")

    total_moment = None
    if compact is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            loads_s = compact.reshape(-1, 2)
            coordinates_s = X_s + vertex_values if motion is not None else X_s
            # 固体侧力矩：以 origin 为参考点，r × F；再用完整网格坐标重算作对照。
            moment_solid = _cross_sum(coordinates_s - origin, loads_s)
            contributions = [_contribution_sum(coordinates_s - origin, loads_s)]
            coordinates_full = np.asarray(topology.solid_mesh.points, dtype=float).copy()
            coordinates_full[topology.solid_nodes] = coordinates_s
            loads_full = full.reshape(-1, 2)
            moment_full = _cross_sum(coordinates_full - origin, loads_full)
            contributions.append(_contribution_sum(coordinates_full - origin, loads_full))
            full_moment_error = abs(moment_solid - moment_full)
            moment_fluid = fluid_moment_error = None
            if fluid is not None:
                coordinates_f = X_f + p2_values if motion is not None else X_f
                loads_f = fluid.reshape(-1, 2)
                moment_fluid = _cross_sum(coordinates_f - origin, loads_f)
                fluid_moment_error = abs(moment_solid - moment_fluid)
                contributions.append(_contribution_sum(coordinates_f - origin, loads_f))
            moment_scale, moment_fallback = _effective_scale(
                explicit.get("moment"),
                [value for value in (moment_solid, moment_full, moment_fluid) if value is not None])
            moment_tolerance = _TOLERANCE_FACTOR * moment_scale
            moment_errors = [full_moment_error] + [error for error in (fluid_moment_error,) if error is not None]
            moment_passed = (all(np.isfinite(error) for error in moment_errors)
                             and all(error <= moment_tolerance for error in moment_errors)
                             and all(np.isfinite(value) for value in (moment_solid, moment_full, moment_fluid)
                                     if value is not None))
        total_moment = {"origin": origin, "solid": moment_solid, "full": moment_full, "fluid": moment_fluid,
                        "full_error": full_moment_error, "fluid_error": fluid_moment_error,
                        "scale": moment_scale, "tolerance": moment_tolerance,
                        "scale_fallback": moment_fallback, "passed": bool(moment_passed)}
        if not moment_passed:
            failures.append("total_moment: " + ", ".join(
                f"{name}={total_moment[name]!r}" for name in
                ("full_error", "fluid_error") if total_moment[name] is not None)
                + f", tolerance={moment_tolerance!r}")

    virtual_work = None
    if virtual_solid is None:
        skipped["virtual_work"] = "no virtual displacement provided"
    else:
        delta_solid = _finite_vector(virtual_solid, 2 * n_solid, "virtual_solid")
        # 固体虚位移经 H 映射为流体 P2 迹虚位移，用于两侧虚功对比。
        delta_fluid = np.asarray(topology.H @ delta_solid)
        if virtual_fluid is not None:
            given_fluid = _finite_vector(virtual_fluid, 2 * n_fluid, "virtual_fluid")
            mapping_scale, mapping_fallback = _effective_scale(
                explicit.get("length"),
                [float(np.max(np.abs(given_fluid))), float(np.max(np.abs(delta_fluid)))])
            mapping_error = float(np.max(np.abs(given_fluid - delta_fluid)))
            mapping_tolerance = _TOLERANCE_FACTOR * mapping_scale
            delta_fluid = given_fluid
        else:
            mapping_error = mapping_tolerance = mapping_fallback = None
        if compact is None or fluid is None:
            skipped["virtual_work"] = "virtual work requires both compact and fluid loads"
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                work_solid = float(delta_solid @ compact)
                work_fluid = float(delta_fluid @ fluid)
                work_scale, work_fallback = _effective_scale(explicit.get("work"), [work_solid, work_fluid])
                work_tolerance = _TOLERANCE_FACTOR * work_scale
                work_error = abs(work_solid - work_fluid)
                work_passed = (np.isfinite(work_error) and work_error <= work_tolerance
                               and (mapping_error is None
                                    or (np.isfinite(mapping_error) and mapping_error <= mapping_tolerance)))
            virtual_work = {"solid": work_solid, "fluid": work_fluid, "error": work_error,
                            "scale": work_scale, "tolerance": work_tolerance,
                            "scale_fallback": work_fallback, "mapping_error": mapping_error,
                            "mapping_tolerance": mapping_tolerance,
                            "virtual_mapping_fallback": mapping_fallback, "passed": bool(work_passed)}
            if not work_passed:
                failures.append(f"virtual_work: error={work_error!r}, mapping_error={mapping_error!r}, "
                                f"tolerance={work_tolerance!r}")

    residual = None
    if motion_residual is None:
        skipped["motion_residual"] = "no (velocity, derivative) pair provided"
    else:
        if not isinstance(motion_residual, (tuple, list)) or len(motion_residual) != 2:
            raise ValueError("motion_residual must be a (velocity, derivative) pair.")
        velocity = _finite_vector(motion_residual[0], 2 * n_solid, "motion_residual[0]")
        derivative = _finite_vector(motion_residual[1], 2 * n_solid, "motion_residual[1]")
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(np.max(np.abs(velocity - derivative)))
            scale, fallback = _effective_scale(explicit.get("velocity"),
                                               [float(np.max(np.abs(velocity))),
                                                float(np.max(np.abs(derivative)))])
            tolerance = _TOLERANCE_FACTOR * scale
            passed = np.isfinite(value) and value <= tolerance
        residual = {"value": value, "scale": scale, "tolerance": tolerance, "scale_fallback": fallback,
                    "passed": bool(passed)}
        if not passed:
            failures.append(f"motion_residual: value={value!r}, tolerance={tolerance!r}")

    with np.errstate(over="ignore", invalid="ignore"):
        normals_s = np.asarray(topology.N_s, dtype=float)
        normals_f = np.asarray(topology.N_f, dtype=float)
        # 参照法向检查：两域外法向应相反 (n_s + n_f ≈ 0) 且各自单位化。
        normal_error = float(np.max(np.linalg.norm(normals_s + normals_f, axis=1)))
        normalization_error = float(max(
            np.max(np.abs(np.linalg.norm(normals_s, axis=1) - 1.0)),
            np.max(np.abs(np.linalg.norm(normals_f, axis=1) - 1.0))))
        current_error = current_normalization_error = None
        if current_normals is not None:
            if not isinstance(current_normals, (tuple, list)) or len(current_normals) != 2:
                raise ValueError("current_normals must be a (solid, fluid) pair of per-edge normal arrays.")
            current_s = _finite_matrix(current_normals[0], n_edges, "current_normals[0]")
            current_f = _finite_matrix(current_normals[1], n_edges, "current_normals[1]")
            current_error = float(np.max(np.linalg.norm(current_s + current_f, axis=1)))
            current_normalization_error = float(max(
                np.max(np.abs(np.linalg.norm(current_s, axis=1) - 1.0)),
                np.max(np.abs(np.linalg.norm(current_f, axis=1) - 1.0))))
        normals_passed = (np.isfinite(normal_error) and normal_error <= _TOLERANCE_FACTOR
                          and np.isfinite(normalization_error)
                          and normalization_error <= _TOLERANCE_FACTOR)
        if current_normals is not None:
            normals_passed = (normals_passed and np.isfinite(current_error)
                              and current_error <= _TOLERANCE_FACTOR
                              and np.isfinite(current_normalization_error)
                              and current_normalization_error <= _TOLERANCE_FACTOR)
    normals = {"reference_error": normal_error, "normalization_error": normalization_error,
               "current_error": current_error, "current_normalization_error": current_normalization_error,
               "tolerance": _TOLERANCE_FACTOR, "passed": bool(normals_passed)}
    if not normals_passed:
        failures.append(f"normals: reference_error={normal_error!r}, "
                        f"normalization_error={normalization_error!r}, current_error={current_error!r}, "
                        f"tolerance={_TOLERANCE_FACTOR!r}")

    geometry_block = None
    if geometry_object is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            # 两条面积比路径（当前长度比与 Nanson）应一致；几何块还检查 Nanson 法向
            # 误差、固流间隙与固体/流体最小雅可比为正。
            ratio_error = float(np.max(np.abs(geometry_object.j_Gamma_current
                                                  - geometry_object.j_Gamma_nanson)))
            ratio_scale, ratio_fallback = _effective_scale(
                None, [float(np.max(np.abs(geometry_object.j_Gamma_current))),
                       float(np.max(np.abs(geometry_object.j_Gamma_nanson)))])
            ratio_tolerance = _TOLERANCE_FACTOR*ratio_scale
            nanson_gap = float(geometry_object.nanson_normal_error)
            gap = float(geometry_object.gap_error)
            gap_tolerance = float(geometry_object.gap_tolerance)
            geometry_passed = (np.isfinite(ratio_error) and ratio_error <= ratio_tolerance
                               and np.isfinite(nanson_gap) and nanson_gap <= _TOLERANCE_FACTOR
                               and np.isfinite(gap) and gap <= gap_tolerance
                               and np.isfinite(geometry_object.minimum_jacobian)
                               and geometry_object.minimum_jacobian > 0
                               and np.isfinite(geometry_object.minimum_cell_determinant)
                               and geometry_object.minimum_cell_determinant > 0)
        geometry_block = {"geometry_id": geometry_object.geometry_id,
                          "candidate_id": geometry_object.candidate_id,
                          "j_Gamma_current": geometry_object.j_Gamma_current,
                          "j_Gamma_nanson": geometry_object.j_Gamma_nanson,
                          "surface_ratio_error": ratio_error, "surface_ratio_scale": ratio_scale,
                          "surface_ratio_tolerance": ratio_tolerance, "surface_ratio_fallback": ratio_fallback,
                          "nanson_normal_error": nanson_gap, "normal_tolerance": _TOLERANCE_FACTOR,
                          "gap_error": gap, "gap_tolerance": gap_tolerance,
                          "minimum_jacobian": geometry_object.minimum_jacobian,
                          "minimum_cell_determinant": geometry_object.minimum_cell_determinant,
                          "passed": bool(geometry_passed)}
        if not geometry_passed:
            failures.append(f"geometry: surface_ratio_error={ratio_error!r}, "
                            f"nanson_normal_error={nanson_gap!r}, gap_error={gap!r}, "
                            f"gap_tolerance={gap_tolerance!r}")

    traction = None
    if load_object is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            # 已装配面力的节点总量应与直接积分总量、流体力总量一致（力与力矩）。
            force_error = float(np.max(np.abs(load_object.total_force-load_object.total_force_quadrature)))
            force_fluid_error = float(np.max(np.abs(load_object.total_force-load_object.total_force_fluid)))
            moment_error = abs(load_object.total_moment-load_object.total_moment_quadrature)
            moment_fluid_error = abs(load_object.total_moment-load_object.total_moment_fluid)
            force_scale, force_fallback = _effective_scale(
                explicit.get("force"), [float(np.linalg.norm(load_object.total_force)),
                                        float(np.linalg.norm(load_object.total_force_quadrature)),
                                        float(np.linalg.norm(load_object.total_force_fluid))])
            force_tolerance = _TOLERANCE_FACTOR*force_scale
            moment_scale, moment_fallback = _effective_scale(
                explicit.get("moment"), [abs(load_object.total_moment), abs(load_object.total_moment_quadrature),
                                         abs(load_object.total_moment_fluid)])
            moment_tolerance = _TOLERANCE_FACTOR*moment_scale
            traction_passed = (all(np.isfinite(value) for value in
                                   (force_error, force_fluid_error, moment_error, moment_fluid_error))
                               and force_error <= force_tolerance and force_fluid_error <= force_tolerance
                               and moment_error <= moment_tolerance and moment_fluid_error <= moment_tolerance)
        traction = {"geometry_id": load_object.geometry_id, "thickness": load_object.thickness,
                    "path": load_object.path, "moment_origin": load_object.moment_origin,
                    "total_force": load_object.total_force, "total_force_fluid": load_object.total_force_fluid,
                    "total_force_quadrature": load_object.total_force_quadrature,
                    "total_moment": load_object.total_moment, "total_moment_fluid": load_object.total_moment_fluid,
                    "total_moment_quadrature": load_object.total_moment_quadrature,
                    "force_quadrature_error": force_error, "force_fluid_error": force_fluid_error,
                    "moment_quadrature_error": moment_error, "moment_fluid_error": moment_fluid_error,
                    "force_scale": force_scale, "force_tolerance": force_tolerance,
                    "force_scale_fallback": force_fallback, "moment_scale": moment_scale,
                    "moment_tolerance": moment_tolerance, "moment_scale_fallback": moment_fallback,
                    "passed": bool(traction_passed)}
        if not traction_passed:
            failures.append(f"traction: force_quadrature_error={force_error!r}, "
                            f"force_fluid_error={force_fluid_error!r}, moment_quadrature_error={moment_error!r}, "
                            f"force_tolerance={force_tolerance!r}, moment_tolerance={moment_tolerance!r}")

    return {"topology_id": topology.topology_id, "configuration": configuration, "moment_origin": origin,
            "mapping": mapping, "total_force": total_force, "total_moment": total_moment,
            "virtual_work": virtual_work, "motion_residual": residual, "normals": normals,
            "geometry": geometry_block, "traction": traction,
            "skipped": skipped, "failures": failures, "pass": not failures}

"""Discrete GCL swept transport between adjacent P1 configurations (T14c).

本模块在相邻的两个 P1 网格构型之间构造离散几何守恒律（GCL）扫掠输运，用于 ALE 网格
运动对守恒方程的贡献。关键约定：
- 扫掠通量 Q_BDF 为各 BDF 区间上 cof(F(s))(x_a - x_b) 对时间积分的加权组合；
- 几何源 g_BDF 为 BDF 体积比（det F）的组合；两者都在参考域上表示；
- 提供 GCL 诊断量：逐单元缺陷、内部边通量缺陷与总体积收支缺陷，用于判断离散守恒性；
- 约束正体积：沿直线插值路径要求 det(F(s)) > 0（可选下界），退化则拒绝候选。
"""

from dataclasses import dataclass

import numpy as np

from fsi_ice.ale.geometry import ALEGeometry
from fsi_ice.fem.p1 import physical_gradients, shape_values
from fsi_ice.fem.quadrature import triangle_quadrature
from fsi_ice.mesh import TriMesh

# 单元三条局部边的顶点遍历顺序（逆时针 CCW）。
LOCAL_EDGES = ((0, 1), (1, 2), (2, 0))  # CCW cell traversal; outward normal is rot_cw.


@dataclass(frozen=True)
class GCLTransport:
    """Frozen swept transport Q_BDF, geometry source g_BDF and GCL diagnostics.

    冻结的扫掠输运数据：参考网格、候选与已接受构型、BDF 阶数/步长/系数、扫掠通量 flux、
    几何源 divergence、逐单元/逐边通量与 GCL 缺陷（cell_defects/edge_defect/budget_defect）
    以及 w_gcl 等。所有数组均只读。
    """

    reference_mesh: TriMesh
    candidate_geometry: ALEGeometry
    accepted_geometries: tuple
    order: int
    dt: float
    step_id: int
    interval_weights: tuple
    coefficients: np.ndarray
    times: np.ndarray
    edges: np.ndarray
    exterior_mask: np.ndarray
    quadrature_points: np.ndarray
    quadrature_weights: np.ndarray
    flux: np.ndarray
    divergence: np.ndarray
    edge_fluxes: np.ndarray
    cell_boundary_fluxes: np.ndarray
    w_gcl: np.ndarray
    cell_defects: np.ndarray
    edge_defect: float
    budget_defect: float
    candidate_id: object


def _freeze(array):
    # 复制并设为只读，避免调用方意外修改冻结的输运数据。
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _cofactor(matrices):
    """det(F) F^-1 for (E, 2, 2) deformation gradients (no numerical inversion).

    2x2 变形梯度的余因子矩阵 cof(F) = det(F) F^{-1}，直接由分量写出以避免数值求逆。
    输入 (E, 2, 2)，输出同形状。
    """
    result = np.empty_like(matrices)
    result[:, 0, 0] = matrices[:, 1, 1]
    result[:, 0, 1] = -matrices[:, 0, 1]
    result[:, 1, 0] = -matrices[:, 1, 0]
    result[:, 1, 1] = matrices[:, 0, 0]
    return result


def _interval_flux(reference_mesh, newer, older, dt):
    """Nodal Q[a,b] = (1/dt) int_0^1 cof(F(s)) (x_a-x_b) ds on reference cells.

    F(s) and the integrand are affine in s for two P1 configurations, so the
    s=1/2 midpoint value integrates exactly.

    中文补充：节点扫掠通量 Q[a,b] = (1/dt) ∫_0^1 cof(F(s))(x_a - x_b) ds（参考单元上）。
    对两个 P1 构型，F(s) 与被积函数关于 s 仿射，故取 s=1/2 的中点值即可精确积分。
    """
    deformation_midpoint = 0.5*(newer.deformation_gradients + older.deformation_gradients)
    cofactors = _cofactor(deformation_midpoint)
    delta = (newer.displacement - older.displacement)[reference_mesh.cells]
    # 中点求积：(1/dt) cof(F(1/2)) (x_a - x_b)，对 P1 构型精确。
    return np.einsum("eij,enj->eni", cofactors, delta)/dt


def _check_path_health(reference_mesh, newer, older, interval, path_volume_ratio_min, candidate_id):
    """Require det(F(s)) > 0 (and >= the optional bound) along the whole straight path.

    要求沿整个直线插值路径 det(F(s)) > 0（以及可选下界 path_volume_ratio_min）。由于
    det(F(s)) 是关于 s 的二次函数，用端点值与二次项系数解析求最小值；下溢或低于下界即
    抛出 ValueError 并给出退化单元下标。返回逐单元的最小 det(F(s))。
    """
    newer_gradients = newer.deformation_gradients
    older_gradients = older.deformation_gradients
    delta = newer_gradients - older_gradients
    # det(F(s)) = quadratic*s^2 + linear*s + constant，其中 quadratic 为 dJ 的行列式项。
    quadratic = delta[:, 0, 0]*delta[:, 1, 1] - delta[:, 0, 1]*delta[:, 1, 0]
    constant = older.volume_ratios
    linear = newer.volume_ratios - constant - quadratic
    minimum = np.minimum(constant, newer.volume_ratios)
    positive_quadratic = quadratic > 0
    if np.any(positive_quadratic):
        stationary = np.divide(-linear, 2*quadratic, out=np.full(len(constant), np.inf), where=positive_quadratic)
        turning = positive_quadratic & (stationary > 0) & (stationary < 1)
        safe = np.where(positive_quadratic, quadratic, 1.0)
        vertex = constant - linear*linear/(4*safe)
        minimum = np.where(turning, np.minimum(minimum, vertex), minimum)
    underflow = np.flatnonzero(minimum <= 0)
    if len(underflow):
        index = int(underflow[0])
        raise ValueError(
            f"candidate {candidate_id!r}: swept path interval {interval} cell {index} has "
            f"min det(F(s))={minimum[index]:.12g} <= 0; the straight interpolation degenerates."
        )
    if path_volume_ratio_min is not None:
        below = np.flatnonzero(minimum < path_volume_ratio_min)
        if len(below):
            index = int(below[0])
            raise ValueError(
                f"candidate {candidate_id!r}: swept path interval {interval} cell {index} has "
                f"min det(F(s))={minimum[index]:.12g} < path_volume_ratio_min={path_volume_ratio_min:.12g}."
            )
    return minimum


def build_gcl_transport(reference_mesh, candidate_geometry, accepted_geometries, stencil, *,
                        path_volume_ratio_min=None):
    """Assemble the BDF swept flux and its GCL defects; inputs stay untouched.

    组装 BDF 扫掠通量与几何守恒律（GCL）缺陷，且不修改任何输入。流程：校验参数一致性 →
    沿各相邻构型区间检查正体积（_check_path_health）→ 加权累加区间扫掠通量得到节点通量 →
    在参考域上求几何源 g_BDF 与逐单元 GCL 缺陷 → 按边汇总通量、统计内部边缺陷与总体积收支缺陷。
    返回 GCLTransport（所有数组只读）。
    """
    if not isinstance(reference_mesh, TriMesh):
        raise ValueError("reference_mesh must be a TriMesh.")
    if path_volume_ratio_min is not None:
        if isinstance(path_volume_ratio_min, bool) or not np.isfinite(path_volume_ratio_min) or path_volume_ratio_min <= 0:
            raise ValueError("path_volume_ratio_min must be a finite positive number or None.")
    if not isinstance(candidate_geometry, ALEGeometry) or candidate_geometry.reference_mesh is not reference_mesh:
        raise ValueError("candidate_geometry must be an ALEGeometry built on reference_mesh.")
    accepted = tuple(accepted_geometries)
    for geometry in accepted:
        if not isinstance(geometry, ALEGeometry) or geometry.reference_mesh is not reference_mesh:
            raise ValueError("Every accepted geometry must be an ALEGeometry built on the same reference_mesh.")
    order = getattr(stencil, "order", None)
    if isinstance(order, bool) or order not in (1, 2):
        raise ValueError("stencil must provide an integer order of 1 or 2.")
    if len(accepted) != order:
        raise ValueError(
            f"accepted_geometries must contain stencil.order={order} configurations, got {len(accepted)}."
        )
    try:
        dt = float(stencil.dt)
    except (TypeError, ValueError):
        raise ValueError("stencil.dt must be a finite positive number.") from None
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("stencil.dt must be a finite positive number.")
    coefficients = np.asarray(stencil.coefficients, dtype=float)
    if coefficients.shape != (order+1,) or not np.all(np.isfinite(coefficients)):
        raise ValueError("stencil.coefficients must be finite and have shape (order+1,).")
    interval_weights = np.asarray(stencil.interval_weights, dtype=float)
    if interval_weights.shape != (order,) or not np.all(np.isfinite(interval_weights)):
        raise ValueError("stencil.interval_weights must be finite and have shape (order,).")

    configurations = (candidate_geometry,) + accepted
    # 相邻构型两两组成 BDF 时间区间：候选与最近一次已接受、已接受之间依此类推。
    intervals = tuple(zip(configurations[:-1], configurations[1:]))
    for interval, (newer, older) in enumerate(intervals):
        _check_path_health(reference_mesh, newer, older, interval, path_volume_ratio_min,
                           candidate_geometry.candidate_id)
    nodal_flux = np.zeros(reference_mesh.cells.shape + (reference_mesh.points.shape[1],))
    # 按 interval_weights 加权累加各区间扫掠通量，得到 BDF 组合的节点扫掠通量。
    for weight, (newer, older) in zip(interval_weights, intervals):
        nodal_flux += weight*_interval_flux(reference_mesh, newer, older, dt)

    points, weights = triangle_quadrature(5)
    flux = np.einsum("qa,eai->eqi", shape_values(points), nodal_flux)

    ratios = [candidate_geometry.volume_ratios, *(geometry.volume_ratios for geometry in accepted)]
    # 几何源 g_BDF = sum_j a_j det(F_j)（各构型体积比的 BDF 组合）。
    divergence = coefficients[0]*ratios[0]
    for coefficient, ratio in zip(coefficients[1:], ratios[1:]):
        divergence = divergence + coefficient*ratio
    reference_gradients = physical_gradients(reference_mesh)
    # 由散度定理（节点通量在参考域上的散度）得到的体积变化，用其与 g_BDF 之差度量 GCL 缺陷。
    divergence_exact = np.einsum("eai,eai->e", nodal_flux, reference_gradients)
    cell_defects = np.abs(divergence - divergence_exact)

    cells = reference_mesh.cells
    local_edges = np.stack((cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]), axis=1)
    # 将三条局部边按顶点排序后归并为唯一无向边；counts==1 的边即外边界边。
    unique_edges, inverse, counts = np.unique(
        np.sort(local_edges, axis=2).reshape(-1, 2), axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.reshape(len(cells), 3)
    edge_fluxes = np.zeros(len(unique_edges))
    cell_boundary_fluxes = np.zeros(len(cells))
    reference_points = reference_mesh.points
    # Sign convention of edge_fluxes: for a sorted edge (min, max) the positive
    # normal is rot_cw(X_max - X_min), which is the outward normal of a CCW cell
    # traversing the edge from min to max. Each incident cell contributes its own
    # outward flux, so interior edges cancel and exterior edges sum to the
    # reference-domain BDF volume budget (divergence theorem per cell).
    #
    # 中文：edge_fluxes 的符号约定——对排序边 (min,max)，正法向为 rot_cw(X_max-X_min)，
    # 即 CCW 单元从 min 到 max 遍历该边时的外法向。每个相邻单元各自贡献外法向通量，
    # 于是内部边相互抵消，仅外边界边求和即得参考域上的 BDF 体积收支（逐单元散度定理）。
    for local, (start, end) in enumerate(LOCAL_EDGES):
        tangent = reference_points[cells[:, end]] - reference_points[cells[:, start]]
        outward = np.column_stack((tangent[:, 1], -tangent[:, 0]))  # rot_cw of CCW traversal
        local_flux = 0.5*np.einsum("ei,ei->e", nodal_flux[:, start, :] + nodal_flux[:, end, :], outward)
        cell_boundary_fluxes += local_flux
        np.add.at(edge_fluxes, inverse[:, local], local_flux)
    exterior_mask = counts == 1
    interior_fluxes = np.abs(edge_fluxes[~exterior_mask])
    edge_defect = float(interior_fluxes.max()) if interior_fluxes.size else 0.0

    # 把节点扫掠通量转换到当前构型上的网格速度：w_gcl = (F · flux) / det(F)。
    w_gcl = np.einsum(
        "eij,eqj->eqi", candidate_geometry.deformation_gradients, flux
    )/candidate_geometry.volume_ratios[:, None, None]

    # 注意：reference_mesh.determinants = 2 * 单元面积，故面积 = determinants/2。
    reference_areas = reference_mesh.determinants/2
    # 总体积收支缺陷 = |外边界扫掠通量之和 - ∫ g_BDF dA|。
    budget_defect = float(abs(edge_fluxes.sum() - np.sum(divergence*reference_areas)))

    return GCLTransport(
        reference_mesh=reference_mesh,
        candidate_geometry=candidate_geometry,
        accepted_geometries=accepted,
        order=order,
        dt=dt,
        step_id=getattr(stencil, "step_id", -1),
        interval_weights=tuple(float(weight) for weight in interval_weights),
        coefficients=_freeze(coefficients),
        times=_freeze(getattr(stencil, "times", np.arange(order + 1)[::-1]*dt)),
        edges=_freeze(unique_edges),
        exterior_mask=_freeze(exterior_mask),
        quadrature_points=_freeze(points),
        quadrature_weights=_freeze(weights),
        flux=_freeze(flux),
        divergence=_freeze(divergence),
        edge_fluxes=_freeze(edge_fluxes),
        cell_boundary_fluxes=_freeze(cell_boundary_fluxes),
        w_gcl=_freeze(w_gcl),
        cell_defects=_freeze(cell_defects),
        edge_defect=edge_defect,
        budget_defect=budget_defect,
        candidate_id=candidate_geometry.candidate_id,
    )

"""ALE flow context: geometry-consistent BDF masses, swept transport and Jacobian blocks.

本模块装配单个候选时间步的 ALE 流场上下文：冻结几何、网格速度、BDF 时间模板与
几何守恒律（GCL）扫掠输运，并据此给出 Navier-Stokes 残差与雅可比贡献。
关键约定：
- 对流项使用相对网格速度 (u - w_kin)；网格速度 w_kin 由 P1 位移的 BDF 时间导数得到；
- 时间离散为 BDF 质量项与历史载荷；候选步不推进任何历史，已接受构型保持只读；
- 网格运动通过 GCL 扫掠通量贡献，
  K_mesh = -rho ∫ N_a (Q_BDF·∇_X N_b + g_BDF N_b) dX（参考域上积分）；
- 压力为物理压力 p，动量方程符号约定见 taylor_hood 模块。
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.sparse import bmat, csr_matrix

from fsi_ice.ale.geometry import ALEGeometry
from fsi_ice.ale.gcl import GCLTransport
from fsi_ice.ale.kinematics import MeshVelocity, evaluate_mesh_velocity
from fsi_ice.fem.p1 import map_points, shape_values
from fsi_ice.fem.p2 import p2_physical_gradients, p2_reference_gradients, p2_values
from fsi_ice.fem.quadrature import triangle_quadrature
from fsi_ice.fluid.taylor_hood import assemble_velocity_mass, convection, scatter_matrix
from fsi_ice.solid.linear import vector_dofs, vector_values
from fsi_ice.time import BDFStencil


@dataclass(frozen=True)
class ALEFlowContext:
    """Candidate ALE geometry, mesh velocity, swept transport and BDF template.

    一个候选时间步所需的全部冻结 ALE 输入：候选几何、网格速度、扫掠输运和 BDF 模板；
    history_velocities 为已接受的 P2 速度历史（只读）。所有字段均为冻结后的只读数据。
    """

    geometry: ALEGeometry
    mesh_velocity: MeshVelocity
    transport: GCLTransport
    stencil: BDFStencil
    history_velocities: tuple

    @cached_property
    def _history_loads(self):
        # Owned by one immutable candidate context, keyed by physical density.
        return {}

    @cached_property
    def _transport_matrices(self):
        return {}


@dataclass(frozen=True)
class _MeshDisplacementDerivative:
    """Fixed-history adapter so evaluate_mesh_velocity can build w_kin from the stencil.

    适配器：把形如 a_0 d_m + term 的 BDF 位移时间导数包装成 derivative(candidate)，
    使 evaluate_mesh_velocity 复用同一接口而无需推进历史。coefficient 为 a_0，term 为已接受项之和。
    """

    coefficient: float
    term: np.ndarray

    def derivative(self, candidate):
        candidate = np.asarray(candidate, dtype=float)
        return self.coefficient*candidate + self.term


def _stencil_from_transport(transport):
    # 由 GCL 传输对象重建同参数的 BDF 模板（阶数、步长、步号、系数与时间点）。
    return BDFStencil(transport.order, transport.dt, transport.step_id, transport.coefficients, transport.times)


def build_ale_flow_context(operators, geometry, transport, history_velocities):
    """Freeze the ALE inputs for one candidate step without advancing any history.

    冻结一个候选步的 ALE 输入：校验几何/输运/算子网格的一致性，复制并只读化历史速度，
    组装网格位移时间导数的已接受项，再求网格速度。不修改任何已接受历史。
    输入：operators（FlowOperators）、geometry（候选 ALEGeometry）、transport（对应 GCLTransport）、
    history_velocities（stencil.order 个 (n_P2, 2) 已接受 P2 速度）。输出：ALEFlowContext。
    """
    if not isinstance(geometry, ALEGeometry):
        raise ValueError("geometry must be an ALEGeometry.")
    if not isinstance(transport, GCLTransport) or transport.candidate_geometry is not geometry:
        raise ValueError("transport must be the GCLTransport built for this candidate geometry.")
    if operators.space.mesh is not geometry.current_mesh:
        raise ValueError("operators must be assembled on geometry.current_mesh.")
    stencil = _stencil_from_transport(transport)
    histories = tuple(history_velocities)
    if len(histories) != stencil.order:
        raise ValueError(f"history_velocities must contain stencil.order={stencil.order} accepted P2 velocities.")
    frozen = []
    for velocity in histories:
        values = np.array(velocity, dtype=float, copy=True)
        if values.shape != (len(operators.space.points), 2) or not np.all(np.isfinite(values)):
            raise ValueError("Accepted velocities must be finite (n_P2, 2) arrays.")
        values.setflags(write=False)
        frozen.append(values)
    # 已接受构型对位移 BDF 时间导数的贡献 sum_{j>=1} a_j d_m^j；候选自身的 a_0 d_m 另行提供。
    term = np.zeros(geometry.displacement.shape)
    for coefficient, accepted in zip(stencil.coefficients[1:], transport.accepted_geometries):
        term += coefficient*accepted.displacement
    # w_kin = D_BDF d_m = a_0 d_m + term，只依赖冻结历史。
    mesh_velocity = evaluate_mesh_velocity(
        geometry, _MeshDisplacementDerivative(stencil.coefficients[0], term)
    )
    return ALEFlowContext(geometry, mesh_velocity, transport, stencil, tuple(frozen))


def validate_ale_velocity_context(ale_context, mesh_context):
    """Require the velocity BDF context and the ALE stencil/histories to describe one step.

    校验速度 BDF 上下文与 ALE 模板/历史描述同一步：阶数、目标时刻、候选权重 a_0
    以及历史项 sum a_j u_j 都必须一致，否则抛出 ValueError。
    """
    stencil = ale_context.stencil
    if mesh_context.order != stencil.order:
        raise ValueError("Velocity BDF context and ALE stencil must share the same order.")
    if not np.isclose(mesh_context.time, stencil.times[0], rtol=0.0, atol=1e-14):
        raise ValueError("Velocity BDF context and ALE stencil must target the same time.")
    if not np.isclose(mesh_context.coefficient, stencil.coefficients[0], rtol=1e-13, atol=0.0):
        raise ValueError("Velocity BDF coefficient and ALE stencil candidate weight must agree.")
    combined = np.zeros(mesh_context.history_term.shape)
    for coefficient, velocity in zip(stencil.coefficients[1:], ale_context.history_velocities):
        combined += coefficient*velocity
    if not np.allclose(mesh_context.history_term, combined, rtol=1e-9, atol=1e-12):
        raise ValueError("Velocity BDF history term does not match the ALE history velocities.")


def history_mass_load(operators, ale_context):
    """sum_j a_j M_j u_j over accepted configurations (candidate handled separately).

    历史质量载荷：对已接受构型逐个组装 P2 速度质量矩阵 M_j（各构型网格不同），
    累加 sum_{j>=1} a_j M_j u_j；候选构型的 a_0 M_0 u_0 由残差主项单独处理。
    """
    if ale_context.geometry.current_mesh is not operators.space.mesh:
        raise ValueError("ALE history cache requires its own candidate operators.")
    key = float(operators.density)
    if key in ale_context._history_loads:
        return ale_context._history_loads[key]
    load = np.zeros(operators.n_velocity)
    coefficients = ale_context.stencil.coefficients
    for coefficient, geometry, velocity in zip(
        coefficients[1:], ale_context.transport.accepted_geometries, ale_context.history_velocities
    ):
        mass = assemble_velocity_mass(geometry.current_mesh, density=operators.density)
        load += coefficient*(mass @ velocity.ravel())
    load.setflags(write=False)
    ale_context._history_loads[key] = load
    return load


def mesh_transport_matrix(operators, transport):
    """K_mesh = -rho int_X N_a [Q_BDF.grad_X N_b + g_BDF N_b] dX on the reference domain.

    网格运动带来的输运通量矩阵：在参考域上积分，用扫掠通量 Q_BDF 与几何源 g_BDF
    构造 P2 速度块。要求 transport 的候选网格与 operators 当前网格一致。返回 CSR 矩阵。
    """
    mesh = operators.space.mesh
    if transport.candidate_geometry.current_mesh is not mesh:
        raise ValueError("The swept transport does not belong to the operators' current mesh.")
    reference_mesh = transport.reference_mesh
    points, weights = triangle_quadrature(5)
    shapes = p2_values(points)
    reference_gradients = p2_physical_gradients(reference_mesh, points)
    advected = np.einsum("eqj,eqbj->eqb", transport.flux, reference_gradients)
    # 扫掠输运核 kernel = Q_BDF·∇_X N_b + g_BDF N_b：前者来自网格扫掠通量，后者为 GCL 体积源。
    kernel = advected + transport.divergence[:, None, None]*shapes
    local_scalar = -operators.density*np.einsum(
        "e,q,qa,eqb->eab", reference_mesh.determinants, weights, shapes, kernel
    )
    local = np.einsum("eab,ij->eaibj", local_scalar, np.eye(2)).reshape(mesh.n_cells, 12, 12)
    dofs = vector_dofs(operators.space.cells)
    return scatter_matrix(local, dofs, dofs, (operators.n_velocity, operators.n_velocity))


def ale_navier_stokes_residual(operators, values, ale_context, *, with_jacobian=True):
    """Full ALE residual and Jacobian; the fixed-grid branch of taylor_hood stays separate.

    组装完整 ALE 残差与雅可比（不动网格分支仍由 taylor_hood 处理）。残差 = 定常项
    (matrix·values - rhs) + 对流 + 历史质量载荷 + a_0 M u + K_mesh u；雅可比在动量块上
    追加对流、a_0 M 与 K_mesh，压力块保持不变。values 为 (n_velocity + n_pressure) 混合向量。
    """
    if ale_context.geometry.current_mesh is not operators.space.mesh:
        raise ValueError("ale_context geometry must be the operators' current mesh.")
    n_velocity = operators.n_velocity
    velocity = values[:n_velocity].reshape(-1, 2)
    residual = operators.matrix @ values - operators.rhs
    convective, convective_jacobian = convection(operators, velocity, with_jacobian=with_jacobian)
    residual[:n_velocity] += convective
    residual[:n_velocity] += history_mass_load(operators, ale_context)
    coefficient = float(ale_context.stencil.coefficients[0])
    residual[:n_velocity] += coefficient*(operators.mass @ values[:n_velocity])
    key = float(operators.density)
    if key not in ale_context._transport_matrices:
        matrix = mesh_transport_matrix(operators, ale_context.transport)
        matrix.data.setflags(write=False)
        matrix.indices.setflags(write=False)
        matrix.indptr.setflags(write=False)
        ale_context._transport_matrices[key] = matrix
    transport_matrix = ale_context._transport_matrices[key]
    residual[:n_velocity] += transport_matrix @ values[:n_velocity]
    if not with_jacobian:
        return residual, None
    momentum = convective_jacobian + coefficient*operators.mass + transport_matrix
    correction = bmat([[momentum, None], [None, csr_matrix((operators.space.mesh.n_nodes,)*2)]], format="csr")
    return residual, operators.matrix + correction


def ale_linear_signature(operators, ale_context, dofs, prescribed):
    """Geometry/constraint signature for ReusedLUSolver invalidation.

    几何/约束签名，供 ReusedLUSolver 判断是否需要重新分解：包含候选几何 id、步号、
    黏度、密度以及约束自由度与约束值的字节表示。返回可比较的元组。
    """
    return (
        repr(ale_context.geometry.candidate_id),
        int(ale_context.stencil.step_id),
        float(operators.viscosity),
        float(operators.density),
        np.asarray(dofs, dtype=np.int64).tobytes(),
        np.asarray(prescribed, dtype=float).tobytes(),
    )


BODY_FORCE_DIFFERENCE_STEP = 1e-6  # spatial central-difference step for callable body forces
# 可调用体积力的空间中心差分步长 h：仅在 ale_geometry_jvp 中对 f(x ± h δx) 做近似切向。


def _cofactor(matrices):
    """det(A) A^-1 for (n, 2, 2) matrices (linear in its argument).

    2x2 矩阵的余因子矩阵 det(A) A^{-1}（对参数线性），用于在无显式求逆的情况下
    得到 det 的变分。输入 (n, 2, 2)，输出同形状。
    """
    result = np.empty_like(matrices)
    result[:, 0, 0] = matrices[:, 1, 1]
    result[:, 0, 1] = -matrices[:, 0, 1]
    result[:, 1, 0] = -matrices[:, 1, 0]
    result[:, 1, 1] = matrices[:, 0, 0]
    return result


def _strain_basis(gradients):
    # 由 P2 速度梯度构造对称梯度（应变）基：3 行对应 [eps_xx, eps_yy, 2 eps_xy]，列按速度分量交错排布。
    strains = np.zeros((gradients.shape[0], gradients.shape[1], 3, 12))
    strains[:, :, 0, 0::2] = gradients[:, :, :, 0]
    strains[:, :, 1, 1::2] = gradients[:, :, :, 1]
    strains[:, :, 2, 0::2] = gradients[:, :, :, 1]
    strains[:, :, 2, 1::2] = gradients[:, :, :, 0]
    return strains


def _divergence_basis(gradients):
    # 由 P2 速度梯度构造散度基 div(N_a)：列与 12 个速度自由度一一对应。
    basis = np.empty((gradients.shape[0], gradients.shape[1], 12))
    basis[:, :, 0::2] = gradients[:, :, :, 0]
    basis[:, :, 1::2] = gradients[:, :, :, 1]
    return basis


def _transport_response_matrix(operators, reference_mesh, flux, divergence):
    """K(Q, g) with the same reference-domain formula as mesh_transport_matrix.

    与 mesh_transport_matrix 使用同一参考域公式，但传入给定的扫掠通量 flux 与几何源
    divergence，用于几何 JVP 中的方向导数装配。
    """
    mesh = operators.space.mesh
    points, weights = triangle_quadrature(5)
    shapes = p2_values(points)
    reference_gradients = p2_physical_gradients(reference_mesh, points)
    advected = np.einsum("eqj,eqbj->eqb", flux, reference_gradients)
    kernel = advected + divergence[:, None, None]*shapes
    local_scalar = -operators.density*np.einsum(
        "e,q,qa,eqb->eab", reference_mesh.determinants, weights, shapes, kernel
    )
    local = np.einsum("eab,ij->eaibj", local_scalar, np.eye(2)).reshape(mesh.n_cells, 12, 12)
    dofs = vector_dofs(operators.space.cells)
    return scatter_matrix(local, dofs, dofs, (operators.n_velocity, operators.n_velocity))


def _transport_directional(geometry, transport, delta_jacobian, delta, points):
    """Nodal d(Q_BDF) and d(g_BDF) of the candidate; accepted configurations are fixed.

    候选构型扫掠通量 Q_BDF 与几何源 g_BDF 的方向导数，已接受构型固定。输入候选几何、
    输运、雅可比增量 delta_jacobian、位移增量 delta 与求积点；返回 (delta_flux, delta_source)。
    """
    reference_mesh = geometry.reference_mesh
    inverse_reference = np.linalg.inv(reference_mesh.jacobians)
    delta_deformation = np.einsum("eij,ejk->eik", delta_jacobian, inverse_reference)
    delta_determinant = np.einsum(
        "eij,eji->e", _cofactor(geometry.deformation_gradients), delta_deformation
    )
    accepted = transport.accepted_geometries[0]
    midpoint = 0.5*(geometry.deformation_gradients + accepted.deformation_gradients)
    cofactor_midpoint = _cofactor(midpoint)
    delta_cofactor_midpoint = _cofactor(0.5*delta_deformation)
    difference = (geometry.displacement - accepted.displacement)[geometry.current_mesh.cells]
    delta_difference = delta[geometry.current_mesh.cells]
    nodal = transport.interval_weights[0]*(
        np.einsum("eij,enj->eni", delta_cofactor_midpoint, difference)
        + np.einsum("eij,enj->eni", cofactor_midpoint, delta_difference)
    )/transport.dt
    flux = np.einsum("qa,eai->eqi", shape_values(points), nodal)
    source = transport.coefficients[0]*delta_determinant
    return flux, source


def ale_geometry_jvp(operators, values, *, ale_context, delta_d_m):
    """Directional derivative of the uneliminated ALE residual w.r.t. the candidate mesh displacement.

    Every geometry block is differentiated analytically: current J_x/det J_x and its
    inverse, P2 gradients, strains/divergence/mass, the physical convection tensor,
    the swept cofactors Q_BDF and source g_BDF of the candidate interval, and the
    K_mesh response assembled on the reference domain.  Accepted geometries and all
    history terms stay fixed.  The spatial evaluation of a callable body force has no
    derivative protocol in this version, so that single block uses central differences
    of f(x +/- h delta_x) with h = BODY_FORCE_DIFFERENCE_STEP and is not an analytic
    tangent for arbitrary callables.  Boundary values do not enter the uneliminated
    residual.  The accompanying mesh-velocity relation is delta_w_kin = a_0 delta_d_m.

    说明（中文）：本函数给出未消元 ALE 残差对候选网格位移 d_m 的方向导数（JVP），即
    残差几何块沿 delta_d_m 的解析切向。所有几何块均解析求导：当前 J_x/det J_x 及其逆、
    P2 梯度、应变/散度/质量、物理对流张量、候选区间的扫掠余因子 Q_BDF 与源 g_BDF，
    以及参考域上装配的 K_mesh 响应；已接受几何与所有历史项保持固定。可调用体积力的空间
    取值在本版本无导数协议，因此该单项用 f(x ± h δx) 的中心差分（h = BODY_FORCE_DIFFERENCE_STEP），
    对任意可调用函数并非严格解析切向。边界值不进入未消元残差。配套网格速度关系为
    δw_kin = a_0 δd_m。输入：operators、混合状态 values、ale_context、位移增量 delta_d_m；
    输出：与残差同尺寸的 ndarray。
    """
    geometry = ale_context.geometry
    transport = ale_context.transport
    mesh = operators.space.mesh
    if transport.candidate_geometry is not geometry or geometry.current_mesh is not mesh:
        raise ValueError("ale_context must describe the operators' current candidate geometry.")
    values = np.asarray(values, dtype=float)
    if values.shape != (operators.n_total,) or not np.all(np.isfinite(values)):
        raise ValueError("Mixed flow state must be a finite vector of the correct size.")
    delta = np.asarray(delta_d_m, dtype=float)
    if delta.shape != geometry.displacement.shape or not np.all(np.isfinite(delta)):
        raise ValueError(
            f"delta_d_m must have shape {geometry.displacement.shape} and be finite, got {delta.shape}."
        )
    reference_mesh = geometry.reference_mesh
    cells = mesh.cells
    n_velocity = operators.n_velocity

    delta_vertices = delta[cells]
    delta_jacobian = np.stack((delta_vertices[:, 1]-delta_vertices[:, 0],
                               delta_vertices[:, 2]-delta_vertices[:, 0]), axis=2)
    inverse_current = np.linalg.inv(mesh.jacobians)
    # d(A^{-1}) = -A^{-1} dA A^{-1}
    delta_inverse = -np.einsum("eij,ejk,ekl->eil", inverse_current, delta_jacobian, inverse_current)
    # d(det J) = cof(J) : dJ，其中 cof(J) = det(J) J^{-1}
    delta_determinant = np.einsum("eij,eji->e", _cofactor(mesh.jacobians), delta_jacobian)

    points, weights = triangle_quadrature(5)
    velocity_shape = p2_values(points)
    pressure_shape = shape_values(points)
    reference_gradients = p2_reference_gradients(points)
    delta_gradients = np.einsum("qai,eij->eqaj", reference_gradients, delta_inverse)
    gradients = operators.gradients
    determinant = mesh.determinants
    density = operators.density

    constitutive = operators.viscosity*np.diag([2.0, 2.0, 1.0])
    # 2D Voigt：constitutive 把 [eps_xx, eps_yy, 2 eps_xy] 映射到 [sigma_xx, sigma_yy, sigma_xy]。
    strains = _strain_basis(gradients)
    delta_strains = _strain_basis(delta_gradients)
    raw_stiffness = np.einsum("q,eqai,ab,eqbj->eij", weights, strains, constitutive, strains)
    delta_raw_stiffness = (
        np.einsum("q,eqai,ab,eqbj->eij", weights, delta_strains, constitutive, strains)
        + np.einsum("q,eqai,ab,eqbj->eij", weights, strains, constitutive, delta_strains)
    )
    delta_stiffness = (determinant[:, None, None]*delta_raw_stiffness
                       + delta_determinant[:, None, None]*raw_stiffness)

    raw_divergence = np.einsum("q,qp,eqv->epv", weights, pressure_shape, _divergence_basis(gradients))
    delta_raw_divergence = np.einsum(
        "q,qp,eqv->epv", weights, pressure_shape, _divergence_basis(delta_gradients)
    )
    delta_divergence = (determinant[:, None, None]*delta_raw_divergence
                        + delta_determinant[:, None, None]*raw_divergence)

    physical_points = map_points(mesh, points)
    # 物理求积点的位移增量：δx = δx_0 + ξ·δJ（ξ 为参考坐标）。
    delta_points = delta[cells[:, 0]][:, None, :] + np.einsum("qj,eij->eqi", points, delta_jacobian)
    body_force = getattr(operators, "body_force", (0.0, 0.0))
    flat_points = physical_points.reshape(-1, 2)
    if callable(body_force):
        flat_direction = delta_points.reshape(-1, 2)
        force = vector_values(body_force, flat_points, "Body force").reshape(physical_points.shape)
        plus = vector_values(body_force, flat_points+BODY_FORCE_DIFFERENCE_STEP*flat_direction,
                             "Body force")
        minus = vector_values(body_force, flat_points-BODY_FORCE_DIFFERENCE_STEP*flat_direction,
                              "Body force")
        delta_force = ((plus-minus)/(2*BODY_FORCE_DIFFERENCE_STEP)).reshape(physical_points.shape)
    else:
        force = vector_values(body_force, flat_points, "Body force").reshape(physical_points.shape)
        delta_force = np.zeros_like(force)
    raw_load = np.einsum("q,eqi,qa->eai", weights, force, velocity_shape)
    delta_raw_load = np.einsum("q,eqi,qa->eai", weights, delta_force, velocity_shape)
    delta_load = (determinant[:, None, None]*delta_raw_load
                  + delta_determinant[:, None, None]*raw_load)

    dofs = vector_dofs(operators.space.cells)
    delta_stiffness_matrix = scatter_matrix(delta_stiffness, dofs, dofs, (n_velocity, n_velocity))
    delta_divergence_matrix = scatter_matrix(delta_divergence, cells, dofs, (mesh.n_nodes, n_velocity))
    delta_rhs = np.zeros(operators.n_total)
    np.add.at(delta_rhs, dofs.ravel(), delta_load.ravel())

    result = np.zeros(operators.n_total)
    velocity = values[:n_velocity]
    pressure = values[n_velocity:]
    result[:n_velocity] += delta_stiffness_matrix @ velocity
    result[:n_velocity] -= delta_divergence_matrix.T @ pressure
    result[n_velocity:] -= delta_divergence_matrix @ velocity
    result[:n_velocity] -= delta_rhs[:n_velocity]

    # 对流张量 T_abc = ∫ N_a N_b ∇N_c 的几何导数：det J 变化项 + 梯度变化项。
    raw_tensor = np.einsum("q,qa,qb,eqcj->eabcj", weights, velocity_shape, velocity_shape, gradients)
    delta_raw_tensor = np.einsum(
        "q,qa,qb,eqcj->eabcj", weights, velocity_shape, velocity_shape, delta_gradients
    )
    delta_tensor = density*(determinant[:, None, None, None, None]*delta_raw_tensor
                            + delta_determinant[:, None, None, None, None]*raw_tensor)
    local_velocity = velocity.reshape(-1, 2)[operators.space.cells]
    delta_convective = np.einsum("eabcj,ebj->eac", delta_tensor, local_velocity) @ local_velocity
    np.add.at(result, dofs.ravel(), delta_convective.ravel())

    local_mass_scalar = np.einsum("q,qa,qb->ab", weights, velocity_shape, velocity_shape)
    raw_mass = np.einsum("ab,ij->aibj", local_mass_scalar, np.eye(2)).reshape(12, 12)
    # 质量矩阵只通过 det J 依赖几何：dM = rho d(det J) M_raw。
    delta_mass = (density*delta_determinant)[:, None, None]*raw_mass
    delta_mass_matrix = scatter_matrix(delta_mass, dofs, dofs, (n_velocity, n_velocity))
    result[:n_velocity] += float(ale_context.stencil.coefficients[0])*(delta_mass_matrix @ velocity)

    delta_flux, delta_source = _transport_directional(geometry, transport, delta_jacobian, delta, points)
    # 把 Q_BDF / g_BDF 的方向导数映射为速度块上的 K_mesh 响应。
    delta_transport = _transport_response_matrix(operators, reference_mesh, delta_flux, delta_source)
    result[:n_velocity] += delta_transport @ velocity
    return result

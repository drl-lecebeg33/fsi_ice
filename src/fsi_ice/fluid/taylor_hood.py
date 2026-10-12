"""Physical pressure convention: -div(2 mu eps(u)) + grad(p) = f.

本模块提供 Taylor-Hood P2/P1 不可压缩流离散：速度用二阶连续元（P2，含中边节点），
压力用一阶连续元（P1，仅顶点）。关键约定：
- 动量方程使用物理压力 p，符号为 -div(2 mu eps(u)) + grad(p) = f；eps(u) 为对称梯度；
- 不可压缩约束 div u = 0 乘以 -1，使鞍点矩阵对称；
- 压力存在一个常数零模，需通过 pressure_pin 钉住一个节点；
- 装配基于单元局部自由度做 scatter_matrix 组装，自由度按 (节点, 分量) 交错排列；
- 速度自由度数为 2*n_P2_nodes，压力自由度数为 n_P1_nodes（即网格顶点数）。
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix

from fsi_ice.fem.p1 import map_points, shape_values
from fsi_ice.fem.p2 import P2Space, p2_physical_gradients, p2_values
from fsi_ice.fem.poisson import solve_dirichlet
from fsi_ice.fem.quadrature import triangle_quadrature
from fsi_ice.solid.linear import vector_dofs, vector_values


def scatter_matrix(local, row_dofs, column_dofs, shape):
    # 把 (n_cells, nrow, ncol) 的局部矩阵按自由度映射组装成 (shape) 的 CSR 全局矩阵。
    # row_dofs/column_dofs 形状为 (n_cells, nrow)/(n_cells, ncol)。
    nrow, ncol = local.shape[1:]
    rows = np.repeat(row_dofs, ncol, axis=1).ravel()
    columns = np.tile(column_dofs, (1, nrow)).ravel()
    return coo_matrix((local.ravel(), (rows, columns)), shape=shape).tocsr()


@dataclass(frozen=True)
class FlowOperators:
    # 冻结的 Taylor-Hood 流场算子集合：space 为 P2 速度空间，matrix/rhs 为定常 Stokes 系统，
    # mass 为 P2 速度质量矩阵，divergence 为 P2 速度到 P1 压力的散度算子；viscosity/density 为物理参数。
    space: P2Space
    matrix: object
    rhs: np.ndarray
    mass: object
    divergence: object
    viscosity: float
    density: float
    quadrature: np.ndarray
    weights: np.ndarray
    gradients: np.ndarray
    body_force: object = (0.0, 0.0)

    @property
    def n_velocity(self):
        # 速度自由度总数：每个 P2 节点两个分量。
        return 2*len(self.space.points)

    @property
    def n_total(self):
        # 混合系统总自由度：速度 + 压力（P1 顶点数）。
        return self.n_velocity+self.space.mesh.n_nodes

    @cached_property
    def convection_tensor(self):
        # Integral N_a*N_b*grad(N_c): exact for the degree-five convective form.
        # 对流张量 T[a,b,c,j] = ∫ N_a N_b (∇N_c)_j dX，五阶多项式形式用五点求积精确。
        shape = p2_values(self.quadrature)
        tensor = np.einsum("q,qa,qb,eqcj->eabcj", self.weights, shape, shape, self.gradients)
        tensor *= (self.density*self.space.mesh.determinants)[:, None, None, None, None]
        tensor.flags.writeable = False
        return tensor


def assemble_stokes(mesh, *, viscosity=1.0, density=1.0, body_force=(0.0, 0.0)):
    """Assemble the steady Taylor-Hood Stokes system on a fixed mesh.

    在固定网格上组装定常 Stokes 系统：黏性块 a、散度块 B、P2 速度质量矩阵 M 与体积力
    右端项，并组装对称鞍点矩阵 [[a, -B^T], [-B, 0]]。返回 FlowOperators。
    单位与网格一致（无量纲或一致单位制）；viscosity、density 必须有限且为正。
    输入：TriMesh 与可选参数；输出：FlowOperators（含 space/matrix/rhs/mass/divergence 等）。
    """
    if not np.isfinite(viscosity) or viscosity <= 0 or not np.isfinite(density) or density <= 0:
        raise ValueError("Viscosity and density must be finite and positive.")
    space = P2Space.from_mesh(mesh)
    points, weights = triangle_quadrature(5)
    velocity_shape = p2_values(points)
    pressure_shape = shape_values(points)
    gradients = p2_physical_gradients(mesh, points)
    strains = np.zeros((mesh.n_cells, len(points), 3, 12))
    # 对称梯度（应变）基：行 0/1/2 对应 [eps_xx, eps_yy, 2 eps_xy]，列按速度分量交错排布。
    strains[:, :, 0, 0::2] = gradients[:, :, :, 0]
    strains[:, :, 1, 1::2] = gradients[:, :, :, 1]
    strains[:, :, 2, 0::2] = gradients[:, :, :, 1]
    strains[:, :, 2, 1::2] = gradients[:, :, :, 0]
    constitutive = viscosity*np.diag([2.0, 2.0, 1.0])
    # 2D Voigt 本构 diag(2μ, 2μ, μ)：把 [eps_xx, eps_yy, 2 eps_xy] 线性映射到应力分量。
    local_a = np.einsum("q,eqai,ab,eqbj->eij", weights, strains, constitutive, strains)
    local_a *= mesh.determinants[:, None, None]
    divergence_basis = np.empty((mesh.n_cells, len(points), 12))
    # 散度基 div(N_a)（对矩阵 B 的列空间）。
    divergence_basis[:, :, 0::2] = gradients[:, :, :, 0]
    divergence_basis[:, :, 1::2] = gradients[:, :, :, 1]
    # 散度矩阵 B: ∫ q_p div(N_v) det J dξ。
    local_b = np.einsum("q,qp,eqv->epv", weights, pressure_shape, divergence_basis)
    local_b *= mesh.determinants[:, None, None]
    local_mass_scalar = np.einsum("q,qa,qb->ab", weights, velocity_shape, velocity_shape)
    local_mass = np.einsum("ab,ij->aibj", local_mass_scalar, np.eye(2)).reshape(12, 12)
    # 质量矩阵 M = rho ∫ N_a N_b I det J dξ。
    local_mass = (density*mesh.determinants[:, None, None])*local_mass
    dofs = vector_dofs(space.cells)
    nv = 2*len(space.points)
    a = scatter_matrix(local_a, dofs, dofs, (nv, nv))
    divergence = scatter_matrix(local_b, mesh.cells, dofs, (mesh.n_nodes, nv))
    mass = scatter_matrix(local_mass, dofs, dofs, (nv, nv))
    # Multiplying the incompressibility equation by -1 makes the saddle matrix symmetric.
    # 中文：把不可压缩方程整体乘以 -1，使鞍点矩阵 [[a,-B^T],[-B,0]] 保持对称。
    matrix = bmat([[a, -divergence.T], [-divergence, None]], format="csr")
    physical_points = map_points(mesh, points)
    force = vector_values(body_force, physical_points.reshape(-1, 2), "Body force").reshape(mesh.n_cells, -1, 2)
    # 体积力右端项 ∫ N_a·f det J dξ，组装到速度自由度。
    load = np.einsum("q,eqi,qa->eai", weights, force, velocity_shape)
    load *= mesh.determinants[:, None, None]
    rhs = np.zeros(nv+mesh.n_nodes)
    np.add.at(rhs, dofs.ravel(), load.ravel())
    return FlowOperators(space, matrix, rhs, mass, divergence, viscosity, density, points, weights, gradients,
                         body_force)


def assemble_velocity_mass(mesh, *, density=1.0):
    """Consistent P2 velocity mass matrix on one mesh (same quadrature as assemble_stokes).

    在单个网格上组装一致的 P2 速度质量矩阵 M = rho ∫ N_a N_b I det J dξ，求积与
    assemble_stokes 相同（五点）。用于 BDF 时间项。density 必须有限且为正。
    """
    if not np.isfinite(density) or density <= 0:
        raise ValueError("Density must be finite and positive.")
    space = P2Space.from_mesh(mesh)
    points, weights = triangle_quadrature(5)
    velocity_shape = p2_values(points)
    local_mass_scalar = np.einsum("q,qa,qb->ab", weights, velocity_shape, velocity_shape)
    local_mass = np.einsum("ab,ij->aibj", local_mass_scalar, np.eye(2)).reshape(12, 12)
    local_mass = (density*mesh.determinants[:, None, None])*local_mass
    dofs = vector_dofs(space.cells)
    return scatter_matrix(local_mass, dofs, dofs, (2*len(space.points),)*2)


def flow_constraints(operators, velocity, tags=None, *, pressure_pin=0, pressure_value=0.0):
    """Build the Dirichlet velocity constraints and the pressure pin for a flow step.

    构造流场约束：在 tags 指定的边界节点上施加给定速度（由 velocity 插值得到），
    并可选地在 pressure_pin 节点把压力钉到 pressure_value（消除压力零模）。
    返回 (dofs, prescribed)：约束自由度索引与其目标值。pressure_pin=None 表示不钉压力。
    """
    nodes = operators.space.boundary_nodes(tags)
    values = vector_values(velocity, operators.space.points[nodes], "Boundary velocity")
    dofs = vector_dofs(nodes)
    prescribed = values.ravel()
    if pressure_pin is not None:
        if isinstance(pressure_pin, bool) or not isinstance(pressure_pin, (int, np.integer)):
            raise ValueError("Pressure pin must be an integer node or None.")
        if not 0 <= pressure_pin < operators.space.mesh.n_nodes or not np.isfinite(pressure_value):
            raise ValueError("Pressure pin/value is invalid.")
        # 压力自由度在混合向量中的偏移为 n_velocity；钉住一个 P1 节点以消去压力常数零模。
        dofs = np.append(dofs, operators.n_velocity+pressure_pin)
        prescribed = np.append(prescribed, pressure_value)
    return dofs, prescribed


@dataclass(frozen=True)
class FlowSolution:
    # 流场解：velocity (n_P2, 2)、pressure (n_P1,)、相对残差与连续性残差，以及非线性迭代历史。
    velocity: np.ndarray
    pressure: np.ndarray
    relative_residual: float
    continuity_residual: float
    nonlinear_history: tuple = ()


def unpack_flow(operators, values, residual, history=()):
    """Split a mixed flow vector into a FlowSolution and compute the continuity defect.

    把混合解向量拆成速度 (n_P2, 2) 与压力 (n_P1,) 并封装为 FlowSolution；
    连续性残差用相对范数 ||B u|| / max(||u||, 1) 度量。
    """
    velocity = values[:operators.n_velocity].reshape(-1, 2)
    pressure = values[operators.n_velocity:]
    # 相对连续性残差，避免量纲与速度幅值影响。
    continuity = np.linalg.norm(operators.divergence @ velocity.ravel()) / max(np.linalg.norm(velocity), 1)
    return FlowSolution(velocity, pressure, float(residual), float(continuity), tuple(history))


def solve_stokes(operators, velocity, tags=None, *, pressure_pin=0, pressure_value=0.0):
    """Solve the linear Stokes system with the given boundary data and return a FlowSolution.

    施加边界速度与压力钉后用 solve_dirichlet 求解线性 Stokes 系统，并检查连续性；
    若边界速度与不可压缩性不相容（相对连续性残差 > 1e-10）则抛 RuntimeError。
    """
    dofs, values = flow_constraints(operators, velocity, tags, pressure_pin=pressure_pin, pressure_value=pressure_value)
    result = solve_dirichlet(operators.matrix, operators.rhs, dofs, values)
    solution = unpack_flow(operators, result.values, result.relative_residual)
    if solution.continuity_residual > 1e-10:
        raise RuntimeError("Velocity boundary data are incompatible with incompressibility.")
    return solution


def convection(operators, velocity, *, with_jacobian=True):
    """P2 convective residual ∫ N_a (u·∇)u dX and its Jacobian.

    组装对流残差 r_a = ∫ N_a (u·∇)u dX 及其对速度的雅可比。with_jacobian=False 时仅返回
    残差（jacobian 为 None）。velocity 必须是有限的 (n_P2_nodes, 2) 数组；返回 (residual, jacobian)。
    """
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape != (len(operators.space.points), 2) or not np.all(np.isfinite(velocity)):
        raise ValueError("Velocity must be a finite (n_P2_nodes,2) array.")
    mesh = operators.space.mesh
    local_velocity = velocity[operators.space.cells]
    tensor = operators.convection_tensor
    # advected[e,a,c] = ∫ N_a (u·∇N_c) dX，随后与 u_c 收缩得到 (u·∇)u 的投影。
    advected = np.einsum("eabcj,ebj->eac", tensor, local_velocity)
    load = advected @ local_velocity
    dofs = vector_dofs(operators.space.cells)
    residual = np.zeros(operators.n_velocity)
    np.add.at(residual, dofs.ravel(), load.ravel())
    if not with_jacobian:
        return residual, None
    # 分离两部分：对 (u·∇) 中显式 u 因子的导数，及对 ∇u 中 u 因子的导数。
    local_jacobian = np.einsum("eab,ij->eaibj", advected, np.eye(2))
    advecting_derivative = tensor.transpose(0, 1, 2, 4, 3) @ local_velocity[:, None, None, :, :]
    local_jacobian += advecting_derivative.transpose(0, 1, 4, 2, 3)
    local_jacobian = local_jacobian.reshape(mesh.n_cells, 12, 12)
    jacobian = scatter_matrix(local_jacobian, dofs, dofs, (operators.n_velocity, operators.n_velocity))
    return residual, jacobian


def navier_stokes_residual(operators, values, *, time_coefficient=0.0, time_rhs=None, with_jacobian=True,
                           ale_context=None):
    """Fixed-grid (or ALE-dispatched) Navier-Stokes residual and Jacobian.

    组装混合残差与雅可比：定常项 (matrix·values - rhs) + 对流；若给 time_coefficient、
    time_rhs 则追加 BDF 时间项 a_0 M u - history；ale_context 不为 None 时转发到
    ale_navier_stokes_residual（二者不可与 time_coefficient/time_rhs 同时使用）。
    values 为 (n_velocity + n_pressure) 的有限混合向量。
    """
    values = np.asarray(values, dtype=float)
    if values.shape != (operators.n_total,) or not np.all(np.isfinite(values)):
        raise ValueError("Mixed flow state must be a finite vector of the correct size.")
    if ale_context is not None:
        if time_rhs is not None or (time_coefficient is not None and time_coefficient != 0.0):
            raise ValueError("ale_context cannot be combined with time_coefficient/time_rhs.")
        from fsi_ice.fluid.ale import ale_navier_stokes_residual

        return ale_navier_stokes_residual(operators, values, ale_context, with_jacobian=with_jacobian)
    convective, convective_jacobian = convection(operators, values[:operators.n_velocity].reshape(-1, 2), with_jacobian=with_jacobian)
    residual = operators.matrix @ values - operators.rhs
    residual[:operators.n_velocity] += convective
    momentum_jacobian = convective_jacobian
    if time_coefficient:
        if not np.isfinite(time_coefficient) or time_coefficient <= 0:
            raise ValueError("Time coefficient must be finite and positive.")
        history = np.asarray(time_rhs, dtype=float)
        if history.shape != (operators.n_velocity,) or not np.all(np.isfinite(history)):
            raise ValueError("Time history RHS must match velocity DOFs.")
        # BDF 时间项 a_0 M u - history 追加到动量方程。
        residual[:operators.n_velocity] += time_coefficient*(operators.mass @ values[:operators.n_velocity])-history
        if with_jacobian:
            momentum_jacobian = momentum_jacobian + time_coefficient*operators.mass
    if not with_jacobian:
        return residual, None
    correction = bmat([[momentum_jacobian, None], [None, csr_matrix((operators.space.mesh.n_nodes,)*2)]], format="csr")
    return residual, operators.matrix+correction


def solve_navier_stokes(operators, velocity, tags=None, *, pressure_pin=0, pressure_value=0.0,
                        initial=None, tolerance=1e-9, max_iterations=30,
                        time_coefficient=0.0, time_rhs=None, linear_solver=None, ale_context=None):
    """Newton solve of the (possibly ALE) steady/transient Navier-Stokes system.

    用带阻尼线搜索的 Newton 迭代求解 Navier-Stokes 系统。initial 为 None 时以 Stokes 解
    起算；每步解牛顿修正并做至多 16 次折半线搜索直至残差下降。relative_residual 用
    max(||rhs||, ||history||, 1) 归一。linear_solver 可选（ReusedLUSolver），ale_context
    用于 ALE 分支；返回 FlowSolution。tolerance/max_iterations 必须有效。
    """
    if not np.isfinite(tolerance) or tolerance <= 0 or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("Invalid nonlinear solve controls.")
    dofs, prescribed = flow_constraints(operators, velocity, tags, pressure_pin=pressure_pin, pressure_value=pressure_value)
    signature = None
    if ale_context is not None:
        if time_rhs is not None or (time_coefficient is not None and time_coefficient != 0.0):
            raise ValueError("ale_context cannot be combined with time_coefficient/time_rhs.")
        from fsi_ice.fluid.ale import ale_linear_signature, history_mass_load

        signature = ale_linear_signature(operators, ale_context, dofs, prescribed)
    if linear_solver is not None:
        if ale_context is None:
            linear_solver.prepare(time_coefficient)
        else:
            linear_solver.prepare(ale_context.stencil.coefficients[0], signature=signature)
    if initial is None:
        starting = solve_dirichlet(operators.matrix, operators.rhs, dofs, prescribed)
        values = starting.values.copy()
    else:
        values = np.array(initial, dtype=float, copy=True)
        if values.shape != (operators.n_total,) or not np.all(np.isfinite(values)):
            raise ValueError("Invalid initial flow state.")
    values[dofs] = prescribed
    # 自由自由度 = 全部自由度去掉约束自由度；残差范数只在该集合上度量。
    free = np.setdiff1d(np.arange(operators.n_total), dofs)
    history = []
    if ale_context is None:
        scale = max(np.linalg.norm(operators.rhs), 1.0)
    else:
        scale = max(np.linalg.norm(operators.rhs), np.linalg.norm(history_mass_load(operators, ale_context)), 1.0)
    for iteration in range(max_iterations+1):
        residual, jacobian = navier_stokes_residual(operators, values, time_coefficient=time_coefficient,
                                                    time_rhs=time_rhs, ale_context=ale_context)
        norm = float(np.linalg.norm(residual[free])/scale)
        history.append(norm)
        if norm <= tolerance:
            result = unpack_flow(operators, values, norm, history)
            if result.continuity_residual > max(1e-10, tolerance):
                raise RuntimeError("Flow solution violates the full continuity equation.")
            return result
        if iteration == max_iterations:
            break
        if linear_solver is None:
            correction = solve_dirichlet(jacobian, -residual, dofs, 0.0).values
        elif ale_context is None:
            correction = linear_solver.solve(jacobian, -residual, dofs)
        else:
            correction = linear_solver.solve(jacobian, -residual, dofs, signature=signature)
        step = 1.0
        # 阻尼线搜索：候选使残差下降则接受，否则步长折半（最多 16 次）。
        for _ in range(16):
            candidate = values+step*correction
            candidate_residual, _ = navier_stokes_residual(operators, candidate, time_coefficient=time_coefficient,
                                                           time_rhs=time_rhs, with_jacobian=False,
                                                           ale_context=ale_context)
            if np.linalg.norm(candidate_residual[free])/scale < norm:
                values = candidate
                break
            step *= 0.5
        else:
            raise RuntimeError("Navier-Stokes Newton line search failed.")
    raise RuntimeError(f"Navier-Stokes did not converge in {max_iterations} iterations; residual={history[-1]:.3e}.")


def flow_error_norms(operators, solution, exact_velocity, exact_pressure):
    """L2 error norms of a flow solution against analytic fields at quadrature points.

    在求积点上用解析解评估 L2 误差：返回 {"velocity_l2", "pressure_l2"}。速度按 P2 插值，
    压力按 P1 插值；度量用 det J * 求积权重。exact_velocity 为可调用向量场，exact_pressure 为可调用标量场。
    """
    mesh = operators.space.mesh
    points = map_points(mesh, operators.quadrature)
    # 在求积点插值：速度用 P2 形函数，压力用 P1 形函数。
    velocity = np.einsum("qa,eai->eqi", p2_values(operators.quadrature), solution.velocity[operators.space.cells])
    pressure = solution.pressure[mesh.cells] @ shape_values(operators.quadrature).T
    expected_velocity = vector_values(exact_velocity, points.reshape(-1, 2), "Exact velocity").reshape(points.shape)
    expected_pressure = np.asarray(exact_pressure(points.reshape(-1, 2)), dtype=float).reshape(pressure.shape)
    if not np.all(np.isfinite(expected_pressure)):
        raise ValueError("Exact pressure must be finite.")
    # L2 度量 = det J * 求积权重。
    measure = mesh.determinants[:, None]*operators.weights
    return {"velocity_l2": float(np.sqrt(np.sum(measure*np.sum((velocity-expected_velocity)**2, axis=2)))),
            "pressure_l2": float(np.sqrt(np.sum(measure*(pressure-expected_pressure)**2)))}

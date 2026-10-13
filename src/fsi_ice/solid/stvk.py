"""Total Lagrangian P1 StVK solid, plane strain and reference-volume loads.

本模块实现总拉格朗日（Total Lagrangian）P1 圣维南–基尔霍夫（StVK）有限变形固体，
当前仅支持平面应变。所有积分在参考构型上进行（用参考面积/体积度量）。内部变量：
变形梯度 F = I + ∇u、右 Cauchy–Green 应变 E = (FᵀF - I)/2、第二 Piola 应力
S = λ tr(E) I + 2μE、第一 Piola 应力 P = F·S；并给出解析一致切线 dP/dF，供
Newton 迭代使用。
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix

from fsi_ice.fem.p1 import physical_gradients
from fsi_ice.fem.poisson import solve_dirichlet
from fsi_ice.solid.linear import vector_dofs, vector_values


def assemble_mass(mesh, material):
    """组装一致质量矩阵（每单元 6x6）。

    中文：P1 三角形标量质量核为 area/12·[[2,1,1],[1,2,1],[1,1,2]]，再乘密度与
    厚度；对两个位移分量做 Kronecker 扩张并交错组装成 2*n_nodes 阶 CSR 矩阵。
    """
    scalar = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12
    local = np.einsum("ab,ij->aibj", scalar, np.eye(2)).reshape(6, 6)
    local = (material.density*material.thickness*mesh.areas)[:, None, None]*local
    dofs = vector_dofs(mesh.cells)
    return coo_matrix((local.ravel(), (np.repeat(dofs, 6, axis=1).ravel(),
                                     np.tile(dofs, (1, 6)).ravel())),
                      shape=(2*mesh.n_nodes,)*2).tocsr()


def stvk_residual(mesh, material, displacement, external=None):
    """计算 StVK 总拉格朗日残差与一致切线刚度矩阵。

    中文：返回 (residual, matrix)。``displacement`` 为 (n_nodes,2) 位移；
    ``external`` 为可选的节点外力（已含符号，会从内力中减去）。residual 为内力减
    外力（长度 2*n_nodes），matrix 为对应的一致切线 CSR 矩阵。要求平面应变且
    变形梯度行列式处处为正，否则抛出 ``ValueError``。
    """
    if material.model != "plane_strain":
        raise ValueError("Finite-strain StVK currently requires plane_strain.")
    displacement = np.asarray(displacement, dtype=float)
    if displacement.shape != mesh.points.shape or not np.all(np.isfinite(displacement)):
        raise ValueError("Displacement must be a finite (n_nodes,2) array.")
    gradients = physical_gradients(mesh)
    # 中文：变形梯度 F = I + ∇u，其中 u 按单元节点取值。
    deformation = np.eye(2)+np.einsum("eai,eaJ->eiJ", displacement[mesh.cells], gradients)
    determinants = np.linalg.det(deformation)
    if np.any(determinants <= 0) or not np.all(np.isfinite(determinants)):
        raise ValueError("Solid deformation has a non-positive Jacobian.")
    # 中文：右 Cauchy–Green 应变 E = (FᵀF - I)/2。
    strain = 0.5*(np.einsum("ekI,ekJ->eIJ", deformation, deformation)-np.eye(2))
    lam, mu = material.lame
    # 中文：第二 Piola 应力 S = λ tr(E) I + 2μ E，再乘 F 得第一 Piola 应力 P。
    second_piola = lam*np.trace(strain, axis1=1, axis2=2)[:, None, None]*np.eye(2)+2*mu*strain
    first_piola = deformation @ second_piola
    measure = mesh.areas*material.thickness
    local_residual = measure[:, None, None]*np.einsum("eaJ,eiJ->eai", gradients, first_piola)
    # dP_iJ/dF_kL: geometric stress term plus the derivative of F*S(E(F)).
    # 中文：一致切线 = 几何项 I⊗S + λ F⊗F + μ F⊗F(转置) + μ F Fᵀ 的对称项。
    tangent = np.einsum("ik,eLJ->eiJkL", np.eye(2), second_piola)
    tangent += lam*np.einsum("eiJ,ekL->eiJkL", deformation, deformation)
    tangent += mu*np.einsum("eiL,ekJ->eiJkL", deformation, deformation)
    tangent += mu*np.einsum("JL,eik->eiJkL", np.eye(2), deformation @ deformation.transpose(0, 2, 1))
    local_matrix = np.einsum("eaJ,eiJkL,ebL->eaibk", gradients, tangent, gradients)
    local_matrix = (measure[:, None, None, None, None]*local_matrix).reshape(mesh.n_cells, 6, 6)
    dofs = vector_dofs(mesh.cells)
    residual = np.zeros(2*mesh.n_nodes)
    np.add.at(residual, dofs.ravel(), local_residual.ravel())
    if external is not None:
        external = np.asarray(external, dtype=float)
        if external.shape != residual.shape or not np.all(np.isfinite(external)):
            raise ValueError("External load must match solid DOFs and be finite.")
        residual -= external
    matrix = coo_matrix((local_matrix.ravel(), (np.repeat(dofs, 6, axis=1).ravel(),
                                             np.tile(dofs, (1, 6)).ravel())),
                        shape=(2*mesh.n_nodes,)*2).tocsr()
    return residual, matrix


@dataclass(frozen=True)
class NonlinearSolidSolution:
    """StVK 非线性固体求解结果。

    中文：``displacement`` 为收敛位移 (n_nodes,2)；``reactions`` 为节点支反力；
    ``nonlinear_history`` 为每次 Newton 迭代的相对残差轨迹（元组）；
    ``minimum_jacobian`` 为最终变形梯度行列式的最小值，用于判断是否接近退化。
    """

    displacement: np.ndarray
    reactions: np.ndarray
    nonlinear_history: tuple
    minimum_jacobian: float


def stvk_energy(mesh, material, displacement, external):
    """StVK 总势能（含外部载荷功）。

    中文：返回标量势能 = ∫ [λ/2 tr(E)² + μ |E|²] dA·t - external·u，其中 E 为
    右 Cauchy–Green 应变，最后一项为外力做功（external 与位移均按交错自由度排列）。
    """
    deformation = np.eye(2)+np.einsum("eai,eaJ->eiJ", displacement[mesh.cells], physical_gradients(mesh))
    # 中文：应变 E = (FᵀF - I)/2，储能密度 = λ/2 tr(E)² + μ tr(E²)。
    strain = 0.5*(deformation.transpose(0, 2, 1) @ deformation-np.eye(2))
    lam, mu = material.lame
    density = 0.5*lam*np.trace(strain, axis1=1, axis2=2)**2+mu*np.sum(strain**2, axis=(1, 2))
    return float(np.dot(mesh.areas*material.thickness, density)-np.dot(np.asarray(external), displacement.ravel()))


def solve_stvk(mesh, material, external, fixed_nodes, *, prescribed=(0.0, 0.0), initial=None,
               tolerance=1e-9, max_iterations=40, absolute_tolerance=None,
               mass_coefficient=0.0, mass_rhs=None):
    """用 Newton 迭代求解 StVK 非线性固体平衡。

    中文：``external`` 为节点外力（交错自由度）；``fixed_nodes`` 为约束节点，
    ``prescribed`` 为其位移值；``initial`` 为位移初值；``tolerance`` 为相对残差容
    差，``absolute_tolerance`` 为可选的自由残差绝对上限。``mass_coefficient`` 与
    ``mass_rhs`` 在时间离散时叠加惯性项 α²M 及其历史右端。返回
    :class:`NonlinearSolidSolution`；未收敛则抛出 ``RuntimeError``。
    """
    if not np.isfinite(tolerance) or tolerance <= 0 or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("Invalid nonlinear solid solve controls.")
    if absolute_tolerance is not None and (not np.isfinite(absolute_tolerance) or absolute_tolerance <= 0):
        raise ValueError("absolute_tolerance must be a finite positive real number when provided.")
    fixed_nodes = np.asarray(fixed_nodes)
    if fixed_nodes.ndim != 1 or not np.issubdtype(fixed_nodes.dtype, np.integer) or np.any(fixed_nodes < 0) or np.any(fixed_nodes >= mesh.n_nodes):
        raise ValueError("Invalid fixed solid nodes.")
    dofs = vector_dofs(fixed_nodes)
    values = np.zeros(mesh.points.shape) if initial is None else np.array(initial, dtype=float, copy=True)
    if values.shape != mesh.points.shape or not np.all(np.isfinite(values)):
        raise ValueError("Invalid solid initial guess.")
    # 中文：约束节点位移直接置为规定值，其余自由度由 Newton 迭代求解。
    values[fixed_nodes] = vector_values(prescribed, mesh.points[fixed_nodes], "Prescribed displacement")
    free = np.setdiff1d(np.arange(values.size), dofs)
    external = np.asarray(external, dtype=float)
    # 中文：相对残差以载荷范数（下限 1）归一，避免载荷过小时过早判定收敛。
    scale = max(np.linalg.norm(external), 1.0)
    mass = None
    if mass_coefficient:
        if not np.isfinite(mass_coefficient) or mass_coefficient < 0:
            raise ValueError("Invalid solid mass coefficient.")
        mass = assemble_mass(mesh, material)
        mass_rhs = np.asarray(mass_rhs, dtype=float)
        if mass_rhs.shape != (values.size,) or not np.all(np.isfinite(mass_rhs)):
            raise ValueError("Invalid solid mass history RHS.")

    def evaluate(state):
        # 中文：返回残差与一致切线；若启用质量项则叠加惯性贡献。
        residual, matrix = stvk_residual(mesh, material, state, external)
        if mass is not None:
            residual += mass_coefficient*(mass @ state.ravel())-mass_rhs
            matrix += mass_coefficient*mass
        return residual, matrix

    def potential(state):
        # 中文：总势能，用于回溯线搜索的下降判据；含惯性势能时一并计入。
        energy = stvk_energy(mesh, material, state, external)
        if mass is not None:
            energy += 0.5*mass_coefficient*np.dot(state.ravel(), mass @ state.ravel())-np.dot(mass_rhs, state.ravel())
        return energy

    history = []
    for iteration in range(max_iterations+1):
        residual, matrix = evaluate(values)
        norm = float(np.linalg.norm(residual[free])/scale)
        history.append(norm)
        # The relative criterion scales with the external load magnitude; the
        # optional absolute bound keeps the raw free residual below a caller-given
        # limit (used by the coupled driver to meet its absolute acceptance gate).
        if norm <= tolerance and (absolute_tolerance is None or
                                  float(np.linalg.norm(residual[free])) <= absolute_tolerance):
            # 中文：收敛后返回位移、支反力、残差历史与最小变形梯度行列式。
            deformation = np.eye(2)+np.einsum("eai,eaJ->eiJ", values[mesh.cells], physical_gradients(mesh))
            return NonlinearSolidSolution(values, residual.reshape(-1, 2), tuple(history),
                                          float(np.linalg.det(deformation).min()))
        if iteration == max_iterations:
            break
        # 中文：Newton 修正由约束系统 K Δu = -r 求得（约束自由度的修正为零）。
        correction = solve_dirichlet(matrix, -residual, dofs, 0.0).values.reshape(-1, 2)
        energy = potential(values)
        slope = float(np.dot(residual, correction.ravel()))
        step = 1.0
        for _ in range(20):
            candidate = values+step*correction
            try:
                candidate_residual, _ = evaluate(candidate)
            except ValueError as error:
                if "Jacobian" not in str(error):
                    raise
                # 中文：试探点出现非正 Jacobian（单元翻转）时拒绝该步并减半步长。
                step *= 0.5
                continue
            # 中文：Armijo 下降条件或残差范数下降即接受，否则步长减半（回溯线搜索）。
            descent = slope < 0 and potential(candidate) <= energy+1e-4*step*slope
            if descent or np.linalg.norm(candidate_residual[free])/scale < norm:
                values = candidate
                break
            step *= 0.5
        else:
            raise RuntimeError("StVK Newton line search failed.")
    raise RuntimeError(f"StVK failed after {max_iterations} iterations; residual={history[-1]:.3e}.")


def solve_static_continuation(mesh, material, external, fixed_nodes, *, load_steps=10, **kwargs):
    """分级加载（延续法）求解静态 StVK 问题。

    中文：把总载荷按 ``load_steps`` 等分为若干级，逐级调用 :func:`solve_stvk`，
    以上一级的解作为下一级初值，改善大变形下的收敛性。返回 (最后一级解, 各级
    Newton 残差历史列表)。``kwargs`` 透传给 :func:`solve_stvk`。
    """
    if not isinstance(load_steps, int) or isinstance(load_steps, bool) or load_steps < 1:
        raise ValueError("load_steps must be a positive integer.")
    initial = None
    histories = []
    for step in range(1, load_steps+1):
        solution = solve_stvk(mesh, material, np.asarray(external)*step/load_steps,
                              fixed_nodes, initial=initial, **kwargs)
        initial = solution.displacement
        histories.append(solution.nonlinear_history)
    return solution, histories

"""P1 weak form: integral(k grad(u).grad(v)) = integral(f v).

本模块实现 P1 三角形上的标量 Poisson 弱形式 ∫ k ∇u·∇v = ∫ f v。
``assemble_poisson`` 组装刚度矩阵与载荷向量；``solve_dirichlet`` 以保持对称性
的方式消去 Dirichlet 自由度并做残差与后向误差校验；``solve_poisson`` 组合两者；
``error_norms`` 计算 L2 误差与 H1 半范误差（非完整 H1 范数）。
"""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from fsi_ice.fem.p1 import map_points, physical_gradients, scalar_values, shape_values
from fsi_ice.fem.quadrature import triangle_quadrature
from fsi_ice.mesh import TriMesh


@dataclass(frozen=True)
class DirichletSolution:
    """Dirichlet 问题求解结果。

    ``values`` 为全部自由度的解向量；``free_nodes`` 为自由节点索引；``
    relative_residual`` 为相对残差；``backward_error`` 为缩放后向误差，用于
    区分良态系统的消去误差与真正求解失败。
    """

    values: np.ndarray
    free_nodes: np.ndarray
    relative_residual: float
    backward_error: float


def assemble_poisson(mesh: TriMesh, source, diffusion=1.0):
    """组装 P1 标量 Poisson 弱形式的刚度矩阵与载荷向量。

    参数：``mesh`` 为 P1 网格；``source`` 是源项 f（常量、数组或可调用 (n,2)->(n,)）；
    ``diffusion`` 是扩散系数 k，必须逐求积点严格为正（支持变系数）。返回
    (matrix, rhs)：CSR 格式对称刚度矩阵与载荷向量。使用 5 阶三角形求积；变系数
    以单元内求积加权平均计入。
    """
    points, weights = triangle_quadrature(5)
    shape = shape_values(points)
    physical_points = map_points(mesh, points)
    flat_points = physical_points.reshape(-1, 2)
    force = scalar_values(source, flat_points, "Source").reshape(mesh.n_cells, -1)
    coefficient = scalar_values(diffusion, flat_points, "Diffusion").reshape(mesh.n_cells, -1)
    if np.any(coefficient <= 0):
        raise ValueError("Diffusion must be strictly positive at all quadrature points.")
    gradients = physical_gradients(mesh)
    determinants = mesh.determinants
    # 中文：变系数 k 取单元内求积加权平均，与单元面积（det）合并成刚度缩放因子。
    integrated_coefficient = coefficient @ weights
    local_matrix = np.einsum("eai,ebi->eab", gradients, gradients)
    local_matrix *= (determinants * integrated_coefficient)[:, None, None]
    # 中文：载荷项按 det·Σ_q w_q f_q N_a 计算，得到单元 3 维一致载荷。
    local_rhs = determinants[:, None] * np.einsum("q,eq,qa->ea", weights, force, shape)
    rows = np.repeat(mesh.cells, 3, axis=1).ravel()
    columns = np.tile(mesh.cells, (1, 3)).ravel()
    matrix = coo_matrix((local_matrix.ravel(), (rows, columns)),
                        shape=(mesh.n_nodes, mesh.n_nodes)).tocsr()
    rhs = np.zeros(mesh.n_nodes)
    np.add.at(rhs, mesh.cells.ravel(), local_rhs.ravel())
    return matrix, rhs


def solve_dirichlet(matrix, rhs, nodes, values):
    """Eliminate prescribed DOFs with b_free -= A_free,boundary g, preserving symmetry.

    中文说明：对线性系统 A u = b 施加 Dirichlet 条件 u[nodes]=values，通过把已
    知值移项到右端（``b_free -= A_free,boundary g``）保持矩阵对称后只解自由块。
    返回 :class:`DirichletSolution`；若自由块奇异或数值校验不通过则抛出
    ``RuntimeError``。``nodes``/``values`` 长度须一致。
    """
    matrix = csr_matrix(matrix, dtype=float, copy=True)
    rhs = np.asarray(rhs, dtype=float)
    raw_nodes = np.asarray(nodes)
    n = matrix.shape[0]
    if matrix.shape != (n, n) or rhs.shape != (n,):
        raise ValueError("System must be square with a matching one-dimensional RHS.")
    if not np.all(np.isfinite(matrix.data)) or not np.all(np.isfinite(rhs)):
        raise ValueError("System contains non-finite entries.")
    if raw_nodes.ndim != 1 or not len(raw_nodes) or not np.issubdtype(raw_nodes.dtype, np.integer):
        raise ValueError("Dirichlet nodes must be a nonempty one-dimensional integer array.")
    nodes = np.array(raw_nodes, dtype=np.int64)
    if np.any(nodes < 0) or np.any(nodes >= n) or len(np.unique(nodes)) != len(nodes):
        raise ValueError("Dirichlet nodes must be unique and within the system.")
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full(len(nodes), float(values))
    if values.shape != (len(nodes),) or not np.all(np.isfinite(values)):
        raise ValueError("Dirichlet values must be finite and match the prescribed nodes.")
    free = np.setdiff1d(np.arange(n), nodes)
    solution = np.zeros(n)
    solution[nodes] = values
    if len(free):
        # 中文：把边界已知值的贡献移到右端，只保留自由自由度方程，保持对称。
        lifted_rhs = rhs[free] - matrix[free][:, nodes] @ values
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                solution[free] = spsolve(matrix[free][:, free], lifted_rhs)
        except (MatrixRankWarning, RuntimeError) as error:
            raise RuntimeError("Dirichlet solve failed: singular or invalid free system.") from error
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("Dirichlet solve produced non-finite values.")
    left = matrix[free] @ solution
    scale = max(float(np.linalg.norm(rhs[free])), float(np.linalg.norm(left)), 1.0)
    residual = float(np.linalg.norm(left - rhs[free]) / scale)
    if residual > 1e-10 and len(free):
        # Slender-solid systems can lose digits through cancellation. Refine the
        # solution using higher-precision residuals while keeping the same check.
        from scipy.sparse.linalg import splu

        factor = splu(matrix[free][:, free].tocsc())
        precise_matrix = matrix[free].astype(np.longdouble)
        for _ in range(3):
            defect = rhs[free].astype(np.longdouble)-precise_matrix @ solution.astype(np.longdouble)
            solution[free] += factor.solve(np.asarray(defect, dtype=float))
            left = precise_matrix @ solution.astype(np.longdouble)
            residual = float(np.linalg.norm(left-rhs[free].astype(np.longdouble))/scale)
            if residual <= 1e-10:
                break
    if len(free):
        # A scaled backward-error check distinguishes cancellation in a
        # well-solved ill-conditioned system from a failed linear solve.
        backward_scale = max(float(np.max(np.asarray(abs(matrix[free]).sum(axis=1))))*
                             float(np.max(abs(solution)))+float(np.max(abs(rhs[free]))), 1.0)
        backward_error = float(np.max(abs(left-rhs[free]))/backward_scale)
    else:
        backward_error = 0.0
    if residual > 1e-10 and backward_error > 1e-14:
        raise RuntimeError(f"Dirichlet solve residual={residual:.3e}, backward error={backward_error:.3e}.")
    return DirichletSolution(solution, free, residual, backward_error)


def solve_poisson(mesh: TriMesh, source, dirichlet, diffusion=1.0):
    """求解齐次/非齐次 Dirichlet 边界下的 P1 Poisson 问题。

    中文：先组装系统，再以全部外边界节点为 Dirichlet 节点，``dirichlet`` 给出
    边界值（常量、数组或可调用 (n,2)->(n,)），最后调用 :func:`solve_dirichlet`
    得到全节点解。
    """
    matrix, rhs = assemble_poisson(mesh, source, diffusion)
    nodes = mesh.boundary_nodes()
    prescribed = scalar_values(dirichlet, mesh.points[nodes], "Dirichlet data")
    return solve_dirichlet(matrix, rhs, nodes, prescribed)


def error_norms(mesh: TriMesh, values, exact, exact_gradient):
    """Quadrature L2 error and H1 seminorm error (not the full H1 norm).

    中文说明：用 5 阶求积计算数值解 ``values`` 相对精确解 ``exact`` 的 L2 误差，
    以及梯度相对 ``exact_gradient`` 的 H1 半范误差（只含梯度项，不含解本身）。
    返回字典 {"l2": ..., "h1_seminorm": ...}。
    """
    values = np.asarray(values, dtype=float)
    if values.shape != (mesh.n_nodes,) or not np.all(np.isfinite(values)):
        raise ValueError("Nodal solution must be finite and match the mesh.")
    points, weights = triangle_quadrature(5)
    physical_points = map_points(mesh, points)
    flat_points = physical_points.reshape(-1, 2)
    exact_values = scalar_values(exact, flat_points, "Exact solution").reshape(mesh.n_cells, -1)
    exact_gradients = np.asarray(exact_gradient(flat_points), dtype=float)
    if exact_gradients.shape != flat_points.shape or not np.all(np.isfinite(exact_gradients)):
        raise ValueError("Exact gradient must return a finite (n, 2) array.")
    exact_gradients = exact_gradients.reshape(mesh.n_cells, -1, 2)
    numerical_values = values[mesh.cells] @ shape_values(points).T
    numerical_gradients = np.einsum("ea,eai->ei", values[mesh.cells], physical_gradients(mesh))
    measure = mesh.determinants[:, None] * weights[None, :]
    l2_squared = np.sum(measure * (numerical_values - exact_values)**2)
    h1_squared = np.sum(measure * np.sum((numerical_gradients[:, None] - exact_gradients)**2, axis=2))
    return {"l2": float(np.sqrt(l2_squared)), "h1_seminorm": float(np.sqrt(h1_squared))}

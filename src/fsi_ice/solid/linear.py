"""P1 small-strain elasticity, with interleaved [ux0,uy0,ux1,uy1,...] DOFs.

本模块实现平面小变形线弹性：常应变 P1 三角形单元，位移自由度按节点交错排列为
[ux0, uy0, ux1, uy1, ...]。材料支持平面应力与平面应变两种模型；单元刚度由
Lamé 参数构成的 Voigt 本构（应变 [εxx, εyy, 2εxy]）给出，载荷支持一致体积力
与边线面力。
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix

from fsi_ice.fem.p1 import map_points, physical_gradients, shape_values
from fsi_ice.fem.poisson import solve_dirichlet
from fsi_ice.fem.quadrature import triangle_quadrature


@dataclass(frozen=True)
class ElasticMaterial:
    """线弹性材料参数（不可变数据类）。

    ``young`` 杨氏模量 E、``poisson`` 泊松比 ν、``model`` 取 "plane_stress" 或
    "plane_strain"、``thickness`` 单元厚度、``density`` 密度。约定 E>0、
    -1<ν<0.5、厚度与密度均为正，且所有参数有限。
    """

    young: float
    poisson: float
    model: str = "plane_strain"
    thickness: float = 1.0
    density: float = 1.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in (self.young, self.poisson, self.thickness, self.density)):
            raise ValueError("Material parameters must be finite.")
        # 中文：-1<ν<0.5 保证弹性张量正定且体积模量有限。
        if self.young <= 0 or not -1 < self.poisson < 0.5 or self.thickness <= 0 or self.density <= 0:
            raise ValueError("Require E>0, -1<nu<0.5, thickness>0 and density>0.")
        if self.model not in {"plane_stress", "plane_strain"}:
            raise ValueError("Material model must be plane_stress or plane_strain.")

    @property
    def lame(self):
        """返回 Lamé 参数 (λ, μ)。

        中文：μ 为剪切模量；平面应变下 λ 取三维值，平面应力下折算为
        2μλ/(λ+2μ) 以与平面应力本构保持一致。
        """
        mu = self.young / (2 * (1 + self.poisson))
        lam = self.young * self.poisson / ((1 + self.poisson) * (1 - 2*self.poisson))
        if self.model == "plane_stress":
            lam = 2 * mu * lam / (lam + 2 * mu)
        return lam, mu

    @property
    def constitutive(self):
        """平面弹性本构矩阵（Voigt 形式，3x3）。

        中文：作用于 Voigt 应变 [εxx, εyy, 2εxy]，返回应力 [σxx, σyy, σxy]；
        矩阵对称正定。平面应力与平面应变的差异已体现在 :attr:`lame` 中。
        """
        lam, mu = self.lame
        # Voigt strain = [epsilon_xx, epsilon_yy, 2 epsilon_xy].
        return np.array([[lam + 2*mu, lam, 0], [lam, lam + 2*mu, 0], [0, 0, mu]])


def vector_values(field, points, name):
    """把二维向量场统一为 (n,2) 数组。

    中文：``field`` 可为长度为 2 的常量向量、数组或可调用 (n,2)->(n,2)；常量
    会复制到每个点。若形状不符或含非有限值则抛出 ``ValueError``；``name`` 用于
    错误信息。
    """
    values = np.asarray(field(points) if callable(field) else field, dtype=float)
    if values.shape == (2,):
        values = np.tile(values, (len(points), 1))
    if values.shape != (len(points), 2) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must provide a finite (n,2) array or length-two constant.")
    return values


def vector_dofs(nodes):
    """由节点索引得到交错排列的位移自由度编号。

    中文：节点 k 对应自由度 2k（ux）与 2k+1（uy）。输入任意形状的整数节点数组，
    输出同前缀形状、末维为 2 的数组。
    """
    nodes = np.asarray(nodes)
    if not np.issubdtype(nodes.dtype, np.integer):
        raise ValueError("Node indices must be integers.")
    return (2*nodes[..., None] + np.array([0, 1])).reshape(*nodes.shape[:-1], -1)


def strain_matrices(mesh):
    """构造每单元 3x6 的应变-位移矩阵 B（Voigt 形式）。

    中文：对每个 P1 单元，B 作用在局部自由度 [ux0,uy0,...,ux2,uy2] 上得到
    [εxx, εyy, 2εxy]。行 0 取 x 方向梯度、行 1 取 y 方向梯度、行 2 为剪切行。
    """
    gradients = physical_gradients(mesh)
    matrices = np.zeros((mesh.n_cells, 3, 6))
    # 中文：奇数/偶数自由度分别对应 ux、uy，按列交错填入 B。
    matrices[:, 0, 0::2] = gradients[:, :, 0]
    matrices[:, 1, 1::2] = gradients[:, :, 1]
    matrices[:, 2, 0::2] = gradients[:, :, 1]
    matrices[:, 2, 1::2] = gradients[:, :, 0]
    return matrices


def assemble_elasticity(mesh, material: ElasticMaterial, body_force=(0.0, 0.0), tractions=None):
    """组装线弹性刚度矩阵与等效载荷向量。

    中文：返回 (matrix, rhs)。``matrix`` 为 2*n_nodes 阶 CSR 刚度矩阵，
    由 ∫ Bᵀ D B det(J) t 的单元积分装配；``rhs`` 含一致体积力（``body_force``，
    常量/数组/可调用）与面力（``tractions``，形如 {边界标签: 面力}）。厚度由
    ``material.thickness`` 计入，面力为当前边界单位长度上的力再乘厚度。
    """
    b_matrix = strain_matrices(mesh)
    local = np.einsum("eai,ab,ebj->eij", b_matrix, material.constitutive, b_matrix)
    # 中文：乘单元面积与厚度，得到单元刚度（面积 = det(J)/2）。
    local *= (mesh.areas * material.thickness)[:, None, None]
    dofs = vector_dofs(mesh.cells)
    rows = np.repeat(dofs, 6, axis=1).ravel()
    columns = np.tile(dofs, (1, 6)).ravel()
    matrix = coo_matrix((local.ravel(), (rows, columns)),
                        shape=(2*mesh.n_nodes, 2*mesh.n_nodes)).tocsr()
    reference, weights = triangle_quadrature(5)
    points = map_points(mesh, reference)
    force = vector_values(body_force, points.reshape(-1, 2), "Body force").reshape(mesh.n_cells, -1, 2)
    # 中文：一致体积力按 N_a 加权积分，再乘 det(J) 与厚度。
    local_rhs = np.einsum("q,eqi,qa->eai", weights, force, shape_values(reference))
    local_rhs *= (mesh.determinants * material.thickness)[:, None, None]
    rhs = np.zeros(2*mesh.n_nodes)
    np.add.at(rhs, dofs.ravel(), local_rhs.ravel())
    # Traction is force per current boundary area; multiply by the unit-depth thickness.
    # 中文：边上面力采用 2 点 Gauss 积分，点上取值后按 0.5 权重累加并乘边长与厚度。
    abscissae = np.array([0.5 - 0.5/np.sqrt(3), 0.5 + 0.5/np.sqrt(3)])
    edge_shapes = np.column_stack((1-abscissae, abscissae))
    for tag, traction in (tractions or {}).items():
        if tag not in mesh.boundaries:
            raise ValueError(f"Unknown traction boundary: {tag}")
        edges = mesh.boundaries[tag]
        vertices = mesh.points[edges]
        points = np.einsum("qa,eai->eqi", edge_shapes, vertices)
        values = vector_values(traction, points.reshape(-1, 2), "Traction").reshape(len(edges), 2, 2)
        lengths = np.linalg.norm(vertices[:, 1]-vertices[:, 0], axis=1)
        load = 0.5 * np.einsum("eqi,qa->eai", values, edge_shapes)
        load *= (lengths * material.thickness)[:, None, None]
        np.add.at(rhs, vector_dofs(edges).ravel(), load.ravel())
    return matrix, rhs


@dataclass(frozen=True)
class ElasticSolution:
    """线弹性求解结果。

    中文：``displacement`` 为节点位移 (n_nodes,2)；``strain`` 为单元应变
    (n_cells,3)（Voigt）；``stress`` 为单元应力 (n_cells,3)；``reactions`` 为
    节点反力 (n_nodes,2)；``relative_residual`` 为线性系统相对残差。
    """

    displacement: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    reactions: np.ndarray
    relative_residual: float


def solve_elasticity(mesh, material, prescribed_nodes, displacement=(0.0, 0.0),
                     *, body_force=(0.0, 0.0), tractions=None):
    """在给定位移边界下求解小变形线弹性问题。

    中文：``prescribed_nodes`` 为施加位移约束的节点；``displacement`` 为其位移值
    （常量/数组/可调用）。返回 :class:`ElasticSolution`，包含位移、单元应变/应力、
    节点反力与相对残差。约束通过 :func:`fsi_ice.fem.poisson.solve_dirichlet`
    以保持对称的方式施加。
    """
    nodes = np.asarray(prescribed_nodes)
    if nodes.ndim != 1 or not np.issubdtype(nodes.dtype, np.integer) or not len(nodes):
        raise ValueError("Prescribed nodes must be a nonempty integer vector.")
    if np.any(nodes < 0) or np.any(nodes >= mesh.n_nodes):
        raise ValueError("Prescribed node lies outside the mesh.")
    matrix, rhs = assemble_elasticity(mesh, material, body_force, tractions)
    values = vector_values(displacement, mesh.points[nodes], "Displacement")
    solution = solve_dirichlet(matrix, rhs, vector_dofs(nodes), values.ravel())
    # 中文：由位移恢复单元应变 ε=Bu 与应力 σ=Dε；反力为 K·u - f。
    strain = np.einsum("eai,ei->ea", strain_matrices(mesh), solution.values[vector_dofs(mesh.cells)])
    stress = strain @ material.constitutive.T
    return ElasticSolution(solution.values.reshape(-1, 2), strain, stress,
                           (matrix @ solution.values - rhs).reshape(-1, 2), solution.relative_residual)


def linear_residual(mesh, material, displacement, *, body_force=(0.0, 0.0), tractions=None):
    """计算线弹性残差 r = K·u - f 及刚度矩阵。

    中文：``displacement`` 为 (n_nodes,2) 位移场；返回 (residual, matrix)，可用于
    耦合迭代中的残差评估与雅可比更新。
    """
    values = np.asarray(displacement, dtype=float)
    if values.shape != (mesh.n_nodes, 2) or not np.all(np.isfinite(values)):
        raise ValueError("Displacement must be a finite (n_nodes,2) array.")
    matrix, rhs = assemble_elasticity(mesh, material, body_force, tractions)
    return matrix @ values.ravel() - rhs, matrix

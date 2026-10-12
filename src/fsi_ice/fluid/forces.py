"""Force exerted by the fluid on a tagged solid boundary, unit depth.

``boundary_cauchy_stress`` is the single Cauchy-stress kernel shared by the
fixed-mesh boundary monitors and the moving-interface traction assembly in
``fsi_ice.interface.traction``: it evaluates
``sigma = -p I + mu (grad u + grad u^T)`` at owner-cell local quadrature
points without averaging the fluid gradient across cells.

本模块计算流体作用在标记（tag）固体边界上的力，按单位深度（二维）给出。
关键约定：
- 柯西应力 sigma = -p I + mu (grad u + grad u^T) 在"拥有该边界边的流体单元"的局部
  求积点上求值，不做跨单元平均，因此单侧迹值即可；
- 法向取流体域外法向 n_fluid，边界力为 -∫ sigma·n_fluid ds（即作用在障碍物上的力）；
- 速度用 P2（含中边节点），压力用 P1（顶点）；边界积分用三点 Gauss-Legendre。
"""

import numpy as np

from fsi_ice.fem.p1 import shape_values
from fsi_ice.fem.p2 import p2_reference_gradients


def boundary_cauchy_stress(operators, solution, owner_cells, reference_points):
    """Cauchy stress of a flow state at boundary quadrature points.

    ``owner_cells`` are the fluid cells that own the boundary edges and
    ``reference_points`` are local coordinates in those cells, either shared
    ``(q, 2)`` or per-owner ``(e, q, 2)``; the result has shape
    ``(e, q, 2, 2)``.  Velocity is evaluated on the P2 nodes of each owner
    cell and pressure on its P1 vertices, so the stress uses the fluid trace
    value from one side only and never an inter-cell average.

    中文补充：在边界求积点返回流场的柯西应力张量。owner_cells 为拥有边界边的流体单元；
    reference_points 为这些单元的局部坐标，可为共享 (q,2) 或逐单元 (e,q,2)，输出 (e,q,2,2)。
    速度在各 owner 单元的 P2 节点上求值、压力在其 P1 顶点上求值，故应力只用单侧迹值。
    """
    mesh = operators.space.mesh
    cells = np.asarray(owner_cells)
    if cells.ndim != 1 or not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("owner_cells must be a one-dimensional integer array.")
    if not len(cells):
        raise ValueError("owner_cells must not be empty.")
    if np.any(cells < 0) or np.any(cells >= mesh.n_cells):
        raise ValueError("owner_cells contains a cell outside the mesh.")
    velocity = np.asarray(solution.velocity, dtype=float)
    if velocity.shape != (len(operators.space.points), 2) or not np.all(np.isfinite(velocity)):
        raise ValueError("Velocity must be a finite (n_P2_nodes, 2) array matching the operators.")
    pressure = np.asarray(solution.pressure, dtype=float)
    if pressure.shape != (mesh.n_nodes,) or not np.all(np.isfinite(pressure)):
        raise ValueError("Pressure must be a finite (n_P1_nodes,) array matching the operators.")
    points = np.asarray(reference_points, dtype=float)
    if points.ndim == 3:
        if points.shape[0] != len(cells) or points.shape[2] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("reference_points must be a finite (e, q, 2) array matching owner_cells.")
    elif points.ndim == 2:
        if points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("reference_points must be a finite (q, 2) array.")
    else:
        raise ValueError("reference_points must have shape (q, 2) or (e, q, 2).")
    inverse_jacobians = np.linalg.inv(mesh.jacobians)
    # P2 参考梯度经逐单元 J^{-1} 变换为物理梯度。
    reference = points if points.ndim == 2 else points.reshape(-1, 2)
    shape = shape_values(reference)
    n_points = reference.shape[0] if points.ndim == 2 else points.shape[1]
    gradients = np.empty((len(cells), n_points, 6, 2))
    if points.ndim == 2:
        reference_gradients = p2_reference_gradients(reference)
        for k, cell in enumerate(cells):
            gradients[k] = reference_gradients @ inverse_jacobians[cell]
        shape = shape[None, :, :]
    else:
        for k, cell in enumerate(cells):
            gradients[k] = p2_reference_gradients(points[k]) @ inverse_jacobians[cell]
        shape = shape.reshape(len(cells), n_points, 3)
    # grad u 在求积点取值：P2 节点速度与物理梯度基收缩（e 单元、q 求积点、i/j 分量）。
    gradient = np.einsum("eai,eqaj->eqij", velocity[operators.space.cells[cells]], gradients)
    # 压力在求积点取值：P1 顶点压力乘形函数。
    values = np.einsum("eqa,ea->eq", shape, pressure[mesh.cells[cells]])
    # sigma = mu (grad u + grad u^T) - p I。
    return operators.viscosity*(gradient+gradient.transpose(0, 1, 3, 2))-values[:, :, None, None]*np.eye(2)


class BoundaryMonitor:
    """Precompute fixed-mesh boundary quadrature for transient monitoring.

    The monitor caches the owner cells, reference quadrature points, boundary
    measures and outward normals of one tagged mesh configuration; the stress
    itself is evaluated through :func:`boundary_cauchy_stress`.  Reusing a
    monitor for another candidate geometry or viscosity is rejected.

    中文补充：预计算一条标记边界在固定网格上的边界积分数据（owner 单元、参考求积点、
    边界度量向量与流体外法向），供瞬态监测量（力/流量）复用；应力由 boundary_cauchy_stress
    现场计算。对另一个候选几何或黏度复用同一 monitor 会被拒绝。
    """

    def __init__(self, operators, tag):
        from fsi_ice.fem.p2 import p2_values

        mesh = operators.space.mesh
        owners = {}
        # 建立"边(排序后顶点对) -> 拥有它的单元及局部边"的映射，用于定位边界边的唯一流体 owner。
        for e, cell in enumerate(mesh.cells):
            for a, b in ((0, 1), (1, 2), (2, 0)):
                owners.setdefault(tuple(sorted((cell[a], cell[b]))), []).append((e, a, b))
        # 三点 Gauss-Legendre 映射到 [0,1] 作为边参数。
        s, w = np.polynomial.legendre.leggauss(3)
        s, w = (s+1)/2, w/2
        reference_vertices = np.array([[0, 0], [1, 0], [0, 1]])
        owner_cells, reference_points, velocity_nodes, velocity_basis, measures, normals = [], [], [], [], [], []
        for edge in mesh.boundaries[tag]:
            entries = owners[tuple(sorted(edge))]
            if len(entries) != 1:
                raise ValueError("Boundary monitor requires exterior edges.")
            e, a, b = entries[0]
            reference = (1-s[:, None])*reference_vertices[a]+s[:, None]*reference_vertices[b]
            tangent = mesh.points[mesh.cells[e, b]]-mesh.points[mesh.cells[e, a]]
            length = np.linalg.norm(tangent)
            owner_cells.append(e)
            reference_points.append(reference)
            velocity_nodes.append(operators.space.cells[e])
            velocity_basis.append(p2_values(reference))
            # 边界度量向量 = 边长 * Gauss 权重 dξ。
            measures.append(length*w)
            # 流体外法向 = rot_cw(切向) = (t_y, -t_x)/|t|（单元为 CCW 定向时指向外侧）。
            normals.append([tangent[1]/length, -tangent[0]/length])
        self.operators = operators
        self.viscosity = operators.viscosity
        self.geometry_signature = (mesh, float(operators.viscosity))
        self.owner_cells = np.array(owner_cells, dtype=np.int64)
        self.reference_points = np.array(reference_points)
        self.velocity_nodes = np.array(velocity_nodes)
        self.velocity_basis = np.array(velocity_basis)
        self.measures = np.array(measures)
        self.normals = np.array(normals)

    def require_current_geometry(self, operators):
        """Reject a stale monitor used for a different candidate mesh or viscosity.

        校验 monitor 与当前算子描述同一几何与黏度，否则抛出 ValueError（需重建 monitor）。
        """
        if operators.space.mesh is not self.geometry_signature[0] or float(operators.viscosity) != self.geometry_signature[1]:
            raise ValueError("BoundaryMonitor was built for a different geometry or viscosity; "
                             "rebuild it for the current candidate.")

    def force(self, solution, *, operators=None):
        """Force on the tagged boundary: -∫ sigma·n_fluid ds (unit depth).

        返回流体作用在标记边界上的合力（单位深度）。应力用边界柯西应力单侧求值，
        法向为流体外法向，整体取负号即得作用在障碍物上的力。传入 operators 时校验几何/黏度。
        """
        if operators is not None:
            self.require_current_geometry(operators)
        else:
            operators = self.operators
        stress = boundary_cauchy_stress(operators, solution, self.owner_cells, self.reference_points)
        # sigma·n_fluid 作用于流体，故其负值作用于障碍物。
        return -np.einsum("eq,eqij,ej->i", self.measures, stress, self.normals)

    def flux(self, solution, *, operators=None):
        """Net outward volumetric flux ∫ u·n_fluid ds across the tagged boundary (unit depth).

        返回穿过标记边界的净外法向体积流量（单位深度）；传入 operators 时校验几何/黏度。
        """
        if operators is not None:
            self.require_current_geometry(operators)
        velocity = np.einsum("eqa,eai->eqi", self.velocity_basis, solution.velocity[self.velocity_nodes])
        return float(np.einsum("eq,eqi,ei", self.measures, velocity, self.normals))


def boundary_force(operators, solution, tag):
    """Net fluid force on a tagged boundary via per-edge Gauss quadrature (unit depth).

    对标记边界的每条外部边做三点 Gauss 积分，累加 -∫ sigma·n_fluid ds，返回二维合力（单位深度）。
    要求边为外部边（唯一 owner）；速度用 P2、压力用 P1。输入：operators、FlowSolution、tag。
    """
    mesh = operators.space.mesh
    owners = {}
    for cell_id, cell in enumerate(mesh.cells):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            owners.setdefault(tuple(sorted((cell[a], cell[b]))), []).append((cell_id, a, b))
    abscissae, weights = np.polynomial.legendre.leggauss(3)
    abscissae, weights = (abscissae+1)/2, weights/2
    vertices = np.array([[0, 0], [1, 0], [0, 1]])
    total = np.zeros(2)
    for edge in mesh.boundaries[tag]:
        entries = owners[tuple(sorted(edge))]
        if len(entries) != 1:
            raise ValueError("Force integration requires exterior edges.")
        cell_id, a, b = entries[0]
        # 沿边 a->b 的参数化 Gauss 点（在局部坐标下）。
        reference = (1-abscissae[:, None])*vertices[a]+abscissae[:, None]*vertices[b]
        stress = boundary_cauchy_stress(operators, solution, np.array([cell_id]), reference[None, :, :])[0]
        tangent = mesh.points[mesh.cells[cell_id, b]]-mesh.points[mesh.cells[cell_id, a]]
        length = np.linalg.norm(tangent)
        outward_fluid = np.array([tangent[1], -tangent[0]])/length
        # 流体外法向 = rot_cw(切向)/|切向|。
        # sigma*n_fluid is the force on the fluid; its negative acts on the obstacle.
        total -= length*np.einsum("q,qij,j->i", weights, stress, outward_fluid)
    return total


def boundary_flux(operators, solution, tag):
    """Net outward volumetric flux ∫ u·n_fluid ds across a tagged boundary (unit depth).

    对标记边界的每条边做三点 Gauss 积分，累加 u·n_fluid ds，返回净外法向体积流量（标量，单位深度）。
    """
    from fsi_ice.fem.p2 import p2_values

    mesh = operators.space.mesh
    owners = {tuple(sorted((cell[a], cell[b]))): (e, a, b)
              for e, cell in enumerate(mesh.cells) for a, b in ((0, 1), (1, 2), (2, 0))}
    abscissae, weights = np.polynomial.legendre.leggauss(3)
    abscissae, weights = (abscissae+1)/2, weights/2
    vertices = np.array([[0, 0], [1, 0], [0, 1]])
    total = 0.0
    for edge in mesh.boundaries[tag]:
        e, a, b = owners[tuple(sorted(edge))]
        reference = (1-abscissae[:, None])*vertices[a]+abscissae[:, None]*vertices[b]
        # 边界上速度的 P2 插值。
        field = p2_values(reference) @ solution.velocity[operators.space.cells[e]]
        tangent = mesh.points[mesh.cells[e, b]]-mesh.points[mesh.cells[e, a]]
        total += float(np.einsum("q,qi,i", weights, field, [tangent[1], -tangent[0]]))
    return total

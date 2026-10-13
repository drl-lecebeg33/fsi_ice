"""Quadratic triangle basis [v0,v1,v2,edge01,edge12,edge20] on affine geometry.

本模块实现仿射三角形上的二次（P2）基。约定局部自由度顺序为
[v0, v1, v2, edge01, edge12, edge20]，前三个为顶点自由度，后三个为边中点
自由度；边按 (0,1)、(1,2)、(2,0) 归并。整个 P2 空间在网格原节点之后追加
各边中点节点，供高阶 Poisson 与误差分析使用。
"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from fsi_ice.fem.p1 import REFERENCE_GRADIENTS, shape_values


def p2_values(points):
    """P2 形函数在参考点处的取值。

    输入 ``points`` 形状 (q,2)；返回形状 (q,6)，列依次对应
    [v0, v1, v2, edge01, edge12, edge20]。其中顶点形函数为
    N_i = L_i(2L_i-1)，边中点为 4 L_i L_j（L 为 P1 重心坐标）。
    """
    barycentric = shape_values(points)
    return np.column_stack((barycentric*(2*barycentric-1),
                            4*barycentric[:, 0]*barycentric[:, 1],
                            4*barycentric[:, 1]*barycentric[:, 2],
                            4*barycentric[:, 2]*barycentric[:, 0]))


def p2_reference_gradients(points):
    """P2 形函数对参考坐标 (xi, eta) 的梯度。

    返回形状 (q,6,2)。前三个顶点形函数梯度为 (4L_i-1)·grad(L_i)；后三个
    边中位形函数由 4(L_i·grad(L_j) + L_j·grad(L_i)) 得到。
    """
    barycentric = shape_values(points)
    result = np.empty((len(points), 6, 2))
    result[:, :3] = (4*barycentric-1)[:, :, None]*REFERENCE_GRADIENTS
    for local, (i, j) in enumerate(((0, 1), (1, 2), (2, 0)), 3):
        result[:, local] = 4*(barycentric[:, i, None]*REFERENCE_GRADIENTS[j]
                             + barycentric[:, j, None]*REFERENCE_GRADIENTS[i])
    return result


def p2_physical_gradients(mesh, points):
    # 中文：以 J^-1 将参考梯度变换到物理坐标，返回形状 (n_cells, q, 6, 2)。
    return np.einsum("qai,eij->eqaj", p2_reference_gradients(points), np.linalg.inv(mesh.jacobians))


@dataclass(frozen=True)
class P2Space:
    """P2 有限元空间：在原网格上追加边中点形成的 6 自由度/单元空间。

    字段：``mesh`` 为原始 P1 网格；``points`` 为含中点的新节点坐标 (N+n_edges,2)；
    ``cells`` 为每单元 6 个全局自由度（前 3 为顶点、后 3 为边中点）；``edge_nodes``
    将排序后的无向边映射到其全局中点节点编号。对象为不可变（frozen，数组只读）。
    """

    mesh: object
    points: np.ndarray
    cells: np.ndarray
    edge_nodes: dict

    @classmethod
    def from_mesh(cls, mesh):
        """由 P1 网格构造 P2 空间：为每条唯一无向边生成一个中点节点。"""
        local_edges = np.stack((mesh.cells[:, [0, 1]], mesh.cells[:, [1, 2]],
                                mesh.cells[:, [2, 0]]), axis=1)
        edges, inverse = np.unique(np.sort(local_edges.reshape(-1, 2), axis=1), axis=0, return_inverse=True)
        midpoints = mesh.points[edges].mean(axis=1)
        points = np.vstack((mesh.points, midpoints))
        cells = np.column_stack((mesh.cells, mesh.n_nodes+inverse.reshape(-1, 3)))
        edge_nodes = {tuple(edge): mesh.n_nodes+i for i, edge in enumerate(edges)}
        points.setflags(write=False)
        cells.setflags(write=False)
        return cls(mesh, points, cells, MappingProxyType(edge_nodes))

    def boundary_nodes(self, tags=None):
        """返回 P2 空间边界节点（含边中点）。

        ``tags`` 为边界标签或标签序列；为 ``None`` 时取全部外边界。返回的整数
        数组包含边界边的两个端点及其边中点节点。
        """
        if tags is None:
            edges = self.mesh._exterior_edges
        else:
            if isinstance(tags, str):
                tags = [tags]
            for tag in tags:
                if tag not in self.mesh.boundaries:
                    raise ValueError(f"Unknown boundary tag: {tag}")
            if not tags:
                return np.array([], dtype=np.int64)
            edges = np.concatenate([self.mesh.boundaries[tag] for tag in tags])
        midpoint_nodes = [self.edge_nodes[tuple(sorted(edge))] for edge in edges]
        return np.unique(np.concatenate((edges.ravel(), midpoint_nodes))).astype(np.int64)

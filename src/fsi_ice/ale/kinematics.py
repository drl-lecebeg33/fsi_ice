"""BDF mesh velocity of a candidate ALE geometry (P1 plus P2 edge interpolation).

本模块由候选 ALE 几何与网格位移的 BDF 时间导数计算网格速度 w_kin。关键约定：
- 顶点网格速度来自 P1 位移的时间导数（由传入的 mesh_context 提供）；
- P2 节点网格速度通过边中点线性插值得到（P2 单元的中边节点速度 = 两端点平均）；
- 所有输出只读，且不推进任何历史。
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeshVelocity:
    """Mesh velocity at P1 vertices and embedded at P2 nodes (m/s).

    网格速度：values_p1 为 P1 顶点速度（与位移同形状），values_p2 为嵌入到 P2 节点
    （顶点在前、中边节点在后）的速度，单位 m/s；两者均为只读数组。
    """

    values_p1: np.ndarray
    values_p2: np.ndarray


def evaluate_mesh_velocity(geometry, mesh_context):
    """Evaluate w_kin = D_BDF d_m on one candidate without advancing any history.

    在单个候选上计算 w_kin = D_BDF d_m，不推进历史。mesh_context.derivative 作用于候选位移
    得到 P1 速度；再由边中点线性插值嵌入 P2 节点。返回只读的 MeshVelocity。
    输入：候选 geometry 与带 derivative() 的 mesh_context。
    """
    values_p1 = np.asarray(mesh_context.derivative(geometry.displacement), dtype=float)
    if values_p1.shape != geometry.displacement.shape or not np.all(np.isfinite(values_p1)):
        raise ValueError("Mesh velocity must be finite and match the P1 displacement shape.")
    space = geometry.current_space
    edges = np.asarray(tuple(space.edge_nodes.keys()), dtype=np.int64)
    if len(edges):
        # P2 中边节点速度 = 两端点速度的线性平均。
        midpoints = 0.5*(values_p1[edges[:, 0]] + values_p1[edges[:, 1]])
        values_p2 = np.vstack((values_p1, midpoints))
    else:
        # 退化情形（无内部边）：P2 速度直接取 P1 速度的副本。
        values_p2 = np.array(values_p1, copy=True)
    if values_p2.shape != space.points.shape or not np.all(np.isfinite(values_p2)):
        raise ValueError("P2 mesh velocity must be finite and match the current P2 space.")
    values_p1 = np.array(values_p1, copy=True)
    values_p2 = np.array(values_p2, copy=True)
    values_p1.setflags(write=False)
    values_p2.setflags(write=False)
    return MeshVelocity(values_p1, values_p2)

"""Scalar nodal P1 element; local DOFs are the three mesh cell nodes.

本模块实现标量 P1（线性）三角形单元。约定参考三角形顶点为 (0,0)、(1,0)、(0,1)，
三个形函数为 N0=1-xi-eta、N1=xi、N2=eta；局部自由度顺序与单元节点顺序一致。
物理单元由仿射映射 x = x0 + J·xi 生成，其中 J 为 2x2 Jacobian。函数的梯度均
按行排列，形状为 (..., 3, 2)。
"""

import numpy as np

from fsi_ice.mesh import TriMesh

# 参考单元三个形函数对 (xi, eta) 的梯度，行对应 N0、N1、N2。
REFERENCE_GRADIENTS = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
REFERENCE_GRADIENTS.setflags(write=False)


def shape_values(reference_points):
    """P1 形函数在给定参考点处的取值。

    输入 ``reference_points`` 形状 (q,2)；返回形状 (q,3)，每行是
    [N0, N1, N2]，三者和恒为 1（单位分解）。
    """
    points = np.asarray(reference_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("Reference points must be a finite (q, 2) array.")
    return np.column_stack((1 - points.sum(axis=1), points))


def physical_gradients(mesh: TriMesh):
    # For x = x0 + J xi, row gradients transform as grad_x(N) = grad_xi(N) J^-1.
    # 中文：物理坐标梯度由参考梯度右乘 J 的逆得到，逐单元求逆，返回形状 (n_cells,3,2)。
    return np.einsum("ai,eij->eaj", REFERENCE_GRADIENTS, np.linalg.inv(mesh.jacobians))


def map_points(mesh: TriMesh, reference_points):
    # 中文：仿射映射物理坐标 x = 单元首顶点 + J·xi，返回形状 (n_cells, q, 2)。
    points = np.asarray(reference_points, dtype=float)
    shape_values(points)  # Validate dimensions and finite values.
    return mesh.points[mesh.cells[:, 0], None, :] + np.einsum("qj,eij->eqi", points, mesh.jacobians)


def scalar_values(field, points, name):
    """Fields accept a flat (n, 2) array and return (n,) values or a scalar.

    中文说明：把标量场统一为长度 n 的数组。``field`` 可为常量、可调用对象或数组；
    若为可调用对象则对展平的 (n,2) 坐标求值；返回标量时会广播为 (n,)。``name``
    仅用于错误信息。若形状或数值不合法则抛出 ``ValueError``。
    """
    values = np.asarray(field(points) if callable(field) else field, dtype=float)
    if values.ndim == 0:
        values = np.full(len(points), float(values))
    if values.shape != (len(points),) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must return finite scalar values with shape (n,).")
    return values

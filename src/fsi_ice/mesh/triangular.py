"""Validated, counterclockwise P1 triangle meshes with tagged exterior edges.

本模块定义带校验的 P1 三角形网格 :class:`TriMesh`：要求节点按逆时针（CCW）排列、
无退化/重复单元且无未使用节点，并可对外边界边打标签以施加边界条件。约定单元
Jacobian 由首顶点出发的两条边构成，其行列式为正、面积 = det/2；边界标签仅允许
出现在网格的外边界边上。
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np


@dataclass(frozen=True)
class TriMesh:
    """不可变的 P1 三角形网格。

    中文：``points`` 为节点坐标 (n,2)；``cells`` 为单元节点索引 (m,3)，须逆时针；
    ``boundaries`` 为边界标签到外边数组的映射。对象构造时完成全部几何与拓扑校验，
    并冻结所有数组，构造后不可修改。
    """

    points: np.ndarray
    cells: np.ndarray
    boundaries: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        points = np.array(self.points, dtype=float, copy=True)
        raw_cells = np.asarray(self.cells)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError("Mesh points must have shape (n>=3, 2).")
        if not np.all(np.isfinite(points)):
            raise ValueError("Mesh coordinates must be finite.")
        if raw_cells.ndim != 2 or raw_cells.shape[1] != 3 or not len(raw_cells):
            raise ValueError("Mesh cells must have shape (m>=1, 3).")
        if not np.issubdtype(raw_cells.dtype, np.integer):
            raise ValueError("Cell indices must be integers.")
        cells = np.array(raw_cells, dtype=np.int64, copy=True)
        if np.any(cells < 0) or np.any(cells >= len(points)):
            raise ValueError("Cell index is outside the mesh points.")
        canonical = np.sort(cells, axis=1)
        if np.any(np.diff(canonical, axis=1) == 0):
            raise ValueError("A cell contains repeated node indices.")
        if len(np.unique(canonical, axis=0)) != len(cells):
            raise ValueError("Duplicate cells are not allowed.")
        if len(np.unique(cells)) != len(points):
            raise ValueError("Unused mesh nodes are not allowed.")
        vertices = points[cells]
        with np.errstate(over="ignore", invalid="ignore"):
            first = vertices[:, 1] - vertices[:, 0]
            second = vertices[:, 2] - vertices[:, 0]
            # 中文：2×2 Jacobian 行列式，其符号即单元定向（正为逆时针）。
            determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
            lengths_squared = np.stack([
                np.sum(first**2, axis=1), np.sum(second**2, axis=1),
                np.sum((vertices[:, 2] - vertices[:, 1])**2, axis=1),
            ], axis=1)
        if not np.all(np.isfinite(determinant)) or not np.all(np.isfinite(lengths_squared)):
            raise ValueError("Mesh geometry overflows floating-point arithmetic.")
        # 中文：以边长平方量级设定相对阈值，识别退化或数值上过扁的单元。
        tolerance = 32 * np.finfo(float).eps * lengths_squared.max(axis=1)
        invalid = np.flatnonzero(np.abs(determinant) <= tolerance)
        if len(invalid):
            raise ValueError(f"Degenerate or numerically flat cell {invalid[0]}.")
        # 中文：行列式为负表示顺时针排列，必须改为 CCW。
        reversed_cells = np.flatnonzero(determinant < 0)
        if len(reversed_cells):
            raise ValueError(f"Negative orientation in cell {reversed_cells[0]}; use CCW nodes.")
        # 中文：枚举每个单元的三条边，将节点索引排序后按行去重统计出现次数。
        edges = np.sort(np.concatenate([
            cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]],
        ]), axis=1)
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        if np.any(counts > 2):
            raise ValueError("Non-manifold mesh edge belongs to more than two cells.")
        # 中文：只被一个单元使用（计数为 1）的边即外边界边。
        exterior = unique_edges[counts == 1]
        exterior_set = {tuple(edge) for edge in exterior}
        boundary_map = {}
        tagged = set()
        # 中文：逐标签校验其边均为外边界边且不被重复标注。
        for tag, raw_edges in self.boundaries.items():
            boundary_edges = np.asarray(raw_edges)
            if not isinstance(tag, str) or not tag:
                raise ValueError("Boundary tags must be nonempty strings.")
            if boundary_edges.ndim != 2 or boundary_edges.shape[1] != 2:
                raise ValueError(f"Boundary {tag} must have shape (k, 2).")
            if not np.issubdtype(boundary_edges.dtype, np.integer):
                raise ValueError(f"Boundary {tag} indices must be integers.")
            boundary_edges = np.array(boundary_edges, dtype=np.int64, copy=True)
            for edge in np.sort(boundary_edges, axis=1):
                key = tuple(edge)
                if key not in exterior_set:
                    raise ValueError(f"Boundary {tag} contains a non-exterior edge {key}.")
                if key in tagged:
                    raise ValueError(f"Exterior edge {key} is tagged more than once.")
                tagged.add(key)
            boundary_edges.setflags(write=False)
            boundary_map[tag] = boundary_edges
        points.setflags(write=False)
        cells.setflags(write=False)
        exterior.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "boundaries", MappingProxyType(boundary_map))
        object.__setattr__(self, "_exterior_edges", exterior)

    @property
    def n_nodes(self):
        return len(self.points)

    @property
    def n_cells(self):
        return len(self.cells)

    @property
    def jacobians(self):
        """单元 Jacobian J，形状 (n_cells,2,2)。

        中文：以首顶点为原点，两列分别为边 (v0→v1) 与 (v0→v2)，即仿射映射
        x = v0 + J·xi 的矩阵。
        """
        vertices = self.points[self.cells]
        return np.stack((vertices[:, 1] - vertices[:, 0],
                         vertices[:, 2] - vertices[:, 0]), axis=2)

    @property
    def determinants(self):
        """各单元 J 的行列式，等于两倍单元面积；逆时针单元为正。"""
        jacobian = self.jacobians
        return jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]

    @property
    def areas(self):
        """各单元面积，等于 det(J)/2。"""
        return 0.5 * self.determinants

    @property
    def h(self):
        """全网格最大单元边长，用作网格尺度 h。"""
        vertices = self.points[self.cells]
        edges = vertices - np.roll(vertices, 1, axis=1)
        return float(np.sqrt(np.sum(edges**2, axis=2)).max())

    @property
    def quality(self):
        """2 sqrt(3) det(J) / sum(edge_length^2); equilateral quality is one.

        中文：等边三角形取值为 1，越接近退化其值越小，可用于评估单元质量。
        """
        vertices = self.points[self.cells]
        lengths_squared = np.sum((vertices - np.roll(vertices, 1, axis=1))**2, axis=2)
        return 2 * np.sqrt(3) * self.determinants / lengths_squared.sum(axis=1)

    def boundary_nodes(self, tags=None):
        """返回边界节点（去重）。

        中文：``tags`` 为边界标签或标签序列；为 ``None`` 时返回全部外边界节点，
        标签为空序列时返回空数组；未知标签抛出 ``ValueError``。
        """
        if tags is None:
            return np.unique(self._exterior_edges)
        if isinstance(tags, str):
            tags = [tags]
        if not tags:
            return np.array([], dtype=np.int64)
        for tag in tags:
            if tag not in self.boundaries:
                raise ValueError(f"Unknown boundary tag: {tag}")
        return np.unique(np.concatenate([self.boundaries[tag].ravel() for tag in tags]))


def rectangle_mesh(nx, ny=None, *, width=1.0, height=1.0):
    """Split each rectangular grid cell along its lower-left/upper-right diagonal.

    中文：在矩形区域 [0,width]×[0,height] 上生成 nx×ny 个矩形网格，每个矩形沿
    左下—右上对角线剖分为两个逆时针三角形。``ny`` 省略时取 ``nx``。返回带
    "bottom"/"right"/"top"/"left" 四个边界标签的 :class:`TriMesh`。
    """
    ny = nx if ny is None else ny
    for name, value in (("nx", nx), ("ny", ny)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError("Rectangle dimensions must be finite and positive.")
    x, y = np.meshgrid(np.linspace(0, width, nx + 1), np.linspace(0, height, ny + 1))
    points = np.column_stack((x.ravel(), y.ravel()))
    node = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
    a, b = node[:-1, :-1].ravel(), node[:-1, 1:].ravel()
    c, d = node[1:, :-1].ravel(), node[1:, 1:].ravel()
    # 中文：每个矩形拆为 (a,b,d) 与 (a,d,c) 两个逆时针三角形。
    cells = np.vstack((np.column_stack((a, b, d)), np.column_stack((a, d, c))))
    # 中文：四条边按逆时针环绕编号，保证每条边界边方向一致。
    boundaries = {
        "bottom": np.column_stack((node[0, :-1], node[0, 1:])),
        "right": np.column_stack((node[:-1, -1], node[1:, -1])),
        "top": np.column_stack((node[-1, 1:], node[-1, :-1])),
        "left": np.column_stack((node[1:, 0], node[:-1, 0])),
    }
    return TriMesh(points, cells, boundaries)

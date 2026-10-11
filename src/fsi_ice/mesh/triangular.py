"""Validated, counterclockwise P1 triangle meshes with tagged exterior edges."""

from dataclasses import dataclass, field
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np


@dataclass(frozen=True)
class TriMesh:
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
            determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
            lengths_squared = np.stack([
                np.sum(first**2, axis=1), np.sum(second**2, axis=1),
                np.sum((vertices[:, 2] - vertices[:, 1])**2, axis=1),
            ], axis=1)
        if not np.all(np.isfinite(determinant)) or not np.all(np.isfinite(lengths_squared)):
            raise ValueError("Mesh geometry overflows floating-point arithmetic.")
        tolerance = 32 * np.finfo(float).eps * lengths_squared.max(axis=1)
        invalid = np.flatnonzero(np.abs(determinant) <= tolerance)
        if len(invalid):
            raise ValueError(f"Degenerate or numerically flat cell {invalid[0]}.")
        reversed_cells = np.flatnonzero(determinant < 0)
        if len(reversed_cells):
            raise ValueError(f"Negative orientation in cell {reversed_cells[0]}; use CCW nodes.")
        edges = np.sort(np.concatenate([
            cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]],
        ]), axis=1)
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        if np.any(counts > 2):
            raise ValueError("Non-manifold mesh edge belongs to more than two cells.")
        exterior = unique_edges[counts == 1]
        exterior_set = {tuple(edge) for edge in exterior}
        boundary_map = {}
        tagged = set()
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
        vertices = self.points[self.cells]
        return np.stack((vertices[:, 1] - vertices[:, 0],
                         vertices[:, 2] - vertices[:, 0]), axis=2)

    @property
    def determinants(self):
        jacobian = self.jacobians
        return jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]

    @property
    def areas(self):
        return 0.5 * self.determinants

    @property
    def h(self):
        vertices = self.points[self.cells]
        edges = vertices - np.roll(vertices, 1, axis=1)
        return float(np.sqrt(np.sum(edges**2, axis=2)).max())

    @property
    def quality(self):
        """2 sqrt(3) det(J) / sum(edge_length^2); equilateral quality is one."""
        vertices = self.points[self.cells]
        lengths_squared = np.sum((vertices - np.roll(vertices, 1, axis=1))**2, axis=2)
        return 2 * np.sqrt(3) * self.determinants / lengths_squared.sum(axis=1)

    def boundary_nodes(self, tags=None):
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
    """Split each rectangular grid cell along its lower-left/upper-right diagonal."""
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
    cells = np.vstack((np.column_stack((a, b, d)), np.column_stack((a, d, c))))
    boundaries = {
        "bottom": np.column_stack((node[0, :-1], node[0, 1:])),
        "right": np.column_stack((node[:-1, -1], node[1:, -1])),
        "top": np.column_stack((node[-1, 1:], node[-1, :-1])),
        "left": np.column_stack((node[1:, 0], node[:-1, 0])),
    }
    return TriMesh(points, cells, boundaries)

"""Quadratic triangle basis [v0,v1,v2,edge01,edge12,edge20] on affine geometry."""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from fsi_ice.fem.p1 import REFERENCE_GRADIENTS, shape_values


def p2_values(points):
    barycentric = shape_values(points)
    return np.column_stack((barycentric*(2*barycentric-1),
                            4*barycentric[:, 0]*barycentric[:, 1],
                            4*barycentric[:, 1]*barycentric[:, 2],
                            4*barycentric[:, 2]*barycentric[:, 0]))


def p2_reference_gradients(points):
    barycentric = shape_values(points)
    result = np.empty((len(points), 6, 2))
    result[:, :3] = (4*barycentric-1)[:, :, None]*REFERENCE_GRADIENTS
    for local, (i, j) in enumerate(((0, 1), (1, 2), (2, 0)), 3):
        result[:, local] = 4*(barycentric[:, i, None]*REFERENCE_GRADIENTS[j]
                             + barycentric[:, j, None]*REFERENCE_GRADIENTS[i])
    return result


def p2_physical_gradients(mesh, points):
    return np.einsum("qai,eij->eqaj", p2_reference_gradients(points), np.linalg.inv(mesh.jacobians))


@dataclass(frozen=True)
class P2Space:
    mesh: object
    points: np.ndarray
    cells: np.ndarray
    edge_nodes: dict

    @classmethod
    def from_mesh(cls, mesh):
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

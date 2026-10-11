"""BDF mesh velocity of a candidate ALE geometry (P1 plus P2 edge interpolation)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeshVelocity:
    """Mesh velocity at P1 vertices and embedded at P2 nodes (m/s)."""

    values_p1: np.ndarray
    values_p2: np.ndarray


def evaluate_mesh_velocity(geometry, mesh_context):
    """Evaluate w_kin = D_BDF d_m on one candidate without advancing any history."""
    values_p1 = np.asarray(mesh_context.derivative(geometry.displacement), dtype=float)
    if values_p1.shape != geometry.displacement.shape or not np.all(np.isfinite(values_p1)):
        raise ValueError("Mesh velocity must be finite and match the P1 displacement shape.")
    space = geometry.current_space
    edges = np.asarray(tuple(space.edge_nodes.keys()), dtype=np.int64)
    if len(edges):
        midpoints = 0.5*(values_p1[edges[:, 0]] + values_p1[edges[:, 1]])
        values_p2 = np.vstack((values_p1, midpoints))
    else:
        values_p2 = np.array(values_p1, copy=True)
    if values_p2.shape != space.points.shape or not np.all(np.isfinite(values_p2)):
        raise ValueError("P2 mesh velocity must be finite and match the current P2 space.")
    values_p1 = np.array(values_p1, copy=True)
    values_p2 = np.array(values_p2, copy=True)
    values_p1.setflags(write=False)
    values_p2.setflags(write=False)
    return MeshVelocity(values_p1, values_p2)

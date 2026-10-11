"""Scalar nodal P1 element; local DOFs are the three mesh cell nodes."""

import numpy as np

from fsi_ice.mesh import TriMesh

REFERENCE_GRADIENTS = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
REFERENCE_GRADIENTS.setflags(write=False)


def shape_values(reference_points):
    points = np.asarray(reference_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("Reference points must be a finite (q, 2) array.")
    return np.column_stack((1 - points.sum(axis=1), points))


def physical_gradients(mesh: TriMesh):
    # For x = x0 + J xi, row gradients transform as grad_x(N) = grad_xi(N) J^-1.
    return np.einsum("ai,eij->eaj", REFERENCE_GRADIENTS, np.linalg.inv(mesh.jacobians))


def map_points(mesh: TriMesh, reference_points):
    points = np.asarray(reference_points, dtype=float)
    shape_values(points)  # Validate dimensions and finite values.
    return mesh.points[mesh.cells[:, 0], None, :] + np.einsum("qj,eij->eqi", points, mesh.jacobians)


def scalar_values(field, points, name):
    """Fields accept a flat (n, 2) array and return (n,) values or a scalar."""
    values = np.asarray(field(points) if callable(field) else field, dtype=float)
    if values.ndim == 0:
        values = np.full(len(points), float(values))
    if values.shape != (len(points),) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must return finite scalar values with shape (n,).")
    return values

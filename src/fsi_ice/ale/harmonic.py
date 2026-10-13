"""Harmonic (Laplace) extension of prescribed P1 boundary displacements."""

from dataclasses import dataclass

import numpy as np

from fsi_ice.fem.poisson import assemble_poisson, solve_dirichlet
from fsi_ice.mesh import TriMesh

LENGTH_SCALE = 1.0  # Frozen validation length scale L*=1 m (m3_freeze.md section 6).


@dataclass(frozen=True)
class MeshDisplacement:
    """Absolute reference-domain P1 displacement d_m with its solve diagnostics."""

    values: np.ndarray
    component_residuals: np.ndarray
    boundary_error: float
    boundary_nodes: np.ndarray


class HarmonicExtension:
    """Reusable Laplace stiffness on one reference mesh for both P1 components."""

    def __init__(self, reference_mesh, *, diffusion=1.0):
        if not isinstance(reference_mesh, TriMesh):
            raise ValueError("reference_mesh must be a TriMesh.")
        if callable(diffusion):
            value = diffusion
        else:
            if isinstance(diffusion, bool):
                raise ValueError("diffusion must be a finite positive scalar or reference field.")
            try:
                value = float(diffusion)
            except (TypeError, ValueError):
                raise ValueError("diffusion must be a finite positive scalar or reference field.") from None
            if not np.isfinite(value) or value <= 0:
                raise ValueError("diffusion must be a finite positive scalar or reference field.")
        self.reference_mesh = reference_mesh
        self.diffusion = value
        self._matrix, self._rhs = assemble_poisson(reference_mesh, 0.0, value)
        self._matrix.data.setflags(write=False)
        self._matrix.indices.setflags(write=False)
        self._matrix.indptr.setflags(write=False)
        self._rhs.setflags(write=False)

    @property
    def stiffness(self):
        """Read-only P1 Laplace stiffness ``K_h`` assembled on the reference mesh.

        This is the same matrix :meth:`extend` solves with (zero source).  The
        arrays are frozen, so a caller can reuse it directly, e.g. as the mesh
        block of the monolithic residual, without copying or risking an
        accidental modification of the shared extension operator.
        """
        return self._matrix

    def extend(self, boundary_nodes, displacement_values):
        """Solve -Div(diffusion grad(d_m)) = 0 with the given exterior values."""
        mesh = self.reference_mesh
        raw_nodes = np.asarray(boundary_nodes)
        if raw_nodes.ndim != 1 or raw_nodes.size == 0 or not np.issubdtype(raw_nodes.dtype, np.integer):
            raise ValueError("Boundary nodes must be a nonempty one-dimensional integer array of P1 vertices.")
        nodes = np.array(raw_nodes, dtype=np.int64)
        if np.any(nodes < 0) or np.any(nodes >= mesh.n_nodes):
            raise ValueError(f"Unknown boundary node; valid P1 vertices are 0..{mesh.n_nodes - 1}.")
        if len(np.unique(nodes)) != len(nodes):
            raise ValueError("Boundary nodes must be unique; duplicate or conflicting prescriptions are rejected.")
        exterior = mesh.boundary_nodes()
        missing = np.setdiff1d(exterior, nodes)
        if missing.size:
            raise ValueError(
                "Missing exterior boundary node(s) "
                f"{missing[:8].tolist() if missing.size > 8 else missing.tolist()}; "
                "every external boundary vertex must be prescribed."
            )
        values = np.asarray(displacement_values, dtype=float)
        if values.shape != (len(nodes), 2):
            raise ValueError(f"displacement_values must have shape ({len(nodes)}, 2), got {values.shape}.")
        if not np.all(np.isfinite(values)):
            raise ValueError("displacement_values must be finite.")
        solution = np.empty((mesh.n_nodes, 2))
        residuals = np.empty(2)
        for component in range(2):
            solved = solve_dirichlet(self._matrix, self._rhs, nodes, values[:, component])
            solution[:, component] = solved.values
            residuals[component] = solved.relative_residual
        boundary_error = float(np.max(np.abs(solution[nodes] - values)) / LENGTH_SCALE)
        solution.setflags(write=False)
        residuals.setflags(write=False)
        prescribed = np.array(nodes, copy=True)
        prescribed.setflags(write=False)
        return MeshDisplacement(solution, residuals, boundary_error, prescribed)


def build_harmonic_extension(reference_mesh, *, diffusion=1.0):
    """Assemble the reference Laplace stiffness once; extend() solves candidates."""
    return HarmonicExtension(reference_mesh, diffusion=diffusion)


def build_interface_weighted_extension(reference_mesh, *, interface_tag, length, power=1.0):
    """Fixed reference weight (length/sqrt(distance**2+length**2))**power.

    Distance is to the reference interface segments, in mesh length units.
    The positive scalar weight stiffens the near-interface region relative to
    the far field. It changes the mesh equation only; it is not a material law.
    Reference weights give exactly the same constant stiffness to partitioned
    extension and the monolithic mesh residual/Jacobian.
    """
    for name, value in (("length", length), ("power", power)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or \
                not np.isfinite(value) or value <= 0:
            raise ValueError(f"Mesh weight {name} must be a finite positive number.")
    if interface_tag not in reference_mesh.boundaries:
        raise ValueError(f"Unknown mesh weight interface tag: {interface_tag}.")
    segments = reference_mesh.points[reference_mesh.boundaries[interface_tag]].copy()
    if not len(segments):
        raise ValueError("Mesh weight interface must contain boundary segments.")
    segments.setflags(write=False)

    def weight(points):
        distance_squared = np.full(len(points), np.inf)
        for start, end in segments:
            direction = end-start
            fraction = np.clip((points-start) @ direction / (direction @ direction), 0., 1.)
            delta = points-start-fraction[:, None]*direction
            distance_squared = np.minimum(distance_squared, np.sum(delta*delta, axis=1))
        return (length/np.sqrt(distance_squared+length*length))**power

    return HarmonicExtension(reference_mesh, diffusion=weight)

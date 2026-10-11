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
        if isinstance(diffusion, bool):
            raise ValueError("diffusion must be a finite positive scalar.")
        try:
            value = float(diffusion)
        except (TypeError, ValueError):
            raise ValueError("diffusion must be a finite positive scalar.") from None
        if not np.isfinite(value) or value <= 0:
            raise ValueError("diffusion must be a finite positive scalar.")
        self.reference_mesh = reference_mesh
        self.diffusion = value
        self._matrix, self._rhs = assemble_poisson(reference_mesh, 0.0, value)

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

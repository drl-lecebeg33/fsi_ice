"""Candidate ALE geometry: reference mesh plus an absolute P1 displacement."""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from fsi_ice.fem.p2 import P2Space
from fsi_ice.mesh import TriMesh

DEFAULT_QUALITY_LIMITS = MappingProxyType({"min_volume_ratio": 0.2, "min_quality": 0.1})


@dataclass(frozen=True)
class ALEGeometry:
    """Frozen candidate configuration with quality and topology diagnostics."""

    reference_mesh: TriMesh
    current_mesh: TriMesh
    reference_space: P2Space
    current_space: P2Space
    displacement: np.ndarray
    deformation_gradients: np.ndarray
    volume_ratios: np.ndarray
    quality: np.ndarray
    minimum_volume_ratio: float
    minimum_quality: float
    quality_limits: MappingProxyType
    candidate_id: object


def _resolve_quality_limits(quality_limits):
    if quality_limits is None:
        return DEFAULT_QUALITY_LIMITS
    try:
        provided = dict(quality_limits)
    except (TypeError, ValueError):
        raise ValueError("quality_limits must be a mapping with min_volume_ratio and min_quality.") from None
    unknown = sorted(set(provided) - set(DEFAULT_QUALITY_LIMITS))
    if unknown:
        raise ValueError(f"quality_limits contains unknown key(s) {unknown}.")
    resolved = {}
    for name, default in DEFAULT_QUALITY_LIMITS.items():
        value = provided.get(name, default)
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"quality_limits[{name!r}] must be a finite positive number.")
        resolved[name] = float(value)
    return MappingProxyType(resolved)


def build_ale_geometry(reference_mesh, displacement, *, quality_limits=None, candidate_id):
    """Build the frozen current configuration x = X + d_m without touching X."""
    if not isinstance(reference_mesh, TriMesh):
        raise ValueError("reference_mesh must be a TriMesh.")
    limits = _resolve_quality_limits(quality_limits)
    values = np.asarray(displacement, dtype=float)
    if values.shape != reference_mesh.points.shape:
        raise ValueError(
            f"candidate {candidate_id!r}: displacement must have shape {reference_mesh.points.shape}, "
            f"got {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"candidate {candidate_id!r}: displacement contains non-finite values.")
    frozen = np.array(values, copy=True)
    frozen.setflags(write=False)
    current_points = reference_mesh.points + frozen
    if not np.all(np.isfinite(current_points)):
        raise ValueError(f"candidate {candidate_id!r}: current coordinates are non-finite.")
    vertices = current_points[reference_mesh.cells]
    first = vertices[:, 1] - vertices[:, 0]
    second = vertices[:, 2] - vertices[:, 0]
    with np.errstate(over="ignore", invalid="ignore"):
        determinant_current = first[:, 0]*second[:, 1] - first[:, 1]*second[:, 0]
    if not np.all(np.isfinite(determinant_current)):
        raise ValueError(f"candidate {candidate_id!r}: current cell determinants over/underflow.")
    if np.any(determinant_current <= 0):
        index = int(np.argmin(determinant_current))
        raise ValueError(
            f"candidate {candidate_id!r}: cell {index} has det(J_x)={determinant_current[index]:.12g}; "
            f"minimum det(J_x)={float(determinant_current.min()):.12g} <= 0 (orientation flip)."
        )
    try:
        current_mesh = TriMesh(current_points, reference_mesh.cells, reference_mesh.boundaries)
    except ValueError as error:
        raise ValueError(f"candidate {candidate_id!r}: rejected current mesh: {error}") from error
    inverse_reference = np.linalg.inv(reference_mesh.jacobians)
    deformation = np.einsum("eij,ejk->eik", current_mesh.jacobians, inverse_reference)
    volume_ratios = deformation[:, 0, 0]*deformation[:, 1, 1] - deformation[:, 0, 1]*deformation[:, 1, 0]
    quality = np.array(current_mesh.quality)
    minimum_volume_ratio = float(np.min(volume_ratios))
    minimum_quality = float(np.min(quality))
    if minimum_volume_ratio < limits["min_volume_ratio"]:
        index = int(np.argmin(volume_ratios))
        raise ValueError(
            f"candidate {candidate_id!r}: cell {index} has volume ratio {volume_ratios[index]:.12g}; "
            f"minimum volume ratio {minimum_volume_ratio:.12g} < {limits['min_volume_ratio']:.12g}."
        )
    if minimum_quality < limits["min_quality"]:
        index = int(np.argmin(quality))
        raise ValueError(
            f"candidate {candidate_id!r}: cell {index} has quality {quality[index]:.12g}; "
            f"minimum quality {minimum_quality:.12g} < {limits['min_quality']:.12g}."
        )
    reference_space = P2Space.from_mesh(reference_mesh)
    current_space = P2Space.from_mesh(current_mesh)
    if not np.array_equal(reference_space.cells, current_space.cells) or (
        dict(reference_space.edge_nodes) != dict(current_space.edge_nodes)
    ):
        raise ValueError(f"candidate {candidate_id!r}: P2 topology changed under the displacement.")
    deformation.setflags(write=False)
    volume_ratios.setflags(write=False)
    quality.setflags(write=False)
    return ALEGeometry(
        reference_mesh=reference_mesh,
        current_mesh=current_mesh,
        reference_space=reference_space,
        current_space=current_space,
        displacement=frozen,
        deformation_gradients=deformation,
        volume_ratios=volume_ratios,
        quality=quality,
        minimum_volume_ratio=minimum_volume_ratio,
        minimum_quality=minimum_quality,
        quality_limits=limits,
        candidate_id=candidate_id,
    )

"""Current interface geometry for a fixed reference topology.

``evaluate_interface_geometry`` freezes one candidate configuration of the wet
interface: per-edge current quadrature points, lengths, outward normals of both
domains, the solid deformation gradient ``F_s``/``J_s`` and the surface ratio
``j_Gamma`` computed along two independent paths (current length ratio and the
Nanson formula ``J_s ||F_s^-T N_s||``).  The reference topology is matched once
and is never re-matched here; the reference meshes, the solid displacement
input and every time history stay untouched.

Candidate versions are rejected instead of repaired: a different connectivity,
a non-CCW current mesh, a non-positive solid Jacobian, a degenerate current
edge or a gap between the solid and fluid current interface raises a
``ValueError`` that reports the offending edge or node.
"""

from dataclasses import dataclass

import numpy as np

from fsi_ice.fem.p1 import physical_gradients
from fsi_ice.interface.topology import InterfaceTopology, _digest

REFERENCE_VERTICES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
REFERENCE_VERTICES.setflags(write=False)
EDGE_COLUMNS = {(0, 1): 3, (1, 2): 4, (2, 0): 5}


def _read_only(array):
    array.setflags(write=False)
    return array


def _quadrature():
    abscissae, weights = np.polynomial.legendre.leggauss(3)
    return (abscissae+1)/2, weights/2


def _edge_local(cells, owner, edge, side):
    """Recover the CCW local endpoint indices of ``edge`` in its owner cell."""
    row = np.asarray(cells)[owner]
    first = np.flatnonzero(row == edge[0])
    second = np.flatnonzero(row == edge[1])
    if len(first) != 1 or len(second) != 1:
        raise ValueError(f"Internal {side} edge ({edge[0]}, {edge[1]}) is not part of owner cell {int(owner)}.")
    return int(first[0]), int(second[0])


@dataclass(frozen=True)
class InterfaceGeometry:
    """Frozen candidate interface geometry for one topology and version.

    Per-edge arrays are indexed by ``edge_pairs`` order.  ``x_q`` are the
    current quadrature coordinates of the solid side, ``lengths`` the current
    solid edge lengths and ``n_s``/``n_f`` the current outward normals of the
    two domains; ``j_Gamma_current`` is ``l/L0`` while ``j_Gamma_nanson`` is
    ``J_s ||F_s^-T N_s||``.  All arrays are read-only and no history is stored.
    """

    topology_id: str
    candidate_id: object
    geometry_id: str
    current_mesh: object
    current_space: object
    solid_displacement: np.ndarray
    abscissae: np.ndarray
    weights: np.ndarray
    solid_owner: np.ndarray
    fluid_owner: np.ndarray
    solid_local: np.ndarray
    fluid_local: np.ndarray
    reference_points_solid: np.ndarray
    reference_points_fluid: np.ndarray
    x_q: np.ndarray
    lengths: np.ndarray
    fluid_lengths: np.ndarray
    reference_lengths: np.ndarray
    n_s: np.ndarray
    n_f: np.ndarray
    F_s: np.ndarray
    J_s: np.ndarray
    j_Gamma_current: np.ndarray
    j_Gamma_nanson: np.ndarray
    nanson_normals: np.ndarray
    vertex_coordinates: np.ndarray
    fluid_p2_coordinates: np.ndarray
    normal_error: float
    normalization_error: float
    nanson_normal_error: float
    surface_ratio_error: float
    gap_error: float
    gap_tolerance: float
    minimum_jacobian: float
    minimum_cell_determinant: float

    def __post_init__(self):
        for name in ("solid_displacement", "abscissae", "weights", "solid_owner", "fluid_owner",
                     "solid_local", "fluid_local", "reference_points_solid", "reference_points_fluid",
                     "x_q", "lengths", "fluid_lengths", "reference_lengths", "n_s", "n_f", "F_s",
                     "J_s", "j_Gamma_current", "j_Gamma_nanson", "nanson_normals",
                     "vertex_coordinates", "fluid_p2_coordinates"):
            _read_only(getattr(self, name))

    @property
    def n_edges(self):
        return len(self.lengths)

    @property
    def n_quadrature(self):
        return len(self.weights)

    def require_current(self, operators):
        """Reject operators assembled on another candidate geometry or version."""
        mesh = getattr(getattr(operators, "space", None), "mesh", None)
        if mesh is not self.current_mesh:
            raise ValueError("InterfaceGeometry was evaluated for a different candidate mesh; "
                             "rebuild the geometry and operators for the current candidate.")
        space = operators.space
        if (not np.array_equal(np.asarray(space.cells), np.asarray(self.current_space.cells))
                or dict(space.edge_nodes) != dict(self.current_space.edge_nodes)):
            raise ValueError("Operators use a different interface topology/version than the geometry.")


def _validate_mesh_like(mesh, side):
    points = np.asarray(mesh.points, dtype=float)
    cells = np.asarray(mesh.cells)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError(f"The current {side} mesh must provide finite (n, 2) coordinates.")
    if cells.ndim != 2 or cells.shape[1] != 3 or not np.issubdtype(cells.dtype, np.integer):
        raise ValueError(f"The current {side} mesh must provide an integer (m, 3) connectivity.")
    if np.any(cells < 0) or np.any(cells >= len(points)):
        raise ValueError(f"The current {side} mesh contains a cell index outside the points.")
    return points, cells


def _current_determinants(points, cells):
    vertices = points[cells]
    first = vertices[:, 1]-vertices[:, 0]
    second = vertices[:, 2]-vertices[:, 0]
    with np.errstate(over="ignore", invalid="ignore"):
        determinant = first[:, 0]*second[:, 1]-first[:, 1]*second[:, 0]
    if not np.all(np.isfinite(determinant)):
        raise ValueError("Current mesh determinants over/underflow.")
    bad = np.flatnonzero(determinant <= 0)
    if len(bad):
        raise ValueError(f"Current mesh cell {int(bad[0])} has det(J_x)={float(determinant[bad[0]]):.12g} <= 0; "
                         "the candidate configuration is flipped or degenerate.")
    return determinant


def evaluate_interface_geometry(topology, current_mesh, current_space, solid_displacement, *, candidate_id):
    """Evaluate the candidate interface geometry for a fixed reference topology.

    ``current_mesh`` is the candidate fluid P1 mesh, ``current_space`` its P2
    space and ``solid_displacement`` the full solid ``(n_s_nodes, 2)``
    displacement.  The reference topology must be built from the reference
    fluid mesh of ``current_mesh``; a different connectivity, a flipped
    candidate, a non-positive solid Jacobian, a degenerate edge or a gap
    between the solid and fluid current interface is rejected with the
    offending edge/node reported.  Neither the reference meshes nor any time
    history is modified.
    """
    if not isinstance(topology, InterfaceTopology):
        raise TypeError("topology must be an InterfaceTopology.")
    points, cells = _validate_mesh_like(current_mesh, "fluid")
    if getattr(current_space, "mesh", None) is not current_mesh:
        raise ValueError("current_space must be the P2 space built from current_mesh.")
    if len(points) != len(np.asarray(topology.fluid_mesh.points)) or not np.array_equal(cells, np.asarray(topology.fluid_mesh.cells)):
        raise ValueError("current_mesh connectivity differs from the reference fluid mesh; the fixed topology "
                         "is not valid for this candidate (re-mesh or rebuild the topology).")
    space_cells = np.asarray(current_space.cells)
    space_points = np.asarray(current_space.points, dtype=float)
    if (space_cells.shape != np.asarray(topology.fluid_space.cells).shape
            or not np.array_equal(space_cells, np.asarray(topology.fluid_space.cells))
            or dict(current_space.edge_nodes) != dict(topology.fluid_space.edge_nodes)):
        raise ValueError("current_space P2 connectivity differs from the reference fluid space; "
                         "the fixed topology is not valid for this candidate.")
    if space_points.shape != np.asarray(topology.fluid_space.points).shape or not np.all(np.isfinite(space_points)):
        raise ValueError("current_space must provide finite P2 coordinates with the reference node count.")
    determinant = _current_determinants(points, cells)

    displacement = np.asarray(solid_displacement, dtype=float)
    expected = np.asarray(topology.solid_mesh.points).shape
    if displacement.shape != expected:
        raise ValueError(f"solid_displacement must have shape {expected}, got {displacement.shape}.")
    if not np.all(np.isfinite(displacement)):
        raise ValueError("solid_displacement must be finite.")
    displacement = np.array(displacement, dtype=float, copy=True)

    solid_points = np.asarray(topology.solid_mesh.points, dtype=float)
    solid_cells = np.asarray(topology.solid_mesh.cells)
    solid_gradients = physical_gradients(topology.solid_mesh)
    abscissae, weights = _quadrature()
    n_edges = topology.n_edges
    n_quadrature = len(weights)
    solid_owner = np.empty(n_edges, dtype=np.int64)
    fluid_owner = np.empty(n_edges, dtype=np.int64)
    solid_local = np.empty((n_edges, 2), dtype=np.int64)
    fluid_local = np.empty((n_edges, 2), dtype=np.int64)
    reference_solid = np.empty((n_edges, n_quadrature, 2))
    reference_fluid = np.empty((n_edges, n_quadrature, 2))
    x_q = np.empty((n_edges, n_quadrature, 2))
    fluid_x_q = np.empty((n_edges, n_quadrature, 2))
    lengths = np.empty(n_edges)
    fluid_lengths = np.empty(n_edges)
    n_s = np.empty((n_edges, 2))
    n_f = np.empty((n_edges, 2))
    F_s = np.empty((n_edges, 2, 2))
    J_s = np.empty(n_edges)
    for index, pair in enumerate(topology.edge_pairs):
        solid_owner[index] = pair.solid_owner
        fluid_owner[index] = pair.fluid_owner
        a_s, b_s = _edge_local(solid_cells, pair.solid_owner, pair.solid_edge, "solid")
        a_f, b_f = _edge_local(cells, pair.fluid_owner, pair.fluid_edge, "fluid")
        solid_local[index] = (a_s, b_s)
        fluid_local[index] = (a_f, b_f)
        # Parameterise the solid edge by the fluid parameter: orientation -1
        # means the stored CCW traversals run in opposite directions.
        s_solid = abscissae if pair.orientation == 1 else 1.0-abscissae
        reference_solid[index] = ((1-s_solid)[:, None]*REFERENCE_VERTICES[a_s]
                                  + s_solid[:, None]*REFERENCE_VERTICES[b_s])
        reference_fluid[index] = ((1-abscissae)[:, None]*REFERENCE_VERTICES[a_f]
                                  + abscissae[:, None]*REFERENCE_VERTICES[b_f])
        current_solid = solid_points[list(pair.solid_edge)]+displacement[list(pair.solid_edge)]
        current_fluid = points[list(pair.fluid_edge)]
        x_q[index] = (1-s_solid)[:, None]*current_solid[0]+s_solid[:, None]*current_solid[1]
        fluid_x_q[index] = (1-abscissae)[:, None]*current_fluid[0]+abscissae[:, None]*current_fluid[1]
        tangent = current_solid[1]-current_solid[0]
        fluid_tangent = current_fluid[1]-current_fluid[0]
        lengths[index] = np.linalg.norm(tangent)
        fluid_lengths[index] = np.linalg.norm(fluid_tangent)
        if not (lengths[index] > 0 and fluid_lengths[index] > 0):
            raise ValueError(f"Interface edge {index} (solid {pair.solid_edge}, fluid {pair.fluid_edge}) is "
                             f"degenerate in the candidate configuration: lengths "
                             f"{lengths[index]:.12g} and {fluid_lengths[index]:.12g}.")
        n_s[index] = [tangent[1]/lengths[index], -tangent[0]/lengths[index]]
        n_f[index] = [fluid_tangent[1]/fluid_lengths[index], -fluid_tangent[0]/fluid_lengths[index]]
        local_nodes = solid_cells[pair.solid_owner]
        F_s[index] = np.eye(2)+np.einsum("ai,aJ->iJ", displacement[local_nodes], solid_gradients[pair.solid_owner])
        J_s[index] = np.linalg.det(F_s[index])
    bad = np.flatnonzero(~np.isfinite(J_s) | (J_s <= 0))
    if len(bad):
        index = int(bad[0])
        pair = topology.edge_pairs[index]
        raise ValueError(f"Interface edge {index} (solid {pair.solid_edge}) has non-positive solid Jacobian "
                         f"J_s={float(J_s[index]):.12g}; the solid candidate is flipped or degenerate.")
    if not np.all(np.isfinite(n_s)) or not np.all(np.isfinite(n_f)):
        raise ValueError("Interface normals are non-finite; check the candidate coordinates.")

    vertex_coordinates = np.asarray(topology.X_s_gamma, dtype=float)+displacement[topology.solid_nodes]
    fluid_vertex_coordinates = space_points[topology.fluid_vertex_nodes]
    vertex_gap = np.linalg.norm(vertex_coordinates-fluid_vertex_coordinates, axis=1)
    solid_row = {int(node): row for row, node in enumerate(topology.solid_nodes)}
    midpoint_gap = np.empty(n_edges)
    for index, pair in enumerate(topology.edge_pairs):
        predicted = 0.5*(vertex_coordinates[solid_row[int(pair.solid_edge[0])]]
                         + vertex_coordinates[solid_row[int(pair.solid_edge[1])]])
        midpoint_gap[index] = np.linalg.norm(predicted-space_points[pair.fluid_midpoint])
    edge_gap = np.linalg.norm(x_q-fluid_x_q, axis=2).ravel()
    gap_error = float(max(vertex_gap.max(), midpoint_gap.max(), edge_gap.max()))
    scale = float(max(np.linalg.norm(vertex_coordinates, axis=1).max(),
                      np.linalg.norm(fluid_vertex_coordinates, axis=1).max(),
                      np.linalg.norm(x_q.reshape(-1, 2), axis=1).max()))
    gap_tolerance = topology.atol+topology.rtol*scale
    if not np.isfinite(gap_error) or gap_error > gap_tolerance:
        raise ValueError(f"Solid and fluid current interfaces do not coincide: maximum gap {gap_error:.6e} "
                         f"exceeds tolerance {gap_tolerance:.6e} (topology atol={topology.atol!r}, "
                         f"rtol={topology.rtol!r}); the candidate geometry is inconsistent or cracked.")

    j_current = lengths/topology.reference_lengths
    transformed = np.linalg.solve(np.transpose(F_s, (0, 2, 1)), np.asarray(topology.N_s)[:, :, None])[:, :, 0]
    transformed_norm = np.linalg.norm(transformed, axis=1)
    nanson_normals = transformed/transformed_norm[:, None]
    j_nanson = J_s*transformed_norm
    normal_error = float(np.max(np.linalg.norm(n_s+n_f, axis=1)))
    normalization_error = float(max(np.max(np.abs(np.linalg.norm(n_s, axis=1)-1.0)),
                                    np.max(np.abs(np.linalg.norm(n_f, axis=1)-1.0))))
    nanson_normal_error = float(np.max(np.linalg.norm(n_s-nanson_normals, axis=1)))
    surface_ratio_error = float(np.max(np.abs(j_current-j_nanson)))
    geometry_id = _digest(("topology", topology.topology_id), ("candidate", str(candidate_id)),
                          ("points", points), ("cells", cells), ("displacement", displacement))
    return InterfaceGeometry(
        topology_id=topology.topology_id, candidate_id=candidate_id, geometry_id=geometry_id,
        current_mesh=current_mesh, current_space=current_space, solid_displacement=_read_only(displacement),
        abscissae=_read_only(abscissae.copy()), weights=_read_only(weights.copy()),
        solid_owner=_read_only(solid_owner), fluid_owner=_read_only(fluid_owner),
        solid_local=_read_only(solid_local), fluid_local=_read_only(fluid_local),
        reference_points_solid=_read_only(reference_solid), reference_points_fluid=_read_only(reference_fluid),
        x_q=_read_only(x_q), lengths=_read_only(lengths), fluid_lengths=_read_only(fluid_lengths),
        reference_lengths=_read_only(np.array(topology.reference_lengths, dtype=float, copy=True)),
        n_s=_read_only(n_s), n_f=_read_only(n_f), F_s=_read_only(F_s), J_s=_read_only(J_s),
        j_Gamma_current=_read_only(j_current), j_Gamma_nanson=_read_only(j_nanson),
        nanson_normals=_read_only(nanson_normals),
        vertex_coordinates=_read_only(vertex_coordinates),
        fluid_p2_coordinates=_read_only(np.array(space_points[topology.fluid_p2_nodes], copy=True)),
        normal_error=normal_error, normalization_error=normalization_error,
        nanson_normal_error=nanson_normal_error, surface_ratio_error=surface_ratio_error,
        gap_error=gap_error, gap_tolerance=float(gap_tolerance),
        minimum_jacobian=float(np.min(J_s)), minimum_cell_determinant=float(np.min(determinant)))

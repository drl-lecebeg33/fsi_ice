"""Interface load assembly from the fluid Cauchy stress trace.

``assemble_interface_load`` evaluates the fluid traction ``t_s = -sigma_f n_f``
at the current interface quadrature points through the shared
``boundary_cauchy_stress`` kernel and integrates it against the fluid P2 trace
basis.  The default current path uses ``t_s`` with the measure ``l w_q``; the
equivalent reference path uses ``T_s0 = j_Gamma t_s`` with ``L0 w_q``.  The
thickness is applied exactly once in both paths.  The integrated dual nodal
force ``g_f_gamma`` (fluid P2 trace) is mapped to the solid by
``f_s_gamma = H.T g_f_gamma``; neither an already integrated nodal force nor
``f_s_gamma`` is integrated or scaled a second time.

``pullback_traction`` performs the current-to-reference conversion check
``T_s0 = j_Gamma t_s`` with the Nanson surface ratio.

``interface_load_jvp`` is the analytic directional derivative (JVP) of
``assemble_interface_load`` with respect to the fluid velocity/pressure, the
fluid mesh displacement and the solid displacement.  It differentiates every
geometry factor by hand -- ``J_x^-1`` and the P2 physical gradients, the fluid
tangent/normal/length, the solid ``F_s``/``J_s`` and the Nanson ratio
``j_Gamma`` -- and never uses black-box differencing, so it can be used as the
T21 interface block without an extra solve.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from fsi_ice.fem.p1 import physical_gradients, shape_values
from fsi_ice.fem.p2 import p2_reference_gradients, p2_values
from fsi_ice.fluid.forces import boundary_cauchy_stress
from fsi_ice.interface.geometry import EDGE_COLUMNS, InterfaceGeometry
from fsi_ice.interface.topology import InterfaceTopology

_DIRECTION_KEYS = ("velocity", "pressure", "mesh_displacement", "solid_displacement")


def _read_only(array):
    array.setflags(write=False)
    return array


def _finite_vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array.astype(float, copy=True)


@dataclass(frozen=True)
class InterfaceLoad:
    """Integrated interface load for one candidate geometry and fluid state.

    ``t_s`` is the current traction (fluid on solid, Pa) and ``T_s0`` the
    reference-area traction ``j_Gamma t_s`` where ``j_Gamma`` is the Nanson
    surface ratio ``J_s ||F_s^-T N_s||`` (``length_ratios`` keeps the
    independent current ratio ``l/L0``).  ``g_f_gamma`` is the already
    integrated fluid P2 trace dual nodal force including the thickness
    ``(2*n_fG,)``; ``f_s_gamma`` is ``H.T g_f_gamma`` on the compact solid
    interface trace.  ``total_force``/``total_moment`` are the solid-side
    totals, the ``_fluid`` variants the fluid trace sums and the
    ``_quadrature`` variants the direct quadrature sums about
    ``moment_origin``; all totals include the thickness once.  All arrays are
    read-only.
    """

    topology_id: str
    geometry_id: str
    candidate_id: object
    thickness: float
    path: str
    t_s: np.ndarray
    T_s0: np.ndarray
    j_Gamma: np.ndarray
    length_ratios: np.ndarray
    g_f_gamma: np.ndarray
    f_s_gamma: np.ndarray
    total_force: np.ndarray
    total_force_fluid: np.ndarray
    total_force_quadrature: np.ndarray
    total_moment: float
    total_moment_fluid: float
    total_moment_quadrature: float
    moment_origin: np.ndarray

    def __post_init__(self):
        for name in ("t_s", "T_s0", "j_Gamma", "length_ratios", "g_f_gamma", "f_s_gamma",
                     "total_force", "total_force_fluid", "total_force_quadrature", "moment_origin"):
            _read_only(getattr(self, name))


def pullback_traction(geometry, traction, *, path="nanson"):
    """Convert current-area traction samples to reference-area tractions.

    Returns ``j_Gamma * traction`` where ``path="nanson"`` uses the Nanson
    surface ratio ``J_s ||F_s^-T N_s||`` and ``path="length"`` the current
    length ratio ``l/L0``.  The result is the traction per unit reference
    area, still expressed in current x/y components.
    """
    if not isinstance(geometry, InterfaceGeometry):
        raise TypeError("geometry must be an InterfaceGeometry.")
    values = np.asarray(traction, dtype=float)
    expected = geometry.x_q.shape
    if values.shape != expected:
        raise ValueError(f"traction must have shape {expected}, got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("traction must be finite.")
    if path == "nanson":
        factor = geometry.j_Gamma_nanson
    elif path == "length":
        factor = geometry.j_Gamma_current
    else:
        raise ValueError(f"Unknown pullback path {path!r}; allowed paths are 'nanson' and 'length'.")
    return _read_only(np.array(factor[:, None, None]*values, copy=True))


def assemble_interface_load(topology, geometry, operators, solution, *, thickness, path="current",
                            moment_origin=(0.0, 0.0)):
    """Assemble the interface load for one candidate fluid state.

    ``operators`` must be assembled on ``geometry.current_mesh`` and
    ``solution`` is the fluid state at that geometry.  ``path="current"``
    integrates ``t_s`` with ``l w_q``; ``path="reference"`` integrates
    ``T_s0 = j_Gamma t_s`` with ``L0 w_q`` for the conversion check.  The
    thickness is multiplied once and the integrated ``f_s_gamma`` is ready to
    be scattered into the solid ``external`` vector without further factors.
    """
    if not isinstance(topology, InterfaceTopology):
        raise TypeError("topology must be an InterfaceTopology.")
    if not isinstance(geometry, InterfaceGeometry):
        raise TypeError("geometry must be an InterfaceGeometry.")
    if geometry.topology_id != topology.topology_id:
        raise ValueError("geometry was evaluated for a different topology.")
    if isinstance(thickness, bool) or not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("thickness must be a finite positive number.")
    if path not in ("current", "reference"):
        raise ValueError(f"Unknown integration path {path!r}; allowed paths are 'current' and 'reference'.")
    origin = _finite_vector(moment_origin, 2, "moment_origin")
    geometry.require_current(operators)
    velocity = getattr(solution, "velocity", None)
    pressure = getattr(solution, "pressure", None)
    if velocity is None or pressure is None:
        raise TypeError("solution must provide velocity and pressure arrays.")

    stress = boundary_cauchy_stress(operators, solution, geometry.fluid_owner, geometry.reference_points_fluid)
    t_s = -np.einsum("eqij,ej->eqi", stress, geometry.n_f)
    T_s0 = pullback_traction(geometry, t_s)
    if path == "current":
        measure = geometry.lengths
        field = t_s
    else:
        measure = geometry.reference_lengths
        field = T_s0

    local = geometry.fluid_local
    columns = np.column_stack((local, [EDGE_COLUMNS[(int(a), int(b))] for a, b in local]))
    basis = p2_values(geometry.reference_points_fluid.reshape(-1, 2)).reshape(geometry.n_edges,
                                                                             geometry.n_quadrature, 6)
    basis = np.take_along_axis(basis, columns[:, None, :], axis=2)
    weighted = basis*(geometry.weights[None, :, None]*measure[:, None, None])*thickness
    loads = np.einsum("eqa,eqi->eai", weighted, field)
    node_ids = np.take_along_axis(np.asarray(operators.space.cells)[geometry.fluid_owner], columns, axis=1)
    position = {int(node): index for index, node in enumerate(topology.fluid_p2_nodes)}
    try:
        rows = np.array([[position[int(node)] for node in row] for row in node_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Fluid trace node {error.args[0]!r} is not part of the topology; "
                         "the operators do not match the interface topology.") from None
    g_f_gamma = np.zeros(2*topology.n_fluid_interface_nodes)
    np.add.at(g_f_gamma, (2*rows[..., None]+np.array([0, 1])).reshape(-1), loads.reshape(-1))
    f_s_gamma = np.asarray(topology.H.T @ g_f_gamma)

    force_solid = f_s_gamma.reshape(-1, 2).sum(axis=0)
    force_fluid = g_f_gamma.reshape(-1, 2).sum(axis=0)
    quad_weight = geometry.weights[None, :]*measure[:, None]
    force_quadrature = thickness*np.einsum("eq,eqi->i", quad_weight, field)
    x_s = geometry.vertex_coordinates-origin
    moment_solid = float(np.sum(x_s[:, 0]*f_s_gamma.reshape(-1, 2)[:, 1]
                                - x_s[:, 1]*f_s_gamma.reshape(-1, 2)[:, 0]))
    x_f = geometry.fluid_p2_coordinates-origin
    moment_fluid = float(np.sum(x_f[:, 0]*g_f_gamma.reshape(-1, 2)[:, 1]
                                - x_f[:, 1]*g_f_gamma.reshape(-1, 2)[:, 0]))
    relative = geometry.x_q-origin
    moment_quadrature = float(thickness*np.sum(quad_weight*(relative[..., 0]*field[..., 1]
                                                            - relative[..., 1]*field[..., 0])))
    return InterfaceLoad(
        topology_id=topology.topology_id, geometry_id=geometry.geometry_id, candidate_id=geometry.candidate_id,
        thickness=float(thickness), path=path, t_s=_read_only(np.array(t_s, copy=True)),
        T_s0=_read_only(np.array(T_s0, copy=True)), j_Gamma=_read_only(np.array(geometry.j_Gamma_nanson, copy=True)),
        length_ratios=_read_only(np.array(geometry.j_Gamma_current, copy=True)),
        g_f_gamma=_read_only(g_f_gamma), f_s_gamma=_read_only(np.array(f_s_gamma, copy=True)),
        total_force=_read_only(np.array(force_solid, copy=True)),
        total_force_fluid=_read_only(np.array(force_fluid, copy=True)),
        total_force_quadrature=_read_only(np.array(force_quadrature, copy=True)),
        total_moment=moment_solid, total_moment_fluid=moment_fluid,
        total_moment_quadrature=moment_quadrature, moment_origin=_read_only(origin))


@dataclass(frozen=True)
class InterfaceLoadDerivative:
    """Directional derivative of one assembled interface load.

    ``delta_t_s`` are the traction samples' derivative ``d t_s`` (Pa),
    ``delta_g_f_gamma`` the derivative of the already integrated fluid P2 trace
    dual nodal force including the thickness (N), ``delta_f_s_gamma`` its
    ``H.T`` image on the compact solid trace (N).  ``delta_total_force`` is the
    solid-side total force derivative (N) and ``delta_total_moment`` the
    derivative of the solid-side total moment about the origin ``(0, 0)``
    (N*m); both match ``InterfaceLoad.total_force``/``total_moment`` for the
    same path.  ``path`` selects whether the measure derivative enters through
    the current lengths or whether the Nanson ratio of the reference path is
    differentiated.  All arrays are read-only and no input or history is
    modified.
    """

    topology_id: str
    geometry_id: str
    thickness: float
    path: str
    delta_t_s: np.ndarray
    delta_g_f_gamma: np.ndarray
    delta_f_s_gamma: np.ndarray
    delta_total_force: np.ndarray
    delta_total_moment: float

    def __post_init__(self):
        for name in ("delta_t_s", "delta_g_f_gamma", "delta_f_s_gamma", "delta_total_force"):
            _read_only(getattr(self, name))


def _direction_array(delta, key, shape):
    value = np.asarray(delta[key], dtype=float)
    if value.shape != shape:
        raise ValueError(f"delta[{key!r}] must have shape {shape}, got {value.shape}.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"delta[{key!r}] must be finite.")
    return value


def _direction_arrays(topology, operators, delta):
    """Validate the direction mapping and fill missing keys with zeros."""
    if not isinstance(delta, Mapping):
        raise TypeError("delta must be a mapping with keys among "
                        "'velocity', 'pressure', 'mesh_displacement', 'solid_displacement'.")
    unknown = sorted(set(delta) - set(_DIRECTION_KEYS))
    if unknown:
        raise ValueError(f"Unknown delta keys {unknown}; allowed keys are {list(_DIRECTION_KEYS)}.")
    if not delta:
        raise ValueError("delta must contain at least one direction key.")
    mesh = operators.space.mesh
    shapes = {"velocity": (len(operators.space.points), 2), "pressure": (mesh.n_nodes,),
              "mesh_displacement": (mesh.n_nodes, 2),
              "solid_displacement": (len(topology.solid_mesh.points), 2)}
    directions = {key: (_direction_array(delta, key, shapes[key]) if key in delta
                        else np.zeros(shapes[key])) for key in _DIRECTION_KEYS}
    if not any(np.any(directions[key]) for key in _DIRECTION_KEYS):
        raise ValueError("delta must contain at least one nonzero direction; "
                         "every provided direction is zero.")
    return directions


def _nanson_direction(F_s, J_s, N_s, delta_F_s):
    """Directional derivative of the Nanson surface ratio ``J_s ||F_s^-T N_s||``."""
    transpose = np.transpose(F_s, (0, 2, 1))
    transformed = np.linalg.solve(transpose, np.asarray(N_s)[:, :, None])[:, :, 0]
    norm = np.linalg.norm(transformed, axis=1)
    delta_transformed = -np.linalg.solve(
        transpose, np.einsum("eji,ej->ei", delta_F_s, transformed)[:, :, None])[:, :, 0]
    delta_norm = np.einsum("ei,ei->e", transformed, delta_transformed)/norm
    cofactor = np.empty_like(F_s)
    cofactor[:, 0, 0] = F_s[:, 1, 1]
    cofactor[:, 0, 1] = -F_s[:, 0, 1]
    cofactor[:, 1, 0] = -F_s[:, 1, 0]
    cofactor[:, 1, 1] = F_s[:, 0, 0]
    delta_jacobian = np.einsum("eij,eji->e", cofactor, delta_F_s)
    return delta_jacobian*norm + J_s*delta_norm


def interface_load_jvp(topology, geometry, operators, solution, delta, *, thickness, path="current"):
    """Analytic directional derivative of ``assemble_interface_load``.

    ``delta`` is a mapping whose keys are a subset of ``"velocity"``
    ``(n_P2, 2)``, ``"pressure"`` ``(n_P1,)``, ``"mesh_displacement"``
    ``(n_fluid_P1, 2)`` and ``"solid_displacement"`` ``(n_solid, 2)``; at
    least one direction must be nonzero and mixed directions are allowed.
    Every implemented block is differentiated analytically: the fluid stress
    by ``delta sigma = -delta p I + mu (delta grad u + delta grad u^T)`` on the
    same P2 physical gradient basis as ``boundary_cauchy_stress``; the mesh
    direction through ``J_x^-1``, the P2 physical gradients and the fluid
    tangent/normal; the solid direction through ``F_s``/``J_s``/Nanson
    ``j_Gamma`` and the solid edge length on the current path (the reference
    path keeps ``L0`` and differentiates ``T_s0 = j_Gamma t_s``, so ``T_s0``
    never enters the current path).  ``operators`` must be assembled on
    ``geometry.current_mesh`` and ``solution`` is the fluid state there.  No
    input array, geometry, mesh or history is modified.
    """
    if not isinstance(topology, InterfaceTopology):
        raise TypeError("topology must be an InterfaceTopology.")
    if not isinstance(geometry, InterfaceGeometry):
        raise TypeError("geometry must be an InterfaceGeometry.")
    if geometry.topology_id != topology.topology_id:
        raise ValueError("geometry was evaluated for a different topology.")
    if isinstance(thickness, bool) or not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("thickness must be a finite positive number.")
    if path not in ("current", "reference"):
        raise ValueError(f"Unknown integration path {path!r}; allowed paths are 'current' and 'reference'.")
    geometry.require_current(operators)
    velocity = getattr(solution, "velocity", None)
    pressure = getattr(solution, "pressure", None)
    if velocity is None or pressure is None:
        raise TypeError("solution must provide velocity and pressure arrays.")
    directions = _direction_arrays(topology, operators, delta)
    delta_velocity = directions["velocity"]
    delta_pressure = directions["pressure"]
    delta_mesh = directions["mesh_displacement"]
    delta_solid = directions["solid_displacement"]

    mesh = operators.space.mesh
    cells = np.asarray(mesh.cells, dtype=np.int64)
    space_cells = np.asarray(operators.space.cells, dtype=np.int64)
    n_edges = topology.n_edges
    n_quadrature = geometry.n_quadrature
    owners = geometry.fluid_owner
    local = geometry.fluid_local
    reference_points = geometry.reference_points_fluid
    reference_gradients = p2_reference_gradients(reference_points.reshape(-1, 2)).reshape(
        n_edges, n_quadrature, 6, 2)
    inverse_jacobians = np.linalg.inv(mesh.jacobians)

    # Current traction through the shared kernel, exactly as the assembly.
    stress = boundary_cauchy_stress(operators, solution, owners, reference_points)
    n_f = geometry.n_f
    t_s = -np.einsum("eqij,ej->eqi", stress, n_f)

    # Fluid state and mesh derivatives of the physical velocity gradients.
    local_velocity = np.asarray(velocity, dtype=float)[space_cells[owners]]
    physical = np.einsum("eqai,eij->eqaj", reference_gradients, inverse_jacobians[owners])
    gradient_delta = np.einsum("eai,eqaj->eqij", delta_velocity[space_cells[owners]], physical)
    if np.any(delta_mesh):
        vertices = delta_mesh[cells]
        delta_jacobian = np.stack((vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]), axis=2)
        delta_inverse = -np.einsum("eij,ejk,ekl->eil", inverse_jacobians, delta_jacobian, inverse_jacobians)
        gradient_delta += np.einsum("eai,eqaj->eqij", local_velocity,
                                    np.einsum("eqai,eij->eqaj", reference_gradients,
                                              delta_inverse[owners]))
        fluid_edges = np.array([[pair.fluid_edge[0], pair.fluid_edge[1]] for pair in topology.edge_pairs],
                               dtype=np.int64)
        fluid_tangent = np.asarray(mesh.points, dtype=float)[fluid_edges[:, 1]] \
            - np.asarray(mesh.points, dtype=float)[fluid_edges[:, 0]]
        delta_tangent = delta_mesh[fluid_edges[:, 1]] - delta_mesh[fluid_edges[:, 0]]
        delta_fluid_length = np.einsum("ei,ei->e", fluid_tangent, delta_tangent)/geometry.fluid_lengths
        delta_n_f = (np.column_stack((delta_tangent[:, 1], -delta_tangent[:, 0]))
                     - n_f*delta_fluid_length[:, None])/geometry.fluid_lengths[:, None]
    else:
        delta_n_f = np.zeros_like(n_f)
    shape = shape_values(reference_points.reshape(-1, 2)).reshape(n_edges, n_quadrature, 3)
    delta_pressure_values = np.einsum("eqa,ea->eq", shape, delta_pressure[cells[owners]])
    delta_stress = (operators.viscosity*(gradient_delta + gradient_delta.transpose(0, 1, 3, 2))
                    - delta_pressure_values[:, :, None, None]*np.eye(2))
    delta_t_s = (-np.einsum("eqij,ej->eqi", delta_stress, n_f)
                 - np.einsum("eqij,ej->eqi", stress, delta_n_f))

    # Solid direction: F_s/J_s/j_Gamma and the current-path edge lengths.
    if np.any(delta_solid):
        solid_cells = np.asarray(topology.solid_mesh.cells, dtype=np.int64)
        solid_gradients = physical_gradients(topology.solid_mesh)
        solid_owners = geometry.solid_owner
        delta_F_s = np.einsum("eai,eaJ->eiJ", delta_solid[solid_cells[solid_owners]],
                              solid_gradients[solid_owners])
        delta_j_Gamma = _nanson_direction(geometry.F_s, geometry.J_s, topology.N_s, delta_F_s)
        solid_edges = np.array([[pair.solid_edge[0], pair.solid_edge[1]] for pair in topology.edge_pairs],
                               dtype=np.int64)
        solid_points = np.asarray(topology.solid_mesh.points, dtype=float)
        current_solid = solid_points[solid_edges] + np.asarray(geometry.solid_displacement)[solid_edges]
        delta_solid_tangent = delta_solid[solid_edges[:, 1]] - delta_solid[solid_edges[:, 0]]
        delta_lengths = np.einsum("ei,ei->e", current_solid[:, 1]-current_solid[:, 0],
                                  delta_solid_tangent)/geometry.lengths
    else:
        delta_j_Gamma = np.zeros(n_edges)
        delta_lengths = np.zeros(n_edges)

    if path == "current":
        measure = geometry.lengths
        field = t_s
        delta_field = delta_t_s
        delta_measure = delta_lengths
    else:
        measure = geometry.reference_lengths
        field = geometry.j_Gamma_nanson[:, None, None]*t_s
        delta_field = delta_j_Gamma[:, None, None]*t_s + geometry.j_Gamma_nanson[:, None, None]*delta_t_s
        delta_measure = np.zeros(n_edges)

    columns = np.column_stack((local, [EDGE_COLUMNS[(int(a), int(b))] for a, b in local]))
    basis = p2_values(reference_points.reshape(-1, 2)).reshape(n_edges, n_quadrature, 6)
    basis = np.take_along_axis(basis, columns[:, None, :], axis=2)
    weighted = basis*(geometry.weights[None, :, None]*measure[:, None, None])*thickness
    delta_weighted = basis*(geometry.weights[None, :, None]*delta_measure[:, None, None])*thickness
    loads = np.einsum("eqa,eqi->eai", weighted, field)
    delta_loads = (np.einsum("eqa,eqi->eai", weighted, delta_field)
                   + np.einsum("eqa,eqi->eai", delta_weighted, field))

    node_ids = np.take_along_axis(space_cells[owners], columns, axis=1)
    position = {int(node): index for index, node in enumerate(topology.fluid_p2_nodes)}
    try:
        rows = np.array([[position[int(node)] for node in row] for row in node_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Fluid trace node {error.args[0]!r} is not part of the topology; "
                         "the operators do not match the interface topology.") from None
    scatter = (2*rows[..., None]+np.array([0, 1])).reshape(-1)
    delta_g_f_gamma = np.zeros(2*topology.n_fluid_interface_nodes)
    np.add.at(delta_g_f_gamma, scatter, delta_loads.reshape(-1))
    delta_f_s_gamma = np.asarray(topology.H.T @ delta_g_f_gamma).astype(float, copy=True)

    g_f_gamma = np.zeros_like(delta_g_f_gamma)
    np.add.at(g_f_gamma, scatter, loads.reshape(-1))
    f_s_gamma = np.asarray(topology.H.T @ g_f_gamma).reshape(-1, 2)
    delta_f = delta_f_s_gamma.reshape(-1, 2)
    delta_vertex = delta_solid[topology.solid_nodes]
    x_s = geometry.vertex_coordinates
    delta_total_moment = float(np.sum(
        delta_vertex[:, 0]*f_s_gamma[:, 1] - delta_vertex[:, 1]*f_s_gamma[:, 0]
        + x_s[:, 0]*delta_f[:, 1] - x_s[:, 1]*delta_f[:, 0]))
    return InterfaceLoadDerivative(
        topology_id=topology.topology_id, geometry_id=geometry.geometry_id, thickness=float(thickness),
        path=path, delta_t_s=np.array(delta_t_s, copy=True), delta_g_f_gamma=delta_g_f_gamma,
        delta_f_s_gamma=delta_f_s_gamma,
        delta_total_force=np.array(delta_f.sum(axis=0), copy=True),
        delta_total_moment=delta_total_moment)

"""T21 monolithic residual, coupling Jacobian and matrix-free JVP tests (A6).

Evidence for the M5/T21 acceptance items of
``cases/m5_validation.json`` (21-01 ... 21-05) plus the layout, invariance,
rejection and real-mesh smoke checks:

* ``test_21_01_*``  shared single-field kernels at one state, zero-coupling and
  fixed-grid degenerate paths;
* ``test_21_02_*``  per-block single-component directional derivatives (fluid
  ``u/p``, solid ``d_s/v_s``, mesh ``d_m``, interface ``g``) including interior,
  wet-interface and constrained rows, zero blocks and unit rows;
* ``test_21_03_*``  global cross-block derivative, the three mesh-dependence
  paths and BDF1/BDF2 histories;
* ``test_21_04_*``  completeness and approximation audit of the assembled
  Jacobian (drop-one-block + the single marked approximation);
* ``test_21_05_*``  interface constraint identities (``H``, ``D_BDF``, ``H^T``,
  thickness once, pressure pin) and the assembled dual force.

Frozen scales of this file: ``L*=1 m`` (length), ``U*=0.05 m/s`` (lid velocity),
``P*=0.01 Pa`` (pressure), ``F*=0.01 N`` (interface nodal load).  Residual rows
use the per-block scales ``R_u: P*, R_p: U*L*, R_sd: U*, R_sv: P*, R_m: L*,
R_g: F*``.  Central-difference scans normalise the physical direction by the
per-variable scales and the residual by the per-block scales before comparing
norms; the best non-degenerate relative error must be <= 1e-5 (``spec.md``
section 5.2).  Deterministic random directions use the recorded seeds
``20261011`` (mixed) and ``20261012`` (per-block scans).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fsi_ice.ale import build_ale_geometry, build_harmonic_extension
from fsi_ice.coupling import (APPROXIMATIONS, BLOCK_NAMES, JACOBIAN_BLOCK_SOURCES, MonolithicBase,
                              MonolithicContext, PressurePin, build_block_layout,
                              evaluate_monolithic, monolithic_jacobian, monolithic_jvp,
                              monolithic_residual, require_candidate)
from fsi_ice.fem.p2 import P2Space
from fsi_ice.fem.poisson import assemble_poisson
from fsi_ice.fluid.ale import (ale_geometry_jvp, ale_navier_stokes_residual, build_ale_flow_context,
                               history_mass_load)
from fsi_ice.fluid.taylor_hood import assemble_stokes, navier_stokes_residual
from fsi_ice.interface import (assemble_interface_load, build_matching_interface,
                               evaluate_interface_geometry, interface_load_jvp)
from fsi_ice.mesh import TriMesh, fsi1_mesh, rectangle_mesh
from fsi_ice.solid.linear import ElasticMaterial, vector_dofs, vector_values
from fsi_ice.solid.stvk import assemble_mass, stvk_residual
from fsi_ice.time import BDFHistory
from fsi_ice.validation.monolithic import (BLOCK_RECTANGLES, DEFAULT_STEPS, EXTENDED_STEPS,
                                           RELATIVE_DERIVATIVE_LIMIT, block_scale_vector,
                                           central_difference_vectors, completeness_audit,
                                           directional_derivative_scan, jacobian_jvp_gap)

L_STAR = 1.0
U_STAR = 0.05
P_STAR = 0.01
F_STAR = 0.01
GATE = RELATIVE_DERIVATIVE_LIMIT
IDENTITY_LIMIT = 1e-12
MIXED_SEED = 20261011
BLOCK_SEED = 20261012
VARIABLE_SCALES = {"u_f": U_STAR, "p_f": P_STAR, "d_s": L_STAR, "v_s": U_STAR,
                   "d_m": L_STAR, "g": F_STAR}
RESIDUAL_SCALES = {"u_f": P_STAR, "p_f": U_STAR*L_STAR, "d_s": U_STAR, "v_s": P_STAR,
                   "d_m": L_STAR, "g": F_STAR}


# ---------------------------------------------------------------------------
# small two-domain problem
# ---------------------------------------------------------------------------

def two_domain(nx=4, ny_fluid=3, ny_solid=2, width=1.0):
    """Fluid ``[0,width] x [0,1]`` above solid ``[0,width] x [-1,0]``, matching y=0."""
    fluid_flat = rectangle_mesh(nx, ny_fluid, width=width, height=1.0)
    solid_flat = rectangle_mesh(nx, ny_solid, width=width, height=1.0)

    def tags(mesh, interface_tag):
        result = {name: mesh.boundaries[name] for name in ("bottom", "top", "left", "right")}
        result["interface"] = result.pop(interface_tag)
        return result

    solid = TriMesh(solid_flat.points-np.array([0.0, 1.0]), solid_flat.cells, tags(solid_flat, "top"))
    fluid = TriMesh(fluid_flat.points, fluid_flat.cells, tags(fluid_flat, "bottom"))
    return solid, fluid


def solid_field(points, amplitude, phase):
    """Smooth solid displacement, zero on the clamped bottom ``y=-1``."""
    x, y = points[:, 0], points[:, 1]+1.0
    return np.column_stack((amplitude*np.sin(phase)*y*np.sin(np.pi*x),
                            amplitude*np.cos(phase)*y*y*np.sin(2*np.pi*x)))


def fluid_field(points, amplitude, phase):
    x, y = points[:, 0], points[:, 1]
    return np.column_stack((amplitude*np.cos(phase)*np.sin(np.pi*x)*np.sin(np.pi*y),
                            amplitude*np.sin(phase)*np.sin(np.pi*x)*(1-np.cos(np.pi*y))/2))


def pressure_field(points, amplitude, phase):
    x, y = points[:, 0], points[:, 1]
    return amplitude*np.sin(phase)*np.cos(np.pi*x)*np.sin(np.pi*y/2)


def lid_velocity(points):
    values = np.zeros_like(points)
    top = np.isclose(points[:, 1], 1.0, rtol=0, atol=1e-12)
    values[top, 0] = U_STAR*np.sin(np.pi*points[top, 0])
    return values


def build_problem(*, nx=4, ny_fluid=3, ny_solid=2, order=1, dt=0.25, thickness=1.0, pin=PressurePin(0, 0.0),
                  viscosity=0.1, density=1.0, young=10.0, poisson=0.3, solid_density=1.0,
                  candidate_id="t21-candidate", amplitude=0.02, zero_history=False, flat_candidate=False,
                  fluid_boundary_velocity=lid_velocity, mesh_boundary_displacement=(0.0, 0.0),
                  solid_body_force=(0.0, 0.0), fluid_body_force=(0.0, 0.0)):
    """One small monolithic BDF step: accepted history, candidate state and context.

    ``zero_history`` uses the initial (zero) accepted state without accepting
    anything; ``flat_candidate`` puts ``d_s = d_m = 0`` so the candidate sits on
    the reference grid (used by the degenerate-path checks).
    """
    solid, fluid = two_domain(nx, ny_fluid, ny_solid)
    fluid_space = P2Space.from_mesh(fluid)
    topology = build_matching_interface(solid, fluid, fluid_space, solid_tag="interface",
                                        fluid_tag="interface", atol=1e-12, rtol=1e-12)
    harmonic = build_harmonic_extension(fluid)
    material = ElasticMaterial(young, poisson, density=solid_density, thickness=thickness)
    layout = build_block_layout(topology, fluid_dirichlet_tags=["top", "left", "right"],
                                solid_dirichlet_tags=["bottom"], pressure_pin=pin)
    n_solid = len(solid.points)
    exterior = fluid.boundary_nodes()
    position = {int(node): index for index, node in enumerate(exterior)}

    def mesh_displacement(displacement):
        boundary = np.zeros((len(exterior), 2))
        for solid_node, fluid_node in zip(topology.solid_nodes, topology.fluid_vertex_nodes):
            boundary[position[int(fluid_node)]] = displacement[solid_node]
        return np.array(harmonic.extend(exterior, boundary).values, copy=True)

    solid_history = BDFHistory(np.zeros((2, n_solid, 2)), dt, order=order)
    fluid_history = BDFHistory(np.zeros((len(fluid_space.points), 2)), dt, order=order)
    accepted = []
    if not zero_history:
        levels = 1 if order == 1 else 2
        for level in range(1, levels+1):
            displacement = solid_field(solid.points, amplitude/level, 0.4*level)
            velocity = solid_field(solid.points, 0.9*amplitude/level, 0.1*level)
            velocity_field = fluid_field(fluid_space.points, amplitude/level, 0.2*level)
            solid_history.accept(solid_history.context(), np.stack((displacement, velocity)))
            fluid_history.accept(fluid_history.context(), velocity_field)
            accepted.insert(0, {"d": displacement, "v": velocity, "u": velocity_field,
                                "geometry": build_ale_geometry(fluid, mesh_displacement(displacement),
                                                               candidate_id=("accepted", level))})
    if not accepted:
        accepted = [{"d": np.zeros((n_solid, 2)), "v": np.zeros((n_solid, 2)),
                     "u": np.zeros((len(fluid_space.points), 2)),
                     "geometry": build_ale_geometry(fluid, np.zeros_like(fluid.points),
                                                    candidate_id=("accepted", 0))}]
    stencil = solid_history.stencil()
    assert fluid_history.stencil().step_id == stencil.step_id
    base = MonolithicBase(reference_mesh=fluid, material=material, fluid_density=density,
                          fluid_viscosity=viscosity,
                          solid_states=tuple((state["d"], state["v"]) for state in accepted),
                          fluid_states=tuple(state["u"] for state in accepted),
                          accepted_geometries=tuple(state["geometry"] for state in accepted),
                          generation=stencil.step_id, solid_body_force=solid_body_force,
                          fluid_body_force=fluid_body_force,
                          fluid_boundary_velocity=fluid_boundary_velocity,
                          mesh_boundary_displacement=mesh_boundary_displacement)
    context = MonolithicContext(layout, base, stencil, topology, harmonic, candidate_id=candidate_id,
                                thickness=thickness, pressure_constraint=pin)

    if flat_candidate:
        d_s = np.zeros((n_solid, 2))
    else:
        d_s = solid_field(solid.points, amplitude, 0.7)
    d_m = mesh_displacement(d_s)
    candidate_geometry = build_ale_geometry(fluid, d_m, candidate_id=candidate_id)
    operators = assemble_stokes(candidate_geometry.current_mesh, viscosity=viscosity, density=density,
                                body_force=fluid_body_force)
    coefficient = float(stencil.coefficients[0])
    history_displacement = np.zeros((n_solid, 2))
    for weight, state in zip(np.asarray(stencil.coefficients)[1:], accepted):
        history_displacement += weight*state["d"]
    v_s = coefficient*d_s+history_displacement
    if flat_candidate:
        u_f = np.zeros((len(fluid_space.points), 2))
    else:
        u_f = fluid_field(operators.space.points, amplitude, 0.5)
    u_f[layout.fluid_dirichlet_nodes] = vector_values(fluid_boundary_velocity,
                                                      operators.space.points[layout.fluid_dirichlet_nodes],
                                                      "Lid velocity")
    u_f[topology.fluid_p2_nodes] = (topology.H @ v_s.ravel()[topology.solid_dofs]).reshape(-1, 2)
    if flat_candidate:
        p_f = np.zeros(fluid.n_nodes)
    else:
        p_f = pressure_field(fluid.points, 0.4*P_STAR, 0.9)
    if pin is not None:
        p_f[pin.node] = pin.value
    x = layout.pack({"u_f": u_f, "p_f": p_f, "d_s": d_s, "v_s": v_s, "d_m": d_m,
                     "g": np.zeros((topology.n_fluid_interface_nodes, 2))})
    x[layout.slice_of("g")] = -evaluate_monolithic(x, context=context).residual[layout.slice_of("g")]
    return SimpleNamespace(solid=solid, fluid=fluid, fluid_space=fluid_space, topology=topology,
                           harmonic=harmonic, material=material, layout=layout, base=base,
                           stencil=stencil, context=context, x=x, operators=operators,
                           accepted=accepted, candidate_geometry=candidate_geometry,
                           variable_scales=block_scale_vector(layout, VARIABLE_SCALES),
                           residual_scales=block_scale_vector(layout, RESIDUAL_SCALES))


def solid_history_terms(problem):
    """``hd``/``hv`` of the accepted solid states (same convention as the module)."""
    coefficients = np.asarray(problem.stencil.coefficients)[1:]
    displacement = np.zeros((problem.solid.n_nodes, 2))
    velocity = np.zeros_like(displacement)
    for weight, (d_state, v_state) in zip(coefficients, problem.base.solid_states):
        displacement += weight*d_state
        velocity += weight*v_state
    return displacement, velocity


def unit_direction(problem, block, dof, scale=None):
    """Physical direction with one unit (or ``scale``) component of one block."""
    direction = np.zeros(problem.layout.n_total)
    index = problem.layout.start_of(block)+int(dof)
    direction[index] = 1.0 if scale is None else float(scale)
    return direction


def make_direction(problem, contributions, *, sync=True):
    """Physical direction from per-block values with the interface compatibility.

    A compatible candidate needs ``d_m|Gamma = H_d d_s|Gamma``; when a direction
    touches the solid or mesh interface, the mesh trace is set to the solid
    trace (the solid displacement wins when both are given).
    """
    layout = problem.layout
    topology = problem.topology
    direction = np.zeros(layout.n_total)
    for name, values in contributions.items():
        direction[layout.slice_of(name)] = np.asarray(values, dtype=float).ravel()
    if sync and ("d_s" in contributions or "d_m" in contributions):
        # A compatible candidate needs d_m|Gamma = H_d d_s|Gamma.  When both
        # blocks are given the solid trace defines the common interface motion;
        # when only one is given the other trace follows it.
        blocks = layout.unpack(direction)
        d_s_full = np.array(blocks["d_s"], copy=True)
        d_m_full = np.array(blocks["d_m"], copy=True)
        if "d_s" in contributions:
            common = d_s_full.ravel()[topology.solid_dofs]
        else:
            common = d_m_full.ravel()[layout.mesh_interface_dofs]
        d_s_full.ravel()[topology.solid_dofs] = common
        d_m_full.ravel()[layout.mesh_interface_dofs] = common
        direction[layout.slice_of("d_s")] = d_s_full.ravel()
        direction[layout.slice_of("d_m")] = d_m_full.ravel()
    return direction


def scan(problem, direction, steps=DEFAULT_STEPS):
    return directional_derivative_scan(problem.x, direction, context=problem.context,
                                       variable_scales=problem.variable_scales,
                                       residual_scales=problem.residual_scales, steps=steps)


def restricted_scan(problem, direction, rows, steps=DEFAULT_STEPS):
    """Best relative error restricted to ``rows`` of the residual."""
    context = problem.context
    q, vectors = central_difference_vectors(problem.x, direction, context=context,
                                            variable_scales=problem.variable_scales, steps=steps)
    analytic = np.asarray(monolithic_jvp(problem.x, q, context=context))
    scale = problem.residual_scales[rows]
    reference = float(np.linalg.norm(analytic[rows]/scale))
    errors = [float(np.linalg.norm((vector[rows]-analytic[rows])/scale)) for vector in vectors]
    if reference <= 1e-12:
        best = int(np.argmin(errors))
        return errors[best], float(steps[best]), errors
    relative = [error/reference for error in errors]
    best = int(np.argmin(relative))
    return relative[best], float(steps[best]), relative


def assert_gate(result, limit=GATE, label=""):
    assert not result.degenerate, f"{label}: unexpected zero analytic response"
    assert result.best_relative_error <= limit, (
        f"{label}: best relative directional error {result.best_relative_error:.3e} at h="
        f"{result.best_relative_step:.0e} exceeds {limit:.0e}; scan={result.relative_errors}")


def assert_scaled_close(actual, expected, scale, limit=IDENTITY_LIMIT, label=""):
    difference = float(np.max(np.abs(np.asarray(actual, dtype=float)-np.asarray(expected, dtype=float))))
    assert difference <= limit*scale, f"{label}: max |diff|={difference:.3e} > {limit:.0e}*{scale:g}"


def replaced_fluid_mask(layout):
    mask = np.zeros(layout.sizes[0], dtype=bool)
    mask[layout.fluid_interface_dofs] = True
    mask[layout.fluid_dirichlet_dofs] = True
    return mask


def reference_candidate(problem):
    """Independent rebuild of the candidate through the public single-field API."""
    layout = problem.layout
    blocks = layout.unpack(problem.x)
    geometry = build_ale_geometry(problem.fluid, np.array(blocks["d_m"], copy=True),
                                  candidate_id=problem.context.candidate_id)
    transport = build_gcl_transport_for(problem, geometry)
    operators = assemble_stokes(geometry.current_mesh, viscosity=problem.base.fluid_viscosity,
                                density=problem.base.fluid_density,
                                body_force=problem.base.fluid_body_force)
    ale_context = build_ale_flow_context(operators, geometry, transport, problem.base.fluid_states)
    interface_geometry = evaluate_interface_geometry(problem.topology, geometry.current_mesh,
                                                     operators.space, np.array(blocks["d_s"], copy=True),
                                                     candidate_id=problem.context.candidate_id)
    state = SimpleNamespace(velocity=np.array(blocks["u_f"], copy=True),
                            pressure=np.array(blocks["p_f"], copy=True))
    load = assemble_interface_load(problem.topology, interface_geometry, operators, state,
                                   thickness=problem.context.thickness)
    values = np.concatenate((blocks["u_f"].ravel(), blocks["p_f"].ravel()))
    return SimpleNamespace(geometry=geometry, operators=operators, ale_context=ale_context,
                           interface_geometry=interface_geometry, load=load, blocks=blocks,
                           values=values, state=state)


def build_gcl_transport_for(problem, geometry):
    from fsi_ice.ale import build_gcl_transport

    return build_gcl_transport(problem.fluid, geometry, problem.base.accepted_geometries, problem.stencil)


def block_columns(problem, block, *, out_rows):
    """Per-DOF ``ale_geometry_jvp`` reference columns for one out block."""
    layout = problem.layout
    candidate = reference_candidate(problem)
    mesh = candidate.operators.space.mesh
    n_velocity = candidate.operators.n_velocity
    size = layout.size_of("d_m")
    columns = np.zeros((layout.size_of(block), size))
    direction = np.zeros((mesh.n_nodes, 2))
    for column in range(size):
        direction.ravel()[column] = 1.0
        delta = ale_geometry_jvp(candidate.operators, candidate.values, ale_context=candidate.ale_context,
                                 delta_d_m=direction)
        direction.ravel()[column] = 0.0
        columns[:, column] = delta[:n_velocity] if block == "u_f" else delta[n_velocity:]
    return columns


# ---------------------------------------------------------------------------
# 21-01 shared kernels and degenerate paths
# ---------------------------------------------------------------------------

def test_21_01_shared_residual_kernels_and_degenerate_paths():
    problem = build_problem()
    layout = problem.layout
    topology = problem.topology
    reference = reference_candidate(problem)
    residual = monolithic_residual(problem.x, context=problem.context)
    mask = replaced_fluid_mask(layout)
    n_velocity = layout.sizes[0]

    # fluid block: identical to ale_navier_stokes_residual on the same candidate
    flow, _ = ale_navier_stokes_residual(reference.operators, reference.values, reference.ale_context,
                                         with_jacobian=False)
    assert_scaled_close(residual[:n_velocity][~mask], flow[:n_velocity][~mask], P_STAR,
                        label="R_u free rows")
    pin = layout.pressure_pin.node
    free_pressure = np.ones(layout.sizes[1], dtype=bool)
    free_pressure[pin] = False
    assert_scaled_close(residual[layout.slice_of("p_f")][free_pressure],
                        flow[n_velocity:][free_pressure], U_STAR*L_STAR, label="R_p free rows")

    # solid blocks: identical to the BDF operator, StVK internal force and H^T load
    coefficient = float(problem.stencil.coefficients[0])
    hd, hv = solid_history_terms(problem)
    blocks = layout.unpack(problem.x)
    free_displacement = np.ones(layout.sizes[2], dtype=bool)
    free_displacement[layout.solid_displacement_dirichlet_dofs] = False
    free_velocity = np.ones(layout.sizes[3], dtype=bool)
    free_velocity[layout.solid_velocity_dirichlet_dofs] = False
    assert_scaled_close(residual[layout.slice_of("d_s")][free_displacement],
                        (coefficient*np.asarray(blocks["d_s"])+hd-np.asarray(blocks["v_s"])).ravel()[
                            free_displacement], U_STAR, label="R_sd free rows")
    np.testing.assert_array_equal(residual[layout.slice_of("d_s")][layout.solid_displacement_dirichlet_dofs],
                                  np.asarray(blocks["d_s"]).ravel()[layout.solid_displacement_dirichlet_dofs])
    mesh = topology.solid_mesh
    mass = assemble_mass(mesh, problem.material)
    internal, _ = stvk_residual(mesh, problem.material, np.asarray(blocks["d_s"]))
    interface_load = np.zeros(2*mesh.n_nodes)
    interface_load[topology.solid_dofs] = topology.H.T @ np.asarray(blocks["g"]).ravel()
    expected = mass @ (coefficient*np.asarray(blocks["v_s"]).ravel()+hv.ravel())+internal-interface_load
    assert_scaled_close(residual[layout.slice_of("v_s")][free_velocity], expected[free_velocity], P_STAR,
                        label="R_sv free rows")
    np.testing.assert_array_equal(residual[layout.slice_of("v_s")][layout.solid_velocity_dirichlet_dofs],
                                  np.asarray(blocks["v_s"]).ravel()[layout.solid_velocity_dirichlet_dofs])

    # mesh block: harmonic stiffness on free rows, exact interface/boundary rows
    stiffness = assemble_poisson(problem.fluid, 0.0, problem.harmonic.diffusion)[0]
    expected_mesh = np.empty(2*problem.fluid.n_nodes)
    for component in range(2):
        expected_mesh[component::2] = stiffness @ np.asarray(blocks["d_m"])[:, component]
    free_mesh = np.ones(2*problem.fluid.n_nodes, dtype=bool)
    free_mesh[layout.mesh_interface_dofs] = False
    free_mesh[layout.mesh_dirichlet_dofs] = False
    assert_scaled_close(residual[layout.slice_of("d_m")][free_mesh], expected_mesh[free_mesh], L_STAR,
                        label="R_m free rows")
    np.testing.assert_array_equal(residual[layout.slice_of("d_m")][layout.mesh_interface_dofs],
                                  (np.asarray(blocks["d_m"]).ravel()[layout.mesh_interface_dofs]
                                   - np.asarray(blocks["d_s"]).ravel()[topology.solid_dofs]))

    # load block: identical to the assembled dual nodal force
    assert_scaled_close(residual[layout.slice_of("g")],
                        np.asarray(blocks["g"]).ravel()-reference.load.g_f_gamma, F_STAR, label="R_g")
    np.testing.assert_array_equal(residual[layout.slice_of("g")], np.asarray(blocks["g"]).ravel()
                                  - reference.load.g_f_gamma)

    # degenerate path 1: fixed grid (d_m = 0) equals the fixed-grid transient residual
    flat = build_problem(zero_history=True, flat_candidate=True)
    flat_reference = reference_candidate(flat)
    flat_residual = monolithic_residual(flat.x, context=flat.context)
    flat_mask = replaced_fluid_mask(flat.layout)
    fixed_grid, _ = navier_stokes_residual(
        flat_reference.operators, flat_reference.values,
        time_coefficient=float(flat.stencil.coefficients[0]),
        time_rhs=-history_mass_load(flat_reference.operators, flat_reference.ale_context))
    assert_scaled_close(flat_residual[:flat.layout.sizes[0]][~flat_mask], fixed_grid[:flat.layout.sizes[0]][~flat_mask],
                        P_STAR, label="fixed-grid R_u")

    # degenerate path 2: zero coupling state (all fields zero) has zero residual
    zero = build_problem(zero_history=True, flat_candidate=True, fluid_boundary_velocity=(0.0, 0.0),
                         pin=PressurePin(0, 0.0))
    np.testing.assert_array_equal(monolithic_residual(np.zeros(zero.layout.n_total), context=zero.context),
                                  np.zeros(zero.layout.n_total))

    # degenerate path 3: zero coupling rows with a nonzero fluid state
    np.testing.assert_array_equal(
        flat_residual[flat.layout.start_of("u_f")+flat.layout.fluid_interface_dofs], 0.0)
    np.testing.assert_array_equal(
        flat_residual[flat.layout.start_of("d_m")+flat.layout.mesh_interface_dofs], 0.0)
    np.testing.assert_array_equal(flat_residual[flat.layout.slice_of("g")], 0.0)


# ---------------------------------------------------------------------------
# 21-02 per-block directional derivatives and row structure
# ---------------------------------------------------------------------------

def test_21_02_block_directional_derivatives_single_components():
    problem = build_problem()
    layout = problem.layout
    topology = problem.topology
    interface_nodes = set(int(node) for node in layout.fluid_dirichlet_nodes)
    interior_p2 = next(int(node) for node in range(len(problem.fluid_space.points))
                       if node not in interface_nodes
                       and not np.isin(node, layout.fluid_dirichlet_nodes)
                       and not np.isin(node, topology.fluid_p2_nodes))
    pin = layout.pressure_pin.node
    interior_p1 = next(int(node) for node in range(problem.fluid.n_nodes)
                       if node != pin and not np.isin(node, layout.mesh_dirichlet_nodes)
                       and not np.isin(node, topology.fluid_vertex_nodes))
    interior_solid = next(int(node) for node in range(problem.solid.n_nodes)
                          if not np.isin(node, topology.solid_nodes)
                          and not np.isin(node, layout.solid_displacement_dirichlet_nodes))
    interface_solid = int(topology.solid_nodes[0])
    results = {}
    cases = {
        "u_f interior": unit_direction(problem, "u_f", 2*interior_p2, U_STAR),
        "u_f dirichlet": unit_direction(problem, "u_f", 2*int(layout.fluid_dirichlet_nodes[0]), U_STAR),
        "u_f interface": unit_direction(problem, "u_f", int(layout.fluid_interface_dofs[0]), U_STAR),
        "p_f interior": unit_direction(problem, "p_f", interior_p1, P_STAR),
        "p_f pin": unit_direction(problem, "p_f", pin, P_STAR),
        "v_s interface": unit_direction(problem, "v_s", 2*interface_solid, U_STAR),
        "d_s interior": unit_direction(problem, "d_s", 2*interior_solid, L_STAR),
        "g trace": unit_direction(problem, "g", 0, F_STAR),
    }
    # solid interface component is scanned together with its compatible mesh trace
    q = unit_direction(problem, "d_s", 2*interface_solid, L_STAR)
    cases["d_s interface (compatible mesh)"] = make_direction(problem, {"d_s": q[layout.slice_of("d_s")]})
    cases["d_m interior"] = make_direction(problem, {"d_m": np.asarray(
        unit_direction(problem, "d_m", 2*interior_p1, L_STAR)[layout.slice_of("d_m")]).reshape(-1, 2)})
    cases["d_m interface (compatible solid)"] = make_direction(problem, {"d_m": np.asarray(
        unit_direction(problem, "d_m", 2*int(topology.fluid_vertex_nodes[0]), L_STAR)[layout.slice_of("d_m")]
    ).reshape(-1, 2)})
    for label, direction in cases.items():
        result = scan(problem, direction)
        results[label] = result
        assert_gate(result, label=label)
    assert min(result.best_relative_error for result in results.values()) <= GATE


def test_21_02_block_directional_derivatives_zero_blocks_and_unit_rows():
    problem = build_problem()
    layout = problem.layout
    topology = problem.topology
    jacobian = monolithic_jacobian(problem.x, context=problem.context)
    rng = np.random.default_rng(BLOCK_SEED)
    q = np.zeros(layout.n_total)
    for name in BLOCK_NAMES:
        q[layout.slice_of(name)] = rng.normal(size=layout.size_of(name))*problem.variable_scales[layout.slice_of(name)]
    q = make_direction(problem, {name: layout.unpack(q)[name] for name in BLOCK_NAMES})
    jvp = np.asarray(monolithic_jvp(problem.x, q, context=problem.context))
    blocks = layout.unpack(q)
    u0, p0, ds0, vs0, dm0, g0 = layout.starts

    # unit rows: prescribed DOFs have row e_d
    for dof in layout.fluid_dirichlet_dofs.tolist():
        assert jvp[u0+dof] == pytest.approx(blocks["u_f"].ravel()[dof], abs=0.0, rel=1e-15)
    for dof in layout.solid_displacement_dirichlet_dofs.tolist():
        assert jvp[ds0+dof] == pytest.approx(blocks["d_s"].ravel()[dof], abs=0.0, rel=1e-15)
        assert jvp[vs0+dof] == pytest.approx(blocks["v_s"].ravel()[dof], abs=0.0, rel=1e-15)
    for dof in layout.mesh_dirichlet_dofs.tolist():
        assert jvp[dm0+dof] == pytest.approx(blocks["d_m"].ravel()[dof], abs=0.0, rel=1e-15)
    assert jvp[p0+layout.pressure_pin.node] == pytest.approx(
        blocks["p_f"].ravel()[layout.pressure_pin.node], abs=0.0, rel=1e-15)

    # coupling rows: fluid trace = H v_s and mesh trace = d_s
    np.testing.assert_allclose(jvp[u0+layout.fluid_interface_dofs],
                               blocks["u_f"].ravel()[layout.fluid_interface_dofs]
                               - topology.H @ blocks["v_s"].ravel()[topology.solid_dofs],
                               rtol=0, atol=1e-15)
    np.testing.assert_allclose(jvp[dm0+layout.mesh_interface_dofs],
                               blocks["d_m"].ravel()[layout.mesh_interface_dofs]
                               - blocks["d_s"].ravel()[topology.solid_dofs], rtol=0, atol=1e-15)

    # structural zero blocks of the assembled Jacobian
    def rectangle(rows, columns):
        row_slice = slice(layout.start_of(rows), layout.start_of(rows)+layout.size_of(rows))
        column_slice = slice(layout.start_of(columns), layout.start_of(columns)+layout.size_of(columns))
        return jacobian[row_slice, column_slice]

    zero_pairs = [("u_f", "d_s"), ("u_f", "g"), ("p_f", "d_s"), ("p_f", "v_s"), ("p_f", "g"),
                  ("d_s", "u_f"), ("d_s", "d_m"), ("d_s", "g"), ("v_s", "u_f"), ("v_s", "d_m"),
                  ("d_m", "u_f"), ("d_m", "p_f"), ("d_m", "g"), ("g", "v_s")]
    for rows, columns in zero_pairs:
        assert rectangle(rows, columns).nnz == 0, f"expected zero block J[{rows}, {columns}]"

    # the only entries of J[u_f, v_s] and J[d_m, d_s] are the coupling constraints
    hops = topology.H.tocoo()
    expected_u_vs = np.zeros((layout.sizes[0], layout.sizes[3]))
    expected_u_vs[layout.fluid_interface_dofs[hops.row], topology.solid_dofs[hops.col]] = -hops.data
    np.testing.assert_allclose(np.asarray(rectangle("u_f", "v_s").todense()), expected_u_vs, rtol=0, atol=1e-15)
    expected_m_ds = np.zeros((layout.sizes[4], layout.sizes[2]))
    expected_m_ds[layout.mesh_interface_dofs, topology.solid_dofs] = -1.0
    np.testing.assert_allclose(np.asarray(rectangle("d_m", "d_s").todense()), expected_m_ds, rtol=0, atol=1e-15)
    # and the original block rows are cleared at the replaced rows
    h_dense = np.asarray(topology.H.todense())
    for index, dof in enumerate(layout.fluid_interface_dofs.tolist()):
        row = np.asarray(jacobian[u0+dof].todense()).ravel()
        expected = np.zeros(layout.n_total)
        expected[u0+dof] = 1.0
        expected[vs0+topology.solid_dofs] = -h_dense[index]
        np.testing.assert_allclose(row, expected, rtol=0, atol=1e-14)
    for index, dof in enumerate(layout.mesh_interface_dofs.tolist()):
        row = np.asarray(jacobian[dm0+dof].todense()).ravel()
        expected = np.zeros(layout.n_total)
        expected[dm0+dof] = 1.0
        expected[ds0+int(topology.solid_dofs[index])] = -1.0
        np.testing.assert_allclose(row, expected, rtol=0, atol=1e-14)
    for dof in layout.fluid_dirichlet_dofs.tolist():
        row = np.asarray(jacobian[u0+dof].todense()).ravel()
        expected = np.zeros(layout.n_total)
        expected[u0+dof] = 1.0
        np.testing.assert_allclose(row, expected, rtol=0, atol=1e-14)
    for dof in layout.solid_displacement_dirichlet_dofs.tolist():
        row = np.asarray(jacobian[ds0+dof].todense()).ravel()
        expected = np.zeros(layout.n_total)
        expected[ds0+dof] = 1.0
        np.testing.assert_allclose(row, expected, rtol=0, atol=1e-14)
    for dof in layout.mesh_dirichlet_dofs.tolist():
        row = np.asarray(jacobian[dm0+dof].todense()).ravel()
        expected = np.zeros(layout.n_total)
        expected[dm0+dof] = 1.0
        np.testing.assert_allclose(row, expected, rtol=0, atol=1e-14)

    # rows of the prescribed DOFs are unit rows in the assembled matrix
    for dof in layout.fluid_dirichlet_dofs.tolist():
        row = np.asarray(jacobian[u0+dof].todense()).ravel()
        assert np.count_nonzero(row) == 1 and row[u0+dof] == 1.0
    for dof in layout.solid_velocity_dirichlet_dofs.tolist():
        row = np.asarray(jacobian[vs0+dof].todense()).ravel()
        assert np.count_nonzero(row) == 1 and row[vs0+dof] == 1.0
    row = np.asarray(jacobian[p0+layout.pressure_pin.node].todense()).ravel()
    assert np.count_nonzero(row) == 1 and row[p0+layout.pressure_pin.node] == 1.0


def test_21_02_block_directional_derivatives_random_directions():
    """Every block gets a deterministic random direction (seed recorded)."""
    problem = build_problem()
    layout = problem.layout
    rng = np.random.default_rng(BLOCK_SEED)
    results = {}
    for index, name in enumerate(BLOCK_NAMES):
        values = (rng.normal(size=layout.shape_of(name))
                  * problem.variable_scales[layout.slice_of(name)].reshape(layout.shape_of(name)))
        direction = make_direction(problem, {name: values})
        result = scan(problem, direction)
        results[name] = result
        assert_gate(result, label=f"random direction in {name!r} (seed={BLOCK_SEED})")
    assert len(results) == len(BLOCK_NAMES)
    assert min(result.best_relative_error for result in results.values()) <= GATE


def test_21_02_block_directional_derivatives_zero_response_absolute():
    """Mesh DOFs outside the interface owner cells have a structurally zero load response."""
    problem = build_problem()
    layout = problem.layout
    candidate = reference_candidate(problem)
    owner_nodes = set(int(node) for node in
                      np.asarray(candidate.operators.space.mesh.cells)[candidate.interface_geometry.fluid_owner].ravel())
    outside = next(int(node) for node in range(problem.fluid.n_nodes) if node not in owner_nodes)
    direction = make_direction(problem, {"d_m": np.asarray(
        unit_direction(problem, "d_m", 2*outside, L_STAR)[layout.slice_of("d_m")]).reshape(-1, 2)})
    q = direction/problem.variable_scales
    jvp = np.asarray(monolithic_jvp(problem.x, q, context=problem.context))
    np.testing.assert_array_equal(jvp[layout.slice_of("g")], 0.0)
    _, vectors = central_difference_vectors(problem.x, direction, context=problem.context,
                                            variable_scales=problem.variable_scales, steps=DEFAULT_STEPS)
    analytic = np.asarray(monolithic_jvp(problem.x, q, context=problem.context))
    rows = layout.slice_of("g")
    for vector in vectors:
        assert np.max(np.abs((vector[rows]-analytic[rows])/problem.residual_scales[rows])) <= 1e-10
    result = scan(problem, direction, steps=(1e-4, 1e-5))
    assert result.degenerate or result.best_relative_error <= GATE


# ---------------------------------------------------------------------------
# 21-03 global derivative, mesh paths, BDF1/BDF2
# ---------------------------------------------------------------------------

def mixed_direction(problem, seed=MIXED_SEED):
    layout = problem.layout
    rng = np.random.default_rng(seed)
    contributions = {}
    for name in BLOCK_NAMES:
        contributions[name] = (rng.normal(size=layout.shape_of(name))
                               * problem.variable_scales[layout.slice_of(name)].reshape(layout.shape_of(name)))
    return make_direction(problem, contributions)


def test_21_03_global_directional_derivative_mixed_direction():
    problem = build_problem()
    layout = problem.layout
    direction = mixed_direction(problem)
    result = scan(problem, direction, steps=EXTENDED_STEPS)
    assert_gate(result, label=f"mixed direction seed={MIXED_SEED}")
    assert result.trend == "decreasing_then_increasing", result.relative_errors
    assert result.best_relative_step <= 1e-6
    # assembled Jacobian and matrix-free JVP agree on the same mixed direction
    jacobian = monolithic_jacobian(problem.x, context=problem.context)
    _, relative = jacobian_jvp_gap(problem.x, direction, context=problem.context, jacobian=jacobian,
                                   variable_scales=problem.variable_scales,
                                   residual_scales=problem.residual_scales)
    assert relative <= 1e-12


def test_21_03_global_directional_derivative_mesh_dependence_paths():
    problem = build_problem()
    layout = problem.layout
    topology = problem.topology
    rng = np.random.default_rng(MIXED_SEED+1)
    interior_mesh_nodes = [int(node) for node in range(problem.fluid.n_nodes)
                           if not np.isin(node, topology.fluid_vertex_nodes)
                           and not np.isin(node, layout.mesh_dirichlet_nodes)]
    interface_nodes = [int(node) for node in topology.fluid_vertex_nodes]
    interface_solid = [int(node) for node in topology.solid_nodes]
    # (i) interior-only mesh direction
    d_m = np.zeros((problem.fluid.n_nodes, 2))
    d_m[interior_mesh_nodes] = rng.normal(size=(len(interior_mesh_nodes), 2))*L_STAR
    interior_only = make_direction(problem, {"d_m": d_m}, sync=False)
    # (ii) interface-only mesh direction with the compatible solid trace
    d_m_face = np.zeros((problem.fluid.n_nodes, 2))
    d_m_face[interface_nodes] = rng.normal(size=(len(interface_nodes), 2))*L_STAR
    d_s_face = np.zeros((problem.solid.n_nodes, 2))
    d_s_face[interface_solid] = d_m_face[interface_nodes]
    interface_only = make_direction(problem, {"d_m": d_m_face, "d_s": d_s_face})
    # (iii) compatible mixed direction: d_m interior + interface = solid trace
    d_s = np.zeros((problem.solid.n_nodes, 2))
    d_s[interface_solid] = rng.normal(size=(len(interface_solid), 2))*L_STAR
    d_m_mixed = np.zeros((problem.fluid.n_nodes, 2))
    d_m_mixed[interior_mesh_nodes] = rng.normal(size=(len(interior_mesh_nodes), 2))*L_STAR
    d_m_mixed[interface_nodes] = d_s[interface_solid]
    mixed = make_direction(problem, {"d_m": d_m_mixed, "d_s": d_s})
    u_rows = slice(layout.starts[0], layout.starts[1])
    p_rows = slice(layout.starts[1], layout.starts[2])
    g_rows = layout.slice_of("g")
    fluid_rows = slice(layout.starts[0], layout.starts[2])
    for label, direction in (("interior-only d_m", interior_only),
                             ("interface-only d_m + d_s", interface_only),
                             ("compatible mixed d_m/d_s", mixed)):
        result = scan(problem, direction, steps=EXTENDED_STEPS)
        assert_gate(result, label=label)
        # the mesh geometry term enters exactly once through ale_geometry_jvp
        best, step, errors = restricted_scan(problem, direction, fluid_rows, steps=EXTENDED_STEPS)
        assert best <= GATE, f"{label}: fluid rows {best:.3e} at h={step:.0e} {errors}"
        # the load term enters exactly once through interface_load_jvp
        best_load, step_load, errors_load = restricted_scan(problem, direction, g_rows, steps=EXTENDED_STEPS)
        assert best_load <= GATE, f"{label}: load rows {best_load:.3e} at h={step_load:.0e} {errors_load}"
        # and the residual is linear in the direction (no double counting in the JVP)
        q = direction/problem.variable_scales
        analytic = np.asarray(monolithic_jvp(problem.x, q, context=problem.context))
        jvp_p = analytic[p_rows]
        del jvp_p


def test_21_03_global_directional_derivative_bdf1_and_bdf2():
    for order in (1, 2):
        problem = build_problem(order=order)
        layout = problem.layout
        coefficient = float(problem.stencil.coefficients[0])
        assert problem.stencil.order == order
        assert coefficient == pytest.approx(1.0/0.25 if order == 1 else 1.5/0.25, rel=1e-14)
        direction = mixed_direction(problem, seed=MIXED_SEED+order)
        result = scan(problem, direction, steps=EXTENDED_STEPS)
        assert_gate(result, label=f"BDF{order} mixed direction")
        # BDF identities of the d_s/v_s blocks hold for a random direction
        rng = np.random.default_rng(MIXED_SEED+10+order)
        q = make_direction(problem, {name: rng.normal(size=layout.shape_of(name))
                                     * problem.variable_scales[layout.slice_of(name)].reshape(layout.shape_of(name))
                                     for name in BLOCK_NAMES})
        blocks = layout.unpack(q)
        jvp = np.asarray(monolithic_jvp(problem.x, q, context=problem.context))
        free_displacement = np.ones(layout.sizes[2], dtype=bool)
        free_displacement[layout.solid_displacement_dirichlet_dofs] = False
        expected = (coefficient*blocks["d_s"].ravel()-blocks["v_s"].ravel())[free_displacement]
        np.testing.assert_allclose(jvp[layout.slice_of("d_s")][free_displacement], expected,
                                   rtol=1e-13, atol=1e-15)
        free_velocity = np.ones(layout.sizes[3], dtype=bool)
        free_velocity[layout.solid_velocity_dirichlet_dofs] = False
        mesh = problem.topology.solid_mesh
        state = layout.unpack(problem.x)
        _, stiffness = stvk_residual(mesh, problem.material, np.asarray(state["d_s"]))
        mass = assemble_mass(mesh, problem.material)
        trace_load = np.zeros(2*mesh.n_nodes)
        trace_load[problem.topology.solid_dofs] = problem.topology.H.T @ blocks["g"].ravel()
        expected_velocity = (stiffness @ blocks["d_s"].ravel()
                             + coefficient*(mass @ blocks["v_s"].ravel())-trace_load)[free_velocity]
        np.testing.assert_allclose(jvp[layout.slice_of("v_s")][free_velocity], expected_velocity,
                                   rtol=1e-12, atol=1e-14)


def test_21_03_global_directional_derivative_jacobian_matches_jvp_on_batch():
    problem = build_problem()
    layout = problem.layout
    rng = np.random.default_rng(MIXED_SEED+3)
    batch = np.stack([mixed_direction(problem, seed=MIXED_SEED+index)/problem.variable_scales
                      for index in range(3)])
    jvp = np.asarray(monolithic_jvp(problem.x, batch, context=problem.context))
    assert jvp.shape == batch.shape
    jacobian = monolithic_jacobian(problem.x, context=problem.context)
    np.testing.assert_allclose(jvp, np.asarray(jacobian @ batch.T).T, rtol=1e-12, atol=1e-12)
    single = np.asarray(monolithic_jvp(problem.x, batch[0], context=problem.context))
    np.testing.assert_array_equal(single, jvp[0])
    del rng, layout


# ---------------------------------------------------------------------------
# 21-04 completeness and approximation audit
# ---------------------------------------------------------------------------

def test_21_04_jacobian_completeness_and_approximation_audit_sources():
    covered = set()
    for entry in JACOBIAN_BLOCK_SOURCES.values():
        assert isinstance(entry, dict) and entry.get("source") and entry.get("kind")
        covered.update(part.strip() for part in entry["rows"].split(","))
    assert covered == set(BLOCK_NAMES)
    assert APPROXIMATIONS and all("body force" in text for text in APPROXIMATIONS)


def test_21_04_jacobian_completeness_and_approximation_audit_drop_one_block():
    problem = build_problem()
    layout = problem.layout
    direction = mixed_direction(problem, seed=MIXED_SEED+4)
    reports = completeness_audit(problem.x, direction, context=problem.context,
                                 variable_scales=problem.variable_scales,
                                 residual_scales=problem.residual_scales, steps=EXTENDED_STEPS)
    assert reports["full"].best_relative_error <= GATE
    jacobian = monolithic_jacobian(problem.x, context=problem.context)
    q = direction/problem.variable_scales
    contributions = {}
    for name, (rows, columns) in BLOCK_RECTANGLES.items():
        row_slice = slice(layout.starts[BLOCK_NAMES.index(rows.split(",")[0])],
                          layout.starts[BLOCK_NAMES.index(rows.split(",")[-1])]
                          + layout.sizes[BLOCK_NAMES.index(rows.split(",")[-1])])
        column_slice = slice(layout.starts[BLOCK_NAMES.index(columns.split(",")[0])],
                             layout.starts[BLOCK_NAMES.index(columns.split(",")[-1])]
                             + layout.sizes[BLOCK_NAMES.index(columns.split(",")[-1])])
        piece = np.asarray(jacobian[row_slice, column_slice] @ q[column_slice]).ravel()
        contribution = float(np.linalg.norm(piece/problem.residual_scales[row_slice]))
        contributions[name] = contribution
        if contribution <= GATE:
            continue
        # removing a needed block changes the prediction by its own contribution,
        # so the drop-one absolute error must recover at least half of it
        assert reports[name].best_absolute_error >= 0.5*contribution, (
            name, contribution, reports[name].best_absolute_error)
        assert reports[name].best_relative_error > GATE
    exercised = [name for name, value in contributions.items() if value > GATE]
    assert len(exercised) >= 6, contributions
    assert contributions["fluid_geometry_j_um_j_pm"] > GATE
    assert contributions["load_fluid_l_u_l_p"] > GATE
    assert contributions["load_mesh_l_dm"] > GATE
    assert contributions["solid_load_s_h_t"] > GATE


def test_21_04_jacobian_completeness_and_approximation_audit_callable_body_force():
    def body_force(points):
        return np.column_stack((0.01+0.5*points[:, 1], 0.02+0.1*points[:, 0]))

    problem = build_problem(fluid_body_force=body_force)
    direction = mixed_direction(problem, seed=MIXED_SEED+5)
    result = scan(problem, direction)
    assert_gate(result, label="callable fluid body force (marked approximation)")


# ---------------------------------------------------------------------------
# 21-05 interface constraint identities
# ---------------------------------------------------------------------------

def test_21_05_interface_constraint_identities_h_bdf_ht_thickness_pin():
    problem = build_problem()
    layout = problem.layout
    topology = problem.topology
    residual = monolithic_residual(problem.x, context=problem.context)
    blocks = layout.unpack(problem.x)
    # H: fluid P2 trace = H (solid trace)
    np.testing.assert_array_equal(residual[layout.start_of("u_f")+layout.fluid_interface_dofs], 0.0)
    # D_BDF: v_s = a0 d_s + hd gives R_sd = 0
    np.testing.assert_array_equal(residual[layout.slice_of("d_s")], 0.0)
    # H_d = I on the matched P1 trace
    np.testing.assert_array_equal(residual[layout.start_of("d_m")+layout.mesh_interface_dofs], 0.0)
    # H^T: a g perturbation changes R_sv by exactly -S_s H^T
    delta_g = 0.317*np.asarray(problem.variable_scales[layout.slice_of("g")]).reshape(-1, 2)
    perturbed = problem.x.copy()
    perturbed[layout.slice_of("g")] += delta_g.ravel()
    changed = monolithic_residual(perturbed, context=problem.context)
    expected = np.zeros(2*topology.solid_mesh.n_nodes)
    expected[topology.solid_dofs] = -(topology.H.T @ delta_g.ravel())
    np.testing.assert_allclose(changed[layout.slice_of("v_s")]-residual[layout.slice_of("v_s")],
                               expected, rtol=0, atol=1e-15)

    # thickness 0.2/1/2: the interface load enters linearly and exactly once
    g = np.asarray(blocks["g"]).ravel()
    loads = {}
    for thickness in (0.2, 1.0, 2.0):
        scaled = build_problem(thickness=thickness, candidate_id=f"t21-thickness-{thickness}")
        scaled_blocks = scaled.layout.unpack(scaled.x)
        scaled_residual = monolithic_residual(scaled.x, context=scaled.context)
        loads[thickness] = scaled_blocks["g"].ravel()-scaled_residual[scaled.layout.slice_of("g")]
    np.testing.assert_allclose(loads[2.0], 2.0*loads[1.0], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(loads[0.2], 0.2*loads[1.0], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(loads[1.0], g, rtol=1e-12, atol=1e-15)

    # pressure gauge: pin row p_i - value, and an unpinned layout keeps the raw row
    pin = layout.pressure_pin
    assert residual[layout.start_of("p_f")+pin.node] == 0.0
    shifted = build_problem(pin=PressurePin(2, 0.25*P_STAR))
    shifted_residual = monolithic_residual(shifted.x, context=shifted.context)
    shifted_blocks = shifted.layout.unpack(shifted.x)
    assert shifted_residual[shifted.layout.start_of("p_f")+2] == (
        shifted_blocks["p_f"].ravel()[2]-0.25*P_STAR)
    unpinned = build_problem(pin=None)
    assert unpinned.layout.pressure_pin is None
    assert unpinned.context.pressure_constraint is None


def test_21_05_interface_constraint_identities_load_block_derivatives():
    """The g rows use exactly the four interface_load_jvp directions."""
    problem = build_problem()
    layout = problem.layout
    candidate = reference_candidate(problem)
    rng = np.random.default_rng(MIXED_SEED+6)
    contributions = {}
    for name in ("u_f", "p_f", "d_s", "d_m"):
        contributions[name] = (rng.normal(size=layout.shape_of(name))
                               * problem.variable_scales[layout.slice_of(name)].reshape(layout.shape_of(name)))
    direction = make_direction(problem, contributions)
    q = direction/problem.variable_scales
    blocks = layout.unpack(q)
    expected = np.zeros(layout.sizes[5])
    for key, name in (("velocity", "u_f"), ("pressure", "p_f"), ("solid_displacement", "d_s"),
                      ("mesh_displacement", "d_m")):
        expected += np.asarray(interface_load_jvp(
            problem.topology, candidate.interface_geometry, candidate.operators, candidate.state,
            {key: np.asarray(blocks[name])}, thickness=problem.context.thickness).delta_g_f_gamma)
    jvp = np.asarray(monolithic_jvp(problem.x, q, context=problem.context))
    g_rows = layout.slice_of("g")
    np.testing.assert_allclose(jvp[g_rows], blocks["g"].ravel()-expected, rtol=1e-13, atol=1e-15)
    # and the residual g rows follow the same linear map
    # current and reference integration paths are equivalent at the same candidate
    # (the monolithic residual uses ``current``; the assembled dual force and its
    # directional derivative must not depend on which equivalent path is used)
    reference_load = assemble_interface_load(
        problem.topology, candidate.interface_geometry, candidate.operators, candidate.state,
        thickness=problem.context.thickness, path="reference")
    np.testing.assert_allclose(reference_load.g_f_gamma, candidate.load.g_f_gamma,
                               rtol=1e-11, atol=1e-18)
    q = direction/problem.variable_scales
    blocks = layout.unpack(q)
    current_derivative = np.zeros(layout.sizes[5])
    reference_derivative = np.zeros(layout.sizes[5])
    for key, name in (("velocity", "u_f"), ("pressure", "p_f"), ("solid_displacement", "d_s"),
                      ("mesh_displacement", "d_m")):
        delta = {key: np.asarray(blocks[name])}
        current_derivative += np.asarray(interface_load_jvp(
            problem.topology, candidate.interface_geometry, candidate.operators, candidate.state, delta,
            thickness=problem.context.thickness, path="current").delta_g_f_gamma)
        reference_derivative += np.asarray(interface_load_jvp(
            problem.topology, candidate.interface_geometry, candidate.operators, candidate.state, delta,
            thickness=problem.context.thickness, path="reference").delta_g_f_gamma)
    np.testing.assert_allclose(reference_derivative, current_derivative, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(np.asarray(monolithic_jvp(problem.x, q, context=problem.context))[g_rows],
                               np.asarray(blocks["g"]).ravel()-reference_derivative, rtol=1e-8, atol=1e-14)
    best, step, errors = restricted_scan(problem, direction, g_rows)
    assert best <= GATE, f"load rows {best:.3e} at h={step:.0e} {errors}"


# ---------------------------------------------------------------------------
# layout, batching, invariance, rejections, accessor
# ---------------------------------------------------------------------------

def test_layout_pack_unpack_scatter_and_read_only():
    problem = build_problem()
    layout = problem.layout
    assert sum(layout.sizes) == layout.n_total
    position = 0
    for name in BLOCK_NAMES:
        assert layout.start_of(name) == position
        assert layout.slice_of(name) == slice(position, position+layout.size_of(name))
        position += layout.size_of(name)
    blocks = layout.unpack(problem.x)
    assert set(blocks) == set(BLOCK_NAMES)
    for name in BLOCK_NAMES:
        assert blocks[name].shape == layout.shape_of(name)
        assert not blocks[name].flags.writeable
        with pytest.raises(ValueError):
            blocks[name].ravel()[0] = 1.0
    repacked = layout.pack(blocks)
    np.testing.assert_array_equal(repacked, problem.x)
    repacked[0] = 123.0
    assert problem.x[0] != 123.0
    with pytest.raises(ValueError):
        layout.pack({"u_f": np.zeros(layout.shape_of("u_f"))})
    bad = dict(blocks)
    bad["p_f"] = np.zeros((layout.sizes[1]+1,))
    with pytest.raises(ValueError):
        layout.pack(bad)
    nan = dict(blocks)
    nan["p_f"] = np.full(layout.sizes[1], np.nan)
    with pytest.raises(ValueError):
        layout.pack(nan)
    scattered = layout.scatter("g", blocks["g"])
    assert scattered.shape == (layout.n_total,)
    np.testing.assert_array_equal(scattered[layout.slice_of("g")], blocks["g"].ravel())
    other = np.ones(layout.n_total, dtype=bool)
    other[layout.slice_of("g")] = False
    np.testing.assert_array_equal(scattered[other], 0.0)
    np.testing.assert_array_equal(layout.gather("d_s", problem.x), blocks["d_s"])
    with pytest.raises(ValueError):
        layout.unpack(problem.x[:-1])


def test_geometry_columns_match_direct_directional_derivatives():
    """The batched J_um/J_pm equal one ale_geometry_jvp column per mesh DOF."""
    problem = build_problem()
    layout = problem.layout
    jacobian = monolithic_jacobian(problem.x, context=problem.context)
    mask = replaced_fluid_mask(layout)
    n_velocity = layout.sizes[0]
    n_pressure = layout.sizes[1]
    direct_u = block_columns(problem, "u_f", out_rows=slice(0, n_velocity))
    direct_p = block_columns(problem, "p_f", out_rows=slice(0, n_pressure))
    assembled_u = np.asarray(jacobian[:n_velocity, layout.slice_of("d_m")].todense())
    assembled_p = np.asarray(jacobian[n_velocity:n_velocity+n_pressure, layout.slice_of("d_m")].todense())
    np.testing.assert_allclose(assembled_u[~mask], direct_u[~mask], rtol=0, atol=1e-14)
    free_pressure = np.ones(n_pressure, dtype=bool)
    free_pressure[layout.pressure_pin.node] = False
    np.testing.assert_allclose(assembled_p[free_pressure], direct_p[free_pressure], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(assembled_p[layout.pressure_pin.node], 0.0)


def test_load_blocks_match_full_column_scan():
    """L_u/L_p/L_dm/L_ds equal one interface_load_jvp call for every block DOF."""
    problem = build_problem()
    layout = problem.layout
    candidate = reference_candidate(problem)
    jacobian = monolithic_jacobian(problem.x, context=problem.context)
    g_rows = layout.slice_of("g")
    for key, name in (("velocity", "u_f"), ("pressure", "p_f"), ("solid_displacement", "d_s"),
                      ("mesh_displacement", "d_m")):
        size = layout.size_of(name)
        columns = np.zeros((size, layout.sizes[5]))
        value = np.zeros(layout.shape_of(name))
        flat = value.ravel()
        for dof in range(size):
            flat[dof] = 1.0
            columns[dof] = np.asarray(interface_load_jvp(
                problem.topology, candidate.interface_geometry, candidate.operators, candidate.state,
                {key: value}, thickness=problem.context.thickness).delta_g_f_gamma)
            flat[dof] = 0.0
        assembled = -np.asarray(jacobian[g_rows, layout.slice_of(name)].todense())
        np.testing.assert_allclose(assembled.T, columns, rtol=0, atol=1e-14)


def test_context_and_candidate_rejections():
    problem = build_problem()
    layout = problem.layout
    # stale candidate: a stencil from a later BDF generation
    stale_history = BDFHistory(np.zeros((2, problem.solid.n_nodes, 2)), 0.25, order=1)
    for _ in range(2):
        stale_history.accept(stale_history.context(), np.zeros((2, problem.solid.n_nodes, 2)))
    assert stale_history.stencil().step_id != problem.base.generation
    with pytest.raises(ValueError, match="stale candidate"):
        MonolithicContext(layout, problem.base, stale_history.stencil(), problem.topology,
                          problem.harmonic, candidate_id="t21", thickness=1.0,
                          pressure_constraint=layout.pressure_pin)
    # mixed reference mesh
    other = build_problem(nx=3, candidate_id="t21-other")
    with pytest.raises(ValueError, match="harmonic"):
        MonolithicContext(layout, problem.base, problem.stencil, problem.topology, other.harmonic,
                          candidate_id="t21", thickness=1.0, pressure_constraint=layout.pressure_pin)
    with pytest.raises(ValueError, match="topology"):
        MonolithicContext(other.layout, other.base, other.stencil, problem.topology, problem.harmonic,
                          candidate_id="t21", thickness=1.0, pressure_constraint=layout.pressure_pin)
    # pressure pin mismatch and invalid thickness / candidate id
    with pytest.raises(ValueError, match="pressure_constraint"):
        MonolithicContext(layout, problem.base, problem.stencil, problem.topology, problem.harmonic,
                          candidate_id="t21", thickness=1.0, pressure_constraint=PressurePin(1, 0.0))
    for thickness in (0.0, -1.0, True):
        with pytest.raises(ValueError, match="thickness"):
            MonolithicContext(layout, problem.base, problem.stencil, problem.topology, problem.harmonic,
                              candidate_id="t21", thickness=thickness, pressure_constraint=layout.pressure_pin)
    with pytest.raises(ValueError, match="candidate_id"):
        MonolithicContext(layout, problem.base, problem.stencil, problem.topology, problem.harmonic,
                          candidate_id=None, thickness=1.0, pressure_constraint=layout.pressure_pin)
    # invalid x
    with pytest.raises(ValueError, match="x must"):
        monolithic_residual(np.zeros(layout.n_total+1), context=problem.context)
    with pytest.raises(ValueError, match="x must"):
        monolithic_residual(np.full(layout.n_total, np.nan), context=problem.context)
    # flipped candidate: a mirror displacement flips every cell
    flipped = problem.x.copy()
    points = problem.fluid.points
    mirror = -2.0*(points-points.mean(axis=0))
    flipped[layout.slice_of("d_m")] = mirror.ravel()
    with pytest.raises(ValueError, match="det"):
        monolithic_residual(flipped, context=problem.context)
    # quality rejection: a severe compression keeps the orientation but fails the volume ratio
    squeezed = problem.x.copy()
    squeezed[layout.slice_of("d_m")] = (-0.9*(points-points.mean(axis=0))).ravel()
    with pytest.raises(ValueError, match="volume ratio"):
        monolithic_residual(squeezed, context=problem.context)
    # mixed candidate versions
    candidate = reference_candidate(problem)
    with pytest.raises(ValueError, match="stale candidate geometry"):
        require_candidate(problem.context, geometry=other.candidate_geometry)
    with pytest.raises(ValueError, match="different topology"):
        require_candidate(problem.context, interface_geometry=other_interface_geometry(problem, other))
    second = reference_candidate_with_shift(problem)
    with pytest.raises(ValueError, match="different interface geometry|stale"):
        require_candidate(problem.context, interface_geometry=candidate.interface_geometry, load=second)
    require_candidate(problem.context, geometry=candidate.geometry,
                      interface_geometry=candidate.interface_geometry, load=candidate.load)


def other_interface_geometry(problem, other):
    """Interface geometry of another topology re-labelled with this candidate id."""
    blocks = other.layout.unpack(other.x)
    return evaluate_interface_geometry(other.topology, other.candidate_geometry.current_mesh,
                                       other.operators.space, np.asarray(blocks["d_s"]),
                                       candidate_id=problem.context.candidate_id)


def reference_candidate_with_shift(problem):
    """A load assembled at a compatible shifted state but under the same candidate id."""
    layout = problem.layout
    topology = problem.topology
    shifted = problem.x.copy()
    blocks = layout.unpack(shifted)
    d_s = np.array(blocks["d_s"], copy=True)
    d_s[topology.solid_nodes, 1] += 1e-5
    d_m = np.array(blocks["d_m"], copy=True)
    d_m.ravel()[layout.mesh_interface_dofs] = d_s.ravel()[topology.solid_dofs]
    shifted[layout.slice_of("d_s")] = d_s.ravel()
    shifted[layout.slice_of("d_m")] = d_m.ravel()
    blocks = layout.unpack(shifted)
    geometry = build_ale_geometry(problem.fluid, np.array(blocks["d_m"], copy=True),
                                  candidate_id=problem.context.candidate_id)
    operators = assemble_stokes(geometry.current_mesh, viscosity=problem.base.fluid_viscosity,
                                density=problem.base.fluid_density)
    interface_geometry = evaluate_interface_geometry(problem.topology, geometry.current_mesh,
                                                     operators.space, np.array(blocks["d_s"], copy=True),
                                                     candidate_id=problem.context.candidate_id)
    state = SimpleNamespace(velocity=np.array(blocks["u_f"], copy=True),
                            pressure=np.array(blocks["p_f"], copy=True))
    return assemble_interface_load(problem.topology, interface_geometry, operators, state,
                                   thickness=problem.context.thickness)


def test_invariance_repeatability_and_read_only_outputs():
    problem = build_problem()
    x_before = problem.x.copy()
    states_before = [(np.array(d, copy=True), np.array(v, copy=True))
                     for d, v in problem.base.solid_states]
    fluid_before = [np.array(state, copy=True) for state in problem.base.fluid_states]
    mesh_before = np.array(problem.fluid.points, copy=True)
    topology_h_before = np.array(problem.topology.h.toarray(), copy=True)
    rng = np.random.default_rng(MIXED_SEED+7)
    q = make_direction(problem, {name: rng.normal(size=problem.layout.shape_of(name))
                                 * problem.variable_scales[problem.layout.slice_of(name)].reshape(
                                     problem.layout.shape_of(name)) for name in BLOCK_NAMES})
    first = evaluate_monolithic(problem.x, context=problem.context, mode="assembled")
    second = evaluate_monolithic(problem.x, context=problem.context, mode="assembled")
    jvp_first = monolithic_jvp(problem.x, q, context=problem.context)
    jvp_second = monolithic_jvp(problem.x, q, context=problem.context)
    np.testing.assert_array_equal(first.residual, second.residual)
    np.testing.assert_array_equal(first.jacobian.data, second.jacobian.data)
    np.testing.assert_array_equal(first.jacobian.indices, second.jacobian.indices)
    np.testing.assert_array_equal(first.jacobian.indptr, second.jacobian.indptr)
    np.testing.assert_array_equal(jvp_first, jvp_second)
    np.testing.assert_array_equal(problem.x, x_before)
    for (d_old, v_old), (d_new, v_new) in zip(states_before, problem.base.solid_states):
        np.testing.assert_array_equal(d_old, d_new)
        np.testing.assert_array_equal(v_old, v_new)
    for old, new in zip(fluid_before, problem.base.fluid_states):
        np.testing.assert_array_equal(old, new)
    np.testing.assert_array_equal(problem.fluid.points, mesh_before)
    np.testing.assert_array_equal(problem.topology.h.toarray(), topology_h_before)
    # read-only outputs
    assert not first.residual.flags.writeable
    assert not first.jacobian.data.flags.writeable
    assert not first.jacobian.indices.flags.writeable
    assert not first.jacobian.indptr.flags.writeable
    assert not jvp_first.flags.writeable
    assert first.candidate_id == problem.context.candidate_id
    assert set(first.block_norms) == set(BLOCK_NAMES)
    assert first.geometry_diagnostics["minimum_volume_ratio"] > 0
    assert first.timings["total_seconds"] >= first.timings["candidate_seconds"] >= 0
    with pytest.raises(ValueError):
        first.residual[0] = 1.0
    # repeated evaluation of the same inputs is bitwise identical to the batched residual
    np.testing.assert_array_equal(monolithic_residual(problem.x, context=problem.context), first.residual)
    with pytest.raises(ValueError):
        evaluate_monolithic(problem.x, context=problem.context, mode="unknown")
    with pytest.raises(ValueError):
        evaluate_monolithic(problem.x, context=problem.context, mode="jvp")
    with pytest.raises(ValueError):
        monolithic_jvp(problem.x, np.zeros((2, 2, problem.layout.n_total)), context=problem.context)


def test_harmonic_stiffness_accessor():
    _solid, fluid = two_domain()
    harmonic = build_harmonic_extension(fluid, diffusion=1.7)
    expected, _ = assemble_poisson(fluid, 0.0, 1.7)
    assert np.array_equal(harmonic.stiffness.toarray(), expected.toarray())
    assert not harmonic.stiffness.data.flags.writeable
    with pytest.raises(ValueError):
        harmonic.stiffness.data[0] = 1.0
    # extend still solves the same operator: an affine field is reproduced exactly
    exterior = fluid.boundary_nodes()
    coefficients = np.array([[0.3, -0.2], [0.1, 0.4]])
    values = coefficients @ fluid.points[exterior].T
    solution = harmonic.extend(exterior, values.T)
    np.testing.assert_allclose(solution.values, (coefficients @ fluid.points.T).T, rtol=0, atol=1e-11)


# ---------------------------------------------------------------------------
# optional smoke: real FSI1 mesh
# ---------------------------------------------------------------------------

def test_real_fsi1_mesh_smoke(tmp_path):
    """Read-only residual/JVP/Jacobian smoke on the FSI1 h=0.02 mesh (no Newton)."""
    mesh_size = 0.02
    pair = fsi1_mesh(mesh_size, tmp_path/"mesh")
    fluid, solid = pair.fluid, pair.solid
    fluid_space = P2Space.from_mesh(fluid)
    topology = build_matching_interface(solid, fluid, fluid_space, solid_tag="solid-interface",
                                        fluid_tag="fluid-interface", atol=1e-12, rtol=0)
    harmonic = build_harmonic_extension(fluid)
    material = ElasticMaterial(1400000.0, 0.4, density=1000.0, thickness=1.0)
    layout = build_block_layout(topology, fluid_dirichlet_tags=["inlet", "walls", "cylinder-rigid-wall"],
                                solid_dirichlet_tags=["solid-clamp"], pressure_pin=None)
    zero_solid = np.zeros((len(solid.points), 2))
    zero_fluid = np.zeros((len(fluid_space.points), 2))
    zero_mesh = np.zeros_like(fluid.points)
    solid_history = BDFHistory(np.zeros((2, len(solid.points), 2)), 1.0, order=1)
    solid_history.accept(solid_history.context(), np.zeros((2, len(solid.points), 2)))
    stencil = solid_history.stencil()
    base = MonolithicBase(reference_mesh=fluid, material=material, fluid_density=1000.0,
                          fluid_viscosity=1.0, solid_states=((zero_solid, zero_solid),),
                          fluid_states=(zero_fluid,), accepted_geometries=(
                              build_ale_geometry(fluid, zero_mesh, candidate_id=("fsi1", 0)),),
                          generation=stencil.step_id)
    context = MonolithicContext(layout, base, stencil, topology, harmonic, candidate_id=("fsi1", mesh_size),
                                thickness=1.0, pressure_constraint=None)
    x = np.zeros(layout.n_total)
    points = fluid_space.points
    velocity = np.zeros((len(points), 2))
    velocity[:, 0] = 0.2*np.sin(np.pi*points[:, 0]/2.5)*np.sin(np.pi*points[:, 1]/0.41)
    velocity[:, 1] = 0.05*np.cos(np.pi*points[:, 0]/2.5)*np.sin(np.pi*points[:, 1]/0.41)
    x[layout.slice_of("u_f")] = velocity.ravel()
    pressure = 40.0*np.sin(np.pi*fluid.points[:, 0]/2.5)*np.cos(np.pi*fluid.points[:, 1]/0.41)
    x[layout.slice_of("p_f")] = pressure
    x[layout.slice_of("g")] = 1e-3
    rng = np.random.default_rng(MIXED_SEED+8)
    q = rng.normal(size=layout.n_total)
    result = evaluate_monolithic(x, context=context)
    jvp = evaluate_monolithic(x, context=context, mode="jvp", directions=q)
    assembled = evaluate_monolithic(x, context=context, mode="assembled")
    assert np.all(np.isfinite(result.residual))
    assert np.all(np.isfinite(jvp.jvp_values))
    assert np.all(np.isfinite(assembled.jacobian.data))
    gap = np.asarray(assembled.jacobian @ q).ravel()-np.asarray(jvp.jvp_values).ravel()
    relative = float(np.linalg.norm(gap)/max(np.linalg.norm(jvp.jvp_values), 1e-300))
    assert relative <= 1e-8
    record = {"mesh_size": mesh_size, "interface_edges": topology.n_edges,
              "fluid_cells": fluid.n_cells, "solid_cells": solid.n_cells,
              "n_total": int(layout.n_total), "sizes": list(layout.sizes),
              "nnz": int(assembled.jacobian.nnz), "jvp_gap_relative": relative,
              "residual_norm": float(np.linalg.norm(result.residual)),
              "geometry": {key: float(value) for key, value in dict(result.geometry_diagnostics).items()},
              "mode_seconds": {"residual": float(result.timings["total_seconds"]),
                               "jvp": float(jvp.timings["total_seconds"]),
                               "assembled": float(assembled.timings["total_seconds"])},
              "timings": {key: float(value) for key, value in dict(assembled.timings).items()}}
    output = Path("results/m5/t21")
    output.mkdir(parents=True, exist_ok=True)
    (output/"fsi1_smoke.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n",
                                          encoding="utf-8")

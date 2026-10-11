"""T15a ALE fluid residual/tangent, GCL path health, mass/monitor diagnostics and rollback.

Frozen thresholds follow docs/development/m3_freeze.md section 6 and
docs/development/validation_plan.md (15-01...15-06, T15a scope). Validation scales:
L*=1 m, T*=1 s, U*=L*/T*, rho*=1 kg/m^3, mu*=0.01 Pa s, P*=max(rho*U*^2, mu*U*/L*).
"""

import numpy as np
import pytest

from fsi_ice.ale.gcl import build_gcl_transport
from fsi_ice.ale.geometry import build_ale_geometry
from fsi_ice.fem.p1 import physical_gradients as p1_physical_gradients
from fsi_ice.fem.p2 import P2Space, p2_physical_gradients, p2_values
from fsi_ice.fem.poisson import solve_dirichlet
from fsi_ice.fem.quadrature import triangle_quadrature
from fsi_ice.fluid.ale import (
    ALEFlowContext,
    ale_geometry_jvp,
    build_ale_flow_context,
    history_mass_load,
    mesh_transport_matrix,
)
from fsi_ice.fluid.dynamics import solve_flow_step
from fsi_ice.fluid.forces import BoundaryMonitor, boundary_force, boundary_flux
from fsi_ice.fluid.linear_solver import ReusedLUSolver
from fsi_ice.fluid.taylor_hood import (
    assemble_stokes,
    convection,
    flow_constraints,
    flow_error_norms,
    navier_stokes_residual,
)
from fsi_ice.mesh import rectangle_mesh
from fsi_ice.time import BDFHistory

L_STAR = 1.0
T_STAR = 1.0
U_STAR = L_STAR/T_STAR
RHO_STAR = 1.0
MU_STAR = 0.01
P_STAR = max(RHO_STAR*U_STAR**2, MU_STAR*U_STAR/L_STAR)
CONSTANT_VELOCITY = np.array([1.0, 0.37])

FIXED_MESH_RESIDUAL_LIMIT = 1e-12
FIXED_MESH_SOLUTION_LIMIT = 1e-10
CONSTANT_ROW_LIMIT = 1e-11
CONSTANT_SPEED_LIMIT = 1e-9*U_STAR
CONSTANT_PRESSURE_LIMIT = 1e-8*P_STAR
FREE_RESIDUAL_LIMIT = 1e-8
PHYSICAL_FLUX_LIMIT = 1e-9*U_STAR*L_STAR
INSTANTANEOUS_LIMIT = 1e-9*U_STAR*L_STAR
GCL_CELL_LIMIT = 1e-11
GCL_EDGE_LIMIT = 1e-12*U_STAR*L_STAR
GCL_BUDGET_LIMIT = 1e-10*U_STAR*L_STAR
MONITOR_LIMIT = 1e-11
CACHE_RESIDUAL_LIMIT = 1e-10
RETRY_LIMIT = 1e-10
DERIVATIVE_LIMIT = 1e-5
BDF1_ORDER_MIN = 0.9
BDF2_ORDER_MIN = 1.8

DEFORMATION = np.array([[0.1, 0.15], [-0.05, 0.08]])
DEFORMATION_OFFSET = np.array([0.02, -0.01])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _scaling_motion(points, time):
    return 0.05*np.sin(2*np.pi*time/T_STAR)*points


def _internal_motion(points, time):
    envelope = 0.03*np.sin(np.pi*points[:, 0]/L_STAR)*np.sin(np.pi*points[:, 1]/L_STAR)
    phase = 2*np.pi*time/T_STAR
    return envelope[:, None]*np.array([np.sin(phase), np.cos(phase)])


def _swirl(points, amplitude=0.5):
    x, y = np.pi*points[:, 0]/L_STAR, np.pi*points[:, 1]/L_STAR
    return amplitude*np.column_stack((np.sin(x)*np.cos(y), -np.cos(x)*np.sin(y)))


def _constant_velocity(operators, velocity=CONSTANT_VELOCITY):
    return np.tile(velocity, (len(operators.space.points), 1))


def _state(operators, velocity, pressure=0.0):
    values = np.zeros(operators.n_total)
    values[:operators.n_velocity] = velocity.ravel()
    values[operators.n_velocity:] = pressure
    return values


def _exterior_edges(mesh):
    edges = np.sort(np.concatenate([
        mesh.cells[:, [0, 1]], mesh.cells[:, [1, 2]], mesh.cells[:, [2, 0]],
    ]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]


def _boundary_flux(operators, velocity):
    """Independent 3-point Gauss integral of velocity.n over all exterior edges."""
    mesh = operators.space.mesh
    owners = {}
    for e, cell in enumerate(mesh.cells):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            owners.setdefault(tuple(sorted((cell[a], cell[b]))), []).append((e, a, b))
    abscissae, weights = np.polynomial.legendre.leggauss(3)
    abscissae, weights = (abscissae+1)/2, weights/2
    vertices = np.array([[0, 0], [1, 0], [0, 1]])
    total = 0.0
    for edge in _exterior_edges(mesh):
        entries = owners[tuple(edge)]
        assert len(entries) == 1
        e, a, b = entries[0]
        reference = (1-abscissae[:, None])*vertices[a]+abscissae[:, None]*vertices[b]
        field = p2_values(reference) @ velocity[operators.space.cells[e]]
        tangent = mesh.points[mesh.cells[e, b]]-mesh.points[mesh.cells[e, a]]
        normal = np.array([tangent[1], -tangent[0]])
        total += float(np.einsum("q,qi,i", weights, field, normal))
    return total


def _divergence_integral(operators, velocity):
    points, weights = triangle_quadrature(5)
    gradients = p2_physical_gradients(operators.space.mesh, points)
    divergence = np.einsum("eai,eqai->eq", velocity[operators.space.cells], gradients)
    measure = operators.space.mesh.determinants[:, None]*weights[None, :]
    return float(np.sum(measure*divergence))


def _momentum_row_scale(operators, transport, stencil, velocity=CONSTANT_VELOCITY):
    coefficients = np.asarray(stencil.coefficients)
    ratios = [transport.candidate_geometry.volume_ratios]
    ratios.extend(geometry.volume_ratios for geometry in transport.accepted_geometries)
    weighted = np.zeros(operators.space.mesh.n_cells)
    for coefficient, ratio in zip(coefficients, ratios):
        weighted += np.abs(coefficient)*ratio
    points, weights = triangle_quadrature(5)
    integral = np.einsum("q,qa->a", weights, np.abs(p2_values(points)))
    reference = transport.reference_mesh
    local = np.einsum("a,e->ea", integral, weighted*reference.determinants)
    node_scale = np.zeros(len(operators.space.points))
    np.add.at(node_scale, operators.space.cells.ravel(), local.ravel())
    return node_scale[:, None]*np.abs(velocity)[None, :] + RHO_STAR*U_STAR**2*L_STAR


def _pressure_row_scale(operators):
    mesh = operators.space.mesh
    gradients = np.abs(p1_physical_gradients(mesh))
    local = np.sum(gradients, axis=2)*(mesh.determinants/2)[:, None]
    scale = np.zeros(mesh.n_nodes)
    np.add.at(scale, mesh.cells.ravel(), local.ravel())
    return U_STAR*scale + 1e-12*U_STAR*L_STAR


def _build_step(mesh, motion, d_history, accepted, *, step, viscosity=MU_STAR, density=RHO_STAR,
                body_force=(0.0, 0.0), path_volume_ratio_min=None):
    stencil = d_history.stencil()
    geometry = build_ale_geometry(mesh, motion(mesh.points, stencil.times[0]),
                                  candidate_id=("step", step))
    transport = build_gcl_transport(mesh, geometry, accepted[:stencil.order], stencil,
                                    path_volume_ratio_min=path_volume_ratio_min)
    operators = assemble_stokes(geometry.current_mesh, viscosity=viscosity, density=density,
                                body_force=body_force)
    return stencil, geometry, transport, operators


def _run_step(mesh, motion, d_history, u_history, accepted, *, step, velocity, tolerance=1e-11,
              initial=None, viscosity=MU_STAR, density=RHO_STAR, body_force=(0.0, 0.0),
              pressure_pin=0, pressure_value=0.0, path_volume_ratio_min=None, max_iterations=30):
    u_context = u_history.context()
    stencil, geometry, transport, operators = _build_step(
        mesh, motion, d_history, accepted, step=step, viscosity=viscosity, density=density,
        body_force=body_force, path_volume_ratio_min=path_volume_ratio_min)
    assert np.isclose(u_context.time, stencil.times[0], rtol=0.0, atol=1e-14)
    ale_context = build_ale_flow_context(operators, geometry, transport, u_history.states[:stencil.order])
    solution = solve_flow_step(operators, velocity, u_context, ale_context=ale_context, initial=initial,
                               tolerance=tolerance, pressure_pin=pressure_pin, pressure_value=pressure_value,
                               max_iterations=max_iterations)
    return solution, u_context, stencil, geometry, transport, operators, ale_context


def _accept(mesh, motion, d_history, u_history, accepted, geometry, solution):
    u_history.accept(u_history.context(), solution.velocity)
    d_history.accept(d_history.context(), geometry.displacement)
    accepted.insert(0, geometry)


def _constant_history(mesh, motion, order, dt, *, steps_before=0):
    d_history = BDFHistory(motion(mesh.points, 0.0), dt, order=order)
    initial = np.tile(CONSTANT_VELOCITY, (len(P2Space.from_mesh(mesh).points), 1))
    u_history = BDFHistory(initial, dt, order=order)
    accepted = [build_ale_geometry(mesh, motion(mesh.points, 0.0), candidate_id=("initial", dt, order))]
    for index in range(steps_before):
        d_context = d_history.context()
        u_history.accept(u_history.context(), initial)
        d_history.accept(d_context, motion(mesh.points, d_context.time))
        accepted.insert(0, build_ale_geometry(mesh, motion(mesh.points, d_context.time),
                                              candidate_id=("history", index)))
    return d_history, u_history, accepted


# --------------------------------------------------------------------------
# 15-01 fixed-grid degeneration
# --------------------------------------------------------------------------

def test_15_01_fixed_grid_degeneration_matches_existing_bdf():
    mesh = rectangle_mesh(4)
    dt = 0.1
    space = P2Space.from_mesh(mesh)
    generator = np.random.default_rng(15)
    first = 0.1*generator.standard_normal((len(space.points), 2))
    second = 0.05*generator.standard_normal((len(space.points), 2))
    u_history = BDFHistory(first, dt, order=2)
    d_history = BDFHistory(np.zeros_like(mesh.points), dt, order=2)
    u_history.accept(u_history.context(), second)
    d_history.accept(d_history.context(), np.zeros_like(mesh.points))
    context = u_history.context()
    stencil = d_history.stencil()
    assert stencil.order == 2

    geometry = build_ale_geometry(mesh, np.zeros_like(mesh.points), candidate_id="degenerate")
    transport = build_gcl_transport(mesh, geometry, [geometry, geometry], stencil)
    fixed = assemble_stokes(mesh, viscosity=MU_STAR, density=RHO_STAR)
    ale = assemble_stokes(geometry.current_mesh, viscosity=MU_STAR, density=RHO_STAR)
    np.testing.assert_array_equal(fixed.matrix.toarray(), ale.matrix.toarray())
    np.testing.assert_array_equal(fixed.mass.toarray(), ale.mass.toarray())
    assert np.count_nonzero(transport.flux) == 0
    assert np.max(np.abs(transport.divergence)) <= 1e-12
    assert np.max(np.abs(transport.edge_fluxes)) == 0

    values = generator.standard_normal(fixed.n_total)
    fixed_residual, fixed_jacobian = navier_stokes_residual(
        fixed, values, time_coefficient=context.coefficient,
        time_rhs=-(fixed.mass @ context.history_term.ravel()))
    ale_context = build_ale_flow_context(ale, geometry, transport, u_history.states[:stencil.order])
    ale_residual, ale_jacobian = navier_stokes_residual(ale, values, ale_context=ale_context)
    n_velocity = fixed.n_velocity
    momentum_scale = max(float(np.max(np.abs(fixed_residual[:n_velocity]))),
                         float(np.max(np.abs(ale_residual[:n_velocity]))), 1.0)
    assert np.max(np.abs(fixed_residual[:n_velocity]-ale_residual[:n_velocity])) <= \
        FIXED_MESH_RESIDUAL_LIMIT*momentum_scale
    pressure_scale = max(float(np.max(np.abs(fixed_residual[n_velocity:]))), 1.0)
    assert np.max(np.abs(fixed_residual[n_velocity:]-ale_residual[n_velocity:])) <= \
        FIXED_MESH_RESIDUAL_LIMIT*pressure_scale
    jacobian_scale = max(float(np.max(np.abs(fixed_jacobian.toarray()))), 1.0)
    difference = (fixed_jacobian-ale_jacobian).toarray()
    assert np.max(np.abs(difference)) <= FIXED_MESH_RESIDUAL_LIMIT*jacobian_scale

    fixed_solution = solve_flow_step(fixed, CONSTANT_VELOCITY, context, initial=values, tolerance=1e-12)
    ale_solution = solve_flow_step(ale, CONSTANT_VELOCITY, context, ale_context=ale_context,
                                   initial=values, tolerance=1e-12)
    assert np.max(np.abs(fixed_solution.velocity-ale_solution.velocity)) <= FIXED_MESH_SOLUTION_LIMIT*U_STAR
    assert np.max(np.abs(fixed_solution.pressure-ale_solution.pressure)) <= FIXED_MESH_SOLUTION_LIMIT*P_STAR


def test_15_01_forbidden_time_arguments_and_geometry_mismatch():
    mesh = rectangle_mesh(3)
    dt = 0.05
    d_history, u_history, accepted = _constant_history(mesh, _scaling_motion, 2, dt, steps_before=1)
    stencil, geometry, transport, operators = _build_step(mesh, _scaling_motion, d_history, accepted, step=0)
    ale_context = build_ale_flow_context(operators, geometry, transport, u_history.states[:stencil.order])
    values = _state(operators, _constant_velocity(operators))
    with pytest.raises(ValueError, match="time_coefficient|time_rhs"):
        navier_stokes_residual(operators, values, time_coefficient=1.0, ale_context=ale_context)
    with pytest.raises(ValueError, match="time_coefficient|time_rhs"):
        navier_stokes_residual(operators, values, time_rhs=np.zeros(operators.n_velocity),
                               ale_context=ale_context)
    with pytest.raises(ValueError, match="time_coefficient|time_rhs"):
        solve_flow_step(operators, CONSTANT_VELOCITY, u_history.context(), ale_context=ale_context,
                        time_rhs=np.zeros(operators.n_velocity))
    other_mesh = rectangle_mesh(3)
    other_operators = assemble_stokes(other_mesh, viscosity=MU_STAR, density=RHO_STAR)
    with pytest.raises(ValueError, match="geometry|current_mesh|mesh"):
        build_ale_flow_context(other_operators, geometry, transport, u_history.states[:stencil.order])
    with pytest.raises(ValueError, match="current mesh|geometry"):
        navier_stokes_residual(other_operators, values, ale_context=ale_context)
    with pytest.raises(ValueError):
        build_ale_flow_context(operators, geometry, transport, u_history.states[:1])


# --------------------------------------------------------------------------
# 15-02 constant residual substitution and constant preservation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("motion", [_scaling_motion, _internal_motion], ids=["scaling", "internal"])
@pytest.mark.parametrize("order", [1, 2])
def test_15_02_constant_residual_substitution(order, motion):
    mesh = rectangle_mesh(3)
    dt = 0.05
    d_history, u_history, accepted = _constant_history(mesh, motion, order, dt, steps_before=order-1)
    stencil, geometry, transport, operators = _build_step(mesh, motion, d_history, accepted, step=0)
    assert stencil.order == order
    ale_context = build_ale_flow_context(operators, geometry, transport, u_history.states[:stencil.order])
    assert np.linalg.norm(history_mass_load(operators, ale_context)) > 0.1

    values = _state(operators, _constant_velocity(operators))
    residual, _ = navier_stokes_residual(operators, values, ale_context=ale_context)
    n_velocity = operators.n_velocity
    momentum = np.abs(residual[:n_velocity].reshape(-1, 2))/_momentum_row_scale(operators, transport, stencil)
    pressure = np.abs(residual[n_velocity:])/(RHO_STAR*U_STAR**2*L_STAR)
    assert np.max(momentum) <= CONSTANT_ROW_LIMIT
    assert np.max(pressure) <= CONSTANT_ROW_LIMIT


@pytest.mark.parametrize("motion", [_scaling_motion, _internal_motion], ids=["scaling", "internal"])
@pytest.mark.parametrize("order", [1, 2])
def test_15_02_constant_preservation_over_one_second(order, motion):
    mesh = rectangle_mesh(3)
    maxima = {"speed": 0.0, "pressure": 0.0, "residual": 0.0, "continuity": 0.0, "flux": 0.0}
    for dt in (0.05, 0.025, 0.0125):
        d_history = BDFHistory(motion(mesh.points, 0.0), dt, order=order)
        initial = np.tile(CONSTANT_VELOCITY, (len(P2Space.from_mesh(mesh).points), 1))
        u_history = BDFHistory(initial, dt, order=order)
        accepted = [build_ale_geometry(mesh, motion(mesh.points, 0.0), candidate_id=("initial", dt))]
        steps = int(round(1.0*T_STAR/dt))
        for step in range(steps):
            solution, _, _, geometry, _, operators, ale_context = _run_step(
                mesh, motion, d_history, u_history, accepted, step=step, velocity=CONSTANT_VELOCITY,
                initial=None, tolerance=1e-11)
            speed = float(np.max(np.abs(solution.velocity-CONSTANT_VELOCITY)))
            pressure = float(np.max(np.abs(solution.pressure)))
            flux = abs(_boundary_flux(operators, solution.velocity))
            maxima["speed"] = max(maxima["speed"], speed)
            maxima["pressure"] = max(maxima["pressure"], pressure)
            maxima["residual"] = max(maxima["residual"], solution.relative_residual)
            maxima["continuity"] = max(maxima["continuity"], solution.continuity_residual)
            maxima["flux"] = max(maxima["flux"], flux)
            assert speed <= CONSTANT_SPEED_LIMIT
            assert pressure <= CONSTANT_PRESSURE_LIMIT
            assert solution.relative_residual <= FREE_RESIDUAL_LIMIT
            assert solution.continuity_residual <= 1e-9
            assert flux <= PHYSICAL_FLUX_LIMIT
            if step == 2:
                perturbed = _state(operators, _constant_velocity(operators))
                perturbed[:operators.n_velocity] += 1e-5*np.sin(np.arange(operators.n_velocity))
                recovered = solve_flow_step(operators, CONSTANT_VELOCITY, u_history.context(),
                                            ale_context=ale_context, initial=perturbed,
                                            tolerance=1e-12, max_iterations=30)
                assert np.max(np.abs(recovered.velocity-CONSTANT_VELOCITY)) <= CONSTANT_SPEED_LIMIT
            _accept(mesh, motion, d_history, u_history, accepted, geometry, solution)
        assert abs(u_history.time-1.0*T_STAR) <= 1e-12
    assert maxima["speed"] <= CONSTANT_SPEED_LIMIT
    assert maxima["pressure"] <= CONSTANT_PRESSURE_LIMIT


# --------------------------------------------------------------------------
# 15-03 variable volume: instantaneous relation, sweep budget, w_gcl convergence
# --------------------------------------------------------------------------

@pytest.mark.parametrize("motion", [_scaling_motion, _internal_motion], ids=["scaling", "internal"])
@pytest.mark.parametrize("order", [1, 2])
def test_15_03_instantaneous_relation_and_sweep_budget(order, motion):
    mesh = rectangle_mesh(3)
    dt = 0.05
    d_history, u_history, accepted = _constant_history(mesh, motion, order, dt, steps_before=order-1)
    for step in range(4):
        stencil, geometry, transport, operators = _build_step(mesh, motion, d_history, accepted, step=step)
        ale_context = build_ale_flow_context(operators, geometry, transport, u_history.states[:stencil.order])
        velocity = _constant_velocity(operators)
        mesh_velocity = ale_context.mesh_velocity.values_p2
        relative_flux = _boundary_flux(operators, velocity-mesh_velocity)
        divergence = _divergence_integral(operators, mesh_velocity)
        instantaneous = abs(relative_flux+divergence)/(U_STAR*L_STAR)
        assert instantaneous <= 1e-9

        coefficients = np.asarray(stencil.coefficients)
        ratios = [geometry.volume_ratios]
        ratios.extend(geometry.volume_ratios for geometry in transport.accepted_geometries)
        areas = transport.reference_mesh.determinants/2
        volume_change = sum(coefficient*float(np.sum(ratio*areas))
                            for coefficient, ratio in zip(coefficients, ratios))
        swept = float(transport.edge_fluxes.sum())
        assert abs(swept-volume_change) <= GCL_BUDGET_LIMIT
        assert transport.budget_defect <= GCL_BUDGET_LIMIT

        u_history.accept(u_history.context(), _constant_velocity(operators))
        d_history.accept(d_history.context(), geometry.displacement)
        accepted.insert(0, geometry)


@pytest.mark.parametrize("order", [1, 2])
def test_15_03_effective_velocity_difference_converges_with_dt(order):
    final_differences = []
    for dt in (0.05, 0.025, 0.0125):
        mesh = rectangle_mesh(4)
        d_history = BDFHistory(_scaling_motion(mesh.points, 0.0), dt, order=order)
        u_history = BDFHistory(np.tile(CONSTANT_VELOCITY, (len(P2Space.from_mesh(mesh).points), 1)),
                               dt, order=order)
        accepted = [build_ale_geometry(mesh, _scaling_motion(mesh.points, 0.0), candidate_id=("w", dt))]
        difference = 0.0
        for step in range(int(round(0.5*T_STAR/dt))):
            stencil, geometry, transport, operators = _build_step(
                mesh, _scaling_motion, d_history, accepted, step=step)
            u_history.accept(u_history.context(), np.tile(CONSTANT_VELOCITY, (len(operators.space.points), 1)))
            ale_context = build_ale_flow_context(operators, geometry, transport, u_history.states[:stencil.order])
            shapes = p2_values(transport.quadrature_points)
            mesh_velocity = np.einsum(
                "qa,eai->eqi", shapes, ale_context.mesh_velocity.values_p2[operators.space.cells])
            difference = float(np.max(np.abs(transport.w_gcl-mesh_velocity)))
            d_history.accept(d_history.context(), geometry.displacement)
            accepted.insert(0, geometry)
        final_differences.append(difference)
    assert final_differences[1] < final_differences[0]
    assert final_differences[2] < final_differences[1]
    orders = [np.log2(coarse/fine) for coarse, fine in zip(final_differences, final_differences[1:])]
    threshold = BDF1_ORDER_MIN if order == 1 else BDF2_ORDER_MIN
    assert min(orders) >= threshold


# --------------------------------------------------------------------------
# 15-GCL / 15-SWEEP diagnostics on the fluid trajectory
# --------------------------------------------------------------------------

@pytest.mark.parametrize("motion", [_scaling_motion, _internal_motion], ids=["scaling", "internal"])
@pytest.mark.parametrize("order", [1, 2])
def test_15_gcl_and_sweep_defects_on_moving_domain(order, motion):
    mesh = rectangle_mesh(4)
    dt = 0.05
    d_history = BDFHistory(motion(mesh.points, 0.0), dt, order=order)
    accepted = [build_ale_geometry(mesh, motion(mesh.points, 0.0), candidate_id=("gcl", 0))]
    worst_cell = worst_raw = worst_edge = worst_budget = 0.0
    for step in range(4):
        stencil, geometry, transport, operators = _build_step(mesh, motion, d_history, accepted, step=step)
        coefficients = np.asarray(stencil.coefficients)
        ratios = [geometry.volume_ratios]
        ratios.extend(geometry.volume_ratios for geometry in transport.accepted_geometries)
        weighted = sum(np.abs(a)*j for a, j in zip(coefficients, ratios))
        scale = np.maximum(weighted, 1.0/T_STAR)
        normalized = float(np.max(transport.cell_defects/scale))
        raw = float(np.max(transport.cell_defects))
        worst_cell = max(worst_cell, normalized)
        worst_raw = max(worst_raw, raw)
        worst_edge = max(worst_edge, transport.edge_defect)
        worst_budget = max(worst_budget, transport.budget_defect)
        # Frozen check is normalized; the dimensional defect and its 1/T* relative
        # version are reported alongside it (no extra frozen threshold).
        assert normalized <= GCL_CELL_LIMIT, (normalized, float(raw), float(raw*T_STAR))
        assert transport.edge_defect <= GCL_EDGE_LIMIT
        assert transport.budget_defect <= GCL_BUDGET_LIMIT
        d_history.accept(d_history.context(), geometry.displacement)
        accepted.insert(0, geometry)
    assert worst_cell <= GCL_CELL_LIMIT, (worst_cell, worst_raw, worst_raw*T_STAR)
    assert worst_edge <= GCL_EDGE_LIMIT
    assert worst_budget <= GCL_BUDGET_LIMIT


# --------------------------------------------------------------------------
# 15-MASS mass and continuity diagnostics
# --------------------------------------------------------------------------

def test_15_mass_diagnostics_on_constant_state():
    mesh = rectangle_mesh(3)
    dt = 0.05
    d_history, u_history, accepted = _constant_history(mesh, _scaling_motion, 2, dt, steps_before=1)
    solution, _, _, _, _, operators, _ = _run_step(
        mesh, _scaling_motion, d_history, u_history, accepted, step=0, velocity=CONSTANT_VELOCITY,
        initial=None, tolerance=1e-11)
    rows = operators.divergence @ solution.velocity.ravel()
    boundary = _boundary_flux(operators, solution.velocity)
    assert abs(float(np.sum(rows))-boundary) <= PHYSICAL_FLUX_LIMIT
    assert np.max(np.abs(rows)/_pressure_row_scale(operators)) <= 1e-9
    assert solution.continuity_residual <= 1e-9
    assert solution.pressure[0] == 0.0


@pytest.mark.parametrize("order", [1, 2])
def test_15_mass_diagnostics_on_nontrivial_flow(order):
    mesh = rectangle_mesh(4)
    dt = 0.05
    d_history = BDFHistory(_internal_motion(mesh.points, 0.0), dt, order=order)
    initial = np.zeros((len(P2Space.from_mesh(mesh).points), 2))
    u_history = BDFHistory(initial, dt, order=order)
    accepted = [build_ale_geometry(mesh, _internal_motion(mesh.points, 0.0), candidate_id=("mass", 0))]
    solution = None
    operators = None
    for step in range(3):
        solution, _, stencil, geometry, transport, operators, ale_context = _run_step(
            mesh, _internal_motion, d_history, u_history, accepted, step=step,
            velocity=lambda points: _swirl(points), tolerance=1e-12)
        _accept(mesh, _internal_motion, d_history, u_history, accepted, geometry, solution)
    rows = operators.divergence @ solution.velocity.ravel()
    boundary = _boundary_flux(operators, solution.velocity)
    assert abs(float(np.sum(rows))-boundary) <= PHYSICAL_FLUX_LIMIT
    assert np.max(np.abs(rows)/_pressure_row_scale(operators)) <= 1e-9
    assert solution.continuity_residual <= 1e-9
    assert np.max(np.abs(solution.velocity)) > 0.1
    assert solution.pressure[0] == 0.0


# --------------------------------------------------------------------------
# 15-MON current-geometry boundary monitors
# --------------------------------------------------------------------------

def test_15_monitor_affine_analytic_flux_force_and_stale_rejection():
    mesh = rectangle_mesh(4)
    displacement = mesh.points @ DEFORMATION.T + DEFORMATION_OFFSET
    geometry = build_ale_geometry(mesh, displacement, candidate_id="monitor-a")
    operators = assemble_stokes(geometry.current_mesh, viscosity=0.25, density=RHO_STAR)
    velocity_matrix = np.array([[0.2, 0.5], [-0.3, 0.1]])
    velocity_offset = np.array([0.1, -0.2])
    pressure_value = 0.3

    def velocity(points):
        return points @ velocity_matrix.T + velocity_offset

    from fsi_ice.fluid.taylor_hood import FlowSolution

    solution = FlowSolution(velocity(operators.space.points),
                            np.full(operators.space.mesh.n_nodes, pressure_value), 0.0, 0.0)
    stress = 0.25*(velocity_matrix+velocity_matrix.T)-pressure_value*np.eye(2)
    for tag in operators.space.mesh.boundaries:
        monitor = BoundaryMonitor(operators, tag)
        force = monitor.force(solution)
        reference_force = boundary_force(operators, solution, tag)
        flux = monitor.flux(solution)
        reference_flux = boundary_flux(operators, solution, tag)
        assert np.max(np.abs(force-reference_force)) <= MONITOR_LIMIT*max(np.linalg.norm(reference_force), 1.0)
        assert abs(flux-reference_flux) <= MONITOR_LIMIT*max(abs(reference_flux), U_STAR*L_STAR)

        normal_integral = np.zeros(2)
        analytic_flux = 0.0
        for edge in operators.space.mesh.boundaries[tag]:
            a, b = operators.space.mesh.points[edge[0]], operators.space.mesh.points[edge[1]]
            tangent = b-a
            normal_integral += np.array([tangent[1], -tangent[0]])
            analytic_flux += float((velocity_offset+velocity_matrix @ a) @ np.array([tangent[1], -tangent[0]])
                                   + 0.5*(velocity_matrix @ tangent) @ np.array([tangent[1], -tangent[0]]))
        analytic_force = -stress @ normal_integral
        assert np.max(np.abs(force-analytic_force)) <= MONITOR_LIMIT*max(np.linalg.norm(analytic_force), 1.0)
        assert abs(flux-analytic_flux) <= MONITOR_LIMIT*max(abs(analytic_flux), U_STAR*L_STAR)
        assert monitor.force(solution, operators=operators) is not None

    moved = build_ale_geometry(mesh, 1.5*displacement, candidate_id="monitor-b")
    moved_operators = assemble_stokes(moved.current_mesh, viscosity=0.25, density=RHO_STAR)
    moved_solution = FlowSolution(velocity(moved_operators.space.points),
                                  np.full(moved_operators.space.mesh.n_nodes, pressure_value), 0.0, 0.0)
    monitor = BoundaryMonitor(operators, "bottom")
    with pytest.raises(ValueError, match="different geometry|rebuild"):
        monitor.force(solution, operators=moved_operators)
    with pytest.raises(ValueError, match="different geometry|rebuild"):
        monitor.flux(moved_solution, operators=moved_operators)
    rebuilt = BoundaryMonitor(moved_operators, "bottom")
    np.testing.assert_allclose(rebuilt.force(moved_solution, operators=moved_operators),
                               boundary_force(moved_operators, moved_solution, "bottom"), atol=0, rtol=1e-13)

    returned = build_ale_geometry(mesh, displacement, candidate_id="monitor-c")
    returned_operators = assemble_stokes(returned.current_mesh, viscosity=0.25, density=RHO_STAR)
    with pytest.raises(ValueError, match="different geometry|rebuild"):
        monitor.flux(solution, operators=returned_operators)


# --------------------------------------------------------------------------
# 15-06 cache invalidation
# --------------------------------------------------------------------------

def test_15_06_linear_solver_signature_invalidation():
    mesh = rectangle_mesh(4)
    dt = 0.05
    d_history, u_history, accepted = _constant_history(mesh, _internal_motion, 2, dt, steps_before=1)
    stencil, geometry, transport, operators = _build_step(mesh, _internal_motion, d_history, accepted, step=0)
    ale_context = build_ale_flow_context(operators, geometry, transport, u_history.states[:stencil.order])
    generator = np.random.default_rng(60)
    state = 0.01*generator.standard_normal(operators.n_total)
    _, jacobian = navier_stokes_residual(operators, state, ale_context=ale_context)
    dofs, prescribed = flow_constraints(operators, CONSTANT_VELOCITY)
    rhs = generator.standard_normal(operators.n_total)
    expected = solve_dirichlet(jacobian, rhs, dofs, 0).values
    free = np.setdiff1d(np.arange(operators.n_total), dofs)

    def true_residual(values):
        return float(np.linalg.norm(jacobian[free][:, free] @ values[free]-rhs[free])
                     / max(np.linalg.norm(rhs[free]), 1.0))

    solver = ReusedLUSolver(refresh_interval=20, velocity_dofs=operators.n_velocity, density=RHO_STAR)
    signature_a = ("geometry-a", 0, MU_STAR)
    signature_b = ("geometry-b", 1, MU_STAR)
    signature_viscosity = ("geometry-b", 1, 2*MU_STAR)
    solver.prepare(1.0, signature=signature_a)
    values = solver.solve(jacobian, rhs, dofs, signature=signature_a)
    assert np.max(np.abs(values-expected)) <= CACHE_RESIDUAL_LIMIT
    assert true_residual(values) <= CACHE_RESIDUAL_LIMIT
    assert solver.factorizations == 1
    with pytest.raises(ValueError, match="signature"):
        solver.solve(jacobian, rhs, dofs, signature=signature_b)
    solver.prepare(1.0, signature=signature_b)
    values = solver.solve(jacobian, rhs, dofs, signature=signature_b)
    assert np.max(np.abs(values-expected)) <= CACHE_RESIDUAL_LIMIT
    assert solver.factorizations == 2
    solver.prepare(1.0, signature=signature_viscosity)
    values = solver.solve(jacobian, rhs, dofs, signature=signature_viscosity)
    assert np.max(np.abs(values-expected)) <= CACHE_RESIDUAL_LIMIT
    assert solver.factorizations == 3
    solver.prepare(1.0, signature=signature_a)
    values = solver.solve(jacobian, rhs, dofs, signature=signature_a)
    assert np.max(np.abs(values-expected)) <= CACHE_RESIDUAL_LIMIT
    assert solver.factorizations == 4


def test_15_06_moving_mesh_rebuild_and_cold_agreement():
    mesh = rectangle_mesh(3)
    dt = 0.05
    order = 2
    n_velocity = 2*len(P2Space.from_mesh(mesh).points)
    d_history = BDFHistory(_internal_motion(mesh.points, 0.0), dt, order=order)
    u_history = BDFHistory(np.zeros((len(P2Space.from_mesh(mesh).points), 2)), dt, order=order)
    accepted = [build_ale_geometry(mesh, _internal_motion(mesh.points, 0.0), candidate_id=("cache", 0))]
    solver = ReusedLUSolver(refresh_interval=20, velocity_dofs=n_velocity, density=RHO_STAR)
    factorizations = []
    for step in range(3):
        cold, u_context, _, geometry, _, operators, ale_context = _run_step(
            mesh, _internal_motion, d_history, u_history, accepted, step=step,
            velocity=lambda points: _swirl(points), tolerance=1e-12)
        warm = solve_flow_step(operators, lambda points: _swirl(points), u_context, ale_context=ale_context,
                               tolerance=1e-12, linear_solver=solver)
        assert np.max(np.abs(warm.velocity-cold.velocity)) <= CACHE_RESIDUAL_LIMIT*U_STAR
        assert np.max(np.abs(warm.pressure-cold.pressure)) <= CACHE_RESIDUAL_LIMIT*P_STAR
        factorizations.append(solver.factorizations)
        _accept(mesh, _internal_motion, d_history, u_history, accepted, geometry, cold)
    assert factorizations == [1, 2, 3]
    assert solver.iterations > 0
    changed = solve_flow_step(operators, lambda points: _swirl(points), u_context, ale_context=ale_context,
                              tolerance=1e-12, linear_solver=solver, pressure_value=0.25)
    assert solver.factorizations == 4
    assert np.all(np.isfinite(changed.velocity))


# --------------------------------------------------------------------------
# 15-ROLLBACK single-field rollback and retry
# --------------------------------------------------------------------------

def test_15_rollback_repeated_candidates_snapshot_and_retry():
    mesh = rectangle_mesh(3)
    dt = 0.05
    order = 2
    d_history = BDFHistory(_internal_motion(mesh.points, 0.0), dt, order=order)
    initial = np.zeros((len(P2Space.from_mesh(mesh).points), 2))
    u_history = BDFHistory(initial, dt, order=order)
    accepted = [build_ale_geometry(mesh, _internal_motion(mesh.points, 0.0), candidate_id=("rollback", 0))]
    u_snapshot = u_history.snapshot()
    d_snapshot = d_history.snapshot()
    history_states = tuple(state.copy() for state in u_history.states)

    baseline, _, _, _, _, _, _ = _run_step(mesh, _internal_motion, d_history, u_history, accepted, step=0,
                                           velocity=lambda points: _swirl(points), tolerance=1e-11)

    u_history.restore(u_snapshot)
    d_history.restore(d_snapshot)
    for before, after in zip(history_states, u_history.states):
        np.testing.assert_array_equal(before, after)
    assert (u_history.time, u_history.snapshot()[1]) == (u_snapshot[0], u_snapshot[1])

    candidate = _run_step(mesh, _internal_motion, d_history, u_history, accepted, step=0,
                          velocity=lambda points: _swirl(points), tolerance=1e-11)
    repeated = _run_step(mesh, _internal_motion, d_history, u_history, accepted, step=0,
                         velocity=lambda points: _swirl(points), tolerance=1e-11)
    np.testing.assert_array_equal(candidate[0].velocity, repeated[0].velocity)
    for before, after in zip(history_states, u_history.states):
        np.testing.assert_array_equal(before, after)
    assert (u_history.time, u_history.snapshot()[1]) == (u_snapshot[0], u_snapshot[1])

    failed_snapshot_u = u_history.snapshot()
    failed_snapshot_d = d_history.snapshot()
    with pytest.raises(RuntimeError, match="converge|line search"):
        _run_step(mesh, _internal_motion, d_history, u_history, accepted, step=0,
                  velocity=lambda points: _swirl(points), tolerance=1e-15, max_iterations=1)
    for before, after in zip(failed_snapshot_u[2], u_history.states):
        np.testing.assert_array_equal(before, after)
    for before, after in zip(failed_snapshot_d[2], d_history.states):
        np.testing.assert_array_equal(before, after)
    assert (u_history.time, u_history.snapshot()[1]) == (failed_snapshot_u[0], failed_snapshot_u[1])
    assert (d_history.time, d_history.snapshot()[1]) == (failed_snapshot_d[0], failed_snapshot_d[1])

    retry = _run_step(mesh, _internal_motion, d_history, u_history, accepted, step=0,
                      velocity=lambda points: _swirl(points), tolerance=1e-11)
    assert np.max(np.abs(retry[0].velocity-baseline.velocity)) <= RETRY_LIMIT*U_STAR
    assert np.max(np.abs(retry[0].pressure-baseline.pressure)) <= RETRY_LIMIT*P_STAR


# --------------------------------------------------------------------------
# residual/Jacobian directional derivative self-check
# --------------------------------------------------------------------------

def test_15_tangent_directional_derivative_matches_central_difference():
    mesh = rectangle_mesh(3)
    dt = 0.05
    d_history, u_history, accepted = _constant_history(mesh, _internal_motion, 2, dt, steps_before=1)
    stencil, geometry, transport, operators = _build_step(mesh, _internal_motion, d_history, accepted, step=0)
    ale_context = build_ale_flow_context(operators, geometry, transport, u_history.states[:stencil.order])
    generator = np.random.default_rng(151)
    state = 0.1*generator.standard_normal(operators.n_total)
    direction = generator.standard_normal(operators.n_total)
    residual, jacobian = navier_stokes_residual(operators, state, ale_context=ale_context)
    tangent = jacobian @ direction
    errors = []
    for step in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        plus, _ = navier_stokes_residual(operators, state+step*direction, ale_context=ale_context,
                                         with_jacobian=False)
        minus, _ = navier_stokes_residual(operators, state-step*direction, ale_context=ale_context,
                                          with_jacobian=False)
        difference = (plus-minus)/(2*step)
        errors.append(np.linalg.norm(difference-tangent)/max(np.linalg.norm(tangent), RHO_STAR*U_STAR**2*L_STAR))
    assert np.linalg.norm(tangent) > 0.01*RHO_STAR*U_STAR**2*L_STAR
    assert min(errors) <= DERIVATIVE_LIMIT


# --------------------------------------------------------------------------
# GCL path health
# --------------------------------------------------------------------------

def test_gcl_path_health_rejects_interior_degeneracy():
    mesh = rectangle_mesh(3)
    dt = 0.05
    accepted = build_ale_geometry(mesh, np.zeros_like(mesh.points), candidate_id="path-accepted")
    stencil = BDFHistory(np.zeros_like(mesh.points), dt, order=1).context().stencil
    turned = build_ale_geometry(mesh, -2*mesh.points, candidate_id="path-turned")
    with pytest.raises(ValueError, match="swept path|cell"):
        build_gcl_transport(mesh, turned, [accepted], stencil)
    with pytest.raises(ValueError, match="path"):
        build_gcl_transport(mesh, accepted, [accepted], stencil, path_volume_ratio_min=0.0)


def test_gcl_path_volume_ratio_min_is_enforced():
    mesh = rectangle_mesh(3)
    accepted = build_ale_geometry(mesh, -0.1*mesh.points, candidate_id="path-acc")
    candidate = build_ale_geometry(mesh, -0.2*mesh.points, candidate_id="path-cand")
    stencil = BDFHistory(-0.1*mesh.points, 0.05, order=1).context().stencil
    transport = build_gcl_transport(mesh, candidate, [accepted], stencil, path_volume_ratio_min=0.5)
    assert transport.order == 1
    with pytest.raises(ValueError, match="swept path interval 0 cell"):
        build_gcl_transport(mesh, candidate, [accepted], stencil, path_volume_ratio_min=0.7)
    with pytest.raises(ValueError, match="path_volume_ratio_min"):
        build_gcl_transport(mesh, candidate, [accepted], stencil, path_volume_ratio_min=0.0)


# --------------------------------------------------------------------------
# context object behaviour
# --------------------------------------------------------------------------

def test_ale_context_freezes_inputs_and_transport_primitive_is_consistent():
    mesh = rectangle_mesh(3)
    dt = 0.05
    d_history, u_history, accepted = _constant_history(mesh, _internal_motion, 2, dt, steps_before=1)
    stencil, geometry, transport, operators = _build_step(mesh, _internal_motion, d_history, accepted, step=0)
    historical = u_history.states[:stencil.order]
    ale_context = build_ale_flow_context(operators, geometry, transport, historical)
    assert isinstance(ale_context, ALEFlowContext)
    for frozen, original in zip(ale_context.history_velocities, historical):
        assert not frozen.flags.writeable
        assert not np.shares_memory(frozen, original)
        np.testing.assert_array_equal(frozen, original)
    assert ale_context.stencil.order == stencil.order and ale_context.stencil.dt == stencil.dt
    assert ale_context.mesh_velocity.values_p2.shape == geometry.current_space.points.shape

    matrix = mesh_transport_matrix(operators, transport)
    assert matrix.shape == (operators.n_velocity, operators.n_velocity)
    values = _state(operators, _constant_velocity(operators))
    residual, _ = navier_stokes_residual(operators, values, ale_context=ale_context)
    n_velocity = operators.n_velocity
    manual = operators.matrix @ values-operators.rhs
    manual[:n_velocity] += convection(operators, values[:n_velocity].reshape(-1, 2), with_jacobian=False)[0]
    manual[:n_velocity] += history_mass_load(operators, ale_context)
    manual[:n_velocity] += ale_context.stencil.coefficients[0]*(operators.mass @ values[:n_velocity])
    manual[:n_velocity] += matrix @ values[:n_velocity]
    assert np.max(np.abs(residual-manual)) <= 1e-12*max(np.linalg.norm(manual), 1.0)


# --------------------------------------------------------------------------
# 15-04 ALE manufactured solution (Piola map of a reference divergence-free field)
# --------------------------------------------------------------------------

MMS_SPATIAL_DT = 0.003125
MMS_SPATIAL_TIME = 0.125
MMS_TEMPORAL_LEVEL = 32
MMS_TEMPORAL_TIME = 0.5
MMS_TEMPORAL_DTS = (0.1, 0.05, 0.025)
VELOCITY_SPACE_ORDER_MIN = 2.5
PRESSURE_SPACE_ORDER_MIN = 1.8


class _PiolaManufacturedSolution:
    """Smooth divergence-free ALE manufactured solution on the unit square.

    Reference field psi = U*L* sin^2(pi x/L*) sin^2(pi y/L*) with
    uhat = (d_y psi, -d_x psi) e^{-t/T*}; a smooth self-map with fixed boundary
    d = eps(t) g, g = (s, s), s = sin(pi x/L*) sin(pi y/L*), eps = A sin(2 pi t/T*)
    defines the divergence-preserving Piola map u = F uhat / det(F).  The pressure is
    p = rho U*^2 sin(pi x/L*) sin(pi y/L*) e^{-t/T*}.  The body force is the analytic
    Eulerian residual rho (du/dt|x + (u.grad)u) - mu lap(u) + grad(p), evaluated at
    current points through the inverse map; expressions are fixed here and never
    reconstructed from the discrete residual.
    """

    def __init__(self, *, amplitude=0.03, period=T_STAR):
        self.amplitude = amplitude
        self.period = period
        self.length = L_STAR
        self.velocity_scale = U_STAR
        self.time_scale = T_STAR
        self.density = RHO_STAR
        self.mu = MU_STAR

    @staticmethod
    def _cofactor(matrices):
        result = np.empty_like(matrices)
        result[:, 0, 0] = matrices[:, 1, 1]
        result[:, 0, 1] = -matrices[:, 0, 1]
        result[:, 1, 0] = -matrices[:, 1, 0]
        result[:, 1, 1] = matrices[:, 0, 0]
        return result

    @staticmethod
    def _det_directional(adjugate, matrix):
        return np.einsum("nij,nji->n", adjugate, matrix)

    def epsilon(self, time):
        return self.amplitude*np.sin(2*np.pi*time/self.period)

    def epsilon_rate(self, time):
        return self.amplitude*(2*np.pi/self.period)*np.cos(2*np.pi*time/self.period)

    def displacement(self, points, time):
        s = self._s(points)
        return self.epsilon(time)*np.column_stack((s, s))

    def _s(self, points):
        return (np.sin(np.pi*points[:, 0]/self.length)
                * np.sin(np.pi*points[:, 1]/self.length))

    def _reference(self, xi, time):
        k = np.pi/self.length
        U, T = self.velocity_scale, self.time_scale
        x = k*xi[:, 0]
        y = k*xi[:, 1]
        decay = np.exp(-time/T)
        a, b = np.sin(x)**2, np.sin(y)**2
        p, q = np.sin(2*x), np.sin(2*y)
        c2x, c2y = np.cos(2*x), np.cos(2*y)
        uhat = U*np.pi*np.column_stack((a*q, -p*b))*decay
        uhat_x = U*np.pi*k*np.column_stack((p*q, -2*c2x*b))*decay
        uhat_y = U*np.pi*k*np.column_stack((2*a*c2y, -p*q))*decay
        uhat_xx = U*np.pi*k*k*np.column_stack((2*c2x*q, 4*p*b))*decay
        uhat_xy = U*np.pi*k*k*np.column_stack((2*p*c2y, -2*c2x*q))*decay
        uhat_yy = U*np.pi*k*k*np.column_stack((-4*a*q, -2*p*c2y))*decay
        pressure = self.density*U**2*np.sin(x)*np.sin(y)*decay
        pressure_gradient = (self.density*U**2*k
                             * np.column_stack((np.cos(x)*np.sin(y), np.sin(x)*np.cos(y)))*decay)
        return dict(uhat=uhat, uhat_x=uhat_x, uhat_y=uhat_y, uhat_xx=uhat_xx,
                    uhat_xy=uhat_xy, uhat_yy=uhat_yy, uhat_t=-uhat/T,
                    pressure=pressure, pressure_gradient=pressure_gradient)

    def _map(self, xi, time):
        k = np.pi/self.length
        x = k*xi[:, 0]
        y = k*xi[:, 1]
        e = self.epsilon(time)
        rate = self.epsilon_rate(time)
        s = np.sin(x)*np.sin(y)
        sX = k*np.cos(x)*np.sin(y)
        sY = k*np.sin(x)*np.cos(y)
        sXX = -k*k*s
        sYY = -k*k*s
        sXY = k*k*np.cos(x)*np.cos(y)
        sXXX = -k**3*np.cos(x)*np.sin(y)
        sXXY = -k**3*np.sin(x)*np.cos(y)
        sXYY = -k**3*np.cos(x)*np.sin(y)
        sYYY = -k**3*np.sin(x)*np.cos(y)
        n = len(x)

        def matrix(first, second):
            out = np.empty((n, 2, 2))
            out[:, 0, 0] = first
            out[:, 1, 0] = first
            out[:, 0, 1] = second
            out[:, 1, 1] = second
            return out

        G = matrix(sX, sY)
        Gx, Gy = matrix(sXX, sXY), matrix(sXY, sYY)
        Gxx, Gxy, Gyy = matrix(sXXX, sXXY), matrix(sXXY, sXYY), matrix(sXYY, sYYY)
        F = np.eye(2)[None, :, :] + e*G
        Fx, Fy = e*Gx, e*Gy
        Fxx, Fxy, Fyy = e*Gxx, e*Gxy, e*Gyy
        Ft = rate*G
        det = F[:, 0, 0]*F[:, 1, 1]-F[:, 0, 1]*F[:, 1, 0]
        adjugate = self._cofactor(F)
        hessian = np.empty((n, 2, 2, 2))
        hessian[:, 0, 0, 0] = e*sXX
        hessian[:, 0, 0, 1] = e*sXY
        hessian[:, 0, 1, 0] = e*sXY
        hessian[:, 0, 1, 1] = e*sYY
        hessian[:, 1] = hessian[:, 0]
        return dict(F=F, inverse=np.linalg.inv(F), det=det, hessian=hessian,
                    Fx=Fx, Fy=Fy, Fxx=Fxx, Fxy=Fxy, Fyy=Fyy, Ft=Ft,
                    det_x=self._det_directional(adjugate, Fx),
                    det_y=self._det_directional(adjugate, Fy),
                    det_t=self._det_directional(adjugate, Ft),
                    det_xx=(self._det_directional(adjugate, Fxx)
                            + self._det_directional(self._cofactor(Fx), Fx)),
                    det_xy=(self._det_directional(adjugate, Fxy)
                            + self._det_directional(self._cofactor(Fx), Fy)),
                    det_yy=(self._det_directional(adjugate, Fyy)
                            + self._det_directional(self._cofactor(Fy), Fy)),
                    w=rate*np.column_stack((s, s)))

    def _composed(self, xi, time):
        ref = self._reference(xi, time)
        mp = self._map(xi, time)
        F, inverse, j = mp["F"], mp["inverse"], mp["det"]
        u, ux, uy = ref["uhat"], ref["uhat_x"], ref["uhat_y"]
        uxx, uxy, uyy = ref["uhat_xx"], ref["uhat_xy"], ref["uhat_yy"]

        def contract(matrix, vector):
            return np.einsum("nij,nj->ni", matrix, vector)

        N = contract(F, u)
        Nx = contract(mp["Fx"], u) + contract(F, ux)
        Ny = contract(mp["Fy"], u) + contract(F, uy)
        Nxx = contract(mp["Fxx"], u) + 2*contract(mp["Fx"], ux) + contract(F, uxx)
        Nxy = (contract(mp["Fxy"], u) + contract(mp["Fx"], uy) + contract(mp["Fy"], ux)
               + contract(F, uxy))
        Nyy = contract(mp["Fyy"], u) + 2*contract(mp["Fy"], uy) + contract(F, uyy)
        Nt = contract(mp["Ft"], u) + contract(F, ref["uhat_t"])
        jx, jy, jt = mp["det_x"], mp["det_y"], mp["det_t"]
        jxx, jxy, jyy = mp["det_xx"], mp["det_xy"], mp["det_yy"]
        j2 = j*j
        velocity = N/j[:, None]
        cx = Nx/j[:, None] - N*jx[:, None]/j2[:, None]
        cy = Ny/j[:, None] - N*jy[:, None]/j2[:, None]
        ct = Nt/j[:, None] - N*jt[:, None]/j2[:, None]
        cxx = (Nxx/j[:, None] - (2*Nx*jx[:, None])/j2[:, None] - (N*jxx[:, None])/j2[:, None]
               + (2*N*(jx*jx)[:, None])/j2[:, None]/j[:, None])
        cxy = (Nxy/j[:, None] - (Nx*jy[:, None]+Ny*jx[:, None])/j2[:, None]
               - (N*jxy[:, None])/j2[:, None] + (2*N*(jx*jy)[:, None])/j2[:, None]/j[:, None])
        cyy = (Nyy/j[:, None] - (2*Ny*jy[:, None])/j2[:, None] - (N*jyy[:, None])/j2[:, None]
               + (2*N*(jy*jy)[:, None])/j2[:, None]/j[:, None])
        derivative = np.stack((cx, cy), axis=2)
        gradient = np.einsum("nim,nml->nil", derivative, inverse)
        hessian = np.empty((len(j), 2, 2, 2))
        hessian[:, 0, 0, 0] = cxx[:, 0]
        hessian[:, 0, 0, 1] = cxy[:, 0]
        hessian[:, 0, 1, 0] = cxy[:, 0]
        hessian[:, 0, 1, 1] = cyy[:, 0]
        hessian[:, 1, 0, 0] = cxx[:, 1]
        hessian[:, 1, 0, 1] = cxy[:, 1]
        hessian[:, 1, 1, 0] = cxy[:, 1]
        hessian[:, 1, 1, 1] = cyy[:, 1]
        metric = np.einsum("nmi,nli->nml", inverse, inverse)
        trace_hessian = np.einsum("nijl,njl->ni", mp["hessian"], metric)
        curvature = -np.einsum("nmi,ni->nm", inverse, trace_hessian)
        laplacian = (np.einsum("nml,niml->ni", metric, hessian)
                     + np.einsum("nim,nm->ni", derivative, curvature))
        return dict(velocity=velocity, velocity_gradient=gradient, laplacian=laplacian,
                    velocity_time_derivative=ct - np.einsum("nl,nil->ni", mp["w"], gradient),
                    pressure=ref["pressure"],
                    pressure_gradient=np.einsum("nmi,nm->ni", inverse, ref["pressure_gradient"]))

    def reference_from_physical(self, points, time):
        xi = np.array(points, dtype=float, copy=True)
        e = self.epsilon(time)
        for _ in range(12):
            s = self._s(xi)
            residual = xi + e*np.column_stack((s, s)) - points
            jacobian = np.eye(2)[None, :, :] + e*self._G(xi)
            step = np.linalg.solve(jacobian, residual[..., None])[..., 0]
            xi = xi - step
            if np.max(np.abs(step)) < 1e-15:
                break
        return xi

    def _G(self, xi):
        k = np.pi/self.length
        x = k*xi[:, 0]
        y = k*xi[:, 1]
        sX = k*np.cos(x)*np.sin(y)
        sY = k*np.sin(x)*np.cos(y)
        out = np.empty((len(x), 2, 2))
        out[:, 0, 0] = sX
        out[:, 1, 0] = sX
        out[:, 0, 1] = sY
        out[:, 1, 1] = sY
        return out

    def _at_physical(self, points, time):
        return self._composed(self.reference_from_physical(points, time), time)

    def velocity(self, points, time):
        return self._at_physical(points, time)["velocity"]

    def pressure(self, points, time):
        return self._at_physical(points, time)["pressure"]

    def body_force(self, points, time):
        data = self._at_physical(points, time)
        convection = np.einsum("nl,nil->ni", data["velocity"], data["velocity_gradient"])
        return (self.density*(data["velocity_time_derivative"] + convection)
                - self.mu*data["laplacian"] + data["pressure_gradient"])


def _run_manufactured(level, dt, order, t_end, *, amplitude=0.03, tolerance=1e-11):
    mms = _PiolaManufacturedSolution(amplitude=amplitude)
    mesh = rectangle_mesh(level)
    space = P2Space.from_mesh(mesh)
    d_history = BDFHistory(mms.displacement(mesh.points, 0.0), dt, order=order)
    u_history = BDFHistory(mms.velocity(space.points, 0.0), dt, order=order)
    accepted = [build_ale_geometry(mesh, mms.displacement(mesh.points, 0.0), candidate_id=("mms", 0))]
    initial = None
    solution = None
    operators = None
    stencil_orders = []
    for step in range(int(round(t_end/dt))):
        u_context = u_history.context()
        target = u_context.time
        stencil = d_history.stencil()
        stencil_orders.append(stencil.order)
        geometry = build_ale_geometry(mesh, mms.displacement(mesh.points, target), candidate_id=("mms", step))
        transport = build_gcl_transport(mesh, geometry, accepted[:stencil.order], stencil)
        operators = assemble_stokes(geometry.current_mesh, viscosity=mms.mu, density=RHO_STAR,
                                    body_force=lambda points, t=target: mms.body_force(points, t))
        ale_context = build_ale_flow_context(operators, geometry, transport, u_history.states[:stencil.order])
        solution = solve_flow_step(operators, lambda points, t=target: mms.velocity(points, t), u_context,
                                   ale_context=ale_context, initial=initial, tolerance=tolerance,
                                   pressure_pin=0, pressure_value=0.0)
        initial = np.r_[solution.velocity.ravel(), solution.pressure]
        u_history.accept(u_context, solution.velocity)
        d_history.accept(d_history.context(), geometry.displacement)
        accepted.insert(0, geometry)
    errors = flow_error_norms(operators, solution, lambda points: mms.velocity(points, t_end),
                              lambda points: mms.pressure(points, t_end))
    return mesh, errors, stencil_orders


def test_15_04_manufactured_solution_is_smooth_divergence_free_and_analytic():
    mms = _PiolaManufacturedSolution()
    generator = np.random.default_rng(1504)
    points = 0.1 + 0.8*generator.random((32, 2))
    time = 0.37
    data = mms._composed(mms.reference_from_physical(points, time), time)
    velocity = mms.velocity(points, time)
    gradient = data["velocity_gradient"]
    assert np.max(np.abs(gradient[:, 0, 0]+gradient[:, 1, 1])) <= 1e-12
    assert np.max(np.abs(velocity)) > 0.1

    step = 1e-4
    for axis in range(2):
        shift = np.zeros(2)
        shift[axis] = step
        fd_gradient = (mms.velocity(points+shift, time)-mms.velocity(points-shift, time))/(2*step)
        assert np.max(np.abs(fd_gradient-gradient[:, :, axis])) <= 1e-6*np.max(np.abs(gradient))
        fd_pressure = (mms.pressure(points+shift, time)-mms.pressure(points-shift, time))/(2*step)
        assert np.max(np.abs(fd_pressure-data["pressure_gradient"][:, axis])) <= \
            1e-6*np.max(np.abs(data["pressure_gradient"]))
    fd_time = (mms.velocity(points, time+step)-mms.velocity(points, time-step))/(2*step)
    assert np.max(np.abs(fd_time-data["velocity_time_derivative"])) <= 1e-6*np.max(np.abs(data["velocity_time_derivative"]))
    laplacian = data["laplacian"]
    fd_laplacian = np.zeros_like(velocity)
    for axis in range(2):
        shift = np.zeros(2)
        shift[axis] = step
        fd_laplacian += (mms.velocity(points+shift, time)-2*velocity
                         + mms.velocity(points-shift, time))/step**2
    assert np.max(np.abs(fd_laplacian-laplacian)) <= 1e-5*np.max(np.abs(laplacian))

    boundary = np.array([[0.0, 0.3], [1.0, 0.6], [0.4, 0.0], [0.7, 1.0]])
    assert np.max(np.abs(mms.displacement(boundary, time))) <= 1e-14
    boundary_velocity = mms.velocity(boundary, time)
    normals = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    assert np.max(np.abs(np.sum(boundary_velocity*normals, axis=1))) <= 1e-12
    assert np.all(np.isfinite(mms.body_force(points, time)))


def test_15_04_spatial_convergence_on_three_meshes():
    rows = []
    for level in (8, 16, 32):
        mesh, errors, _ = _run_manufactured(level, MMS_SPATIAL_DT, 2, MMS_SPATIAL_TIME)
        rows.append((level, mesh.h, errors))
    for (_, _, coarse), (_, _, fine) in zip(rows, rows[1:]):
        assert fine["velocity_l2"] < coarse["velocity_l2"]
        assert fine["pressure_l2"] < coarse["pressure_l2"]
        velocity_order = np.log2(coarse["velocity_l2"]/fine["velocity_l2"])
        pressure_order = np.log2(coarse["pressure_l2"]/fine["pressure_l2"])
        assert velocity_order >= VELOCITY_SPACE_ORDER_MIN, (rows, velocity_order)
        assert pressure_order >= PRESSURE_SPACE_ORDER_MIN, (rows, pressure_order)


@pytest.mark.parametrize("order", [1, 2])
def test_15_04_temporal_convergence_on_finest_mesh(order):
    errors = []
    for dt in MMS_TEMPORAL_DTS:
        _, norms, stencil_orders = _run_manufactured(MMS_TEMPORAL_LEVEL, dt, order, MMS_TEMPORAL_TIME)
        errors.append(norms["velocity_l2"])
        expected = [1]+[2]*(len(stencil_orders)-1) if order == 2 else [1]*len(stencil_orders)
        assert stencil_orders == expected
    orders = [np.log2(coarse/fine) for coarse, fine in zip(errors, errors[1:])]
    threshold = BDF1_ORDER_MIN if order == 1 else BDF2_ORDER_MIN
    assert min(orders) >= threshold, (errors, orders)


# --------------------------------------------------------------------------
# 15-05 geometry JVP
# --------------------------------------------------------------------------

def _jvp_body_force(points):
    x, y = np.pi*points[:, 0]/L_STAR, np.pi*points[:, 1]/L_STAR
    return np.column_stack((0.1*np.sin(x)*np.cos(y), 0.05*np.cos(x)*np.sin(y)))


def _jvp_configuration(body_force, *, level=4, dt=0.05):
    mesh = rectangle_mesh(level)
    space = P2Space.from_mesh(mesh)
    d_history = BDFHistory(_internal_motion(mesh.points, 0.0), dt, order=2)
    u_history = BDFHistory(0.3*_swirl(space.points), dt, order=2)
    d_history.accept(d_history.context(), _internal_motion(mesh.points, dt))
    u_history.accept(u_history.context(), 0.25*_swirl(space.points))
    stencil = d_history.stencil()
    displacement = _internal_motion(mesh.points, stencil.times[0])
    accepted = [
        build_ale_geometry(mesh, _internal_motion(mesh.points, dt), candidate_id=("jvp-accepted-1",)),
        build_ale_geometry(mesh, _internal_motion(mesh.points, 0.0), candidate_id=("jvp-accepted-0",)),
    ]
    geometry = build_ale_geometry(mesh, displacement, candidate_id=("jvp-candidate",))
    transport = build_gcl_transport(mesh, geometry, accepted, stencil)
    operators = assemble_stokes(geometry.current_mesh, viscosity=MU_STAR, density=RHO_STAR,
                                body_force=body_force)
    ale_context = build_ale_flow_context(operators, geometry, transport, u_history.states[:stencil.order])
    generator = np.random.default_rng(1505)
    values = np.r_[0.1*generator.standard_normal(operators.n_velocity),
                   0.05*generator.standard_normal(operators.space.mesh.n_nodes)]
    direction = 0.01*generator.standard_normal(displacement.shape)

    def residual_at(candidate_displacement):
        candidate = build_ale_geometry(mesh, candidate_displacement, candidate_id=("jvp-fd",))
        candidate_transport = build_gcl_transport(mesh, candidate, accepted, stencil)
        candidate_operators = assemble_stokes(candidate.current_mesh, viscosity=MU_STAR,
                                              density=RHO_STAR, body_force=body_force)
        candidate_context = build_ale_flow_context(candidate_operators, candidate, candidate_transport,
                                                   u_history.states[:stencil.order])
        return navier_stokes_residual(candidate_operators, values, ale_context=candidate_context)[0]

    return mesh, geometry, operators, ale_context, values, direction, residual_at


@pytest.mark.parametrize("body_force_kind", ["callable", "zero"])
def test_15_05_geometry_jvp_matches_rebuilt_residual_scan(body_force_kind):
    body_force = _jvp_body_force if body_force_kind == "callable" else (
        lambda points: np.zeros((len(points), 2)))
    mesh, geometry, operators, ale_context, values, direction, residual_at = _jvp_configuration(body_force)
    jvp = ale_geometry_jvp(operators, values, ale_context=ale_context, delta_d_m=direction)
    assert np.linalg.norm(jvp) > 0.01*RHO_STAR*U_STAR**2*L_STAR
    errors = []
    for eps in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        plus = residual_at(geometry.displacement+eps*direction)
        minus = residual_at(geometry.displacement-eps*direction)
        finite_difference = (plus-minus)/(2*eps)
        errors.append(np.linalg.norm(finite_difference-jvp)
                      / max(np.linalg.norm(jvp), RHO_STAR*U_STAR**2*L_STAR))
    assert min(errors) <= 1e-5, (errors, body_force_kind)
    assert errors[0] > errors[1] > errors[2], errors
    assert errors[-1] > min(errors), errors


def test_15_05_geometry_jvp_zero_direction_rejections_and_preservation():
    body_force = lambda points: np.zeros((len(points), 2))
    mesh, geometry, operators, ale_context, values, direction, _ = _jvp_configuration(body_force)
    zero = ale_geometry_jvp(operators, values, ale_context=ale_context,
                            delta_d_m=np.zeros_like(direction))
    np.testing.assert_array_equal(zero, np.zeros(operators.n_total))

    with pytest.raises(ValueError, match="shape|finite"):
        ale_geometry_jvp(operators, values, ale_context=ale_context,
                         delta_d_m=np.zeros((mesh.n_nodes, 3)))
    with pytest.raises(ValueError, match="finite"):
        broken = np.array(direction, copy=True)
        broken[1, 0] = np.nan
        ale_geometry_jvp(operators, values, ale_context=ale_context, delta_d_m=broken)
    with pytest.raises(ValueError, match="state"):
        ale_geometry_jvp(operators, values[:-1], ale_context=ale_context, delta_d_m=direction)
    other = build_ale_geometry(mesh, 1.2*geometry.displacement, candidate_id=("jvp-other",))
    other_operators = assemble_stokes(other.current_mesh, viscosity=MU_STAR, density=RHO_STAR,
                                      body_force=body_force)
    with pytest.raises(ValueError, match="candidate|current mesh|geometry"):
        ale_geometry_jvp(other_operators, values, ale_context=ale_context, delta_d_m=direction)

    matrix_before = operators.matrix.data.copy()
    flux_before = ale_context.transport.flux.copy()
    displacement_before = geometry.displacement.copy()
    result = ale_geometry_jvp(operators, values, ale_context=ale_context, delta_d_m=direction)
    assert np.linalg.norm(result) > 0
    np.testing.assert_array_equal(operators.matrix.data, matrix_before)
    np.testing.assert_array_equal(ale_context.transport.flux, flux_before)
    np.testing.assert_array_equal(geometry.displacement, displacement_before)

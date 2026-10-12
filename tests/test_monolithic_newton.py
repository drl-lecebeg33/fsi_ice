"""T22 monolithic Newton integration and steady-march tests (A6).

Small matching two-domain problem (fluid ``[0,1]x[0,1]`` above solid
``[0,1]x[-1,0]``, ``n_total=264``) with a soft-but-damped material
(``E=1e5``, ``rho_s=10``, ``mu_f=5``) that settles to a genuine coupled fixed
point within a few pseudo-time steps.  Covered evidence:

* single-step Newton: declared tolerance, quadratic residual trend, complete
  iteration records, deterministic counters, read-only results;
* steady march: the scaled BDF time term stops the march, two ``dt`` reach the
  same fixed point (per-block scaled difference recorded), the fixed-point
  context (adjacent identical history layers) reduces to the steady discrete
  equations and accepts the marched state with zero Newton iterations;
* failure paths: line-search failure, singular Jacobian, non-finite input and
  residual, iteration cap, march failure with completed steps -- all structured
  with phase/iteration/norms/diagnostics and no pollution of the base state;
* explicit tangent routes: ``assembled`` (splu) default, ``jvp`` (GMRES) with an
  explicit preconditioner matching the assembled solution, and a fast
  structured failure of the unpreconditioned matrix-free route.

All numbers asserted here were measured on the project ``.venv``
(Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1) with
``OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1``; the validation record
``docs/validation/t22.md`` and ``results/m5/t22/`` hold the reproducible tables.
"""

import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, splu

import fsi_ice.coupling.monolithic_newton as monolithic_newton
from fsi_ice.ale import build_ale_geometry, build_harmonic_extension
from fsi_ice.coupling import (MonolithicBase, MonolithicContext, MonolithicNewtonFailure,
                              NewtonFailure, NewtonOptions, PressurePin, build_block_layout,
                              evaluate_monolithic, interface_compatibility_gap, march_to_steady,
                              project_interface_compatibility, scaled_time_terms,
                              solve_monolithic_step)
from fsi_ice.fem.p2 import P2Space
from fsi_ice.fluid.taylor_hood import assemble_stokes
from fsi_ice.interface import build_matching_interface
from fsi_ice.mesh import TriMesh, fsi1_mesh, rectangle_mesh
from fsi_ice.solid.linear import ElasticMaterial, vector_values
from fsi_ice.time import BDFHistory
from fsi_ice.validation.monolithic import block_scale_vector

L_STAR = 1.0
U_STAR = 0.5
P_STAR = 0.05
F_STAR = 0.05
LID_VELOCITY = 0.5
DT = 0.5
STEADY_TOLERANCE = 1e-10
DT_COMPARE_TOLERANCE = 1e-6
JVP_COMPARE_TOLERANCE = 1e-8

STEP_OPTIONS = NewtonOptions(max_iterations=30, rtol=1e-8, atol=1e-10)
TIGHT_OPTIONS = NewtonOptions(max_iterations=40, rtol=1e-10, atol=1e-12)

SCALES = {"u_f": U_STAR, "p_f": P_STAR, "d_s": L_STAR, "v_s": U_STAR, "d_m": L_STAR,
          "g": F_STAR}


# ---------------------------------------------------------------------------
# small two-domain problem (self-contained; mirrors the frozen T21 test builder)
# ---------------------------------------------------------------------------

def _two_domain(nx=4, ny_fluid=3, ny_solid=2, width=1.0):
    fluid_flat = rectangle_mesh(nx, ny_fluid, width=width, height=1.0)
    solid_flat = rectangle_mesh(nx, ny_solid, width=width, height=1.0)

    def tags(mesh, interface_tag):
        result = {name: mesh.boundaries[name] for name in ("bottom", "top", "left", "right")}
        result["interface"] = result.pop(interface_tag)
        return result

    solid = TriMesh(solid_flat.points-np.array([0.0, 1.0]), solid_flat.cells, tags(solid_flat, "top"))
    fluid = TriMesh(fluid_flat.points, fluid_flat.cells, tags(fluid_flat, "bottom"))
    return solid, fluid


def _solid_field(points, amplitude, phase):
    x, y = points[:, 0], points[:, 1]+1.0
    return np.column_stack((amplitude*np.sin(phase)*y*np.sin(np.pi*x),
                            amplitude*np.cos(phase)*y*y*np.sin(2*np.pi*x)))


def _fluid_field(points, amplitude, phase):
    x, y = points[:, 0], points[:, 1]
    return np.column_stack((amplitude*np.cos(phase)*np.sin(np.pi*x)*np.sin(np.pi*y),
                            amplitude*np.sin(phase)*np.sin(np.pi*x)*(1-np.cos(np.pi*y))/2))


def _pressure_field(points, amplitude, phase):
    x, y = points[:, 0], points[:, 1]
    return amplitude*np.sin(phase)*np.cos(np.pi*x)*np.sin(np.pi*y/2)


def _lid_velocity(points):
    values = np.zeros_like(points)
    top = np.isclose(points[:, 1], 1.0, rtol=0, atol=1e-12)
    values[top, 0] = LID_VELOCITY*np.sin(np.pi*points[top, 0])
    return values


def build_problem(*, nx=4, ny_fluid=3, ny_solid=2, dt=DT, pin=PressurePin(0, 0.0),
                  viscosity=5.0, density=1.0, young=1e5, poisson=0.3, solid_density=10.0,
                  candidate_id="t22-candidate", amplitude=0.02):
    """One small monolithic BDF step: accepted history, candidate and context."""
    solid, fluid = _two_domain(nx, ny_fluid, ny_solid)
    fluid_space = P2Space.from_mesh(fluid)
    topology = build_matching_interface(solid, fluid, fluid_space, solid_tag="interface",
                                        fluid_tag="interface", atol=1e-12, rtol=1e-12)
    harmonic = build_harmonic_extension(fluid)
    material = ElasticMaterial(young, poisson, density=solid_density, thickness=1.0)
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

    # One accepted smooth layer (generation 0) is the march initial history.
    solid_history = BDFHistory(np.zeros((2, n_solid, 2)), dt, order=1)
    fluid_history = BDFHistory(np.zeros((len(fluid_space.points), 2)), dt, order=1)
    displacement = _solid_field(solid.points, amplitude, 0.4)
    velocity = _solid_field(solid.points, 0.9*amplitude, 0.1)
    velocity_field = _fluid_field(fluid_space.points, amplitude, 0.2)
    solid_history.accept(solid_history.context(), np.stack((displacement, velocity)))
    fluid_history.accept(fluid_history.context(), velocity_field)
    stencil = solid_history.stencil()
    base = MonolithicBase(reference_mesh=fluid, material=material, fluid_density=density,
                          fluid_viscosity=viscosity,
                          solid_states=((displacement, velocity),),
                          fluid_states=(velocity_field,),
                          accepted_geometries=(
                              build_ale_geometry(fluid, mesh_displacement(displacement),
                                                 candidate_id=("accepted", 0)),),
                          generation=stencil.step_id, fluid_boundary_velocity=_lid_velocity)
    context = MonolithicContext(layout, base, stencil, topology, harmonic, candidate_id=candidate_id,
                                thickness=1.0, pressure_constraint=pin)

    # Candidate warm start: the same construction as the frozen T21 builder
    # (v_s consistent with the accepted layer, g the assembled dual load).
    d_s = _solid_field(solid.points, amplitude, 0.7)
    d_m = mesh_displacement(d_s)
    candidate_geometry = build_ale_geometry(fluid, d_m, candidate_id=candidate_id)
    operators = assemble_stokes(candidate_geometry.current_mesh, viscosity=viscosity, density=density)
    coefficient = float(stencil.coefficients[0])
    history_displacement = np.zeros((n_solid, 2))
    for weight, state in zip(np.asarray(stencil.coefficients)[1:], ((displacement, velocity),)):
        history_displacement += weight*state[0]
    v_s = coefficient*d_s+history_displacement
    u_f = _fluid_field(operators.space.points, amplitude, 0.5)
    u_f[layout.fluid_dirichlet_nodes] = vector_values(
        _lid_velocity, operators.space.points[layout.fluid_dirichlet_nodes], "Lid velocity")
    u_f[topology.fluid_p2_nodes] = (topology.H @ v_s.ravel()[topology.solid_dofs]).reshape(-1, 2)
    p_f = _pressure_field(fluid.points, 0.4*P_STAR, 0.9)
    if pin is not None:
        p_f[pin.node] = pin.value
    x = layout.pack({"u_f": u_f, "p_f": p_f, "d_s": d_s, "v_s": v_s, "d_m": d_m,
                     "g": np.zeros((topology.n_fluid_interface_nodes, 2))})
    x[layout.slice_of("g")] = -evaluate_monolithic(x, context=context).residual[layout.slice_of("g")]
    scales = block_scale_vector(layout, SCALES)
    return SimpleNamespace(solid=solid, fluid=fluid, fluid_space=fluid_space, topology=topology,
                           harmonic=harmonic, material=material, layout=layout, base=base,
                           stencil=stencil, context=context, x=x, scales=scales)


def fixed_point_context(problem, x, *, dt=DT, order=2):
    """Context whose two accepted layers both equal ``x`` (steady reduction).

    With identical adjacent layers the BDF time term cancels
    (``a_0 + a_1 + a_2 = 0``), so the step residual equals the steady discrete
    residual and a fixed point must be accepted with zero Newton iterations.
    """
    layout = problem.layout
    blocks = layout.unpack(x)
    solid_history = BDFHistory(np.stack((np.asarray(blocks["d_s"]),
                                         np.asarray(blocks["v_s"]))), dt, order=order)
    solid_history.accept(solid_history.context(),
                         np.stack((np.asarray(blocks["d_s"]), np.asarray(blocks["v_s"]))))
    fluid_history = BDFHistory(np.asarray(blocks["u_f"]), dt, order=order)
    fluid_history.accept(fluid_history.context(), np.asarray(blocks["u_f"]))
    stencil = solid_history.stencil()
    geometry = build_ale_geometry(problem.fluid, np.asarray(blocks["d_m"]),
                                  quality_limits=problem.base.quality_limits,
                                  candidate_id=("t22-fixed-point-geometry", 0))
    base = MonolithicBase(
        reference_mesh=problem.fluid, material=problem.material,
        fluid_density=problem.base.fluid_density,
        fluid_viscosity=problem.base.fluid_viscosity,
        solid_states=tuple((state[0], state[1]) for state in solid_history.states[:stencil.order]),
        fluid_states=tuple(fluid_history.states[:stencil.order]),
        accepted_geometries=tuple([geometry]*stencil.order), generation=stencil.step_id,
        fluid_boundary_velocity=problem.base.fluid_boundary_velocity)
    return MonolithicContext(layout, base, stencil, problem.topology, problem.harmonic,
                             candidate_id=("t22-fixed-point", 0),
                             thickness=problem.context.thickness,
                             pressure_constraint=problem.context.pressure_constraint)


def snapshot_state(problem):
    """Deep copies of everything the solver must leave untouched."""
    base = problem.base
    return {
        "x": np.array(problem.x, copy=True),
        "solid": [(np.array(d, copy=True), np.array(v, copy=True)) for d, v in base.solid_states],
        "fluid": [np.array(u, copy=True) for u in base.fluid_states],
        "points": np.array(problem.fluid.points, copy=True),
        "H": np.array(problem.topology.h.toarray(), copy=True),
    }


def assert_state_unchanged(problem, snapshot):
    base = problem.base
    np.testing.assert_array_equal(problem.x, snapshot["x"])
    for old, (d, v) in zip(snapshot["solid"], base.solid_states):
        np.testing.assert_array_equal(old[0], d)
        np.testing.assert_array_equal(old[1], v)
    for old, u in zip(snapshot["fluid"], base.fluid_states):
        np.testing.assert_array_equal(old, u)
    np.testing.assert_array_equal(problem.fluid.points, snapshot["points"])
    np.testing.assert_array_equal(problem.topology.h.toarray(), snapshot["H"])


def marched(problem, **kwargs):
    arguments = dict(dt=DT, order=2, max_steps=60, steady_tolerance=STEADY_TOLERANCE,
                     options=STEP_OPTIONS, scales=problem.scales, jacobian_mode="assembled")
    arguments.update(kwargs)
    return march_to_steady(problem.context, problem.x, **arguments)


@pytest.fixture(scope="module")
def problem():
    return build_problem()


# ---------------------------------------------------------------------------
# single-step Newton
# ---------------------------------------------------------------------------

def test_step_converges_to_declared_tolerance_and_records(problem):
    result = solve_monolithic_step(problem.context, problem.x, options=STEP_OPTIONS,
                                   scales=problem.scales, jacobian_mode="assembled")
    assert result.success is True and result.reason == "converged"
    assert result.residual_norm <= result.threshold
    assert result.threshold == result.records[-1].threshold == result.records[0].threshold
    assert result.records[0].iteration == 0 and result.records[0].damping is None
    assert result.records[0].step_norm is None
    assert result.newton_iterations == len(result.records)-1 == result.counters.newton_iterations
    assert result.residual_history == tuple(row.residual_norm for row in result.records)
    assert result.raw_residual_history == tuple(row.raw_residual_norm for row in result.records)
    assert all(len(row.trial_residual_norms) == row.trial_count for row in result.records[1:])
    # every accepted step in this warm start uses the full undamped Newton step
    assert all(row.damping == 1.0 for row in result.records[1:])
    # declared convergence: the accepted residual passes the recorded threshold
    assert max(result.residual_history) == result.residual_history[0]
    assert result.interface_gap <= 1e-12
    assert set(result.block_norms) == {"u_f", "p_f", "d_s", "v_s", "d_m", "g"}
    assert all(np.isfinite(value) for value in result.block_norms.values())
    # counters: assembled route, one diagnostic residual evaluation included
    counters = result.counters
    assert counters.steps == 1 and counters.failed_steps == 0
    assert counters.evaluations == (counters.residual_evaluations
                                    + counters.assembled_evaluations
                                    + counters.jvp_evaluations)
    assert counters.assembled_evaluations == counters.linear_solves == counters.jacobian_reads
    assert counters.jvp_evaluations == 0 and counters.linear_iterations == 0
    assert counters.line_search_trials >= counters.newton_iterations
    assert counters.failed_trials == counters.line_search_trials-counters.newton_iterations
    assert result.linear_solver == "splu" and result.uses_krylov is False
    # timing buckets are nonnegative and monotone in the nested order
    timings = dict(result.timings)
    assert set(timings) == {"evaluate_seconds", "linear_seconds", "newton_seconds", "total_seconds"}
    assert timings["total_seconds"] >= timings["newton_seconds"] >= 0.0
    assert timings["evaluate_seconds"] > 0.0


def test_step_quadratic_convergence_trend(problem):
    result = solve_monolithic_step(problem.context, problem.x, options=TIGHT_OPTIONS,
                                   scales=problem.scales, jacobian_mode="assembled")
    assert result.reason == "converged"
    norms = result.raw_residual_history
    assert all(later < earlier for earlier, later in zip(norms, norms[1:]))
    # quadratic tail: r_{k+1} <= 10 * r_k^2 for the last two accepted steps
    for index in range(len(norms)-3, len(norms)-1):
        assert norms[index+1] <= 10.0*norms[index]**2
    assert norms[-1] <= norms[0]*1e-5


def test_step_rejects_unknown_route_and_misconfigured_preconditioner(problem):
    with pytest.raises(ValueError, match="jacobian_mode"):
        solve_monolithic_step(problem.context, problem.x, scales=problem.scales,
                              jacobian_mode="full")
    with pytest.raises(ValueError, match="preconditioner is only used"):
        solve_monolithic_step(problem.context, problem.x, scales=problem.scales,
                              jacobian_mode="assembled", preconditioner=lambda x: None)
    with pytest.raises(TypeError, match="preconditioner"):
        solve_monolithic_step(problem.context, problem.x, scales=problem.scales,
                              jacobian_mode="jvp", preconditioner=42)


# ---------------------------------------------------------------------------
# steady march
# ---------------------------------------------------------------------------

def test_march_reaches_steady_state(problem):
    snapshot = snapshot_state(problem)
    result = marched(problem)
    assert result.steady_reached is True and result.success is True and result.reason == "steady"
    assert result.final_time_term <= STEADY_TOLERANCE
    assert set(result.final_time_terms) == {"u_f", "d_s", "v_s"}
    assert len(result.steps) >= 2
    for record in result.steps:
        assert record.residual_norm <= record.threshold
        assert record.counters.steps == 1 and record.counters.failed_steps == 0
        assert np.all(np.isfinite(record.x))
    assert result.steps[-1].time_term == result.final_time_term
    # generations increase by one per accepted step; BDF2 after the BDF1 startup
    assert [record.generation for record in result.steps] == list(range(len(result.steps)))
    assert [record.order for record in result.steps[:2]] == [1, 2]
    # counters aggregate the steps
    assert result.counters.steps == len(result.steps)
    assert result.counters.failed_steps == 0
    assert (result.counters.newton_iterations
            == sum(record.counters.newton_iterations for record in result.steps))
    assert result.counters.evaluations == sum(
        record.counters.evaluations for record in result.steps)
    assert dict(result.timings)["total_seconds"] > 0.0
    assert result.linear_solver == "splu" and result.uses_krylov is False
    assert_state_unchanged(problem, snapshot)


def test_march_fixed_point_solves_the_steady_equations(problem):
    march = marched(problem)
    x = march.x
    context = fixed_point_context(problem, x)
    # time terms of the fixed-point stencil at x cancel to machine zero
    solid_states, fluid_states = context.base.solid_states, context.base.fluid_states
    terms = scaled_time_terms(x, context=context, solid_states=solid_states,
                              fluid_states=fluid_states, scales=problem.scales)
    assert max(terms.values()) <= 1e-13
    # the steady residual at the marched state passes the declared threshold
    result = solve_monolithic_step(context, x, options=STEP_OPTIONS, scales=problem.scales,
                                   jacobian_mode="assembled")
    assert result.reason == "converged"
    assert result.residual_norm <= result.threshold
    assert result.residual_norm <= 1e-9
    # x is a fixed point: zero Newton iterations and zero correction
    assert result.newton_iterations == 0
    np.testing.assert_array_equal(result.x, x)
    # ... and the fixed-point step residual equals the steady residual exactly
    # (identical adjacent layers cancel the BDF time term)
    np.testing.assert_allclose(result.residual, evaluate_monolithic(x, context=context).residual,
                               rtol=0, atol=0)


def test_march_dt_independence(problem):
    coarse = marched(problem, dt=0.5)
    fine = marched(problem, dt=0.25)
    assert coarse.steady_reached and fine.steady_reached
    assert max(coarse.final_time_terms.values()) <= STEADY_TOLERANCE
    assert max(fine.final_time_terms.values()) <= STEADY_TOLERANCE
    blocks_coarse = problem.layout.unpack(coarse.x)
    blocks_fine = problem.layout.unpack(fine.x)
    differences = {}
    for name in ("u_f", "p_f", "d_s", "v_s", "d_m", "g"):
        values = (np.asarray(blocks_coarse[name]).ravel()-np.asarray(blocks_fine[name]).ravel())
        scale = problem.scales[problem.layout.slice_of(name)]
        differences[name] = float(np.max(np.abs(values/scale)))
    assert max(differences.values()) <= DT_COMPARE_TOLERANCE, differences
    # displacement and mesh displacement agree orders of magnitude tighter
    assert differences["d_s"] <= 1e-10
    assert differences["d_m"] <= 1e-10
    # both marches converge to the same steady residual level in the shared
    # fixed-point context
    context = fixed_point_context(problem, coarse.x)
    check = solve_monolithic_step(context, fine.x, options=STEP_OPTIONS,
                                  scales=problem.scales, jacobian_mode="assembled")
    assert check.reason == "converged"


def test_march_max_steps_termination(problem):
    result = marched(problem, max_steps=2, steady_tolerance=1e-14)
    assert result.steady_reached is False and result.success is False
    assert result.reason == "max_steps"
    assert len(result.steps) == 2 and result.counters.steps == 2
    assert result.final_time_term > 1e-14
    assert np.all(np.isfinite(result.x))
    # a stricter run may still report the last computed time term on the result
    assert set(result.final_time_terms) == {"u_f", "d_s", "v_s"}


def test_march_is_deterministic_and_read_only(problem):
    first = marched(problem)
    second = marched(problem)
    np.testing.assert_array_equal(first.x, second.x)
    assert first.counters == second.counters
    assert tuple(first.final_time_terms.items()) == tuple(second.final_time_terms.items())
    assert first.steps[-1].time_term == second.steps[-1].time_term
    assert not first.x.flags.writeable and not first.scales.flags.writeable
    with pytest.raises(ValueError):
        first.x[0] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.reason = "changed"


def test_results_are_read_only_and_inputs_isolated(problem):
    x0 = np.array(problem.x, copy=True)
    result = solve_monolithic_step(problem.context, x0, options=STEP_OPTIONS,
                                   scales=problem.scales, jacobian_mode="assembled")
    assert not result.x.flags.writeable and not result.residual.flags.writeable
    assert not result.scales.flags.writeable
    with pytest.raises(ValueError):
        result.x[0] = 1.0
    with pytest.raises(ValueError):
        result.residual[0] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.reason = "changed"
    assert isinstance(result.records, tuple) and isinstance(result.residual_history, tuple)
    x0[...] = -7.0
    assert not np.any(result.x == -7.0)


# ---------------------------------------------------------------------------
# failure paths
# ---------------------------------------------------------------------------

def _poison_residual_trials(original, *, keep_x, active):
    """Residual-mode evaluations away from ``keep_x`` return NaN (rejected trials)."""
    def wrapper(x, *, context, mode, directions=None):
        result = original(x, context=context, mode=mode, directions=directions)
        if active() and mode == "residual" and not np.array_equal(np.asarray(x), keep_x):
            return SimpleNamespace(residual=np.full_like(result.residual, np.nan), jacobian=None)
        return result
    return wrapper


def test_line_search_failure_diagnostics_and_no_pollution(problem, monkeypatch):
    snapshot = snapshot_state(problem)
    baseline = solve_monolithic_step(problem.context, problem.x, options=STEP_OPTIONS,
                                     scales=problem.scales, jacobian_mode="assembled")
    original = monolithic_newton.evaluate_monolithic
    monkeypatch.setattr(monolithic_newton, "evaluate_monolithic",
                        _poison_residual_trials(original, keep_x=np.array(problem.x, copy=True),
                                                active=lambda: True))
    options = NewtonOptions(max_iterations=5, rtol=1e-8, atol=1e-10, max_backtracks=4)
    with pytest.raises(MonolithicNewtonFailure) as info:
        solve_monolithic_step(problem.context, problem.x, options=options,
                              scales=problem.scales, jacobian_mode="assembled")
    failure = info.value
    assert failure.phase == "line_search" and failure.iteration == 1
    assert failure.reason.startswith("backtracking failed")
    assert failure.residual_norm == pytest.approx(baseline.records[0].residual_norm)
    assert np.isfinite(failure.raw_residual_norm)
    trials = failure.diagnostics["trial_residual_norms"]
    assert len(trials) == options.max_backtracks+1 and all(value == np.inf for value in trials)
    assert failure.counters.failed_steps == 1
    assert failure.counters.failed_trials == options.max_backtracks+1
    assert failure.counters.linear_solves == 1
    assert failure.jacobian_mode == "assembled"
    assert failure.candidate_id == problem.context.candidate_id
    assert failure.step_index is None and failure.march_steps == ()
    assert dict(failure.timings)["total_seconds"] >= 0.0
    np.testing.assert_array_equal(failure.x, problem.x)
    assert_state_unchanged(problem, snapshot)
    monkeypatch.undo()
    again = solve_monolithic_step(problem.context, problem.x, options=STEP_OPTIONS,
                                  scales=problem.scales, jacobian_mode="assembled")
    np.testing.assert_array_equal(again.x, baseline.x)
    assert again.counters == baseline.counters


def test_singular_jacobian_injection_fails_in_linear_solve(problem, monkeypatch):
    original = monolithic_newton.evaluate_monolithic

    def wrapper(x, *, context, mode, directions=None):
        result = original(x, context=context, mode=mode, directions=directions)
        if mode == "assembled":
            matrix = result.jacobian.tocsr(copy=True)
            matrix.data[matrix.indptr[0]:matrix.indptr[1]] = 0.0
            return SimpleNamespace(residual=result.residual, jacobian=matrix)
        return result

    monkeypatch.setattr(monolithic_newton, "evaluate_monolithic", wrapper)
    with pytest.raises(MonolithicNewtonFailure) as info:
        solve_monolithic_step(problem.context, problem.x, options=STEP_OPTIONS,
                              scales=problem.scales, jacobian_mode="assembled")
    failure = info.value
    assert failure.phase == "linear_solve" and failure.iteration == 1
    assert failure.diagnostics["solver"] == "splu"
    assert "singular" in failure.diagnostics["error"].lower()
    assert failure.counters.failed_steps == 1 and failure.counters.linear_solves == 1
    assert failure.counters.linear_iterations == 0


def test_non_finite_input_and_residual_fail_structurally(problem, monkeypatch):
    # (a) non-finite starting point is rejected before any Newton step
    bad = np.array(problem.x, copy=True)
    bad[problem.layout.slice_of("p_f")] = np.nan
    with pytest.raises(MonolithicNewtonFailure) as info:
        solve_monolithic_step(problem.context, bad, options=STEP_OPTIONS,
                              scales=problem.scales, jacobian_mode="assembled")
    failure = info.value
    assert failure.phase == "evaluate" and failure.iteration == 0
    assert failure.diagnostics["contract"] == "x0"
    assert failure.diagnostics["non_finite_entries"] == problem.layout.size_of("p_f")
    assert failure.counters.failed_steps == 1 and failure.counters.evaluations == 0

    # (b) a non-finite residual at the starting point fails in the residual phase
    original = monolithic_newton.evaluate_monolithic

    def wrapper(x, *, context, mode, directions=None):
        result = original(x, context=context, mode=mode, directions=directions)
        if mode == "residual":
            poisoned = np.array(result.residual, copy=True)
            poisoned[0] = np.nan
            return SimpleNamespace(residual=poisoned, jacobian=result.jacobian)
        return result

    monkeypatch.setattr(monolithic_newton, "evaluate_monolithic", wrapper)
    with pytest.raises(MonolithicNewtonFailure) as info:
        solve_monolithic_step(problem.context, problem.x, options=STEP_OPTIONS,
                              scales=problem.scales, jacobian_mode="assembled")
    failure = info.value
    assert failure.phase == "residual" and failure.iteration == 0
    assert failure.residual_norm == np.inf
    assert failure.diagnostics["non_finite_entries"] == 1
    assert failure.history == () and failure.counters.evaluations == 1


def test_iteration_cap_failure(problem):
    options = NewtonOptions(max_iterations=1, rtol=1e-12, atol=0.0)
    with pytest.raises(MonolithicNewtonFailure) as info:
        solve_monolithic_step(problem.context, problem.x, options=options,
                              scales=problem.scales, jacobian_mode="assembled")
    failure = info.value
    assert failure.phase == "convergence" and failure.iteration == 1
    assert "maximum iterations" in failure.reason
    assert failure.diagnostics["max_iterations"] == 1
    assert failure.counters.newton_iterations == 1
    assert len(failure.history) == 2  # start record plus the single iteration
    assert np.isfinite(failure.residual_norm) and np.isfinite(failure.raw_residual_norm)


def test_march_failure_reports_completed_steps(problem, monkeypatch):
    original = monolithic_newton.evaluate_monolithic
    starts = {}

    def wrapper(x, *, context, mode, directions=None):
        result = original(x, context=context, mode=mode, directions=directions)
        step = context.candidate_id[2]
        if mode == "residual":
            starts.setdefault(context.candidate_id, np.array(x, copy=True))
            if step >= 2 and not np.array_equal(np.asarray(x), starts[context.candidate_id]):
                return SimpleNamespace(residual=np.full_like(result.residual, np.nan),
                                       jacobian=None)
        return result

    monkeypatch.setattr(monolithic_newton, "evaluate_monolithic", wrapper)
    with pytest.raises(MonolithicNewtonFailure) as info:
        marched(problem, max_steps=5)
    failure = info.value
    assert failure.phase == "line_search"
    assert failure.step_index == 2
    assert len(failure.march_steps) == 2
    assert failure.counters.steps == 2 and failure.counters.failed_steps == 1
    assert failure.counters.newton_iterations >= len(failure.march_steps)
    assert failure.jacobian_mode == "assembled"
    # the failing step starts from the second completed step's solution
    np.testing.assert_array_equal(failure.x, failure.march_steps[1].x)
    assert "step_index" in failure.diagnostics and failure.diagnostics["step_index"] == 2
    assert failure.diagnostics["counters"]["failed_steps"] == 1
    assert failure.diagnostics["counters"]["steps"] == 2


# ---------------------------------------------------------------------------
# interface compatibility policy
# ---------------------------------------------------------------------------

def test_interface_compatibility_require_and_project(problem):
    infeasible = np.array(problem.x, copy=True)
    trace = problem.layout.slice_of("d_m")
    values = infeasible[trace].reshape(problem.layout.shape_of("d_m"))
    values.ravel()[problem.layout.mesh_interface_dofs] += 1e-3
    gap = interface_compatibility_gap(infeasible, context=problem.context)
    assert gap == pytest.approx(1e-3)
    with pytest.raises(MonolithicNewtonFailure) as info:
        solve_monolithic_step(problem.context, infeasible, options=STEP_OPTIONS,
                              scales=problem.scales, jacobian_mode="assembled")
    failure = info.value
    assert failure.phase == "residual" and failure.iteration == 0
    assert failure.diagnostics["requirement"] == "d_m|Gamma = H_d d_s|Gamma"
    assert failure.diagnostics["policy"] == "require"
    assert failure.diagnostics["interface_gap"] == pytest.approx(1e-3)
    assert failure.counters.evaluations == 0

    projected = project_interface_compatibility(infeasible, context=problem.context)
    assert interface_compatibility_gap(projected, context=problem.context) == 0.0
    assert interface_compatibility_gap(infeasible, context=problem.context) == pytest.approx(1e-3)
    assert not projected.flags.writeable
    result = solve_monolithic_step(problem.context, infeasible, options=STEP_OPTIONS,
                                   scales=problem.scales, jacobian_mode="assembled",
                                   compatibility="project")
    assert result.reason == "converged"
    assert result.compatibility_policy == "project"
    assert result.compatibility_gap == pytest.approx(1e-3)
    assert result.interface_gap <= 1e-12


# ---------------------------------------------------------------------------
# explicit tangent routes
# ---------------------------------------------------------------------------

def _exact_preconditioner(context, counter):
    def factory(x):
        counter.append(1)
        matrix = evaluate_monolithic(x, context=context, mode="assembled").jacobian
        factor = splu(matrix.tocsc())
        return LinearOperator(matrix.shape, matvec=lambda q: factor.solve(q), dtype=float)
    return factory


def test_jvp_mode_with_explicit_preconditioner_matches_assembled(problem):
    assembled = solve_monolithic_step(problem.context, problem.x, options=STEP_OPTIONS,
                                      scales=problem.scales, jacobian_mode="assembled")
    builds = []
    options = NewtonOptions(max_iterations=30, rtol=1e-8, atol=1e-10,
                            linear_solver="gmres", gmres_restart=100,
                            gmres_rtol=1e-10, gmres_atol=1e-12)
    krylov = solve_monolithic_step(problem.context, problem.x, options=options,
                                   scales=problem.scales, jacobian_mode="jvp",
                                   preconditioner=_exact_preconditioner(problem.context, builds))
    assert krylov.reason == "converged"
    assert krylov.uses_krylov is True and krylov.linear_solver == "gmres"
    assert krylov.newton_iterations == assembled.newton_iterations
    counters = krylov.counters
    assert counters.assembled_evaluations == 0  # no silent assembly on the jvp route
    assert counters.jvp_evaluations > 0 and counters.jvp_directions == counters.jvp_evaluations
    assert counters.linear_iterations >= counters.linear_solves >= 1
    assert builds and len(builds) == counters.linear_solves
    difference = float(np.max(np.abs(np.asarray(krylov.x)-np.asarray(assembled.x))
                              / problem.scales))
    assert difference <= JVP_COMPARE_TOLERANCE
    assert assembled.counters.assembled_evaluations > 0


def test_jvp_mode_without_preconditioner_fails_without_fallback(problem):
    options = NewtonOptions(max_iterations=2, rtol=1e-8, atol=1e-10,
                            linear_solver="gmres", gmres_restart=50, gmres_maxiter=1,
                            gmres_rtol=1e-12, gmres_atol=1e-14)
    snapshot = snapshot_state(problem)
    with pytest.raises(MonolithicNewtonFailure) as info:
        solve_monolithic_step(problem.context, problem.x, options=options,
                              scales=problem.scales, jacobian_mode="jvp")
    failure = info.value
    assert failure.phase == "linear_solve" and failure.iteration == 1
    assert failure.diagnostics["solver"] == "gmres"
    assert failure.diagnostics["info"] >= 1
    assert failure.diagnostics["iterations"] >= 1
    assert failure.counters.linear_iterations >= 1
    assert failure.counters.assembled_evaluations == 0
    assert failure.counters.jvp_evaluations >= 1
    assert failure.counters.failed_steps == 1
    assert_state_unchanged(problem, snapshot)


def test_march_passes_the_explicit_jvp_route_through(problem):
    builds = []
    options = NewtonOptions(max_iterations=30, rtol=1e-8, atol=1e-10,
                            linear_solver="gmres", gmres_restart=100,
                            gmres_rtol=1e-10, gmres_atol=1e-12)
    result = marched(problem, max_steps=1, steady_tolerance=1e-14, options=options,
                     jacobian_mode="jvp",
                     preconditioner=_exact_preconditioner(problem.context, builds))
    assert result.reason == "max_steps" and len(result.steps) == 1
    assert result.linear_solver == "gmres" and result.uses_krylov is True
    counters = result.counters
    assert counters.jvp_evaluations > 0 and counters.jvp_directions == counters.jvp_evaluations
    assert counters.assembled_evaluations == 0  # the adapter never assembles on the jvp route
    assert counters.linear_iterations >= counters.linear_solves >= 1
    assert builds


def test_step_counters_are_deterministic(problem):
    first = solve_monolithic_step(problem.context, problem.x, options=STEP_OPTIONS,
                                  scales=problem.scales, jacobian_mode="assembled")
    second = solve_monolithic_step(problem.context, problem.x, options=STEP_OPTIONS,
                                   scales=problem.scales, jacobian_mode="assembled")
    assert first.counters == second.counters
    np.testing.assert_array_equal(first.x, second.x)
    np.testing.assert_array_equal(first.residual, second.residual)
    assert first.residual_history == second.residual_history
    assert dict(first.timings).keys() == dict(second.timings).keys()
    assert first.threshold == second.threshold


# ---------------------------------------------------------------------------
# optional real FSI1 h=0.02 smoke (gated; not part of the default gate)
# ---------------------------------------------------------------------------
# Measured on the project .venv with one BLAS thread: a single assembled step on
# the h=0.02 mesh (n_total=12905) converges in 4 Newton iterations in ~78 s
# (4 assembled Jacobians ~18 s each); a two-step BDF1 march takes roughly two to
# three minutes.  Run with FSI_ICE_T22_SMOKE=1; the default test run skips it and
# the measured budgets live in docs/validation/t22.md section 9.

FSI1_SMOKE = os.environ.get("FSI_ICE_T22_SMOKE") == "1"


@pytest.mark.skipif(not FSI1_SMOKE,
                    reason="optional real FSI1 h=0.02 march smoke; set FSI_ICE_T22_SMOKE=1 "
                           "(measured: single step ~78 s, two-step march ~3 min)")
def test_real_fsi1_h002_march_smoke(tmp_path):
    from fsi_ice.validation.fluid import inlet_profile

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
    scales = block_scale_vector(layout, {"u_f": 0.2, "p_f": 40.0, "d_s": 0.41, "v_s": 0.2,
                                         "d_m": 0.41, "g": 14.29426})
    dt = 0.1
    zero_solid = np.zeros((len(solid.points), 2))
    zero_fluid = np.zeros((len(fluid_space.points), 2))
    zero_mesh = np.zeros_like(fluid.points)
    solid_history = BDFHistory(np.zeros((2, len(solid.points), 2)), dt, order=1)
    solid_history.accept(solid_history.context(), np.zeros((2, len(solid.points), 2)))
    stencil = solid_history.stencil()
    base = MonolithicBase(reference_mesh=fluid, material=material, fluid_density=1000.0,
                          fluid_viscosity=1.0, solid_states=((zero_solid, zero_solid),),
                          fluid_states=(zero_fluid,),
                          accepted_geometries=(build_ale_geometry(
                              fluid, zero_mesh, candidate_id=("fsi1-smoke", 0)),),
                          generation=stencil.step_id,
                          fluid_boundary_velocity=lambda points: inlet_profile(points, 0.2))
    context = MonolithicContext(layout, base, stencil, topology, harmonic,
                                candidate_id=("fsi1-t22-smoke", mesh_size), thickness=1.0,
                                pressure_constraint=None)
    x0 = np.zeros(layout.n_total)
    velocity = np.zeros((len(fluid_space.points), 2))
    velocity[layout.fluid_dirichlet_nodes] = inlet_profile(
        fluid_space.points[layout.fluid_dirichlet_nodes], 0.2)
    x0[layout.slice_of("u_f")] = velocity.ravel()

    options = NewtonOptions(max_iterations=25, rtol=1e-6, atol=1e-8)
    result = march_to_steady(context, x0, dt=dt, order=1, max_steps=2, steady_tolerance=1e-8,
                             options=options, scales=scales, jacobian_mode="assembled")
    assert result.reason == "max_steps" and result.steady_reached is False
    assert len(result.steps) == 2 and result.counters.steps == 2
    assert result.counters.failed_steps == 0
    assert all(record.residual_norm <= record.threshold for record in result.steps)
    assert result.steps[-1].time_term < result.steps[0].time_term
    assert all(record.interface_gap < 1e-12 for record in result.steps)
    assert result.counters.assembled_evaluations == result.counters.linear_solves
    record = {
        "mesh_size": mesh_size, "n_total": int(layout.n_total), "sizes": list(layout.sizes),
        "dt": dt, "steady_tolerance": 1e-8,
        "options": {"rtol": options.rtol, "atol": options.atol,
                    "max_iterations": options.max_iterations},
        "reason": result.reason, "steady_reached": bool(result.steady_reached),
        "steps": [{"index": item.index, "generation": item.generation, "order": item.order,
                   "newton_iterations": item.newton_iterations,
                   "residual_norm": float(item.residual_norm), "threshold": float(item.threshold),
                   "time_term": float(item.time_term),
                   "time_terms": {key: float(value) for key, value in dict(item.time_terms).items()},
                   "interface_gap": float(item.interface_gap),
                   "counters": item.counters.as_dict(),
                   "timings": {key: float(value) for key, value in dict(item.timings).items()}}
                  for item in result.steps],
        "counters": result.counters.as_dict(),
        "timings": {key: float(value) for key, value in dict(result.timings).items()},
    }
    output = Path("results/m5/t22")
    output.mkdir(parents=True, exist_ok=True)
    (output/"fsi1_smoke.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n",
                                          encoding="utf-8")

"""Independent review of the frozen M5/T21 monolithic deliverables (A7).

The problem construction (two-domain mesh, accepted BDF history, candidate
state and context) is reused read-only from ``tests/test_monolithic_residual.py``;
the directions, perturbation scans (``eps = 1e-2 ... 1e-7``), tolerances and
assertions below are designed independently by the reviewer.  Nothing under
``src/`` is modified, no implementer test is duplicated: the scans call
``monolithic_residual`` / ``monolithic_jvp`` / ``monolithic_jacobian`` directly
and the central differences are implemented here.

Frozen gates: best non-degenerate relative directional error <= ``1e-5``
(``spec.md`` section 5.2, ``m3_freeze.md`` section 6) and algebraic identities
<= ``1e-12 * S`` (m3 convention).  Reviewer seeds: ``20261019`` (mixed and
per-block random directions), ``20261021`` (interface chain).  Evidence of the
implementer (``results/m5/t21/scan_evidence.json``, ``docs/validation/t21.md``,
``results/m5/t21/fsi1_smoke.json``) is sampled in the last test.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fsi_ice.ale import build_ale_geometry, build_gcl_transport
from fsi_ice.coupling import (BLOCK_NAMES, PressurePin, build_block_layout, monolithic_jacobian,
                              monolithic_jvp, monolithic_residual, require_candidate)
from fsi_ice.fluid.ale import ale_navier_stokes_residual, build_ale_flow_context
from fsi_ice.fluid.taylor_hood import assemble_stokes
from fsi_ice.interface import assemble_interface_load, evaluate_interface_geometry
from fsi_ice.time import BDFHistory

from test_monolithic_residual import build_problem

REVIEW_SEED = 20261019
L_STAR = 1.0
U_STAR = 0.05
P_STAR = 0.01
F_STAR = 0.01
GATE = 1e-5
IDENTITY_LIMIT = 1e-12
JACOBIAN_JVP_GAP_LIMIT = 1e-12
ZERO_RESPONSE_LIMIT = 1e-9
STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
VARIABLE_SCALES = {"u_f": U_STAR, "p_f": P_STAR, "d_s": L_STAR, "v_s": U_STAR,
                   "d_m": L_STAR, "g": F_STAR}
RESIDUAL_SCALES = {"u_f": P_STAR, "p_f": U_STAR*L_STAR, "d_s": U_STAR, "v_s": P_STAR,
                   "d_m": L_STAR, "g": F_STAR}


# ---------------------------------------------------------------------------
# reviewer-side helpers (independent code, frozen problem from the reused helper)
# ---------------------------------------------------------------------------

def _scales(problem, per_block):
    vector = np.empty(problem.layout.n_total)
    for name in BLOCK_NAMES:
        vector[problem.layout.slice_of(name)] = per_block[name]
    return vector


def _sync_interface(problem, direction):
    """Copy the solid trace into the mesh trace (``d_m|Gamma = H_d d_s|Gamma``)."""
    layout = problem.layout
    topology = problem.topology
    blocks = layout.unpack(direction)
    d_s = np.array(blocks["d_s"], copy=True)
    d_m = np.array(blocks["d_m"], copy=True)
    d_m.ravel()[layout.mesh_interface_dofs] = d_s.ravel()[topology.solid_dofs]
    result = np.array(direction, copy=True)
    result[layout.slice_of("d_s")] = d_s.ravel()
    result[layout.slice_of("d_m")] = d_m.ravel()
    return result


def _smooth_direction(problem):
    """Deterministic smooth direction with a nonzero component in every block."""
    layout = problem.layout
    direction = np.zeros(layout.n_total)
    cx, cy = problem.fluid_space.points[:, 0], problem.fluid_space.points[:, 1]
    direction[layout.slice_of("u_f")] = np.column_stack((
        0.4*U_STAR*np.sin(np.pi*cx)*np.sin(2.0*np.pi*cy),
        0.4*U_STAR*cx*(1.0-cx)*(1.0-np.cos(np.pi*cy)))).ravel()
    px, py = problem.fluid.points[:, 0], problem.fluid.points[:, 1]
    direction[layout.slice_of("p_f")] = 0.6*P_STAR*np.sin(np.pi*px)*np.cos(np.pi*py)
    sx, sy = problem.solid.points[:, 0], problem.solid.points[:, 1]+1.0
    profile = 0.1*L_STAR*sy
    direction[layout.slice_of("d_s")] = np.column_stack((
        profile*np.sin(np.pi*sx), profile*np.cos(2.0*np.pi*sx))).ravel()
    direction[layout.slice_of("v_s")] = np.column_stack((
        0.3*U_STAR*sy*np.cos(np.pi*sx), -0.3*U_STAR*sy*np.sin(np.pi*sx))).ravel()
    direction[layout.slice_of("d_m")] = np.column_stack((
        0.1*L_STAR*np.sin(np.pi*py)*np.cos(np.pi*px),
        -0.1*L_STAR*py*np.sin(np.pi*px))).ravel()
    interface = np.linspace(0.0, 1.0, problem.topology.n_fluid_interface_nodes)
    direction[layout.slice_of("g")] = np.column_stack((
        0.5*F_STAR*np.sin(2.0*np.pi*interface),
        -0.5*F_STAR*np.cos(np.pi*interface))).ravel()
    return _sync_interface(problem, direction)


def _random_direction(problem, seed):
    """Deterministic random direction in every block, rescaled by the frozen scales."""
    layout = problem.layout
    rng = np.random.default_rng(seed)
    direction = np.zeros(layout.n_total)
    for name in BLOCK_NAMES:
        direction[layout.slice_of(name)] = (rng.normal(size=layout.size_of(name))
                                            * VARIABLE_SCALES[name])
    return _sync_interface(problem, direction)


def _interface_direction(problem, seed):
    """Interface chain only: ``d_s``/``v_s`` on the wet trace plus ``g``."""
    layout = problem.layout
    topology = problem.topology
    rng = np.random.default_rng(seed)
    direction = np.zeros(layout.n_total)
    displacement = np.zeros((problem.solid.n_nodes, 2))
    displacement.ravel()[topology.solid_dofs] = rng.normal(size=len(topology.solid_dofs))*L_STAR
    velocity = np.zeros((problem.solid.n_nodes, 2))
    velocity.ravel()[topology.solid_dofs] = rng.normal(size=len(topology.solid_dofs))*U_STAR
    direction[layout.slice_of("d_s")] = displacement.ravel()
    direction[layout.slice_of("v_s")] = velocity.ravel()
    direction[layout.slice_of("g")] = (rng.normal(size=(topology.n_fluid_interface_nodes, 2))*F_STAR).ravel()
    return _sync_interface(problem, direction)


def _errors(problem, q, analytic, steps=STEPS):
    """Own central-difference scan: ``[(h, relative, absolute)]`` scaled by block row scales."""
    residual_scale = _scales(problem, RESIDUAL_SCALES)
    analytic = np.asarray(analytic, dtype=float)
    reference = float(np.linalg.norm(analytic/residual_scale))
    assert reference > 0.0, "the analytic response is structurally zero; use an absolute check"
    rows = []
    for step in steps:
        step = float(step)
        plus = np.asarray(monolithic_residual(problem.x+step*q, context=problem.context), dtype=float)
        minus = np.asarray(monolithic_residual(problem.x-step*q, context=problem.context), dtype=float)
        difference = (plus-minus)/(2.0*step)
        absolute = float(np.linalg.norm((difference-analytic)/residual_scale))
        rows.append((step, absolute/reference, absolute))
    return rows


def _scan(problem, direction, steps=STEPS):
    q = np.asarray(direction, dtype=float)/_scales(problem, VARIABLE_SCALES)
    analytic = np.asarray(monolithic_jvp(problem.x, q, context=problem.context), dtype=float)
    return q, analytic, _errors(problem, q, analytic, steps=steps)


def _best(table):
    step, relative, _ = min(table, key=lambda row: row[1])
    return relative, step


def _trend(table):
    values = [row[1] for row in table]
    best = int(np.argmin(values))
    if 0 < best < len(values)-1 and values[0] > values[best] and values[-1] > values[best]:
        return "decreasing_then_increasing"
    return "not_resolved"


def _print_table(title, table):
    print(f"[review] {title}")
    for step, relative, absolute in table:
        print(f"  h={step:.0e}  rel={relative:.3e}  abs={absolute:.3e}")


def _print_identity(title, difference, scale, limit=IDENTITY_LIMIT):
    verdict = "PASS" if difference <= limit*scale else "FAIL"
    print(f"[review] {title}: max|diff|={difference:.6e}  limit={limit:.0e}*{scale:g}  {verdict}")


def _reference(problem, x=None):
    """Reviewer-side rebuild of the candidate objects through the public single-field API."""
    layout = problem.layout
    context = problem.context
    base = problem.base
    values = problem.x if x is None else np.asarray(x, dtype=float)
    blocks = layout.unpack(values)
    geometry = build_ale_geometry(base.reference_mesh, np.array(blocks["d_m"], copy=True),
                                  quality_limits=base.quality_limits, candidate_id=context.candidate_id)
    transport = build_gcl_transport(base.reference_mesh, geometry, base.accepted_geometries, context.stencil)
    operators = assemble_stokes(geometry.current_mesh, viscosity=base.fluid_viscosity,
                                density=base.fluid_density, body_force=base.fluid_body_force)
    ale_context = build_ale_flow_context(operators, geometry, transport, base.fluid_states)
    flow, _ = ale_navier_stokes_residual(
        operators, np.concatenate((blocks["u_f"].ravel(), blocks["p_f"].ravel())), ale_context,
        with_jacobian=False)
    interface_geometry = evaluate_interface_geometry(context.topology, geometry.current_mesh, operators.space,
                                                     np.array(blocks["d_s"], copy=True),
                                                     candidate_id=context.candidate_id)
    state = SimpleNamespace(velocity=np.array(blocks["u_f"], copy=True),
                            pressure=np.array(blocks["p_f"], copy=True))
    load = assemble_interface_load(context.topology, interface_geometry, operators, state,
                                   thickness=context.thickness, path="current")
    return SimpleNamespace(geometry=geometry, operators=operators, ale_context=ale_context, flow=flow,
                           interface_geometry=interface_geometry, load=load, state=state, blocks=blocks)


def _free_velocity_mask(problem):
    mask = np.ones(2*problem.solid.n_nodes, dtype=bool)
    mask[problem.layout.solid_velocity_dirichlet_dofs] = False
    return mask


# ---------------------------------------------------------------------------
# 21-02 / 21-03: independent directional derivative scans
# ---------------------------------------------------------------------------

def test_review_directional_derivatives_full_scan():
    problem = build_problem()
    directions = {
        "smooth (all blocks, deterministic)": _smooth_direction(problem),
        f"mixed random (seed={REVIEW_SEED})": _random_direction(problem, REVIEW_SEED),
        f"interface chain (seed={REVIEW_SEED+2})": _interface_direction(problem, REVIEW_SEED+2),
    }
    results = {}
    for label, direction in directions.items():
        q, analytic, table = _scan(problem, direction)
        assert np.any(np.asarray(direction) != 0.0), label
        best, best_step = _best(table)
        results[label] = (best, best_step, _trend(table))
        _print_table(f"jvp scan: {label} (best rel={best:.3e} @ h={best_step:.0e}, "
                     f"trend={_trend(table)})", table)
        assert best <= GATE, f"{label}: best relative error {best:.3e} at h={best_step:.0e} > {GATE:.0e}"
        del q, analytic

    # assembled Jacobian against the same central differences (independent of the JVP path)
    jacobian = monolithic_jacobian(problem.x, context=problem.context)
    for label, direction in directions.items():
        q = np.asarray(direction, dtype=float)/_scales(problem, VARIABLE_SCALES)
        table = _errors(problem, q, jacobian @ q)
        best, best_step = _best(table)
        _print_table(f"assembled J scan: {label} (best rel={best:.3e} @ h={best_step:.0e})", table)
        assert best <= GATE, f"{label}: assembled best {best:.3e} at h={best_step:.0e} > {GATE:.0e}"
    # J @ q and the matrix-free JVP agree on the same directions
    for label, direction in directions.items():
        q = np.asarray(direction, dtype=float)/_scales(problem, VARIABLE_SCALES)
        gap = np.asarray(jacobian @ q) - np.asarray(monolithic_jvp(problem.x, q, context=problem.context))
        scale = _scales(problem, RESIDUAL_SCALES)
        relative = float(np.linalg.norm(gap/scale)/np.linalg.norm(np.asarray(monolithic_jvp(
            problem.x, q, context=problem.context))/scale))
        print(f"[review] J vs JVP gap ({label}): {relative:.3e}")
        assert relative <= JACOBIAN_JVP_GAP_LIMIT, label
    assert len(results) == 3


def test_review_zero_response_direction_absolute():
    """A mesh node outside the interface owner cells has a structurally zero load response."""
    problem = build_problem()
    layout = problem.layout
    context = problem.context
    cells = np.asarray(problem.fluid.cells, dtype=np.int64)
    wet = set(int(node) for node in problem.topology.fluid_vertex_nodes)
    owner_nodes = set()
    for cell in cells.tolist():
        if wet.intersection(cell):
            owner_nodes.update(int(node) for node in cell)
    candidates = [int(node) for node in range(problem.fluid.n_nodes)
                  if node not in owner_nodes and not np.isin(node, layout.mesh_dirichlet_nodes)]
    assert candidates, "no mesh node outside the owner band"
    x = problem.fluid.points[candidates, 0]
    y = problem.fluid.points[candidates, 1]
    node = candidates[int(np.lexsort((np.abs(x-0.5), -y))[0])]
    block = np.zeros((problem.fluid.n_nodes, 2))
    block[node] = (L_STAR, -0.5*L_STAR)
    direction = np.zeros(layout.n_total)
    direction[layout.slice_of("d_m")] = block.ravel()

    q = direction/_scales(problem, VARIABLE_SCALES)
    analytic = np.asarray(monolithic_jvp(problem.x, q, context=context), dtype=float)
    g_rows = layout.slice_of("g")
    residual_scale = _scales(problem, RESIDUAL_SCALES)
    np.testing.assert_array_equal(analytic[g_rows], 0.0)
    assert np.any(analytic)  # the full response is not trivially zero
    table = []
    for step in STEPS:
        step = float(step)
        plus = np.asarray(monolithic_residual(problem.x+step*q, context=context), dtype=float)
        minus = np.asarray(monolithic_residual(problem.x-step*q, context=context), dtype=float)
        difference = (plus-minus)/(2.0*step)
        table.append((step, float(np.max(np.abs((difference-analytic)[g_rows]/residual_scale[g_rows])))))
    print(f"[review] zero-response direction: mesh node {node} at "
          f"{tuple(np.round(problem.fluid.points[node], 6))} (owner band excluded)")
    for step, error in table:
        print(f"  h={step:.0e}  |fd-analytic|_max/S_g={error:.3e}")
    assert max(error for _, error in table) <= ZERO_RESPONSE_LIMIT


# ---------------------------------------------------------------------------
# 21-05 / 21-01: identity spot checks
# ---------------------------------------------------------------------------

def test_review_identity_h_and_htranspose():
    problem = build_problem()
    layout = problem.layout
    topology = problem.topology
    rng = np.random.default_rng(REVIEW_SEED+3)
    h_dense = np.asarray(topology.H.todense())
    n_fluid_trace = h_dense.shape[0]
    n_solid_trace = h_dense.shape[1]
    base = np.asarray(monolithic_residual(problem.x, context=problem.context))
    u_rows = layout.start_of("u_f")+layout.fluid_interface_dofs
    vs_rows = layout.slice_of("v_s")

    # H: perturbing v_s only moves the wet fluid trace rows by exactly -H dv
    dv = rng.normal(size=n_solid_trace)*U_STAR
    direction = np.zeros(layout.n_total)
    v_block = np.zeros((problem.solid.n_nodes, 2))
    v_block.ravel()[topology.solid_dofs] = dv
    direction[vs_rows] = v_block.ravel()
    delta = np.asarray(monolithic_residual(problem.x+direction, context=problem.context))-base
    h_difference = float(np.max(np.abs(delta[u_rows]+h_dense @ dv)))
    _print_identity("H: Delta R_u,Gamma = -H dv", h_difference, U_STAR)

    # H^T: perturbing g only moves the free solid velocity rows by exactly -S_s H^T dg
    dg = rng.normal(size=n_fluid_trace)*F_STAR
    direction = np.zeros(layout.n_total)
    direction[layout.slice_of("g")] = dg
    delta = np.asarray(monolithic_residual(problem.x+direction, context=problem.context))-base
    expected = np.zeros(2*problem.solid.n_nodes)
    expected[topology.solid_dofs] = -(h_dense.T @ dg)
    free = _free_velocity_mask(problem)
    ht_difference = float(np.max(np.abs(delta[vs_rows][free]-expected[free])))
    _print_identity("H^T: Delta R_sv = -S_s H^T dg (free rows)", ht_difference, F_STAR)
    np.testing.assert_array_equal(delta[vs_rows][~free], 0.0)
    np.testing.assert_allclose(delta[layout.slice_of("g")], dg, rtol=0, atol=1e-15)
    untouched = np.ones(layout.n_total, dtype=bool)
    untouched[layout.start_of("v_s")+np.flatnonzero(free)] = False
    untouched[layout.slice_of("g")] = False
    np.testing.assert_array_equal(delta[untouched], 0.0)

    # assembled matrix rows: [u_f|Gamma] = I, [u_f|Gamma, v_s|Gamma] = -H, [d_m|Gamma] = [I, -I]
    jacobian = monolithic_jacobian(problem.x, context=problem.context)
    u_identity = np.asarray(jacobian[u_rows][:, layout.start_of("u_f")+layout.fluid_interface_dofs].todense())
    u_coupling = np.asarray(jacobian[u_rows][:, layout.start_of("v_s")+topology.solid_dofs].todense())
    identity_gap = float(np.max(np.abs(u_identity-np.eye(n_fluid_trace))))
    coupling_gap = float(np.max(np.abs(u_coupling+h_dense)))
    _print_identity("assembled wet trace row [u_f|Gamma] = I", identity_gap, 1.0, limit=1e-14)
    _print_identity("assembled wet trace row [v_s|Gamma] = -H", coupling_gap, 1.0, limit=1e-14)
    n_mesh_trace = 2*len(topology.fluid_vertex_nodes)
    m_rows = layout.start_of("d_m")+layout.mesh_interface_dofs
    m_identity = np.asarray(jacobian[m_rows][:, layout.start_of("d_m")+layout.mesh_interface_dofs].todense())
    m_coupling = np.asarray(jacobian[m_rows][:, layout.start_of("d_s")+topology.solid_dofs].todense())
    _print_identity("assembled mesh trace [d_m|Gamma] = I", float(np.max(np.abs(m_identity-np.eye(n_mesh_trace)))),
                    1.0, limit=1e-14)
    _print_identity("assembled mesh trace [d_s|Gamma] = -I", float(np.max(np.abs(m_coupling+np.eye(n_mesh_trace)))),
                    1.0, limit=1e-14)


def test_review_identity_bdf_compatibility_row():
    problem = build_problem()
    layout = problem.layout
    coefficients = np.asarray(problem.stencil.coefficients, dtype=float)
    a0 = float(coefficients[0])
    history = np.zeros(2*problem.solid.n_nodes)
    for weight, (d_state, _velocity) in zip(coefficients[1:], problem.base.solid_states):
        history += weight*d_state.ravel()
    compatible = problem.x.copy()
    d_s = np.asarray(layout.unpack(problem.x)["d_s"]).ravel()
    compatible[layout.slice_of("v_s")] = a0*d_s+history
    residual = np.asarray(monolithic_residual(compatible, context=problem.context))
    free = np.ones(2*problem.solid.n_nodes, dtype=bool)
    free[layout.solid_displacement_dirichlet_dofs] = False
    difference = float(np.max(np.abs(residual[layout.slice_of("d_s")][free])))
    _print_identity("BDF: R_sd = a0 d_s + sum a_j d_j - v_s = 0", difference, U_STAR)
    assert difference <= IDENTITY_LIMIT*U_STAR

    # assembled Jacobian of the compatibility row: a0 I on d_s, -I on v_s (free rows)
    jacobian = monolithic_jacobian(problem.x, context=problem.context)
    rows = layout.start_of("d_s")+np.flatnonzero(free)
    ds_gap = float(np.max(np.abs(np.asarray(jacobian[rows][:, layout.slice_of("d_s")].todense())
                                 - a0*np.eye(2*problem.solid.n_nodes)[free])))
    vs_gap = float(np.max(np.abs(np.asarray(jacobian[rows][:, layout.slice_of("v_s")].todense())
                                 + np.eye(2*problem.solid.n_nodes)[free])))
    _print_identity("assembled J[d_s, d_s] = a0 I (free rows)", ds_gap, 1.0, limit=1e-14)
    _print_identity("assembled J[d_s, v_s] = -I (free rows)", vs_gap, 1.0, limit=1e-14)


def test_review_identity_thickness_counted_once():
    loads = {}
    for thickness in (0.25, 1.0, 4.0):
        problem = build_problem(thickness=thickness, candidate_id=f"review-thickness-{thickness}")
        blocks = problem.layout.unpack(problem.x)
        residual = np.asarray(monolithic_residual(problem.x, context=problem.context))
        loads[thickness] = np.asarray(blocks["g"]).ravel()-residual[problem.layout.slice_of("g")]
    scale = float(np.max(np.abs(loads[1.0])))
    doubled = float(np.max(np.abs(loads[4.0]-4.0*loads[1.0])))/scale
    quarter = float(np.max(np.abs(loads[0.25]-0.25*loads[1.0])))/scale
    _print_identity("thickness 4.0 vs 4 x 1.0", doubled, scale)
    _print_identity("thickness 0.25 vs 0.25 x 1.0", quarter, scale)
    np.testing.assert_allclose(loads[4.0], 4.0*loads[1.0], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(loads[0.25], 0.25*loads[1.0], rtol=1e-12, atol=0.0)

    # the monolithic load rows equal the independently assembled candidate load
    problem = build_problem()
    reference = _reference(problem)
    blocks = problem.layout.unpack(problem.x)
    residual = np.asarray(monolithic_residual(problem.x, context=problem.context))
    load = np.asarray(blocks["g"]).ravel()-residual[problem.layout.slice_of("g")]
    assembled_gap = float(np.max(np.abs(load-reference.load.g_f_gamma)))
    _print_identity("R_g = g - g_f_gamma(candidate)", assembled_gap, F_STAR)


def test_review_identity_pressure_pin_row():
    problem = build_problem(pin=PressurePin(3, -0.7*P_STAR))
    layout = problem.layout
    blocks = layout.unpack(problem.x)
    row = layout.start_of("p_f")+3
    residual = np.asarray(monolithic_residual(problem.x, context=problem.context))
    value_gap = float(np.abs(residual[row]-(np.asarray(blocks["p_f"]).ravel()[3]+0.7*P_STAR)))
    _print_identity("pin row R_p = p_3 - (-0.7 P*)", value_gap, P_STAR)
    assert value_gap <= IDENTITY_LIMIT*P_STAR

    # a p_3 perturbation moves the pin row by exactly the same amount; the assembled
    # pin row is the unit row (pressure changes other rows through the flow coupling)
    direction = np.zeros(layout.n_total)
    direction[row] = 0.31*P_STAR
    delta = np.asarray(monolithic_residual(problem.x+direction, context=problem.context))-residual
    _print_identity("pin row derivative = 1", float(np.abs(delta[row]-0.31*P_STAR)), P_STAR)
    jacobian = monolithic_jacobian(problem.x, context=problem.context)
    dense_row = np.asarray(jacobian[row].todense()).ravel()
    unit_gap = float(np.max(np.abs(dense_row-np.eye(layout.n_total)[row])))
    _print_identity("assembled pin row is the unit row", unit_gap, 1.0, limit=1e-14)

    # without a pin the raw continuity row stays in place
    raw = build_problem(pin=None)
    assert raw.layout.pressure_pin is None
    assert raw.context.pressure_constraint is None
    raw_reference = _reference(raw)
    raw_residual = np.asarray(monolithic_residual(raw.x, context=raw.context))
    n_velocity = raw.layout.sizes[0]
    raw_gap = float(np.abs(raw_residual[raw.layout.start_of("p_f")]
                           - np.asarray(raw_reference.flow[n_velocity])))
    _print_identity("pin=None keeps the raw R_p row", raw_gap, U_STAR*L_STAR)


# ---------------------------------------------------------------------------
# rejection conditions
# ---------------------------------------------------------------------------

def test_review_rejection_flip_and_quality():
    problem = build_problem()
    layout = problem.layout
    points = problem.fluid.points
    baseline = np.asarray(monolithic_residual(problem.x, context=problem.context))
    assert np.all(np.isfinite(baseline))

    flipped = problem.x.copy()
    flip_block = np.column_stack((-3.0*points[:, 0], np.zeros(problem.fluid.n_nodes)))
    flipped[layout.slice_of("d_m")] = flip_block.ravel()
    with pytest.raises(ValueError) as flip_error:
        monolithic_residual(flipped, context=problem.context)
    print(f"[review] orientation flip rejected: {flip_error.value}")
    assert "det(J_x)" in str(flip_error.value)

    squeezed = problem.x.copy()
    center = points.mean(axis=0)
    squeezed[layout.slice_of("d_m")] = (-0.98*(points-center)).ravel()
    with pytest.raises(ValueError) as quality_error:
        monolithic_residual(squeezed, context=problem.context)
    print(f"[review] quality rejection: {quality_error.value}")
    assert "volume ratio" in str(quality_error.value)

    with pytest.raises(ValueError) as shape_error:
        monolithic_residual(problem.x[:-1], context=problem.context)
    print(f"[review] invalid x rejected: {shape_error.value}")
    assert "shape" in str(shape_error.value)


def test_review_rejection_stale_and_mixed_versions():
    problem = build_problem()
    context = problem.context
    reference = _reference(problem)
    require_candidate(context, geometry=reference.geometry, interface_geometry=reference.interface_geometry,
                      load=reference.load)

    stale_history = BDFHistory(np.zeros((2, problem.solid.n_nodes, 2)), 0.25, order=1)
    for _ in range(2):
        stale_history.accept(stale_history.context(), np.zeros((2, problem.solid.n_nodes, 2)))
    assert stale_history.stencil().step_id != problem.base.generation
    with pytest.raises(ValueError) as stale_error:
        type(context)(context.layout, context.base, stale_history.stencil(), context.topology,
                      context.harmonic, candidate_id="review-stale", thickness=context.thickness,
                      pressure_constraint=context.layout.pressure_pin)
    print(f"[review] stale stencil rejected: {stale_error.value}")
    assert "stale candidate" in str(stale_error.value)

    stale_geometry = build_ale_geometry(problem.fluid, np.zeros_like(problem.fluid.points),
                                        candidate_id="review-old-candidate")
    with pytest.raises(ValueError) as version_error:
        require_candidate(context, geometry=stale_geometry)
    print(f"[review] stale geometry rejected: {version_error.value}")
    assert "stale candidate geometry" in str(version_error.value)

    other = build_problem(nx=3, candidate_id="review-other-mesh")
    other_reference = _reference(other)
    foreign_interface = evaluate_interface_geometry(
        other.topology, other_reference.geometry.current_mesh, other_reference.operators.space,
        np.array(other_reference.blocks["d_s"], copy=True), candidate_id=context.candidate_id)
    with pytest.raises(ValueError) as topology_error:
        require_candidate(context, interface_geometry=foreign_interface)
    print(f"[review] foreign topology rejected: {topology_error.value}")
    assert "different topology" in str(topology_error.value)

    shifted = problem.x.copy()
    blocks = problem.layout.unpack(shifted)
    d_s = np.array(blocks["d_s"], copy=True)
    d_s[problem.topology.solid_nodes, 1] += 1e-5
    d_m = np.array(blocks["d_m"], copy=True)
    d_m.ravel()[problem.layout.mesh_interface_dofs] = d_s.ravel()[problem.topology.solid_dofs]
    shifted[problem.layout.slice_of("d_s")] = d_s.ravel()
    shifted[problem.layout.slice_of("d_m")] = d_m.ravel()
    shifted_load = _reference(problem, shifted).load
    with pytest.raises(ValueError) as mixed_error:
        require_candidate(context, interface_geometry=reference.interface_geometry, load=shifted_load)
    print(f"[review] mixed load/geometry rejected: {mixed_error.value}")
    assert "different interface geometry" in str(mixed_error.value)


# ---------------------------------------------------------------------------
# evidence sampling of the implementer report
# ---------------------------------------------------------------------------

def test_review_scan_evidence_matches_report():
    """Sample the frozen evidence JSON against the numbers reported in t21.md."""
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads((root/"results/m5/t21/scan_evidence.json").read_text(encoding="utf-8"))
    smoke = json.loads((root/"results/m5/t21/fsi1_smoke.json").read_text(encoding="utf-8"))
    per_block = evidence["per_block_random_directions"]
    single = evidence["per_block_single_components"]
    paths = evidence["mesh_paths"]
    rectangles = evidence["completeness"]["rectangles"]
    snapshot = evidence["smoke"]
    samples = [
        ("t21.md 4.3 mixed best relative error", evidence["mixed_direction"]["best_relative_error"],
         7.39e-11, 5e-3),
        ("t21.md 4.3 mixed best step", evidence["mixed_direction"]["best_relative_step"], 1e-6, 0.0),
        ("t21.md 4.3 J vs JVP gap", evidence["mixed_direction"]["jacobian_vs_jvp_relative_gap"], 1.6e-16, 5e-3),
        ("t21.md 4.2 random d_s", per_block["d_s"]["best_relative_error"], 5.67e-11, 5e-3),
        ("t21.md 4.2 random d_m", per_block["d_m"]["best_relative_error"], 2.94e-11, 5e-3),
        ("t21.md 4.2 random u_f", per_block["u_f"]["best_relative_error"], 1.94e-16, 5e-3),
        ("t21.md 4.1 u_f interior", single["u_f interior"]["best_relative_error"], 2.96e-16, 5e-3),
        ("t21.md 4.1 d_s interior", single["d_s interior"]["best_relative_error"], 1.87e-11, 5e-3),
        ("t21.md 4.1 d_m interior", single["d_m interior"]["best_relative_error"], 1.04e-10, 5e-3),
        ("t21.md 4.1 g trace", single["g trace"]["best_relative_error"], 1.57e-16, 5e-3),
        ("t21.md 4.4 interior-only d_m", paths["interior_only_d_m"]["best_relative_error"], 4.40e-11, 5e-3),
        ("t21.md 4.4 interface-only d_m", paths["interface_only_d_m_with_compatible_d_s"]["best_relative_error"],
         9.73e-11, 5e-3),
        ("t21.md 4.4 compatible mixed d_m/d_s", paths["compatible_mixed_d_m_d_s"]["best_relative_error"],
         3.93e-11, 5e-3),
        ("t21.md 4.5 BDF1", evidence["bdf"]["bdf1"]["best_relative_error"], 5.64e-11, 5e-3),
        ("t21.md 4.5 BDF2", evidence["bdf"]["bdf2"]["best_relative_error"], 5.33e-11, 5e-3),
        ("t21.md 5 contribution J_um/J_pm", rectangles["fluid_geometry_j_um_j_pm"]["scaled_contribution"],
         1.417e+01, 5e-3),
        ("t21.md 5 contribution S_s H^T", rectangles["solid_load_s_h_t"]["scaled_contribution"],
         2.705e+02, 5e-3),
        ("t21.md 5 dropped J_um/J_pm relative error",
         rectangles["fluid_geometry_j_um_j_pm"]["d dropped_best_relative_error"], 1.23e-03, 5e-3),
        ("t21.md 4.7 load path difference", evidence["load_paths"]["load_max_abs_diff"], 4.3e-19, 2e-2),
        ("t21.md 4.7 derivative path difference", evidence["load_paths"]["derivative_max_abs_diff"],
         1.7e-16, 5e-2),
        ("t21.md 6 smoke n_total (scan snapshot)", snapshot["n_total"], 12905, 0.0),
        ("t21.md 6 smoke nnz (scan snapshot)", snapshot["nnz"], 377096, 0.0),
        ("t21.md 6 smoke J vs JVP", snapshot["jvp_gap_relative"], 1.79e-16, 5e-3),
        ("t21.md 6 smoke residual norm", snapshot["residual_norm"], 7.595, 5e-3),
        ("t21.md 6 smoke min determinant", snapshot["geometry"]["minimum_cell_determinant"], 4.73e-05, 5e-3),
        ("t21.md 6 smoke min quality", snapshot["geometry"]["minimum_quality"], 0.728, 5e-3),
        ("t21.md 6 smoke residual seconds", snapshot["mode_seconds"]["residual"], 0.27, 5e-2),
        ("t21.md 6 smoke jvp seconds", snapshot["mode_seconds"]["jvp"], 0.69, 5e-2),
        ("t21.md 6 smoke assembled seconds", snapshot["mode_seconds"]["assembled"], 19.43, 5e-2),
        ("t21.md 6 smoke min volume ratio", snapshot["geometry"]["minimum_volume_ratio"], 1.0, 0.0),
    ]
    for label, actual, reported, relative in samples:
        if relative == 0.0:
            assert actual == pytest.approx(reported, abs=1e-12), (
                f"{label}: evidence {actual!r} != report {reported!r}")
        else:
            assert actual == pytest.approx(reported, rel=relative), (
                f"{label}: evidence {actual!r} vs report {reported!r}")
    assert evidence["mixed_direction"]["trend"] == "decreasing_then_increasing"
    assert evidence["geometry_columns_batch_vs_direct"]["j_um_free_max_abs_diff"] == 0.0
    assert evidence["geometry_columns_batch_vs_direct"]["j_pm_free_max_abs_diff"] == 0.0

    # the live smoke record is regenerated by every smoke run: deterministic fields only
    assert smoke["n_total"] == 12905 and smoke["nnz"] == 377096
    assert smoke["sizes"] == [8738, 1139, 226, 226, 2278, 298]
    assert smoke["geometry"]["minimum_volume_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert smoke["jvp_gap_relative"] == pytest.approx(1.79e-16, rel=5e-3)

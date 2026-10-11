"""T17 full-field transaction tests: atomic accept, rollback and retry (17-01..17-10).

The physical baseline is a small *real* solid (StVK/BDF ``d/v``) and fluid
(Taylor-Hood / ALE BDF residual on a prescribed moving mesh) with the frozen
interface ``h``/``H`` mapping; lifecycle faults are injected with doubles that
record whether the injection point actually fired.  Thresholds follow
``docs/development/m3_freeze.md`` section 6 and ``validation_plan.md`` section 5:
array and history rollback is ``np.array_equal``, retry-vs-cold-start and
same-input repeats are ``<= 1e-10`` (normalized), and 17-01..17-10 all need at
least one triggered fault or exact-value check.
"""

import dataclasses
import json

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fsi_ice.ale import build_ale_geometry, build_gcl_transport, evaluate_mesh_velocity
from fsi_ice.fem.p2 import P2Space
from fsi_ice.fluid.ale import ale_navier_stokes_residual, build_ale_flow_context
from fsi_ice.fluid.dynamics import solve_flow_step
from fsi_ice.fluid.forces import BoundaryMonitor
from fsi_ice.fluid.linear_solver import ReusedLUSolver
from fsi_ice.fluid.taylor_hood import assemble_stokes, flow_constraints
from fsi_ice.interface import build_matching_interface, interface_diagnostics, map_load, map_motion
from fsi_ice.mesh import TriMesh, rectangle_mesh
from fsi_ice.solid.dynamics import solve_solid_step
from fsi_ice.solid.linear import ElasticMaterial, assemble_elasticity
from fsi_ice.time.bdf import BDFHistory
from fsi_ice.time.transaction import (AcceptedState, StateCoordinator, StepConflict, StepPlan,
                                      TransactionError)

DT = 0.05
L_STAR = 1.0
T_STAR = 1.0
U_STAR = L_STAR/T_STAR
FLUID_VISCOSITY = 0.1
FLUID_DENSITY = 1.0
SOLID_BODY_FORCE = (0.0, -10.0)
TRACTION = np.array([0.3, -0.1])
RETRY_TOLERANCE = 1e-10          # frozen same-input/retry-vs-baseline limit
CONTINUITY_LIMIT = 1e-8          # recomputed discrete continuity of a converged ALE state
RESIDUAL_LIMIT = 1e-7            # recomputed free-row true residual of a converged ALE state
PREPARE_HOOKS = ("after_solid_accept", "after_fluid_accept", "after_mesh_accept", "before_freeze")
STATES = [pytest.param(1, False, id="bdf1-startup"), pytest.param(2, True, id="bdf2-mature")]


class CheckFailure(ValueError):
    """Raised by physics checks with an extra diagnostics mapping."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class Injector:
    """Records whether one named lifecycle injection point was triggered."""

    def __init__(self, point):
        self.point = point
        self.fired = 0
        self.view = None

    def trigger(self, point, view):
        if point != self.point:
            return False
        self.fired += 1
        self.view = view
        return True


def _tags(mesh, interface_tag):
    tags = {name: mesh.boundaries[name] for name in ("bottom", "top", "left", "right")}
    tags["interface"] = tags.pop(interface_tag)
    return tags


def _two_domain(nx=3, ny_fluid=2, ny_solid=2):
    """Fluid [0,1]x[0,0.8] above solid [0,1]x[-0.5,0], sharing y=0."""
    fluid_flat = rectangle_mesh(nx, ny_fluid, width=1.0, height=0.8)
    solid_flat = rectangle_mesh(nx, ny_solid, width=1.0, height=0.5)
    solid = TriMesh(solid_flat.points - np.array([0.0, 0.5]), solid_flat.cells, _tags(solid_flat, "top"))
    fluid = TriMesh(fluid_flat.points, fluid_flat.cells, _tags(fluid_flat, "bottom"))
    topology = build_matching_interface(solid, fluid, P2Space.from_mesh(fluid), solid_tag="interface",
                                        fluid_tag="interface", atol=1e-12, rtol=1e-12)
    return solid, fluid, topology


class Physics:
    """Small real solid/fluid residual plus prescribed mesh motion and frozen mapping."""

    def __init__(self):
        self.solid, self.fluid, self.topology = _two_domain()
        self.space = P2Space.from_mesh(self.fluid)
        self.material = ElasticMaterial(1e4, 0.3, density=10.0)
        _, self.solid_force = assemble_elasticity(self.solid, self.material,
                                                  body_force=list(SOLID_BODY_FORCE))
        self.fixed = self.solid.boundary_nodes("bottom")
        self.cache = {}
        self.cache_hits = 0
        self.revisions = []
        self.ledger = {name: 0 for name in ("displacement", "geometry", "mesh_velocity", "mesh_current",
                                            "solid", "fluid", "fluid_operators", "interface", "confirm",
                                            "monitor", "checks")}
        self.monitors = []
        self.solvers = []
        self.solid_results = []
        self.fluid_solutions = []
        self.fluid_views = None
        self.initial_deltas = []

    # ------------------------------------------------------------ prescribed fields
    def body_force(self, points):
        return np.column_stack((0.05*np.sin(np.pi*points[:, 1]/0.8),
                                0.01*np.cos(np.pi*points[:, 0])*np.sin(np.pi*points[:, 1]/0.8)))

    def mesh_displacement(self, points, time):
        envelope = (np.sin(np.pi*points[:, 0])*np.sin(np.pi*points[:, 1]/0.8))[:, None]
        return 0.02*np.sin(2*np.pi*time/T_STAR + 1.0)*envelope*np.array([1.0, 0.5])

    def mesh_velocity_field(self, points, time):
        envelope = (np.sin(np.pi*points[:, 0])*np.sin(np.pi*points[:, 1]/0.8))[:, None]
        return 0.02*2*np.pi*np.cos(2*np.pi*time/T_STAR + 1.0)/T_STAR*envelope*np.array([1.0, 0.5])

    def initial_state(self, order, time=0.0):
        displacement = self.mesh_displacement(self.fluid.points, time)
        return AcceptedState.initial(
            solid=np.zeros((2, self.solid.n_nodes, 2)),
            velocity=np.zeros((len(self.space.points), 2)),
            pressure=np.zeros(self.fluid.n_nodes),
            mesh_displacement=displacement, mesh_current=self.fluid.points + displacement,
            mesh_velocity=self.mesh_velocity_field(self.fluid.points, time),
            time=time, dt=DT, order=order)

    def interface_dual_loads(self):
        load = np.zeros((self.topology.n_fluid_interface_nodes, 2))
        vertex_row = {int(node): row for row, node in enumerate(self.topology.fluid_vertex_nodes)}
        for offset, pair in enumerate(self.topology.edge_pairs):
            load[self.topology.n_solid_interface_nodes + offset] += pair.length*TRACTION*(2/3)
            for node in pair.fluid_edge:
                load[vertex_row[int(node)]] += pair.length*TRACTION*(1/6)
        return load

    # ------------------------------------------------------------------ plan
    def plan(self, *, injector=None, hooks=None, serialize=None):
        p = self
        n_solid_gamma = self.topology.n_solid_interface_nodes
        n_fluid_gamma = self.topology.n_fluid_interface_nodes

        def trigger(point, view):
            return injector is not None and injector.trigger(point, view)

        def displacement(view):
            p.ledger["displacement"] += 1
            if trigger("displacement_nan", view):
                return np.full((p.fluid.n_nodes, 2), np.nan)
            if trigger("displacement_flip", view):
                return np.column_stack((-1.5*p.fluid.points[:, 0], np.zeros(p.fluid.n_nodes)))
            if trigger("displacement_quality", view):
                return np.column_stack((5.0*p.fluid.points[:, 1], np.zeros(p.fluid.n_nodes)))
            return p.mesh_displacement(p.fluid.points, view.target_time)

        def geometry(view):
            p.ledger["geometry"] += 1
            p.revisions.append(view.revision)
            return build_ale_geometry(p.fluid, view.displacement, candidate_id=view.revision)

        def mesh_velocity(view):
            p.ledger["mesh_velocity"] += 1
            if trigger("mesh_velocity_raise", view):
                raise RuntimeError("injected mesh velocity failure")
            return evaluate_mesh_velocity(view.geometry, view.mesh_context).values_p1

        def mesh_current(view):
            p.ledger["mesh_current"] += 1
            return view.geometry.current_mesh.points

        def solve_solid(view):
            p.ledger["solid"] += 1
            candidate, result = solve_solid_step(p.solid, p.material, p.solid_force, p.fixed,
                                                 view.solid_context)
            p.solid_results.append(result)
            if trigger("solid_raise", view):
                # the real solve succeeded first; the injection proves the post-solve abort
                raise RuntimeError("injected Newton line-search failure")
            if trigger("solid_flip", view):
                flipped = np.array(candidate, copy=True)
                flipped[0] = np.column_stack((-1.5*p.solid.points[:, 0], np.zeros(p.solid.n_nodes)))
                return flipped
            return candidate

        def solve_fluid(view):
            p.ledger["fluid"] += 1
            entry = p.cache.get(view.revision)
            if entry is None:
                operators = assemble_stokes(view.geometry.current_mesh, viscosity=FLUID_VISCOSITY,
                                            density=FLUID_DENSITY, body_force=p.body_force)
                dofs, _ = flow_constraints(operators, np.zeros(2))
                entry = {"operators": operators, "dofs": dofs,
                         "free": np.setdiff1d(np.arange(operators.n_total), dofs),
                         "scale": max(float(np.linalg.norm(operators.rhs)), 1.0)}
                p.cache[view.revision] = entry
                p.ledger["fluid_operators"] += 1
            else:
                p.cache_hits += 1
            operators = entry["operators"]
            stencil = view.stencil
            accepted = [build_ale_geometry(p.fluid, values,
                                           candidate_id=("accepted", view.attempt_id, index))
                        for index, values in enumerate(view.base.mesh_history[:stencil.order])]
            transport = build_gcl_transport(p.fluid, view.geometry, accepted, stencil)
            flow_context = build_ale_flow_context(operators, view.geometry, transport,
                                                  view.base.fluid_history[:stencil.order])
            entry["ale"] = flow_context
            if trigger("fluid_raise", view):
                p.monitors.append(BoundaryMonitor(operators, "interface"))
                p.solvers.append(ReusedLUSolver(refresh_interval=20,
                                                velocity_dofs=operators.n_velocity,
                                                density=FLUID_DENSITY))
                raise RuntimeError("injected fluid solver failure after monitor/LU caches")
            if trigger("fluid_singular", view):
                ReusedLUSolver(refresh_interval=20, velocity_dofs=operators.n_velocity,
                               density=FLUID_DENSITY).solve(
                    csr_matrix((operators.n_total, operators.n_total)),
                    np.ones(operators.n_total), np.array([], dtype=np.int64))
            initial = None
            if p.initial_deltas:
                initial = np.asarray(p.initial_deltas.pop(0), dtype=float)
            iterations = 1 if trigger("fluid_iterations", view) else 50
            solution = solve_flow_step(operators, np.zeros(2), view.fluid_context,
                                       ale_context=flow_context, initial=initial,
                                       tolerance=1e-10, max_iterations=iterations)
            p.fluid_solutions.append(solution)
            velocity, pressure = solution.velocity, solution.pressure
            p.fluid_views = (velocity, pressure)
            if trigger("fluid_mass", view):
                velocity = np.array(velocity, copy=True)
                velocity[:, 0] += 0.1*(p.space.points[:, 0] > 0.5)
                pressure = np.array(pressure, copy=True)
            return velocity, pressure

        def build_interface(view):
            p.ledger["interface"] += 1
            motion = map_motion(p.topology, view.solid[0])
            fluid_load = p.interface_dual_loads().ravel()
            solid_load = map_load(p.topology, fluid_load)
            if trigger("interface_nan", view):
                fluid_load = np.array(fluid_load, copy=True)
                fluid_load[1] = np.nan
            if trigger("interface_shape", view):
                solid_load = np.asarray(solid_load)[:-1]
            if trigger("interface_work", view):
                solid_load = np.array(solid_load, copy=True)
                solid_load[0] += 0.01
            return {"solid_motion": motion.vertex_values, "fluid_motion": motion.p2_values,
                    "solid_load": solid_load, "fluid_load": fluid_load}

        def build_predictor(view):
            values = {"mesh_displacement": view.displacement, "solid": view.solid,
                      "velocity": view.velocity, "pressure": view.pressure}
            if trigger("predictor_nan", view):
                pressure = np.array(view.pressure, copy=True)
                pressure[0] = np.nan
                values["pressure"] = pressure
            return values

        def confirm_solid(view):
            p.ledger["confirm"] += 1
            build_ale_geometry(p.solid, view.solid[0], candidate_id=("solid-health", view.revision))

        def check_mass(view):
            p.ledger["checks"] += 1
            entry = p.cache.get(view.revision)
            operators = entry["operators"] if entry is not None else assemble_stokes(
                view.geometry.current_mesh, viscosity=FLUID_VISCOSITY, density=FLUID_DENSITY,
                body_force=p.body_force)
            velocity = view.velocity
            scale = max(float(np.linalg.norm(velocity)), U_STAR)
            error = float(np.linalg.norm(operators.divergence @ velocity.ravel())/scale)
            if error > CONTINUITY_LIMIT:
                raise CheckFailure(f"fluid continuity/mass limit exceeded ({error:.3e})",
                                   {"continuity_error": error, "limit": CONTINUITY_LIMIT})
            return {"continuity_error": error}

        def check_residual(view):
            p.ledger["checks"] += 1
            entry = p.cache[view.revision]
            mixed = np.concatenate((view.velocity.ravel(), view.pressure))
            residual, _ = ale_navier_stokes_residual(entry["operators"], mixed, entry["ale"],
                                                     with_jacobian=False)
            error = float(np.linalg.norm(residual[entry["free"]])/entry["scale"])
            if error > RESIDUAL_LIMIT:
                raise CheckFailure(f"fluid true residual limit exceeded ({error:.3e})",
                                   {"relative_residual": error, "limit": RESIDUAL_LIMIT})
            return {"relative_residual": error}

        def check_interface(view):
            p.ledger["checks"] += 1
            motion = map_motion(p.topology, view.solid[0])
            delta = np.linspace(-1.0, 1.0, 2*n_solid_gamma)
            report = interface_diagnostics(
                p.topology, motion=motion,
                loads={"compact": view.interface["solid_load"], "fluid": view.interface["fluid_load"]},
                virtual_solid=delta, scales={"force": 1.0, "work": 1.0})
            if not report["pass"]:
                raise CheckFailure("interface consistency failed: " + "; ".join(report["failures"]),
                                   {"failures": tuple(report["failures"]),
                                    "force_error": report["total_force"]["fluid_error"],
                                    "work_error": report["virtual_work"]["error"]})
            return {"work_error": report["virtual_work"]["error"],
                    "force_error": report["total_force"]["fluid_error"]}

        def monitor_row(view):
            p.ledger["monitor"] += 1
            return {"time": view.target_time, "step": view.target_step,
                    "solid": float(np.max(np.abs(view.solid[0]))),
                    "fluid": float(np.max(np.abs(view.velocity))),
                    "mesh": float(np.max(np.abs(view.displacement))),
                    "mass_error": None}

        return StepPlan(
            build_displacement=displacement, build_geometry=geometry,
            build_mesh_velocity=mesh_velocity, build_mesh_current=mesh_current,
            solve_solid=solve_solid, solve_fluid=solve_fluid, build_interface=build_interface,
            interface_shapes={"solid_motion": (n_solid_gamma, 2), "fluid_motion": (n_fluid_gamma, 2),
                              "solid_load": (2*n_solid_gamma,), "fluid_load": (2*n_fluid_gamma,)},
            build_predictor=build_predictor,
            predictor_shapes={"mesh_displacement": (self.fluid.n_nodes, 2),
                              "solid": (2, self.solid.n_nodes, 2),
                              "velocity": (len(self.space.points), 2),
                              "pressure": (self.fluid.n_nodes,)},
            confirms={"solid": confirm_solid},
            checks={"mass": check_mass, "residual": check_residual, "interface": check_interface},
            build_monitor=monitor_row, serialize_monitor=serialize, hooks=hooks or {})


# --------------------------------------------------------------------------- helpers

def capture(state):
    """Deep snapshot of every accepted field for exact-rollback assertions."""
    captured = {"time": state.time, "dt": state.dt, "order": state.order, "step": state.step,
                "epoch": state.epoch, "generations": dict(state.generations),
                "interface": dict(state.interface), "predictor": dict(state.predictor),
                "monitoring": state.monitoring}
    for name in ("solid_history", "fluid_history", "mesh_history"):
        captured[name] = tuple(np.array(item, copy=True) for item in getattr(state, name))
    for name in ("pressure", "mesh_current", "mesh_velocity"):
        captured[name] = np.array(getattr(state, name), copy=True)
    return captured


def _assert_rows_equal(actual, expected, path="monitoring"):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            _assert_rows_equal(actual[key], expected[key], f"{path}[{key!r}]")
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), path
        for index, item in enumerate(expected):
            _assert_rows_equal(actual[index], item, f"{path}[{index}]")
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    else:
        assert actual == expected, path


def assert_capture_equals(state, captured):
    assert state.time == captured["time"]
    assert state.dt == captured["dt"]
    assert state.order == captured["order"]
    assert state.step == captured["step"]
    assert state.epoch == captured["epoch"]
    assert dict(state.generations) == captured["generations"]
    for name in ("solid_history", "fluid_history", "mesh_history"):
        actual = getattr(state, name)
        expected = captured[name]
        assert len(actual) == len(expected)
        for layer, reference in zip(actual, expected):
            np.testing.assert_array_equal(layer, reference)
    for name in ("pressure", "mesh_current", "mesh_velocity"):
        np.testing.assert_array_equal(getattr(state, name), captured[name])
    interface = state.interface
    assert set(interface) == set(captured["interface"])
    for key, reference in captured["interface"].items():
        np.testing.assert_array_equal(interface[key], reference)
    predictor = state.predictor
    assert set(predictor) == set(captured["predictor"])
    for key, reference in captured["predictor"].items():
        np.testing.assert_array_equal(predictor[key], reference)
    _assert_rows_equal(state.monitoring, captured["monitoring"])


def assert_failed(tx, coordinator, root, captured, stage, field=None):
    """A failing attempt must be terminal and must leave the live root byte-identical."""
    assert tx.status == "aborted"
    assert coordinator.active is None
    assert coordinator.state is root
    record = coordinator.failures[-1]
    assert record.stage == stage
    assert record.attempt_id == tx.attempt_id
    assert record.base_epoch == captured["epoch"]
    if field is not None:
        assert record.field == field
    assert_capture_equals(coordinator.state, captured)


def run_one(coordinator, plan):
    tx = coordinator.begin(plan=plan)
    tx.evaluate()
    tx.validate()
    tx.prepare()
    return tx, tx.publish()


def run_steps(coordinator, plan, steps):
    accepted = []
    for _ in range(steps):
        accepted.append(run_one(coordinator, plan)[1])
    return accepted


def make_coordinator(physics, order, mature):
    state = physics.initial_state(order)
    if mature:
        assert order == 2
        state = run_steps(StateCoordinator(state), physics.plan(), 1)[-1]
    return StateCoordinator(state)


def plan_kwargs_with_fault(name):
    if name.startswith("inject_"):
        return {"injector": Injector(name[len("inject_"):])}
    if name == "prepare_serialize":
        def explode(row):
            raise RuntimeError("injected monitor serialization failure")
        return {"serialize": explode}
    if name.startswith("prepare_"):
        suffix = name[len("prepare_"):]
        hook = "before_freeze" if suffix == "freeze" else f"after_{suffix}_accept"

        def failing(info):
            raise RuntimeError(f"injected prepare failure at {info['hook']}")
        return {"hooks": {hook: failing}}
    raise AssertionError(f"unknown fault case {name}")


def root_base(array):
    """Ultimate owning ndarray of a view chain (solver u/p share one mixed buffer)."""
    while isinstance(array.base, np.ndarray) and array.base is not array:
        array = array.base
    return array


def array_error(actual, reference):
    """Normalized max error with the frozen ``max(abs(a-b))/S`` definition."""
    scale = max(float(np.max(np.abs(reference))) if reference.size else 0.0, 1.0)
    return float(np.max(np.abs(actual-reference)))/scale if actual.size else 0.0


def assert_states_close(got, reference, tolerance=RETRY_TOLERANCE):
    assert got.time == reference.time
    assert got.dt == reference.dt and got.order == reference.order
    assert got.step == reference.step and got.epoch == reference.epoch
    assert dict(got.generations) == dict(reference.generations)
    worst = 0.0
    for name in ("solid_history", "fluid_history", "mesh_history"):
        for actual, expected in zip(getattr(got, name), getattr(reference, name)):
            worst = max(worst, array_error(actual, expected))
    for name in ("pressure", "mesh_current", "mesh_velocity"):
        worst = max(worst, array_error(getattr(got, name), getattr(reference, name)))
    for key, expected in reference.interface.items():
        worst = max(worst, array_error(got.interface[key], expected))
    for key, expected in reference.predictor.items():
        worst = max(worst, array_error(got.predictor[key], expected))
    assert worst <= tolerance
    return worst


# ------------------------------------------------------------------- 17-01

@pytest.mark.parametrize("order", [1, 2])
def test_17_01_begin_context_and_snapshot_guards(order):
    physics = Physics()
    shape = (physics.solid.n_nodes, 2)
    solid_input = np.zeros((2, *shape))
    velocity_input = np.zeros((len(physics.space.points), 2))
    displacement_input = physics.mesh_displacement(physics.fluid.points, 0.0)
    current_input = physics.fluid.points + displacement_input
    velocity_mesh_input = physics.mesh_velocity_field(physics.fluid.points, 0.0)
    alias_state = AcceptedState.initial(solid=solid_input, velocity=velocity_input,
                                        pressure=np.zeros(physics.fluid.n_nodes),
                                        mesh_displacement=displacement_input,
                                        mesh_current=current_input, mesh_velocity=velocity_mesh_input,
                                        dt=DT, order=order)
    reference_alias = capture(alias_state)
    for array in (solid_input, velocity_input, displacement_input, current_input, velocity_mesh_input):
        array.flags.writeable = True
        array[:] = 777.0
    assert_capture_equals(alias_state, reference_alias)
    assert not np.shares_memory(alias_state.mesh_displacement, displacement_input)
    assert not np.shares_memory(alias_state.solid_history[0], solid_input)

    state = physics.initial_state(order)
    reference = capture(state)
    coordinator = StateCoordinator(state)
    plan = physics.plan()
    with pytest.raises(TypeError, match="StepPlan"):
        coordinator.begin(plan=None)
    for bad_dt in (2*DT, 0.0, -DT, np.nan, np.inf, True):
        with pytest.raises(ValueError):
            coordinator.begin(plan=plan, dt=bad_dt)
        assert coordinator.active is None
        assert_capture_equals(coordinator.state, reference)
    wrong_order = 1 if order == 2 else 2
    with pytest.raises(ValueError, match="fixed at order"):
        coordinator.begin(plan=plan, order=wrong_order)
    with pytest.raises(ValueError):
        coordinator.begin(plan=plan, order=3)
    assert coordinator.failures == ()

    tx = coordinator.begin(plan=plan, dt=DT, order=order)
    assert tx.check_context(tx.solid_context, "solid")
    foreign_history = BDFHistory(state.solid_state[0], DT, order=order, time=0.0)
    with pytest.raises(ValueError, match="owner"):
        tx.check_context(foreign_history.context(), "solid")
    with pytest.raises(ValueError, match="owner"):
        tx.check_context(tx.mesh_context, "solid")
    with pytest.raises(ValueError, match="generation"):
        tx.check_context(dataclasses.replace(tx.solid_context, generation=tx.solid_context.generation+1),
                         "solid")
    with pytest.raises(ValueError, match="time"):
        tx.check_context(dataclasses.replace(tx.solid_context, time=tx.solid_context.time+1.0), "solid")
    wrong_order_context = dataclasses.replace(tx.solid_context,
                                              order=2 if tx.solid_context.order == 1 else 1)
    with pytest.raises(ValueError, match="order"):
        tx.check_context(wrong_order_context, "solid")
    with pytest.raises(ValueError, match="field"):
        tx.check_context(tx.solid_context, "temperature")
    with pytest.raises(TypeError):
        tx.check_context(np.zeros(3), "solid")
    tx.abort(reason="guard-only attempt")
    assert tx.status == "aborted"

    # writable alias into evaluate: the candidate is copied before use
    tx = coordinator.begin(plan=plan)
    displacement_alias = physics.mesh_displacement(physics.fluid.points, tx.target_time)
    reference_displacement = np.array(displacement_alias, copy=True)
    tx.evaluate_geometry(displacement=displacement_alias)
    displacement_alias.flags.writeable = True
    displacement_alias[:] = -100.0
    tx.evaluate_solid()
    tx.evaluate_fluid()
    tx.evaluate_interface()
    tx.validate()
    tx.prepare()
    live = tx.publish()
    np.testing.assert_array_equal(live.mesh_displacement, reference_displacement)

    # snapshots: foreign owner, malformed/arbitrary, stale dt/order/layers
    other = StateCoordinator(physics.initial_state(order), config_fingerprint="other")
    with pytest.raises(ValueError, match="Foreign snapshot"):
        coordinator.restore(other.snapshot())
    with pytest.raises(TypeError, match="StateSnapshot"):
        coordinator.restore((state.time, 0, state.solid_history))
    snapshot = coordinator.snapshot()
    for bad in (dataclasses.replace(snapshot, dt=2*DT), dataclasses.replace(snapshot, dt=-1.0),
                dataclasses.replace(snapshot, dt=np.nan), dataclasses.replace(snapshot, order=3),
                dataclasses.replace(snapshot, solid_states=()),
                dataclasses.replace(snapshot, solid_states=(state.solid_state[0],)*3),
                dataclasses.replace(snapshot, pressure=np.zeros((2, 2))),
                dataclasses.replace(snapshot, config_fingerprint="tampered")):
        with pytest.raises(ValueError):
            coordinator.restore(bad)
        assert coordinator.state is live

    # after one more accepted step the old snapshot is stale; a fresh one replays once
    live = run_one(coordinator, physics.plan())[1]
    with pytest.raises(ValueError, match="Stale snapshot"):
        coordinator.restore(snapshot)
    fresh = coordinator.snapshot()
    old_context = tx.solid_context
    restored = coordinator.restore(fresh)
    assert restored is not live
    assert restored.step == live.step and restored.epoch == live.epoch
    np.testing.assert_array_equal(restored.velocity, live.velocity)
    with pytest.raises(ValueError, match="Stale snapshot"):
        coordinator.restore(fresh)
    next_tx = coordinator.begin(plan=physics.plan())
    with pytest.raises(ValueError, match="owner"):
        next_tx.check_context(old_context, "solid")


# ------------------------------------------------------------------- 17-02

@pytest.mark.parametrize("order,mature", STATES)
def test_17_02_geometry_and_solid_failures_roll_back(order, mature):
    physics = Physics()
    coordinator = make_coordinator(physics, order, mature)
    root = coordinator.state
    captured = capture(root)
    solves_before = physics.ledger["solid"]

    for point, stage, pattern in (("displacement_flip", "geometry", "cell"),
                                  ("displacement_quality", "geometry", "quality"),
                                  ("mesh_velocity_raise", "geometry", "mesh velocity")):
        fault = Injector(point)
        tx = coordinator.begin(plan=physics.plan(injector=fault))
        with pytest.raises((ValueError, RuntimeError), match=pattern):
            tx.evaluate_geometry()
        assert fault.fired == 1
        assert_failed(tx, coordinator, root, captured, stage)
        if point == "displacement_flip":
            assert "cell" in coordinator.failures[-1].reason

    fault = Injector("solid_raise")
    tx = coordinator.begin(plan=physics.plan(injector=fault))
    tx.evaluate_geometry()
    with pytest.raises(RuntimeError, match="line-search"):
        tx.evaluate_solid()
    assert fault.fired == 1
    assert_failed(tx, coordinator, root, captured, "solid", "solid")
    assert "line-search" in coordinator.failures[-1].reason
    # the real solid solve ran and produced finite diagnostics before the injected abort
    assert physics.ledger["solid"] == solves_before + 1
    assert np.isfinite(physics.solid_results[-1].nonlinear_history[-1])

    fault = Injector("solid_flip")
    tx = coordinator.begin(plan=physics.plan(injector=fault))
    tx.evaluate_geometry()
    with pytest.raises(ValueError, match="cell"):
        tx.evaluate_solid()
    assert fault.fired == 1
    assert_failed(tx, coordinator, root, captured, "solid", "solid")
    assert "cell" in coordinator.failures[-1].reason


# ------------------------------------------------------------------- 17-03

@pytest.mark.parametrize("order,mature", STATES)
def test_17_03_fluid_and_cache_failures_do_not_leak(order, mature):
    physics = Physics()
    coordinator = make_coordinator(physics, order, mature)
    root = coordinator.state
    captured = capture(root)
    solves_before = physics.ledger["solid"]

    fault = Injector("fluid_raise")
    tx = coordinator.begin(plan=physics.plan(injector=fault))
    tx.evaluate_geometry()
    solid_candidate = tx.evaluate_solid()
    assert solid_candidate is not None and physics.ledger["solid"] == solves_before + 1
    with pytest.raises(RuntimeError, match="after monitor/LU"):
        tx.evaluate_fluid()
    assert fault.fired == 1
    assert len(physics.monitors) == 1 and len(physics.solvers) == 1
    failed_revision = fault.view.revision
    assert failed_revision in physics.cache
    assert_failed(tx, coordinator, root, captured, "fluid", "fluid")

    hits_before = physics.cache_hits
    builds_before = physics.ledger["fluid_operators"]
    tx = coordinator.begin(plan=physics.plan())
    tx.evaluate()
    good_revision = physics.revisions[-1]
    assert good_revision != failed_revision
    assert physics.revisions.count(failed_revision) == 1
    assert physics.cache_hits == hits_before  # no cache entry from the failed attempt was reused
    assert physics.ledger["fluid_operators"] == builds_before + 1
    assert physics.cache[good_revision]["operators"] is not physics.cache[failed_revision]["operators"]
    tx.validate()
    tx.prepare()
    live = tx.publish()
    assert live.step == captured["step"] + 1

    physics = Physics()
    coordinator = make_coordinator(physics, order, mature)
    root = coordinator.state
    captured = capture(root)
    fault = Injector("fluid_iterations")
    tx = coordinator.begin(plan=physics.plan(injector=fault))
    tx.evaluate_geometry()
    tx.evaluate_solid()
    with pytest.raises(RuntimeError, match="did not converge"):
        tx.evaluate_fluid()
    assert fault.fired == 1
    assert "residual=" in coordinator.failures[-1].reason
    assert_failed(tx, coordinator, root, captured, "fluid", "fluid")

    fault = Injector("fluid_singular")
    tx = coordinator.begin(plan=physics.plan(injector=fault))
    tx.evaluate_geometry()
    tx.evaluate_solid()
    with pytest.raises(RuntimeError, match="factorization failed"):
        tx.evaluate_fluid()
    assert fault.fired == 1
    assert_failed(tx, coordinator, root, captured, "fluid", "fluid")


# ------------------------------------------------------------------- 17-04

@pytest.mark.parametrize("point,stage,field,pattern", [
    ("interface_nan", "validate", "fluid_load", "not finite"),
    ("interface_shape", "interface", "solid_load", "shape"),
    ("interface_work", "validate", "interface", "interface consistency"),
    ("fluid_mass", "validate", "mass", "continuity/mass"),
    ("predictor_nan", "validate", "predictor", "finite"),
])
@pytest.mark.parametrize("order,mature", STATES)
def test_17_04_end_validation_discards_whole_candidates(order, mature, point, stage, field, pattern):
    physics = Physics()
    coordinator = make_coordinator(physics, order, mature)
    root = coordinator.state
    captured = capture(root)
    counters_before = dict(physics.ledger)
    fault = Injector(point)
    tx = coordinator.begin(plan=physics.plan(injector=fault))
    with pytest.raises(ValueError, match=pattern):
        tx.evaluate()
        tx.validate()
    assert fault.fired == 1
    # every single-field solve returned successfully before the final gate rejected the step
    assert physics.ledger["solid"] == counters_before["solid"] + 1
    assert physics.ledger["fluid"] == counters_before["fluid"] + 1
    assert physics.ledger["interface"] == counters_before["interface"] + 1
    assert_failed(tx, coordinator, root, captured, stage, field)
    assert len(coordinator.state.monitoring) == len(captured["monitoring"])


# ------------------------------------------------------------------- 17-05

@pytest.mark.parametrize("hook", PREPARE_HOOKS)
@pytest.mark.parametrize("order,mature", STATES)
def test_17_05_prepare_injections_never_reach_the_root(order, mature, hook):
    physics = Physics()
    coordinator = make_coordinator(physics, order, mature)
    root = coordinator.state
    captured = capture(root)
    base_layers = len(captured["solid_history"])
    seen = {}

    def failing(info):
        seen.update(info)
        raise RuntimeError(f"injected prepare failure at {info['hook']}")

    tx = coordinator.begin(plan=physics.plan(hooks={hook: failing}))
    tx.evaluate()
    tx.validate()
    with pytest.raises(RuntimeError, match="injected prepare failure"):
        tx.prepare()
    assert seen["hook"] == hook
    expected_field = "freeze"
    if hook.startswith("after_"):
        expected_field = hook[len("after_"):-len("_accept")]
        assert seen["field"] == expected_field
        # the private accept really happened before the injection fired
        assert seen["generation"] == captured["generations"][expected_field] + 1
        assert seen["time"] == tx.target_time
        # BDFHistory keeps two layers after accept regardless of the BDF order
        assert seen["layers"] == min(2, base_layers + 1)
    assert_failed(tx, coordinator, root, captured, "prepare", expected_field)
    assert len(coordinator.state.monitoring) == len(captured["monitoring"])


@pytest.mark.parametrize("order,mature", STATES)
def test_17_05_serialization_failure_aborts_prepare(order, mature):
    physics = Physics()
    coordinator = make_coordinator(physics, order, mature)
    root = coordinator.state
    captured = capture(root)

    def explode(row):
        raise RuntimeError("injected monitor serialization failure")

    tx = coordinator.begin(plan=physics.plan(serialize=explode))
    tx.evaluate()
    tx.validate()
    with pytest.raises(RuntimeError, match="serialization failure"):
        tx.prepare()
    assert_failed(tx, coordinator, root, captured, "prepare", "serialize")
    assert coordinator.pending_output is None
    assert coordinator.output_records == ()


# ------------------------------------------------------------------- 17-06

@pytest.mark.parametrize("order,mature", STATES)
def test_17_06_publish_commits_once_and_rejects_reentry(order, mature):
    physics = Physics()
    coordinator = make_coordinator(physics, order, mature)
    base = coordinator.state
    captured = capture(base)
    base_layers = len(captured["solid_history"])
    plan = physics.plan()
    tx = coordinator.begin(plan=plan)
    tx.evaluate()
    tx.validate()
    tx.prepare()
    live = tx.publish()
    assert tx.status == "committed"
    assert live.step == base.step + 1 and live.epoch == base.epoch + 1
    assert live.time == tx.target_time and live.time == base.time + DT
    assert dict(live.generations) == {field: captured["generations"][field] + 1 for field in
                                      ("solid", "fluid", "mesh")}
    expected_layers = 1 if order == 1 else min(2, base_layers + 1)
    assert all(len(getattr(live, f"{field}_history")) == expected_layers
               for field in ("solid", "fluid", "mesh"))
    assert len(live.monitoring) == len(base.monitoring) + 1
    assert live.monitoring[-1]["time"] == tx.target_time
    # the accepted candidate is physically non-trivial: no all-zero state can mask a leak
    for value in (live.solid_state[0], live.solid_state[1], live.velocity, live.pressure,
                  live.mesh_displacement, live.mesh_velocity):
        assert float(np.max(np.abs(value))) > 0.0
    assert max(float(np.max(np.abs(value))) for value in live.interface.values()) > 0.0
    with pytest.raises(TransactionError):
        tx.publish()
    with pytest.raises(TransactionError):
        tx.prepare()
    with pytest.raises(TransactionError):
        tx.abort(reason="too late")
    with pytest.raises(TransactionError):
        tx.check_context(tx.solid_context, "solid")
    assert coordinator.state is live
    assert_capture_equals(coordinator.state, capture(live))
    # the old base context is rejected by the next attempt even at the same shape/time
    next_tx = coordinator.begin(plan=physics.plan())
    with pytest.raises(ValueError, match="owner"):
        next_tx.check_context(tx.solid_context, "solid")
    next_tx.abort(reason="context probe only")
    assert coordinator.state is live


def test_17_06_publish_conflict_writes_nothing():
    physics = Physics()
    state = physics.initial_state(2)
    coordinator = StateCoordinator(state)
    tx = coordinator.begin(plan=physics.plan())
    tx.evaluate()
    tx.validate()
    tx.prepare()
    # another writer advances the accepted root to a newer epoch (recovered/parallel writer)
    second = StateCoordinator(coordinator.state)
    _, advanced = run_one(second, physics.plan())
    assert advanced.epoch == state.epoch + 1
    coordinator.load(advanced)
    with pytest.raises(StepConflict, match="epoch"):
        tx.publish()
    assert tx.status == "aborted"
    assert coordinator.state is advanced
    record = coordinator.failures[-1]
    assert record.stage == "publish"
    assert "epoch" in record.reason


# ------------------------------------------------------------------- 17-07

@pytest.mark.parametrize("order,mature", STATES)
def test_17_07_repeated_evaluation_keeps_history_frozen(order, mature):
    physics = Physics()
    coordinator = make_coordinator(physics, order, mature)
    root = coordinator.state
    captured = capture(root)
    counters_before = dict(physics.ledger)
    tx = coordinator.begin(plan=physics.plan())
    tx.evaluate_geometry()
    tx.evaluate_solid()
    tx.evaluate_fluid()
    tx.evaluate_interface()
    solids = [tx.evaluate_solid() for _ in range(3)]
    fluids = [tx.evaluate_fluid() for _ in range(3)]
    assert physics.ledger["solid"] == counters_before["solid"] + 4
    assert physics.ledger["fluid"] == counters_before["fluid"] + 4
    for candidate in solids[1:]:
        np.testing.assert_array_equal(candidate, solids[0])
    for velocity, pressure in fluids[1:]:
        np.testing.assert_array_equal(velocity, fluids[0][0])
        np.testing.assert_array_equal(pressure, fluids[0][1])
    assert_capture_equals(coordinator.state, captured)
    assert coordinator.state is root and tx.status == "open"

    # changed and restored candidate initial values: the accepted history never moves
    n_total = len(physics.space.points)*2 + physics.fluid.n_nodes
    delta = np.zeros(n_total)
    delta[0] = 1e-3
    delta[-1] = -1e-3
    physics.initial_deltas = [delta]
    perturbed_velocity, perturbed_pressure = tx.evaluate_fluid()
    restored_velocity, restored_pressure = tx.evaluate_fluid()
    assert_capture_equals(coordinator.state, captured)
    np.testing.assert_array_equal(restored_velocity, fluids[0][0])
    np.testing.assert_array_equal(restored_pressure, fluids[0][1])
    assert float(np.max(np.abs(perturbed_velocity - restored_velocity))) <= 1e-6

    tx.evaluate_interface()
    tx.validate()
    tx.prepare()
    live = tx.publish()
    assert dict(live.generations) == {field: captured["generations"][field] + 1 for field in
                                      ("solid", "fluid", "mesh")}
    assert len(live.monitoring) == len(captured["monitoring"]) + 1


# ------------------------------------------------------------------- 17-08

FAULT_CASES = {
    "geometry_flip": ("inject_displacement_flip", ValueError, "cell"),
    "geometry_quality": ("inject_displacement_quality", ValueError, "quality"),
    "solid_solver": ("inject_solid_raise", RuntimeError, "line-search"),
    "solid_health": ("inject_solid_flip", ValueError, "cell"),
    "fluid_solver": ("inject_fluid_raise", RuntimeError, "monitor/LU"),
    "fluid_convergence": ("inject_fluid_iterations", RuntimeError, "did not converge"),
    "validate_mass": ("inject_fluid_mass", ValueError, "continuity/mass"),
    "validate_work": ("inject_interface_work", ValueError, "interface consistency"),
    "prepare_solid": ("prepare_solid", RuntimeError, "injected prepare failure"),
    "prepare_fluid": ("prepare_fluid", RuntimeError, "injected prepare failure"),
    "prepare_mesh": ("prepare_mesh", RuntimeError, "injected prepare failure"),
    "prepare_freeze": ("prepare_freeze", RuntimeError, "injected prepare failure"),
    "serialize": ("prepare_serialize", RuntimeError, "serialization failure"),
}


@pytest.mark.parametrize("case", sorted(FAULT_CASES))
@pytest.mark.parametrize("order,mature", STATES)
def test_17_08_failed_retries_match_cold_start(case, order, mature):
    name, exception, pattern = FAULT_CASES[case]
    kwargs = plan_kwargs_with_fault(name)

    baseline_physics = Physics()
    baseline_coordinator = make_coordinator(baseline_physics, order, mature)
    baseline_states = run_steps(baseline_coordinator, baseline_physics.plan(), 3)

    physics = Physics()
    coordinator = make_coordinator(physics, order, mature)
    failed = []
    for _ in range(3):
        tx = coordinator.begin(plan=physics.plan(**kwargs))
        with pytest.raises(exception, match=pattern):
            tx.evaluate()
            tx.validate()
            tx.prepare()
            tx.publish()
        assert tx.status == "aborted"
        failed.append(tx)
    assert len(coordinator.failures) == 3
    assert [record.attempt_id for record in coordinator.failures] == \
        [tx.attempt_id for tx in failed]
    attempt_ids = [record.attempt_id for record in coordinator.failures]
    assert attempt_ids == sorted(attempt_ids) and len(set(attempt_ids)) == 3

    accepted = run_steps(coordinator, physics.plan(), 3)
    assert len(accepted) == len(baseline_states) == 3
    worst = 0.0
    for got, reference in zip(accepted, baseline_states):
        worst = max(worst, assert_states_close(got, reference))
    assert len(coordinator.state.monitoring) == len(baseline_coordinator.state.monitoring)
    for tx in failed:
        with pytest.raises(TransactionError):
            tx.publish()
        with pytest.raises(TransactionError):
            tx.evaluate()
    probe = coordinator.begin(plan=physics.plan())
    with pytest.raises(ValueError, match="owner"):
        probe.check_context(failed[0].solid_context, "solid")
    assert worst <= RETRY_TOLERANCE


# ------------------------------------------------------------------- 17-09

@pytest.mark.parametrize("order,mature", STATES)
def test_17_09_accepted_root_owns_independent_copies(order, mature):
    physics = Physics()
    coordinator = make_coordinator(physics, order, mature)
    tx = coordinator.begin(plan=physics.plan())
    tx.evaluate_geometry()
    solid_candidate = tx.evaluate_solid()
    velocity, pressure = tx.evaluate_fluid()
    tx.evaluate_interface()
    source_velocity, source_pressure = physics.fluid_views
    # the solver really returns u/p as views on one mixed buffer (disjoint ranges)
    assert root_base(source_velocity) is root_base(source_pressure)
    solution = physics.fluid_solutions[-1]
    reference_velocity = np.array(solution.velocity, copy=True)
    reference_pressure = np.array(solution.pressure, copy=True)

    # candidates returned by evaluate_* are copies: mutating them cannot reach the workspace
    velocity.flags.writeable = True
    velocity[:] = 999.0
    pressure.flags.writeable = True
    pressure[:] = -999.0
    solid_candidate.flags.writeable = True
    solid_candidate[:] = 555.0

    tx.validate()
    tx.prepare()
    live = tx.publish()
    source_velocity.flags.writeable = True
    source_velocity[:] = 5.0
    source_pressure.flags.writeable = True
    source_pressure[:] = -5.0
    np.testing.assert_array_equal(live.velocity, reference_velocity)
    np.testing.assert_array_equal(live.pressure, reference_pressure)
    assert not np.shares_memory(live.velocity, live.pressure)
    assert not np.shares_memory(live.velocity, source_velocity)
    assert not np.shares_memory(live.velocity, live.solid_state[0])
    assert not np.shares_memory(live.mesh_displacement, live.mesh_current)
    assert not np.shares_memory(live.mesh_displacement, live.mesh_velocity)
    if len(live.solid_history) == 2:
        assert not np.shares_memory(live.solid_history[0], live.solid_history[1])
        assert not np.shares_memory(live.fluid_history[0], live.fluid_history[1])

    references = {}
    for name in ("solid_state", "velocity", "pressure", "mesh_displacement", "mesh_current",
                 "mesh_velocity"):
        references[name] = np.array(getattr(live, name), copy=True)
    interface_reference = {key: np.array(value, copy=True) for key, value in live.interface.items()}
    predictor_reference = {key: np.array(value, copy=True) for key, value in live.predictor.items()}

    for name, reference in references.items():
        exposed = getattr(live, name)
        exposed.flags.writeable = True
        exposed[:] = 123.0
        np.testing.assert_array_equal(getattr(live, name), reference)
    for key, reference in interface_reference.items():
        exposed = live.interface[key]
        exposed.flags.writeable = True
        exposed[:] = 321.0
        np.testing.assert_array_equal(live.interface[key], reference)
    for key, reference in predictor_reference.items():
        exposed = live.predictor[key]
        exposed.flags.writeable = True
        exposed[:] = 456.0
        np.testing.assert_array_equal(live.predictor[key], reference)
    rows = live.monitoring
    expected_step = live.step
    rows[-1]["step"] = -1
    assert live.monitoring[-1]["step"] == expected_step
    snapshot = coordinator.snapshot()
    for array in (snapshot.solid_states[0], snapshot.pressure, snapshot.mesh_current):
        array.flags.writeable = True
        array[:] = 7.0
    np.testing.assert_array_equal(live.velocity, references["velocity"])
    np.testing.assert_array_equal(live.solid_state[0], references["solid_state"][0])
    assert live.velocity.dtype == np.float64
    assert dict(live.shapes)["fluid"] == references["velocity"].shape


# ------------------------------------------------------------------- 17-10

@pytest.mark.parametrize("order,mature", STATES)
def test_17_10_output_pending_and_idempotent_flush(order, mature):
    physics = Physics()
    coordinator = make_coordinator(physics, order, mature)
    base = coordinator.state
    captured = capture(base)

    def explode(row):
        raise RuntimeError("disk encode failed")

    tx = coordinator.begin(plan=physics.plan(serialize=explode))
    tx.evaluate()
    tx.validate()
    with pytest.raises(RuntimeError, match="disk encode failed"):
        tx.prepare()
    assert_failed(tx, coordinator, base, captured, "prepare", "serialize")
    assert coordinator.pending_output is None and coordinator.output_records == ()

    writes = []

    def failing_writer(payload):
        writes.append(payload)
        raise OSError("disk full")

    plan = physics.plan(serialize=lambda row: json.dumps(row, sort_keys=True))
    tx = coordinator.begin(plan=plan)
    tx.evaluate()
    tx.validate()
    tx.prepare()
    live = tx.publish()
    ledger_before = dict(physics.ledger)
    assert tx.write_output(failing_writer) is False
    assert len(writes) == 1 and tx.output_status == "pending"
    pending = coordinator.pending_output
    assert pending is not None and pending.epoch == live.epoch and "disk full" in pending.reason
    assert coordinator.state is live and live.step == base.step + 1 and live.epoch == base.epoch + 1
    assert dict(physics.ledger) == ledger_before  # no re-solve, no re-accept

    records = []

    def good_writer(payload):
        records.append(payload)

    assert coordinator.flush_output(good_writer) is True
    assert len(records) == 1 and records[0] == pending.payload
    assert coordinator.pending_output is None and tx.output_status == "written"
    assert len(coordinator.output_records) == 1
    assert coordinator.output_records[0].epoch == live.epoch
    assert coordinator.output_records[0].step == live.step
    assert json.loads(records[0])["step"] == live.step
    # one success record per accepted epoch: duplicates never write again
    assert tx.write_output(good_writer) is False
    assert len(records) == 1 and len(coordinator.output_records) == 1
    assert coordinator.flush_output(good_writer) is False
    assert dict(physics.ledger) == ledger_before

    # the next accepted epoch gets its own record and does not touch the first one
    second = run_one(coordinator, physics.plan(serialize=lambda row: json.dumps(row, sort_keys=True)))[1]
    assert second.epoch == live.epoch + 1
    assert coordinator.write_output(second.epoch, good_writer,
                                    payload=json.dumps(second.monitoring[-1], sort_keys=True)) is True
    assert len(records) == 2 and len(coordinator.output_records) == 2
    assert coordinator.output_records[0].payload == pending.payload

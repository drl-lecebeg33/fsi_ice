"""T24 genuine FE multi-step consistency, history acceptance and failure retry."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import fsi_ice.validation.fsi2 as fsi2_module
import fsi_ice.coupling.transient as transient_module

from fsi_ice.ale import build_harmonic_extension
from fsi_ice.coupling.newton import NewtonOptions
from fsi_ice.coupling.transient import TransientFSI
from fsi_ice.fem.p2 import P2Space
from fsi_ice.interface import build_matching_interface
from fsi_ice.mesh import TriMesh, rectangle_mesh
from fsi_ice.solid.linear import ElasticMaterial
from fsi_ice.validation.fsi2 import (analyze_fsi2, cycle_statistics, inflow_factor,
                                     load_fsi2_config, run_fsi2_cli, validate_fsi2_config)

ROOT = Path(__file__).resolve().parents[1]
COUPLING = {"absolute_tolerance": 1e-14, "relative_tolerance": 1e-12, "max_iterations": 100, "relaxation": .5}
NEWTON = NewtonOptions(atol=1e-12, rtol=1e-12, max_iterations=30)


def small_problem(order=2):
    fm = rectangle_mesh(2, 2, width=1., height=.8)
    sm = rectangle_mesh(2, 2, width=1., height=.3)
    ft = dict(fm.boundaries)
    ft["fluid-interface"] = ft.pop("bottom")
    st = dict(sm.boundaries)
    st["solid-interface"] = st.pop("top")
    fluid = TriMesh(fm.points, fm.cells, ft)
    solid = TriMesh(sm.points-[0., .3], sm.cells, st)
    topology = build_matching_interface(solid, fluid, P2Space.from_mesh(fluid),
        solid_tag="solid-interface", fluid_tag="fluid-interface", atol=1e-12, rtol=0.)

    def velocity(time, points):
        result = np.zeros_like(points)
        left = np.isclose(points[:, 0], 0., atol=1e-12)
        result[left, 0] = .03*(1-np.cos(time))*points[left, 1]*(.8-points[left, 1])
        return result

    return TransientFSI(topology, build_harmonic_extension(fluid), ElasticMaterial(1000., .3, density=10.),
        density=1., viscosity=.1, boundary_velocity=velocity, fluid_tags=["left", "top"],
        solid_nodes=solid.boundary_nodes(["bottom", "left", "right"]), dt=.05, order=order)


@pytest.mark.parametrize("order", [1, 2])
def test_real_fe_four_methods_three_steps_same_equations(order):
    drivers = {method: small_problem(order) for method in ("fixed", "aitken", "iqn", "monolithic")}
    previous = {method: driver.coordinator.state for method, driver in drivers.items()}
    for index in range(3):
        states = {method: driver.step(method, coupling={**COUPLING,
                  "iqn_initial_relaxation": .5, "iqn_restart_tolerance": 1., "iqn_history_size": 10}, newton=NEWTON)
                  for method, driver in drivers.items()}
        baseline = states["monolithic"]
        for method, state in states.items():
            driver = drivers[method]
            actual = driver._initial_vector(state)
            expected = drivers["monolithic"]._initial_vector(baseline)
            threshold = 1e-8*np.maximum(np.abs(actual), np.abs(expected))+1e-10*driver.scales
            assert np.all(np.abs(actual-expected) <= threshold), np.max(np.abs(actual-expected)/threshold)
            assert state.step == index+1 and state.epoch == index+1
            assert set(state.generations.values()) == {index+1}
            assert len(state.monitoring) == index+1
            assert state.monitoring[-1]["order"] == (1 if index == 0 else order)
            assert state.monitoring[-1]["residual"] <= 1e-7
            if method == "iqn":
                assert state.monitoring[-1]["iqn_updates"] > 0
            # The velocity actually accepted is the BDF derivative of displacement.
            # Older layer for BDF2 comes from the previous accepted state.
            if index == 0 or order == 1:
                expected = (state.solid_state[0]-previous[method].solid_state[0])/.05
            else:
                expected = (1.5*state.solid_state[0]-2*previous[method].solid_history[0][0]+
                            .5*previous[method].solid_history[1][0])/.05
            np.testing.assert_allclose(state.solid_state[1], expected, atol=1e-11)
        previous = states


def test_step_passes_gate_derived_inner_absolute_tolerances(monkeypatch):
    driver = small_problem()
    recorded = []
    original_flow, original_solid = transient_module.solve_flow_step, transient_module.solve_solid_step

    def flow_step(*args, **kwargs):
        recorded.append(("flow", kwargs.get("absolute_tolerance")))
        return original_flow(*args, **kwargs)

    def solid_step(*args, **kwargs):
        recorded.append(("solid", kwargs.get("absolute_tolerance")))
        return original_solid(*args, **kwargs)

    monkeypatch.setattr(transient_module, "solve_flow_step", flow_step)
    monkeypatch.setattr(transient_module, "solve_solid_step", solid_step)
    residual_max = 1e-7
    driver.step("iqn", coupling={**COUPLING, "iqn_initial_relaxation": .5,
                                 "iqn_restart_tolerance": 1., "iqn_history_size": 10},
                newton=NEWTON, residual_max=residual_max)
    assert {kind for kind, _ in recorded} == {"flow", "solid"}
    fluid_scale = float(np.max(driver.scales[driver.layout.slice_of("u_f")]))
    solid_scale = float(np.max(driver.scales[driver.layout.slice_of("v_s")]))
    for kind, value in recorded:
        expected = .1*residual_max*(fluid_scale if kind == "flow" else solid_scale)
        assert value == pytest.approx(expected)


@pytest.mark.parametrize("method", ["aitken", "iqn", "monolithic"])
@pytest.mark.parametrize("failure", ["solve", "validate", "prepare"])
def test_transaction_failure_retry_matches_baseline(method, failure):
    driver, baseline = small_problem(), small_problem()
    driver.step(method, coupling=COUPLING, newton=NEWTON)
    baseline.step(method, coupling=COUPLING, newton=NEWTON)
    before = driver.coordinator.state
    fired = []

    def fail(*args):
        fired.append(True)
        raise RuntimeError("injected t24 failure")

    kwargs = {"coupling": COUPLING, "newton": NEWTON}
    if failure == "solve":
        original = driver._context
        driver._context = fail
    elif failure == "validate":
        kwargs["checks"] = {"injected": fail}
    else:
        kwargs["hooks"] = {"before_freeze": fail}
    with pytest.raises(RuntimeError, match="injected"):
        driver.step(method, **kwargs)
    assert fired and driver.coordinator.state is before
    assert driver.coordinator.active is None and len(driver.coordinator.failures) == 1
    assert before.step == 1 and len(before.monitoring) == 1
    if failure == "solve":
        driver._context = original
    actual = driver.step(method, coupling=COUPLING, newton=NEWTON)
    expected = baseline.step(method, coupling=COUPLING, newton=NEWTON)
    np.testing.assert_array_equal(driver._initial_vector(actual), baseline._initial_vector(expected))


def test_inlet_start_and_derivative_are_continuous():
    assert inflow_factor(0) == 0 and inflow_factor(1) == pytest.approx(.5)
    assert inflow_factor(2) == 1 and inflow_factor(5) == 1
    assert 0 <= 1-inflow_factor(2-1e-6) < 1e-11
    for time in (-1, np.nan, np.inf):
        with pytest.raises(ValueError):
            inflow_factor(time)


@pytest.mark.parametrize("key,value", [("dt", 0), ("dt", True), ("dt", np.nan),
    ("end_time", .015), ("order", True), ("method", "unknown"), ("mesh_size", "fine")])
def test_invalid_configuration(key, value):
    config = load_fsi2_config(ROOT/"cases/fsi2.json")
    config[key] = value
    with pytest.raises(ValueError):
        validate_fsi2_config(config)


def test_standard_materials_and_gates_cannot_be_changed_silently():
    config = load_fsi2_config(ROOT/"cases/fsi2.json")
    modified = copy.deepcopy(config)
    modified["solid"]["density"] = 1000
    with pytest.raises(ValueError, match="Standard FSI2"):
        validate_fsi2_config(modified)
    config["acceptance"]["frequency_relative_max"] = .1
    with pytest.raises(ValueError, match="weakened"):
        validate_fsi2_config(config)


@pytest.mark.parametrize("key,value", [("iqn_initial_relaxation", 0),
    ("iqn_alpha_max", np.inf), ("iqn_history_size", -1), ("iqn_history_size", True),
    ("iqn_history_size", 1.5), ("iqn_restart_tolerance", np.nan), ("unknown", 1.)])
def test_invalid_iqn_configuration(key, value):
    config = load_fsi2_config(ROOT/"cases/fsi2.json")
    config["method"] = "iqn"
    config["coupling"][key] = value
    with pytest.raises(ValueError):
        validate_fsi2_config(config)


def test_old_fsi2_configuration_without_iqn_controls_remains_valid():
    config = load_fsi2_config(ROOT/"cases/fsi2.json")
    for key in tuple(config["coupling"]):
        if key.startswith("iqn_"):
            del config["coupling"][key]
    validate_fsi2_config(config)


def test_iqn_failure_mid_iteration_discards_secants(monkeypatch):
    import fsi_ice.coupling.transient as transient
    driver, baseline = small_problem(), small_problem()
    coupling = {**COUPLING, "iqn_initial_relaxation": .5, "iqn_restart_tolerance": 1.}
    driver.step("iqn", coupling=coupling, newton=NEWTON)
    baseline.step("iqn", coupling=coupling, newton=NEWTON)
    before = driver.coordinator.state
    original, calls = transient.solve_flow_step, []

    def fail_third(*args, **kwargs):
        calls.append(True)
        if len(calls) == 3:
            raise RuntimeError("injected IQN field failure with populated secants")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(transient, "solve_flow_step", fail_third)
        with pytest.raises(RuntimeError, match="populated secants"):
            driver.step("iqn", coupling=coupling, newton=NEWTON)
    assert len(calls) == 3
    assert driver.coordinator.state is before and driver.coordinator.active is None
    actual = driver.step("iqn", coupling=coupling, newton=NEWTON)
    expected = baseline.step("iqn", coupling=coupling, newton=NEWTON)
    np.testing.assert_array_equal(driver._initial_vector(actual), baseline._initial_vector(expected))
    assert actual.monitoring[-1]["coupling_iterations"] == expected.monitoring[-1]["coupling_iterations"]


def test_periodic_analysis_is_time_weighted_and_short_runs_incomplete():
    config = load_fsi2_config(ROOT/"cases/fsi2.json")
    t = np.linspace(2., 8., 1201)
    rows = [{"time": float(time), "tip_y": .0816*np.sin(2*np.pi*1.93*time),
             "drag": 215.06+77.65*np.sin(4*np.pi*1.93*time),
             "lift": .61+237.8*np.sin(2*np.pi*1.93*time)} for time in t]
    assert analyze_fsi2(rows, config)["status"] == "PASS"
    assert analyze_fsi2(rows[:10], config)["status"] == "INCOMPLETE"
    growing = [dict(row, tip_y=row["tip_y"]*(1+row["time"]/10)) for row in rows]
    assert analyze_fsi2(growing, config)["status"] == "INCOMPLETE"
    wrong = [dict(row, tip_y=.04*np.sin(2*np.pi*1.93*row["time"])) for row in rows]
    assert analyze_fsi2(wrong, config)["status"] == "FAIL"
    stats = cycle_statistics(t, 2+np.sin(2*np.pi*2*t)+.2*np.cos(4*np.pi*2*t), prominence=.1)
    assert stats["mean"] == pytest.approx(2, abs=1e-4)


def test_input_failure_clears_old_success(tmp_path):
    (tmp_path/"report.json").write_text('{"status":"PASS"}')
    (tmp_path/"solution.npz").write_bytes(b"stale")
    with pytest.raises(FileNotFoundError):
        run_fsi2_cli(tmp_path/"missing.json", tmp_path, progress=None)
    assert not (tmp_path/"report.json").exists() and not (tmp_path/"solution.npz").exists()
    assert json.loads((tmp_path/"failure.json").read_text())["stage"] == "input"


@pytest.mark.parametrize("failure", [None, "setup", "monitor_output", "output"])
@pytest.mark.parametrize("method", ["aitken", "iqn"])
def test_cli_driver_outputs_short_status_and_durable_failure(tmp_path, monkeypatch, failure, method):
    config = load_fsi2_config(ROOT/"cases/fsi2.json")
    config.update(dt=.05, end_time=.15)
    path = tmp_path/"config.json"
    path.write_text(json.dumps(config))
    driver = small_problem()

    def fail(*args, **kwargs):
        raise OSError("injected output/setup failure")

    def build(config, output):
        (output/"mesh").mkdir()
        (output/"mesh"/"fsi1.msh").write_bytes(b"test mesh; not benchmark geometry")
        return driver

    monkeypatch.setattr(fsi2_module, "build_fsi2_problem", fail if failure == "setup" else build)
    if failure == "monitor_output":
        monkeypatch.setattr(fsi2_module, "write_csv", fail)
    if failure == "output":
        monkeypatch.setattr(fsi2_module, "_write_final_solution", fail)
    if failure:
        with pytest.raises(OSError, match="injected"):
            run_fsi2_cli(path, tmp_path, method=method, progress=None)
        record = json.loads((tmp_path/"failure.json").read_text())
        assert record["stage"] == failure
        assert record["accepted_steps"] == {"setup": 0, "monitor_output": 1, "output": 3}[failure]
        assert not (tmp_path/"report.json").exists() and not (tmp_path/"solution.npz").exists()
    else:
        report = run_fsi2_cli(path, tmp_path, method=method, progress=None)
        assert report["status"] == "INCOMPLETE" and report["accepted_steps"] == 3
        assert not report["checks"]["periodic_validation_complete"]
        assert (tmp_path/"monitor.csv").exists() and (tmp_path/"solution.npz").exists()


def test_invalid_override_records_input_failure_without_nonfinite_json(tmp_path):
    with pytest.raises(ValueError):
        run_fsi2_cli(ROOT/"cases/fsi2.json", tmp_path, end_time=np.nan, progress=None)
    record = json.loads((tmp_path/"failure.json").read_text())
    assert record["stage"] == "input" and record["config"] is None


def test_cli_accepts_iqn_and_preserves_incomplete_exit_code(tmp_path, monkeypatch):
    from fsi_ice.cli import main
    calls = []

    def run(config, output, **kwargs):
        calls.append((config, output, kwargs))
        return {"status": "INCOMPLETE"}

    monkeypatch.setattr(fsi2_module, "run_fsi2_cli", run)
    result = main(["fsi2", "--config", str(ROOT/"cases/fsi2.json"), "--output", str(tmp_path),
                   "--method", "iqn", "--mesh-size", "0.03", "--end-time", "0.03"])
    assert result == 3
    assert calls == [(ROOT/"cases/fsi2.json", tmp_path,
                      {"method": "iqn", "mesh_size": .03, "end_time": .03})]


@pytest.mark.parametrize("kwargs", [{"residual_max": np.nan}, {"minimum_quality": 0},
    {"checks": {"coupled_residual": lambda view: True}}])
def test_invalid_physics_gates_cannot_accept_a_step(kwargs):
    driver = small_problem()
    with pytest.raises(ValueError):
        driver.step(**kwargs)
    assert driver.coordinator.state.step == 0 and driver.coordinator.active is None

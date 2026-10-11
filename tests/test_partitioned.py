"""T18 numerical correctness, failure diagnostics, and fixed accepted histories."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fsi_ice.coupling import (CouplingFailure, CouplingOptions, DirichletNeumannMap,
                              PartitionedEvaluation, solve_dirichlet_neumann)
from fsi_ice.validation.partitioned import (LinearCoupledProblem, load_partitioned_config,
                                           run_partitioned_check, validate_config)

ROOT = Path(__file__).resolve().parents[1]


def evaluation(target):
    return PartitionedEvaluation(np.asarray(target), None, None, None)


@pytest.mark.parametrize("omega", [0.25, 0.5, 1.0])
def test_multicomponent_fixed_point_matches_analytic_direct_solution(omega):
    matrix = np.array([[0.25, -0.2], [0.1, 0.3]])
    rhs = np.array([0.7, -0.4])
    expected = np.linalg.solve(np.eye(2)-matrix, rhs)
    initial = np.array([0.1, 0.2])
    scales = np.array([0.2, 2.0])
    old_initial, old_scales = initial.copy(), scales.copy()
    result = solve_dirichlet_neumann(lambda x: evaluation(matrix @ x+rhs), initial, scales=scales,
                                    options=CouplingOptions(relaxation=omega, max_iterations=200,
                                                            absolute_tolerance=1e-12, relative_tolerance=1e-11))
    assert np.max(np.abs(result.interface_values-expected)/scales) < 1e-9
    assert result.history[-1].residual_norm <= result.history[-1].threshold
    np.testing.assert_array_equal(initial, old_initial)
    np.testing.assert_array_equal(scales, old_scales)
    for row in result.history:
        assert row.relaxation == omega
        assert row.residual_norm == np.max(np.abs(row.residual)/scales)


def test_component_unit_changes_preserve_iterations_and_physical_solution():
    matrix = np.array([[0.3, 0.1], [-0.1, 0.2]])
    rhs = np.array([0.4, -0.7])
    scales = np.array([0.2, 3.0])
    convert = np.array([1e3, 1e-3])
    original = solve_dirichlet_neumann(lambda x: evaluation(matrix @ x+rhs), np.zeros(2), scales=scales)
    converted = solve_dirichlet_neumann(lambda x: evaluation(convert*(matrix @ (x/convert)+rhs)),
                                        np.zeros(2), scales=convert*scales)
    np.testing.assert_allclose(converted.interface_values/convert, original.interface_values, atol=1e-14)
    np.testing.assert_allclose([row.residual_norm for row in converted.history],
                               [row.residual_norm for row in original.history], atol=1e-14)


def test_order_and_returned_fields_belong_to_last_evaluated_input():
    calls = []
    def fluid(x):
        calls.append(("fluid", float(x[0])))
        return float(x[0])
    def load(x):
        calls.append(("load", x))
        return 0.25*x+0.3
    def solid(x):
        calls.append(("solid", x))
        return np.array([x])
    def trace(x):
        calls.append(("trace", float(x[0])))
        return x
    adapter = DirichletNeumannMap(fluid, load, solid, trace)
    result = solve_dirichlet_neumann(adapter, [0], scales=[1], options=CouplingOptions(absolute_tolerance=1e-7))
    assert [name for name, _ in calls] == ["fluid", "load", "solid", "trace"]*len(result.history)
    assert result.evaluation.fluid == result.interface_values[0] == calls[-4][1]
    np.testing.assert_array_equal(result.evaluation.target, result.evaluation.solid)
    assert not np.array_equal(result.interface_values, result.evaluation.target)
    np.testing.assert_array_equal(result.history[-1].residual, result.evaluation.target-result.interface_values)


def test_already_converged_input_is_evaluated_once_and_returned_without_update():
    result = solve_dirichlet_neumann(lambda x: evaluation(x), [2.0], scales=[1.0],
                                    options=CouplingOptions(max_iterations=1))
    assert len(result.history) == 1 and result.reason == "interface_residual"
    np.testing.assert_array_equal(result.interface_values, [2.0])


def test_mutating_a_callback_private_input_cannot_corrupt_controller_state():
    def evaluate(x):
        x.setflags(write=True)
        x[:] = 100.0
        return evaluation([1.0])
    initial = np.array([0.0])
    result = solve_dirichlet_neumann(evaluate, initial, scales=[1],
                                    options=CouplingOptions(relaxation=1, max_iterations=2))
    np.testing.assert_array_equal(initial, [0])
    np.testing.assert_array_equal(result.interface_values, [1])
    assert result.history[0].residual_norm == 1


def test_tiny_relaxed_update_cannot_claim_convergence_and_failure_keeps_last_evaluation():
    inputs = []
    def evaluate(x):
        inputs.append(x.copy())
        return evaluation(x+1)
    options = CouplingOptions(relaxation=1e-14, absolute_tolerance=1e-8,
                              relative_tolerance=0, max_iterations=3, case_name="tiny-omega")
    with pytest.raises(CouplingFailure) as caught:
        solve_dirichlet_neumann(evaluate, [0], scales=[1], options=options)
    error = caught.value
    assert error.stage == "convergence" and error.iteration == 3 and len(error.history) == 3
    assert error.history[-1].residual_norm > 0.99
    np.testing.assert_array_equal(error.interface_values, inputs[-1])
    assert "tiny-omega" in str(error) and "maximum iterations" in str(error)


def test_divergence_is_reported_with_full_residual_history():
    with pytest.raises(CouplingFailure) as caught:
        solve_dirichlet_neumann(lambda x: evaluation(2*x+1), [0], scales=[1],
                               options=CouplingOptions(relaxation=1, max_iterations=4))
    np.testing.assert_array_equal([row.residual_norm for row in caught.value.history], [1, 2, 4, 8])


@pytest.mark.parametrize("stage", ["fluid", "load", "solid", "trace"])
def test_each_single_field_stage_reports_the_original_cause(stage):
    functions = {name: (lambda x: x) for name in ("fluid", "load", "solid", "trace")}
    def fail(_):
        raise RuntimeError("injected singular field")
    functions[stage] = fail
    adapter = DirichletNeumannMap(*(functions[name] for name in ("fluid", "load", "solid", "trace")))
    with pytest.raises(CouplingFailure) as caught:
        solve_dirichlet_neumann(adapter, [0], scales=[1])
    assert caught.value.stage == stage and caught.value.iteration == 1
    assert "injected singular field" in str(caught.value) and caught.value.__cause__ is not None


@pytest.mark.parametrize("bad", [np.array([np.nan]), np.array([np.inf]), np.array([1, 2]), np.array([[1]])])
def test_invalid_candidate_output_is_a_terminal_residual_failure(bad):
    with pytest.raises(CouplingFailure) as caught:
        solve_dirichlet_neumann(lambda x: evaluation(bad), [0], scales=[1])
    assert caught.value.stage == "residual" and caught.value.history == ()


def test_scaled_overflow_and_untyped_evaluation_fail_explicitly():
    with pytest.raises(CouplingFailure, match="overflow"):
        solve_dirichlet_neumann(lambda x: evaluation([1e308]), [0], scales=[1e-308])
    with pytest.raises(CouplingFailure, match="PartitionedEvaluation"):
        solve_dirichlet_neumann(lambda x: np.ones(1), [0], scales=[1])


def test_failure_on_a_later_iteration_retains_prior_history_and_latest_input():
    seen = []
    def evaluate(x):
        seen.append(x.copy())
        if len(seen) == 3:
            raise RuntimeError("late failure")
        return evaluation(x+1)
    with pytest.raises(CouplingFailure) as caught:
        solve_dirichlet_neumann(evaluate, [0], scales=[1])
    assert caught.value.iteration == 3 and len(caught.value.history) == 2
    np.testing.assert_array_equal(caught.value.interface_values, seen[-1])


@pytest.mark.parametrize("change", [{"relaxation": 0}, {"relaxation": 1.1}, {"relaxation": True},
                                    {"relaxation": float("nan")}, {"absolute_tolerance": -1},
                                    {"relative_tolerance": float("inf")}, {"max_iterations": 0},
                                    {"max_iterations": True}, {"max_iterations": 1.5}, {"case_name": ""},
                                    {"absolute_tolerance": 0, "relative_tolerance": 0}])
def test_invalid_options_are_rejected(change):
    with pytest.raises(ValueError):
        CouplingOptions(**change)


@pytest.mark.parametrize("initial,scales", [([], []), ([0], [0]), ([0], [-1]), ([0], [np.nan]),
                                           ([np.inf], [1]), ([0, 0], [1])])
def test_invalid_state_and_scale_inputs_are_rejected(initial, scales):
    with pytest.raises(ValueError):
        solve_dirichlet_neumann(lambda x: evaluation(x), initial, scales=scales)


def test_callback_and_option_types_are_validated():
    with pytest.raises(TypeError):
        solve_dirichlet_neumann(None, [0], scales=[1])
    with pytest.raises(TypeError):
        solve_dirichlet_neumann(lambda x: evaluation(x), [0], scales=[1], options={})
    with pytest.raises(TypeError):
        DirichletNeumannMap(None, lambda x: x, lambda x: x, lambda x: x)


@pytest.fixture
def config():
    return load_partitioned_config(ROOT / "cases/partitioned_check.json")


@pytest.mark.parametrize("omega", [0.25, 0.5, 1.0])
def test_real_fe_fields_match_block_direct_solution_and_preserve_bdf_history(config, omega):
    problem = LinearCoupledProblem(config)
    snapshot = problem.history_snapshot()
    options = CouplingOptions(**{**config["coupling"], "relaxation": omega})
    result = problem.solve(options=options)
    metrics = problem.metrics(result, problem.direct())
    for name in ("displacement", "velocity", "pressure", "load"):
        assert metrics[name+"_difference"] <= config["acceptance"]["direct_difference_max"]
    assert metrics["combined_residual"] <= config["acceptance"]["residual_max"]
    assert metrics["motion_error"] <= config["acceptance"]["motion_error_max"]
    assert metrics["continuity"] <= config["acceptance"]["continuity_max"]
    assert metrics["solid_residual"] < 1e-10 and metrics["fluid_residual"] < 1e-10
    assert np.linalg.norm(result.evaluation.load.f_s_gamma) > 1e-4
    assert np.linalg.norm(result.evaluation.solid.displacement) > 1e-6
    assert problem.histories_equal(snapshot)
    repeat = problem.solve(options=options)
    np.testing.assert_array_equal(repeat.evaluation.fluid.velocity, result.evaluation.fluid.velocity)
    np.testing.assert_array_equal(repeat.evaluation.solid.displacement, result.evaluation.solid.displacement)
    assert problem.histories_equal(snapshot)


def test_real_fe_failure_and_retry_do_not_accept_or_contaminate_history(config):
    problem = LinearCoupledProblem(config)
    snapshot = problem.history_snapshot()
    with pytest.raises(CouplingFailure, match="maximum iterations"):
        problem.solve(options=CouplingOptions(max_iterations=1))
    assert problem.histories_equal(snapshot)
    adapter = problem.adapter
    fired = []
    def fail_after_field_solve(x):
        adapter.fluid_solve(x)
        fired.append(True)
        raise RuntimeError("injected failure after real fluid solve")
    problem.adapter = replace(adapter, fluid_solve=fail_after_field_solve)
    with pytest.raises(CouplingFailure, match="injected failure") as caught:
        problem.solve()
    assert caught.value.stage == "fluid" and fired and problem.histories_equal(snapshot)
    problem.adapter = adapter
    retry = problem.solve()
    cold = LinearCoupledProblem(config).solve()
    for a, b in ((retry.interface_values, cold.interface_values),
                 (retry.evaluation.solid.displacement, cold.evaluation.solid.displacement),
                 (retry.evaluation.fluid.velocity, cold.evaluation.fluid.velocity),
                 (retry.evaluation.fluid.pressure, cold.evaluation.fluid.pressure),
                 (retry.evaluation.load.f_s_gamma, cold.evaluation.load.f_s_gamma)):
        np.testing.assert_array_equal(a, b)
    assert problem.histories_equal(snapshot)


def test_validation_artifacts_and_failed_rerun_remove_success_markers(config, tmp_path):
    report = run_partitioned_check(config, tmp_path)
    assert report["status"] == "PASS" and all(report["checks"].values())
    assert (tmp_path / "iterations.csv").is_file()
    with np.load(tmp_path / "solution.npz") as data:
        assert len(data["interface_residuals"]) == report["results"]["iterations"]
    config["coupling"]["max_iterations"] = 1
    with pytest.raises(CouplingFailure):
        run_partitioned_check(config, tmp_path)
    assert not (tmp_path / "report.json").exists() and not (tmp_path / "solution.npz").exists()
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["stage"] == "convergence" and failure["histories_unchanged"]


def test_reference_validation_failure_also_saves_failure_evidence(config, tmp_path, monkeypatch):
    def fail(_):
        raise RuntimeError("injected direct solve failure")
    monkeypatch.setattr(LinearCoupledProblem, "direct", fail)
    with pytest.raises(RuntimeError, match="injected direct"):
        run_partitioned_check(config, tmp_path)
    evidence = json.loads((tmp_path / "failure.json").read_text())
    assert evidence["stage"] == "direct_reference" and evidence["histories_unchanged"]
    assert evidence["code"]["commit"] and evidence["environment"]["python"]


@pytest.mark.parametrize("field,value", [("dt", 0), ("mesh", [True, 2]), ("mesh", [1, 2])])
def test_validation_config_rejects_invalid_time_and_mesh(config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("group,field,value", [("scales", "load", 0), ("fluid", "density", -1),
                                              ("solid", "poisson", 0.5), ("solid", "young", "bad"),
                                              ("acceptance", "residual_max", float("nan"))])
def test_validation_config_rejects_invalid_physics_and_thresholds(config, group, field, value):
    config[group][field] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_partitioned_cli_runs_from_another_directory_and_reports_failed_case(config, tmp_path):
    command = [sys.executable, "-m", "fsi_ice", "partitioned-check", "--config",
               str(ROOT / "cases/partitioned_check.json"), "--output", "output"]
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "output/report.json").read_text())["status"] == "PASS"
    config["coupling"]["max_iterations"] = 1
    path = tmp_path / "fail.json"
    path.write_text(json.dumps(config))
    command[5] = str(path)
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 2 and "maximum iterations" in result.stderr
    assert "Traceback" not in result.stderr and (tmp_path / "output/failure.json").is_file()
    assert not (tmp_path / "output/report.json").exists()


def test_partitioned_cli_returns_one_when_computed_solution_fails_acceptance(config, tmp_path):
    config["acceptance"]["direct_difference_max"] = 1e-20
    path = tmp_path / "strict.json"
    path.write_text(json.dumps(config))
    result = subprocess.run([sys.executable, "-m", "fsi_ice", "partitioned-check", "--config", str(path),
                             "--output", str(tmp_path / "output")], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 1 and "FAIL" in result.stdout
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert report["status"] == "FAIL" and not all(report["checks"].values())

"""T19 scaled Aitken correctness, bounds, failures and unchanged FE physics."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fsi_ice.coupling import (CouplingFailure, CouplingIteration, CouplingOptions,
                              PartitionedEvaluation, solve_dirichlet_neumann)
from fsi_ice.validation.partitioned import LinearCoupledProblem, load_partitioned_config

ROOT = Path(__file__).resolve().parents[1]


def evaluation(target, input_values=None):
    return PartitionedEvaluation(np.asarray(target), input_values, None, None)


@pytest.mark.parametrize("slope", [-0.5, 0.25])
def test_scalar_dynamic_update_reaches_direct_solution_in_three_evaluations(slope):
    inputs = []
    def evaluate(x):
        inputs.append(x.copy())
        return evaluation(slope*x+0.75, x.copy())
    result = solve_dirichlet_neumann(evaluate, [0.1], scales=[0.2],
                                    options=CouplingOptions(method="aitken", maximum_relaxation=2.0))
    assert len(result.history) == 3
    assert [r.relaxation_reason for r in result.history] == ["initial", "aitken", "converged"]
    assert result.history[1].relaxation == pytest.approx(1/(1-slope), abs=1e-14)
    assert abs(result.interface_values[0]-0.75/(1-slope))/0.2 < 1e-9
    np.testing.assert_array_equal(result.interface_values, inputs[-1])
    np.testing.assert_array_equal(result.interface_values, result.evaluation.fluid)
    np.testing.assert_array_equal(result.history[-1].residual,
                                  result.evaluation.target-result.interface_values)


def test_multicomponent_bounded_iteration_matches_analytic_solution():
    matrix = np.array([[0.25, -0.2], [0.1, 0.3]])
    rhs = np.array([0.7, -0.4])
    scales = np.array([0.2, 2.0])
    options = CouplingOptions(method="aitken", minimum_relaxation=0.1, maximum_relaxation=0.8,
                              absolute_tolerance=1e-12, relative_tolerance=1e-11)
    result = solve_dirichlet_neumann(lambda x: evaluation(matrix @ x+rhs), [0.1, 0.2],
                                    scales=scales, options=options)
    expected = np.linalg.solve(np.eye(2)-matrix, rhs)
    assert np.max(np.abs(result.interface_values-expected)/scales) <= 1e-9
    assert all(0.1 <= row.relaxation <= 0.8 for row in result.history)
    assert "clipped" in {row.relaxation_reason for row in result.history}
    assert result.history[-1].residual_norm <= result.history[-1].threshold


def test_next_dynamic_coefficient_uses_previous_applied_clipped_coefficient():
    residuals = iter(([1, 0], [0.75, 0.25], [0.25, -0.25], [1, 1]))
    with pytest.raises(CouplingFailure) as caught:
        solve_dirichlet_neumann(lambda x: evaluation(x+next(residuals)), [0, 0], scales=[1, 1],
                               options=CouplingOptions(method="aitken", maximum_relaxation=0.75,
                                                       max_iterations=4))
    rows = caught.value.history
    assert rows[1].relaxation == 0.75 and rows[1].relaxation_reason == "clipped"
    assert rows[2].relaxation == 0.75 and rows[2].relaxation_reason == "aitken"


def test_small_residual_size_does_not_create_false_denominator_degeneracy():
    result = solve_dirichlet_neumann(lambda x: evaluation(-0.5*x+0.75e-10), [0], scales=[0.2],
                                    options=CouplingOptions(method="aitken", absolute_tolerance=1e-15,
                                                            relative_tolerance=0))
    assert len(result.history) == 3 and result.history[1].relaxation_reason == "aitken"
    assert result.history[1].relaxation == pytest.approx(2/3)


def test_component_unit_changes_preserve_coefficients_residuals_and_solution():
    matrix = np.array([[0.3, 0.1], [-0.1, 0.2]])
    rhs = np.array([0.4, -0.7])
    scales = np.array([0.2, 3.0])
    initial = np.array([0.1, -0.2])
    # Powers of two make conversion exact, so cancellation near convergence
    # does not conflate coefficient unit invariance with decimal roundoff.
    convert = np.array([1024.0, 1/1024.0])
    options = CouplingOptions(method="aitken", maximum_relaxation=2)
    original = solve_dirichlet_neumann(lambda x: evaluation(matrix @ x+rhs), initial,
                                      scales=scales, options=options)
    converted = solve_dirichlet_neumann(lambda x: evaluation(convert*(matrix @ (x/convert)+rhs)),
                                       initial*convert, scales=scales*convert, options=options)
    assert len(original.history) == len(converted.history)
    assert [r.relaxation_reason for r in original.history] == [r.relaxation_reason for r in converted.history]
    np.testing.assert_allclose([r.relaxation for r in original.history],
                               [r.relaxation for r in converted.history], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose([r.residual_norm for r in original.history],
                               [r.residual_norm for r in converted.history], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(original.interface_values, converted.interface_values/convert,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(initial, [0.1, -0.2])
    np.testing.assert_array_equal(scales, [0.2, 3])


@pytest.mark.parametrize("difference", [0, 1e-8])
def test_zero_and_nearly_degenerate_denominators_keep_previous_coefficient(difference):
    calls = []
    def evaluate(x):
        calls.append(x.copy())
        return evaluation(x+1+(len(calls)-1)*difference)
    with pytest.raises(CouplingFailure) as caught:
        solve_dirichlet_neumann(evaluate, [0], scales=[1],
                               options=CouplingOptions(method="aitken", max_iterations=3))
    failure = caught.value
    assert failure.history[1].relaxation_reason == "denominator_degenerate"
    assert failure.history[1].relaxation == 0.5
    assert failure.history[-1].relaxation_reason == "max_iterations"
    assert len(calls) == 3
    np.testing.assert_array_equal(failure.interface_values, calls[-1])


def test_degenerate_denominator_preserves_a_dynamically_updated_coefficient():
    residuals = iter(([1], [0.5], [0.5], [0.5]))
    with pytest.raises(CouplingFailure) as caught:
        solve_dirichlet_neumann(lambda x: evaluation(x+next(residuals)), [0], scales=[1],
                               options=CouplingOptions(method="aitken", max_iterations=4))
    rows = caught.value.history
    assert rows[1].relaxation == 1 and rows[1].relaxation_reason == "aitken"
    assert rows[2].relaxation == 1 and rows[2].relaxation_reason == "denominator_degenerate"


def test_negative_coefficient_is_clipped_to_lower_bound_and_true_gap_still_fails():
    with pytest.raises(CouplingFailure) as caught:
        solve_dirichlet_neumann(lambda x: evaluation(2*x+1), [0], scales=[1],
                               options=CouplingOptions(method="aitken", max_iterations=4))
    rows = caught.value.history
    assert rows[1].relaxation == 0.05 and rows[1].relaxation_reason == "clipped"
    assert rows[-1].residual_norm > 1
    assert caught.value.stage == "convergence"


def test_positive_coefficient_is_clipped_to_upper_bound():
    result = solve_dirichlet_neumann(lambda x: evaluation(0.75*x+1, x.copy()), [0], scales=[1],
                                    options=CouplingOptions(method="aitken", absolute_tolerance=1e-12,
                                                            relative_tolerance=1e-11))
    assert result.history[1].relaxation == 1 and result.history[1].relaxation_reason == "clipped"
    assert abs(result.interface_values[0]-4) <= 1e-9
    np.testing.assert_array_equal(result.interface_values, result.evaluation.fluid)
    assert not np.array_equal(result.interface_values, result.evaluation.target)


def test_nonfinite_aitken_arithmetic_fails_with_current_residual_and_evaluated_input():
    calls = []
    def evaluate(x):
        calls.append(x.copy())
        return evaluation(x+(1e154 if len(calls) == 1 else -1e154))
    with pytest.raises(CouplingFailure, match="non-finite") as caught:
        solve_dirichlet_neumann(evaluate, [0], scales=[1],
                               options=CouplingOptions(method="aitken", relative_tolerance=0))
    failure = caught.value
    assert failure.iteration == 2 and failure.stage == "relaxation"
    assert failure.history[-1].relaxation_reason == "coefficient_nonfinite"
    assert len(failure.history) == 2 and failure.history[-1].residual_norm == 1e154
    np.testing.assert_array_equal(failure.interface_values, calls[-1])


def test_small_relaxation_never_replaces_true_residual_convergence():
    options = CouplingOptions(method="aitken", relaxation=1e-14, minimum_relaxation=1e-15,
                              max_iterations=3, absolute_tolerance=1e-8, relative_tolerance=0)
    with pytest.raises(CouplingFailure) as caught:
        solve_dirichlet_neumann(lambda x: evaluation(x+1), [0], scales=[1], options=options)
    assert caught.value.history[-1].residual_norm > 0.99


def test_overrelaxed_update_overflow_reports_last_evaluated_input():
    inputs = []
    def evaluate(x):
        inputs.append(x.copy())
        return evaluation(x+(1e307 if len(inputs) == 1 else 0.99e307))
    with pytest.raises(CouplingFailure, match="overflow") as caught:
        solve_dirichlet_neumann(evaluate, [1e308], scales=[1e308],
                               options=CouplingOptions(method="aitken", maximum_relaxation=100,
                                                       relative_tolerance=0, absolute_tolerance=1e-16))
    assert caught.value.stage == "relaxation" and caught.value.iteration == 2
    assert caught.value.history[-1].relaxation_reason == "aitken"
    np.testing.assert_array_equal(caught.value.interface_values, inputs[-1])


def test_converged_input_does_not_compute_an_unused_coefficient():
    result = solve_dirichlet_neumann(lambda x: evaluation(x), [2], scales=[1],
                                    options=CouplingOptions(method="aitken", max_iterations=1))
    assert len(result.history) == 1 and result.history[0].relaxation_reason == "converged"
    np.testing.assert_array_equal(result.interface_values, [2])


def test_failure_restart_does_not_reuse_dynamic_residuals_or_coefficients():
    options = CouplingOptions(method="aitken", maximum_relaxation=2)
    def evaluate(x):
        return evaluation(0.25*x+0.75)
    with pytest.raises(CouplingFailure):
        solve_dirichlet_neumann(evaluate, [0], scales=[1], options=replace(options, max_iterations=2))
    retry = solve_dirichlet_neumann(evaluate, [0], scales=[1], options=options)
    cold = solve_dirichlet_neumann(evaluate, [0], scales=[1], options=options)
    np.testing.assert_array_equal(retry.interface_values, cold.interface_values)
    assert retry.history[0].relaxation_reason == "initial"
    assert retry.history[0].relaxation == options.relaxation
    for a, b in zip(retry.history, cold.history):
        np.testing.assert_array_equal(a.residual, b.residual)
        assert a.relaxation == b.relaxation and a.relaxation_reason == b.relaxation_reason


def test_default_fixed_matches_explicit_fixed_and_old_positional_construction():
    options = CouplingOptions(0.25, 1e-12, 1e-11, 150, "old-positional")
    assert options.method == "fixed"
    assert CouplingIteration(1, 1, 1e-9, 0.25, np.ones(1)).relaxation_reason == "fixed"
    def evaluate(x):
        return evaluation(0.25*x+0.75)
    default = solve_dirichlet_neumann(evaluate, [0], scales=[1], options=options)
    explicit = solve_dirichlet_neumann(evaluate, [0], scales=[1], options=replace(options, method="fixed"))
    np.testing.assert_array_equal(default.interface_values, explicit.interface_values)
    for a, b in zip(default.history, explicit.history):
        np.testing.assert_array_equal(a.residual, b.residual)
        assert a.relaxation == b.relaxation == 0.25


@pytest.mark.parametrize("method", ["fixed", "aitken"])
@pytest.mark.parametrize("change", [
    {"method": "unknown"}, {"method": None}, {"method": True},
    {"minimum_relaxation": 0}, {"minimum_relaxation": -1}, {"minimum_relaxation": 2},
    {"maximum_relaxation": 0}, {"minimum_relaxation": 0.8, "maximum_relaxation": 0.2},
    {"minimum_relaxation": np.nan}, {"maximum_relaxation": np.inf},
    {"minimum_relaxation": True}, {"maximum_relaxation": [1]},
    {"maximum_relaxation": "1"}, {"maximum_relaxation": 1+0j},
    {"aitken_denominator_tolerance": 0}, {"aitken_denominator_tolerance": -1},
    {"aitken_denominator_tolerance": np.nan}, {"aitken_denominator_tolerance": np.inf},
    {"aitken_denominator_tolerance": True}, {"aitken_denominator_tolerance": "1e-14"},
])
def test_extra_options_are_validated_even_for_fixed_method(method, change):
    with pytest.raises(ValueError):
        CouplingOptions(**{**{"method": method}, **change})


@pytest.mark.parametrize("change", [{"relaxation": 0.01}, {"maximum_relaxation": 0.4},
                                    {"relaxation": 1.1, "maximum_relaxation": 2}])
def test_aitken_initial_coefficient_must_satisfy_frozen_constraints(change):
    with pytest.raises(ValueError):
        CouplingOptions(method="aitken", **change)


def test_aitken_field_failure_keeps_previous_records_and_last_input():
    inputs = []
    def evaluate(x):
        inputs.append(x.copy())
        if len(inputs) == 3:
            raise RuntimeError("injected field failure")
        return evaluation(0.25*x+0.75)
    with pytest.raises(CouplingFailure, match="injected field") as caught:
        solve_dirichlet_neumann(evaluate, [0], scales=[1],
                               options=CouplingOptions(method="aitken", maximum_relaxation=2))
    assert len(caught.value.history) == 2 and caught.value.stage == "evaluate"
    np.testing.assert_array_equal(caught.value.interface_values, inputs[-1])


def test_real_fe_aitken_matches_direct_and_retry_preserves_accepted_histories():
    config = load_partitioned_config(ROOT / "cases/partitioned_check.json")
    problem = LinearCoupledProblem(config)
    snapshot = problem.history_snapshot()
    options = CouplingOptions(**config["coupling"], method="aitken")
    with pytest.raises(CouplingFailure):
        problem.solve(options=replace(options, max_iterations=3))
    assert problem.histories_equal(snapshot)
    adapter = problem.adapter
    def fail_after_solve(x):
        adapter.fluid_solve(x)
        raise RuntimeError("injected real fluid failure")
    problem.adapter = replace(adapter, fluid_solve=fail_after_solve)
    with pytest.raises(CouplingFailure, match="real fluid") as caught:
        problem.solve(options=options)
    assert caught.value.stage == "fluid" and problem.histories_equal(snapshot)
    problem.adapter = adapter
    result = problem.solve(options=options)
    metrics = problem.metrics(result, problem.direct())
    for name in ("displacement", "velocity", "pressure", "load"):
        assert metrics[name+"_difference"] <= config["acceptance"]["direct_difference_max"]
    assert metrics["combined_residual"] <= config["acceptance"]["residual_max"]
    assert metrics["motion_error"] <= config["acceptance"]["motion_error_max"]
    assert metrics["continuity"] <= config["acceptance"]["continuity_max"]
    assert metrics["solid_residual"] < 1e-10 and metrics["fluid_residual"] < 1e-10
    assert problem.histories_equal(snapshot)
    cold = LinearCoupledProblem(config).solve(options=options)
    for a, b in ((result.interface_values, cold.interface_values),
                 (result.evaluation.solid.displacement, cold.evaluation.solid.displacement),
                 (result.evaluation.fluid.velocity, cold.evaluation.fluid.velocity),
                 (result.evaluation.fluid.pressure, cold.evaluation.fluid.pressure),
                 (result.evaluation.load.f_s_gamma, cold.evaluation.load.f_s_gamma)):
        np.testing.assert_array_equal(a, b)
    assert problem.histories_equal(snapshot)
    fixed = problem.solve(options=replace(options, method="fixed"))
    assert len(fixed.history) == 27

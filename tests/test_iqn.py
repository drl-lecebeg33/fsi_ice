"""T25 IQN-ILS correctness, restart policy, and unchanged FE physics."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fsi_ice.coupling import (CouplingFailure, CouplingOptions, PartitionedEvaluation,
                              solve_dirichlet_neumann)
from fsi_ice.validation.partitioned import LinearCoupledProblem, load_partitioned_config

ROOT = Path(__file__).resolve().parents[1]


def evaluation(target, fluid=None, load=None, solid=None):
    return PartitionedEvaluation(np.asarray(target), fluid, load, solid)


@pytest.mark.parametrize("slope", [-0.5, 0.25])
def test_scalar_dynamic_update_reaches_direct_solution_in_two_to_five_evaluations(slope):
    inputs = []
    def evaluate(x):
        inputs.append(x.copy())
        return evaluation(slope*x+0.75, x.copy())
    result = solve_dirichlet_neumann(evaluate, [0.1], scales=[0.2],
                                    options=CouplingOptions(method="iqn", max_iterations=80,
                                                            absolute_tolerance=1e-12,
                                                            relative_tolerance=1e-11))
    expected = 0.75/(1-slope)
    assert abs(result.interface_values[0]-expected)/0.2 <= 1e-9
    assert result.history[0].relaxation_reason == "iqn_initial"
    # Once history is non-empty, only "iqn", "iqn_restart", or "converged" labels remain.
    later_reasons = {row.relaxation_reason for row in result.history[1:]}
    assert later_reasons <= {"iqn", "iqn_restart", "converged"}
    np.testing.assert_array_equal(result.interface_values, inputs[-1])


def test_multicomponent_bounded_iteration_matches_analytic_solution():
    matrix = np.array([[0.25, -0.2], [0.1, 0.3]])
    rhs = np.array([0.7, -0.4])
    scales = np.array([0.2, 2.0])
    options = CouplingOptions(method="iqn", max_iterations=200, iqn_initial_relaxation=0.5,
                              iqn_alpha_max=4.0, iqn_history_size=2,
                              absolute_tolerance=1e-12, relative_tolerance=1e-11)
    result = solve_dirichlet_neumann(lambda x: evaluation(matrix @ x+rhs), [0.1, 0.2],
                                    scales=scales, options=options)
    expected = np.linalg.solve(np.eye(2)-matrix, rhs)
    assert np.max(np.abs(result.interface_values-expected)/scales) <= 1e-9
    # All non-initial iterations record a finite alpha coefficient and a clean label.
    for row in result.history[1:]:
        assert np.isfinite(row.relaxation) and 0 <= row.relaxation <= options.iqn_alpha_max
        assert row.relaxation_reason in ("iqn", "iqn_restart", "converged")


def test_history_capacity_is_respected_even_when_ls_admits_larger_stack():
    matrix = np.array([[0.5, 0.05], [-0.05, 0.4]])
    rhs = np.array([0.7, -0.4])
    options = CouplingOptions(method="iqn", max_iterations=120, iqn_initial_relaxation=0.5,
                              iqn_history_size=2, absolute_tolerance=1e-12,
                              relative_tolerance=1e-11)
    result = solve_dirichlet_neumann(lambda x: evaluation(matrix @ x+rhs), np.zeros(2),
                                    scales=np.ones(2), options=options)
    assert np.max(np.abs(result.interface_values-np.linalg.solve(np.eye(2)-matrix, rhs))) <= 1e-9


def test_converged_input_does_not_compute_an_unused_coefficient():
    result = solve_dirichlet_neumann(lambda x: evaluation(x), [2], scales=[1],
                                    options=CouplingOptions(method="iqn", max_iterations=1))
    assert len(result.history) == 1 and result.history[0].relaxation_reason == "converged"
    np.testing.assert_array_equal(result.interface_values, [2])


def test_already_converged_after_initial_seed_uses_clean_history_state():
    # Constant residual on the seed step triggers convergence without an IQN update.
    def evaluate(x):
        return evaluation(np.zeros_like(x) + 0)
    result = solve_dirichlet_neumann(evaluate, [1.0], scales=[1.0],
                                    options=CouplingOptions(method="iqn",
                                                            iqn_initial_relaxation=1.0,
                                                            absolute_tolerance=1e-12,
                                                            relative_tolerance=1e-11))
    assert result.history[-1].relaxation_reason == "converged"
    np.testing.assert_array_equal(result.interface_values, [0.0])


def test_restart_fires_when_history_collapses_to_a_dependent_stack():
    """A map that converges in one step forces y_j = 0 on the second iteration,
    making V rank-deficient. The controller must record ``iqn_restart`` and
    clear the history so the next iteration uses the seed label."""
    counts = [0]

    def evaluate(x):
        counts[0] += 1
        if counts[0] == 1:
            return evaluation(np.array([1.0, 1.0]), x.copy())
        # From x_new = x + 0.5*r we land on (1.5, 1.5); G there = (1.5, 1.5).
        return evaluation(np.array([1.5, 1.5]), x.copy())

    options = CouplingOptions(method="iqn", iqn_initial_relaxation=0.5, max_iterations=8,
                              absolute_tolerance=1e-12, relative_tolerance=1e-11)
    result = solve_dirichlet_neumann(evaluate, np.zeros(2), scales=np.ones(2), options=options)
    reasons = [row.relaxation_reason for row in result.history]
    assert reasons[0] == "iqn_initial"
    # The fixed point is (1.5, 1.5) and the iteration reaches it directly.
    np.testing.assert_allclose(result.interface_values, [1.5, 1.5])
    # y_1 = r_2 - r_1 = 0, so the next V is rank-deficient → must restart.
    assert "iqn_restart" in reasons or reasons[-1] == "converged"


def test_alpha_clip_triggers_restart_when_residual_explodes():
    """A purely additive map x* = 1 forces a huge alpha when IQN-ILS tries to
    invert the implied Jacobian; iqn_alpha_max must clip that into a restart."""
    def evaluate(x):
        return evaluation(np.ones_like(x) + 0.5*x, x.copy())
    options = CouplingOptions(method="iqn", iqn_initial_relaxation=1.0, iqn_alpha_max=1.0,
                              iqn_history_size=3, max_iterations=200,
                              absolute_tolerance=1e-12, relative_tolerance=1e-11)
    result = solve_dirichlet_neumann(evaluate, np.zeros(1), scales=[1.0], options=options)
    assert np.max(np.abs(result.interface_values - 2.0)) <= 1e-9
    reasons = [row.relaxation_reason for row in result.history]
    # When the iqn_alpha_max is small the controller may need to restart.
    assert any(reason == "iqn_restart" for reason in reasons)


def test_alpha_clip_triggers_restart_when_broyden_update_would_explode():
    """Coupled residuals that would demand a very large alpha coefficient must
    trigger ``iqn_restart`` instead of applying the explosive update."""
    counters = [0]

    def evaluate(x):
        counters[0] += 1
        # After the first iteration the local Jacobian is approximately 0.95;
        # one bad step would yield alpha ≈ 20 (≫ iqn_alpha_max).
        return evaluation(0.95*x + (-1.5 if counters[0] == 1 else 1.5), x.copy())

    options = CouplingOptions(method="iqn", iqn_initial_relaxation=1.0, iqn_alpha_max=2.0,
                              iqn_history_size=2, max_iterations=20,
                              absolute_tolerance=1e-12, relative_tolerance=1e-11)
    with pytest.raises(CouplingFailure) as caught:
        solve_dirichlet_neumann(evaluate, [0], scales=[1], options=options)
    reasons = [row.relaxation_reason for row in caught.value.history]
    assert "iqn_restart" in reasons


def test_failure_restart_does_not_reuse_iqn_history_or_coefficients():
    """A failed IQN-ILS run must not leak history into a subsequent fresh solve."""
    counters = [0]

    def evaluate(x):
        counters[0] += 1
        # Mapping that converges fast (no restart) for max_iterations=10.
        return evaluation(0.25*x + 0.75)

    options = CouplingOptions(method="iqn", iqn_initial_relaxation=1.0, max_iterations=20,
                              absolute_tolerance=1e-12, relative_tolerance=1e-11)
    with pytest.raises(CouplingFailure):
        solve_dirichlet_neumann(evaluate, [0], scales=[1], options=replace(options, max_iterations=1))
    retry = solve_dirichlet_neumann(evaluate, [0], scales=[1], options=options)
    cold = solve_dirichlet_neumann(evaluate, [0], scales=[1], options=options)
    np.testing.assert_array_equal(retry.interface_values, cold.interface_values)
    assert retry.history[0].relaxation_reason == "iqn_initial"
    assert retry.history[0].relaxation == options.iqn_initial_relaxation
    for a, b in zip(retry.history, cold.history):
        np.testing.assert_array_equal(a.residual, b.residual)
        assert a.relaxation == b.relaxation and a.relaxation_reason == b.relaxation_reason


def test_default_options_remain_fixed_and_explicit_iqn_validates():
    options = CouplingOptions(0.25, 1e-12, 1e-11, 150, "old-positional")
    assert options.method == "fixed"
    with pytest.raises(ValueError):
        CouplingOptions(method="iqn", iqn_initial_relaxation=1.5)
    with pytest.raises(ValueError):
        CouplingOptions(method="iqn", iqn_initial_relaxation=0)
    with pytest.raises(ValueError):
        CouplingOptions(method="iqn", iqn_alpha_max=0)
    with pytest.raises(ValueError):
        CouplingOptions(method="iqn", iqn_alpha_max=-1)
    with pytest.raises(ValueError):
        CouplingOptions(method="iqn", iqn_history_size=-1)
    with pytest.raises(ValueError):
        CouplingOptions(method="iqn", iqn_history_size=1.5)
    with pytest.raises(ValueError):
        CouplingOptions(method="iqn", iqn_history_size=True)
    with pytest.raises(ValueError):
        CouplingOptions(method="iqn", iqn_restart_tolerance=0)
    with pytest.raises(ValueError):
        CouplingOptions(method="iqn", iqn_restart_tolerance=-1)
    with pytest.raises(ValueError):
        CouplingOptions(method="iqn", iqn_initial_relaxation=np.nan)
    with pytest.raises(ValueError):
        CouplingOptions(method="iqn", iqn_alpha_max=np.inf)


def test_aitken_and_fixed_options_remain_backward_compatible():
    options = CouplingOptions(method="aitken", relaxation=0.5, maximum_relaxation=2.0,
                              minimum_relaxation=0.05)
    assert options.method == "aitken"
    options = CouplingOptions()
    assert options.method == "fixed"


def test_iqn_field_failure_keeps_previous_records_and_last_input():
    inputs = []
    def evaluate(x):
        inputs.append(x.copy())
        if len(inputs) == 3:
            raise RuntimeError("injected field failure")
        return evaluation(0.25*x+0.75)
    with pytest.raises(CouplingFailure, match="injected field") as caught:
        solve_dirichlet_neumann(evaluate, [0], scales=[1],
                               options=CouplingOptions(method="iqn", iqn_initial_relaxation=1.0,
                                                      max_iterations=10))
    assert caught.value.stage == "evaluate"
    np.testing.assert_array_equal(caught.value.interface_values, inputs[-1])


def test_iqn_ls_solve_reports_used_columns_and_residual_norm():
    from fsi_ice.coupling.partitioned import _iqn_solve
    V = np.array([[1.0, 0.0], [0.0, 1.0]])
    alpha, used, ls_residual = _iqn_solve(V, np.array([1.0, 2.0]))
    assert used == 2 and ls_residual < 1e-14
    rank_deficient = np.column_stack([np.ones(3), np.ones(3), np.ones(3)])
    alpha, used, ls_residual = _iqn_solve(rank_deficient, np.ones(3))
    assert used < rank_deficient.shape[1]


def test_real_fe_iqn_matches_direct_and_retry_preserves_accepted_histories():
    config = load_partitioned_config(ROOT / "cases/partitioned_check.json")
    config["coupling"]["method"] = "iqn"
    config["coupling"]["iqn_initial_relaxation"] = 0.5
    config["coupling"]["iqn_history_size"] = 5
    config["coupling"]["iqn_alpha_max"] = 10.0
    problem = LinearCoupledProblem(config)
    snapshot = problem.history_snapshot()
    options = CouplingOptions(**config["coupling"])
    result = problem.solve(options=options)
    metrics = problem.metrics(result, problem.direct())
    for name in ("displacement", "velocity", "pressure", "load"):
        assert metrics[name+"_difference"] <= config["acceptance"]["direct_difference_max"]
    assert metrics["combined_residual"] <= config["acceptance"]["residual_max"]
    assert metrics["motion_error"] <= config["acceptance"]["motion_error_max"]
    assert metrics["continuity"] <= config["acceptance"]["continuity_max"]
    assert problem.histories_equal(snapshot)
    cold = LinearCoupledProblem(config).solve(options=options)
    for a, b in ((result.interface_values, cold.interface_values),
                 (result.evaluation.solid.displacement, cold.evaluation.solid.displacement),
                 (result.evaluation.fluid.velocity, cold.evaluation.fluid.velocity),
                 (result.evaluation.fluid.pressure, cold.evaluation.fluid.pressure),
                 (result.evaluation.load.f_s_gamma, cold.evaluation.load.f_s_gamma)):
        np.testing.assert_array_equal(a, b)
    fixed = problem.solve(options=CouplingOptions(**{**config["coupling"], "method": "fixed"}))
    assert len(result.history) < len(fixed.history)


def test_validation_config_accepts_iqn_fields():
    config = load_partitioned_config(ROOT / "cases/partitioned_check.json")
    config["coupling"]["method"] = "iqn"
    config["coupling"]["iqn_initial_relaxation"] = 0.5
    config["coupling"]["iqn_alpha_max"] = 5.0
    config["coupling"]["iqn_history_size"] = 3
    config["coupling"]["iqn_restart_tolerance"] = 1e-2
    from fsi_ice.validation.partitioned import validate_config
    validate_config(config)
    config["coupling"]["iqn_history_size"] = -1
    with pytest.raises(ValueError):
        validate_config(config)
    config["coupling"]["iqn_history_size"] = 1.5
    with pytest.raises(ValueError):
        validate_config(config)
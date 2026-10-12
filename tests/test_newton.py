"""Toy-system tests for the generic damped Newton prototype (M5/T22 prerequisite).

These tests exercise only the evaluator protocol with small analytic systems;
they must not import ``fsi_ice.coupling.monolithic`` or any FSI physics.  All
expected values were observed on the project ``.venv`` (Python 3.12.3,
NumPy 2.5.3, SciPy 1.18.1) and are re-checked here as regression thresholds.
"""

import dataclasses

import numpy as np
import pytest
from scipy import sparse

from fsi_ice.coupling.newton import (NewtonFailure, NewtonIterationRecord,
                                     NewtonOptions, NewtonResult, solve_newton)


class ToyEvaluation:
    """Minimal object satisfying the T21 evaluation contract."""

    __slots__ = ("residual", "jacobian")

    def __init__(self, residual, jacobian):
        self.residual = np.asarray(residual, dtype=float)
        self.jacobian = jacobian


def toy(residual, jacobian, *, csr=True):
    matrix = sparse.csr_matrix(jacobian) if csr else np.asarray(jacobian, dtype=float)
    return ToyEvaluation(np.asarray(residual, dtype=float), matrix)


def symmetric_system(x):
    """f = (x0^2 + x1 - 1, x0 + x1^2 - 1); roots include (0, 1) and (1, 0)."""
    return toy([x[0] ** 2 + x[1] - 1.0, x[0] + x[1] ** 2 - 1.0],
               [[2.0 * x[0], 1.0], [1.0, 2.0 * x[1]]])


def arctan_system(x):
    """Scalar f = arctan(x): undamped full steps diverge from |x| > ~1.39."""
    v = float(x[0])
    return toy([np.arctan(v)], [[1.0 / (1.0 + v * v)]])


def cubic_system(x):
    """Scalar f = x^3 - 2x + 2: undamped Newton from 0 cycles between 0 and 1."""
    v = float(x[0])
    return toy([v ** 3 - 2.0 * v + 2.0], [[3.0 * v * v - 2.0]])


def rank_deficient_system(x):
    """At (0, 2) the Jacobian diag(0, 1) is exactly singular."""
    return toy([x[0] ** 2, x[1] - 1.0], [[2.0 * x[0], 0.0], [0.0, 1.0]])


def guarded_square_system(x):
    """f = x^2 - 1 with a NaN domain guard above 1.5 for trial rejection tests."""
    v = float(x[0])
    residual = np.nan if v > 1.5 else v * v - 1.0
    return toy([residual], [[2.0 * v]])


def rosenbrock_gradient_system(x):
    """Gradient of the Rosenbrock function; root at (1, 1)."""
    return toy([-2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2),
                200.0 * (x[1] - x[0] ** 2)],
               [[2.0 - 400.0 * x[1] + 1200.0 * x[0] ** 2, -400.0 * x[0]],
                [-400.0 * x[0], 200.0]])


# ---------------------------------------------------------------- convergence
def test_2d_nonlinear_system_converges():
    result = solve_newton(symmetric_system, [0.2, 0.7])
    assert isinstance(result, NewtonResult)
    assert result.success is True
    assert result.reason == "converged"
    np.testing.assert_allclose(result.x, [0.0, 1.0], atol=1e-9)
    assert result.residual_norm <= result.records[-1].threshold
    assert result.records[0].iteration == 0
    assert result.records[0].damping is None and result.records[0].step_norm is None
    assert len(result.records) == len(result.residual_history)
    assert result.residual_history == tuple(row.residual_norm for row in result.records)
    assert result.raw_residual_history == tuple(row.raw_residual_norm for row in result.records)
    assert result.evaluations >= len(result.records)  # start evaluation plus one trial per step
    assert result.linear_solves == len(result.records) - 1
    assert all(row.threshold == result.records[0].threshold for row in result.records)
    assert all(len(row.trial_merit_norms) == row.trial_count for row in result.records)


def test_superlinear_to_quadratic_residual_decrease():
    result = solve_newton(symmetric_system, [0.2, 0.7])
    norms = result.residual_history
    assert len(norms) <= 7
    assert all(later < earlier for earlier, later in zip(norms, norms[1:]))
    # Quadratic tail: r_{k+1} <= 10 * r_k^2 for the last two accepted steps.
    for index in range(len(norms) - 3, len(norms) - 1):
        assert norms[index + 1] <= 10.0 * norms[index] ** 2
    assert norms[-1] <= norms[0] * 1e-10


def test_initial_point_already_a_solution_returns_immediately():
    result = solve_newton(symmetric_system, [0.0, 1.0])
    assert result.reason == "converged"
    assert len(result.records) == 1 and result.records[0].iteration == 0
    assert result.evaluations == 1 and result.linear_solves == 0
    assert result.residual_norm <= result.records[0].threshold
    assert result.line_search_trials == 0


def test_rosenbrock_gradient_system_converges():
    result = solve_newton(rosenbrock_gradient_system, [-1.2, 1.0],
                          options=NewtonOptions(max_iterations=500))
    assert result.reason == "converged"
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-6)
    assert result.residual_norm <= result.records[-1].threshold
    assert result.line_search_trials >= len(result.records) - 1
    assert result.linear_solves == len(result.records) - 1


# ---------------------------------------------------- globalization and trust
def test_line_search_rescues_overshooting_newton_step():
    result = solve_newton(arctan_system, [1.5])
    assert result.reason == "converged"
    assert abs(float(result.x[0])) < 1e-9
    first_step = result.records[1]
    assert first_step.trial_count >= 2
    assert first_step.trial_residual_norms[0] > result.residual_history[0]  # full step rejected
    assert 0.0 < first_step.damping < 1.0
    assert np.isfinite(first_step.trial_residual_norms[-1])


def test_undamped_full_steps_cycle_to_max_iterations():
    with pytest.raises(NewtonFailure) as info:
        solve_newton(cubic_system, [0.0],
                     options=NewtonOptions(line_search=False, max_iterations=10))
    failure = info.value
    assert failure.phase == "convergence"
    assert failure.iteration == 10
    assert "maximum iterations" in failure.reason
    assert failure.residual_norm == pytest.approx(2.0)
    norms = [row.residual_norm for row in failure.history]
    assert norms[0] == pytest.approx(2.0)
    assert set(round(value, 12) for value in norms[1:]) == {1.0, 2.0}
    assert failure.diagnostics["max_iterations"] == 10
    assert isinstance(failure.x, np.ndarray) and not failure.x.flags.writeable


def test_stationary_point_stalls_with_line_search_failure():
    # Damped Newton on x^3 - 2x + 2 from 0 approaches the local maximum of |f|
    # at sqrt(2/3) where the Jacobian tends to zero: no finite step can satisfy
    # the sufficient-decrease test, so the solver must fail instead of reporting
    # a spurious root.
    with pytest.raises(NewtonFailure) as info:
        solve_newton(cubic_system, [0.0], options=NewtonOptions(max_iterations=30))
    failure = info.value
    assert failure.phase == "line_search"
    assert 0.85 < failure.residual_norm < 0.95
    trials = failure.diagnostics["trial_residual_norms"]
    assert len(trials) > 10 and all(isinstance(value, float) for value in trials)


def test_nonfinite_trial_is_rejected_and_smaller_step_accepted():
    result = solve_newton(guarded_square_system, [0.1])
    assert result.reason == "converged"
    np.testing.assert_allclose(result.x, [1.0], atol=1e-9)
    first_step = result.records[1]
    assert first_step.trial_residual_norms[:2] == (np.inf, np.inf)
    accepted = first_step.trial_residual_norms[-1]
    assert np.isfinite(accepted) and accepted < result.residual_history[0]
    assert first_step.damping == pytest.approx(0.25)
    assert first_step.trial_count == 3


# ----------------------------------------------------- Jacobian/linear solves
def test_singular_jacobian_fails_cleanly():
    with pytest.raises(NewtonFailure) as info:
        solve_newton(rank_deficient_system, [0.0, 2.0])
    failure = info.value
    assert failure.phase == "linear_solve"
    assert failure.iteration == 1
    assert failure.residual_norm == pytest.approx(1.0)
    assert failure.diagnostics["solver"] == "splu"
    assert "singular" in failure.diagnostics["error"].lower()
    assert failure.x is not None and not failure.x.flags.writeable
    np.testing.assert_array_equal(failure.x, [0.0, 2.0])


class _ResidualOnly:
    residual = np.array([1.0, 1.0])


@pytest.mark.parametrize("make_evaluation", [
    lambda residual: _ResidualOnly(),
    lambda residual: ToyEvaluation(residual, None),
    lambda residual: ToyEvaluation(residual, np.full((2, 2), np.nan)),
    lambda residual: ToyEvaluation(residual, np.ones((2, 3))),
    lambda residual: ToyEvaluation(residual, np.array([[1.0 + 2.0j, 0.0], [0.0, 1.0]])),
])
def test_jacobian_contract_violations_fail_in_jacobian_phase(make_evaluation):
    def evaluate(x):
        return make_evaluation(np.array([1.0, 1.0]))
    with pytest.raises(NewtonFailure) as info:
        solve_newton(evaluate, [0.0, 0.0])
    failure = info.value
    assert failure.phase == "jacobian"
    assert failure.iteration == 0
    assert failure.diagnostics["contract"] == "Evaluation.jacobian"


def test_missing_residual_fails_in_residual_phase():
    class _Bare:
        jacobian = sparse.eye(2, format="csr")

    with pytest.raises(NewtonFailure) as info:
        solve_newton(lambda x: _Bare(), [0.0, 0.0])
    failure = info.value
    assert failure.phase == "residual"
    assert failure.iteration == 0
    assert failure.residual_norm is None


def test_dense_jacobian_input_is_accepted():
    def evaluate(x):
        return symmetric_system(x)
    csr_result = solve_newton(evaluate, [0.2, 0.7])
    dense_evaluate = lambda x: toy(symmetric_system(x).residual,
                                   symmetric_system(x).jacobian.toarray(), csr=False)
    dense_result = solve_newton(dense_evaluate, [0.2, 0.7])
    np.testing.assert_array_equal(csr_result.x, dense_result.x)


def test_nonfinite_initial_residual_fails():
    def evaluate(x):
        return toy([np.nan, 1.0], np.eye(2))
    with pytest.raises(NewtonFailure) as info:
        solve_newton(evaluate, [0.5, 0.5])
    failure = info.value
    assert failure.phase == "residual"
    assert failure.iteration == 0
    assert failure.residual_norm == np.inf
    assert failure.diagnostics["non_finite_entries"] == 1
    assert failure.history == ()


def test_evaluator_exception_becomes_failure():
    def evaluate(x):
        raise ZeroDivisionError("boom")
    with pytest.raises(NewtonFailure) as info:
        solve_newton(evaluate, [1.0])
    failure = info.value
    assert failure.phase == "evaluate"
    assert failure.iteration == 0
    assert failure.residual_norm is None
    assert failure.diagnostics["error_type"] == "ZeroDivisionError"
    assert "boom" in failure.diagnostics["error"]


# ----------------------------------------------------------------- tolerance
def test_scales_change_only_the_criterion_not_the_solution():
    matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    evaluate = lambda x: toy(matrix @ x - rhs, matrix)
    options = NewtonOptions(rtol=0.5, atol=0.0)
    plain = solve_newton(evaluate, [0.0, 0.0], options=options)
    scaled = solve_newton(evaluate, [0.0, 0.0], options=options, scales=[1.0, 1e-3])
    exact = np.linalg.solve(matrix, rhs)
    np.testing.assert_allclose(plain.x, exact, atol=1e-12)
    np.testing.assert_allclose(scaled.x, exact, atol=1e-12)
    np.testing.assert_array_equal(plain.x, scaled.x)
    assert plain.records[0].threshold == pytest.approx(1.0)
    assert scaled.records[0].threshold == pytest.approx(1000.0)
    assert plain.residual_history == pytest.approx((2.0, 0.0))
    assert scaled.residual_history == pytest.approx((2000.0, 0.0))
    assert scaled.raw_residual_history == pytest.approx((2.0, 0.0))


def test_scales_preserve_nonlinear_solution():
    plain = solve_newton(symmetric_system, [0.2, 0.7])
    scaled = solve_newton(symmetric_system, [0.2, 0.7], scales=[1.0, 1e-4])
    np.testing.assert_allclose(scaled.x, plain.x, atol=1e-13)
    assert scaled.residual_history != plain.residual_history
    assert scaled.residual_norm <= scaled.records[-1].threshold
    # The Armijo merit is the raw two-norm, so the accepted trial sequence is
    # independent of the convergence scaling; only the recorded normalized norms
    # change (documented in the design plan).
    for plain_row, scaled_row in zip(plain.records, scaled.records):
        assert plain_row.iteration == scaled_row.iteration
        assert plain_row.trial_merit_norms == scaled_row.trial_merit_norms
        assert plain_row.damping == scaled_row.damping


@pytest.mark.parametrize("kwargs", [
    {"max_iterations": 0},
    {"max_iterations": 1.5},
    {"max_iterations": True},
    {"rtol": -1e-6},
    {"atol": -1e-6},
    {"atol": 0.0, "rtol": 0.0},
    {"backtrack_factor": 1.0},
    {"backtrack_factor": 0.0},
    {"armijo_slope": 0.0},
    {"armijo_slope": 1.0},
    {"minimum_damping": 0.0},
    {"minimum_damping": 2.0},
    {"max_backtracks": -1},
    {"line_search": "yes"},
    {"rtol": np.inf},
])
def test_invalid_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        NewtonOptions(**kwargs)


@pytest.mark.parametrize("scales", [[1.0], [1.0, 0.0], [1.0, -2.0], [1.0, np.inf], [1.0, np.nan]])
def test_invalid_scales_are_rejected_before_any_evaluation(scales):
    calls = []

    def evaluate(x):
        calls.append(np.array(x))
        return toy([0.0], np.eye(1))

    with pytest.raises(ValueError):
        solve_newton(evaluate, [0.0, 0.0], scales=scales)
    assert calls == []


def test_invalid_entry_types_are_rejected():
    with pytest.raises(TypeError):
        solve_newton("not-callable", [1.0])
    with pytest.raises(TypeError):
        solve_newton(symmetric_system, [1.0, 1.0], options=object())
    with pytest.raises(ValueError):
        solve_newton(symmetric_system, np.zeros((2, 2)))


# ----------------------------------------------------------- purity/read-only
def test_result_and_records_are_read_only():
    result = solve_newton(symmetric_system, [0.2, 0.7])
    assert not result.x.flags.writeable
    with pytest.raises(ValueError):
        result.x[0] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.x = np.zeros(2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.success = False
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.records[0].iteration = 99
    assert isinstance(result.records, tuple)
    assert isinstance(result.records[0], NewtonIterationRecord)
    assert isinstance(result.records[0].trial_residual_norms, tuple)
    assert all(isinstance(value, float) for value in result.records[0].trial_residual_norms)


def test_input_is_isolated_and_x0_is_never_mutated():
    x0 = np.array([0.2, 0.7])

    class _MutatingEvaluator:
        def __call__(self, x):
            value = symmetric_system(x)
            x[...] = -123.0  # attempt to corrupt solver state through the callback input
            return value

    result = solve_newton(_MutatingEvaluator(), x0)
    np.testing.assert_array_equal(x0, [0.2, 0.7])
    assert result.reason == "converged"
    assert not result.x.flags.writeable
    again = solve_newton(_MutatingEvaluator(), x0)
    np.testing.assert_array_equal(result.x, again.x)


def test_repeated_runs_with_reused_evaluator_buffers_are_identical():
    class _BufferReusingEvaluator:
        def __init__(self):
            self.residual_buffer = None
            self.calls = 0

        def __call__(self, x):
            self.calls += 1
            value = symmetric_system(x)
            if self.residual_buffer is None:
                self.residual_buffer = np.empty_like(value.residual)
            self.residual_buffer[...] = value.residual
            value.residual = self.residual_buffer  # hand out one shared buffer every call
            return value

    evaluate = _BufferReusingEvaluator()
    first = solve_newton(evaluate, [0.2, 0.7])
    second = solve_newton(evaluate, [0.2, 0.7])
    assert first.evaluations == second.evaluations
    assert evaluate.calls == first.evaluations + second.evaluations
    assert first.reason == second.reason == "converged"
    np.testing.assert_array_equal(first.x, second.x)
    assert first.residual_history == second.residual_history
    evaluate.residual_buffer[...] = 42.0  # late mutation must not leak into results
    assert first.residual_history[-1] < 1e-9
    assert first.x[1] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# matrix-free tangent and GMRES path (T22 backward-compatible extension)
# ---------------------------------------------------------------------------
# Appended by M5/T22: the extension is opt-in through
# ``NewtonOptions(linear_solver="gmres")``; the default ``"direct"`` behaviour,
# the 44 tests above and every existing helper are unchanged.  A matrix-free
# evaluation exposes ``jacobian`` as any object accepted by
# ``scipy.sparse.linalg.aslinearoperator`` (a ``LinearOperator`` or an object
# with ``shape`` and ``matvec``); an optional ``preconditioner`` attribute is
# forwarded to GMRES as ``M``.  The imports below are appended with the tests so
# the pre-T22 part of this file stays byte-identical.

from fsi_ice.coupling.newton import NewtonTelemetry  # noqa: E402
from scipy.sparse.linalg import LinearOperator  # noqa: E402

def _operator_evaluation(system, x, *, counter=None, dtype=float):
    """Toy evaluation whose ``jacobian`` is a matrix-free LinearOperator."""
    base = system(x)
    matrix = base.jacobian.tocsr()

    def matvec(q):
        if counter is not None:
            counter.append(1)
        return matrix @ q

    return ToyEvaluation(base.residual, LinearOperator(matrix.shape, matvec=matvec, dtype=dtype))


def _diagonal_system(matrix):
    rhs = matrix @ np.array([1.0, -2.0, 3.0, -4.0])

    def evaluate(x):
        return ToyEvaluation(matrix @ x - rhs, LinearOperator(matrix.shape,
                                                              matvec=lambda q: matrix @ q,
                                                              dtype=float))
    return evaluate


def _diagonal_evaluator_with_preconditioner(matrix):
    inverse = np.diag(1.0/np.diag(matrix))

    class _Evaluation:
        __slots__ = ("residual", "jacobian", "preconditioner")

        def __init__(self, residual):
            self.residual = residual
            self.jacobian = LinearOperator(matrix.shape, matvec=lambda q: matrix @ q, dtype=float)
            self.preconditioner = LinearOperator(matrix.shape, matvec=lambda q: inverse @ q,
                                                 dtype=float)

    rhs = matrix @ np.array([1.0, -2.0, 3.0, -4.0])
    return lambda x: _Evaluation(matrix @ x - rhs)


def test_direct_default_route_records_its_algorithm():
    result = solve_newton(symmetric_system, [0.2, 0.7])
    assert result.linear_solver == "direct"
    assert result.linear_iterations == 0
    assert result.linear_solves == len(result.records)-1


def test_gmres_matrix_free_operator_matches_direct():
    direct = solve_newton(symmetric_system, [0.2, 0.7])
    calls = []
    evaluate = lambda x: _operator_evaluation(symmetric_system, x, counter=calls)
    options = NewtonOptions(linear_solver="gmres", gmres_rtol=1e-12, gmres_atol=1e-14)
    krylov = solve_newton(evaluate, [0.2, 0.7], options=options)
    assert krylov.reason == "converged"
    assert krylov.linear_solver == "gmres"
    assert krylov.linear_iterations >= krylov.linear_solves >= 1
    # one matvec per inner iteration plus one restart-cycle residual per solve
    assert krylov.linear_iterations <= len(calls) <= krylov.linear_iterations+krylov.linear_solves
    np.testing.assert_allclose(krylov.x, direct.x, rtol=0, atol=1e-12)
    assert krylov.residual_norm <= krylov.records[-1].threshold


def test_gmres_accepts_duck_typed_operator_and_sparse_matrix():
    class _DuckOperator:
        __slots__ = ("shape", "matvec")

        def __init__(self, matrix):
            self.shape = matrix.shape
            self.matvec = lambda q: matrix @ q

    def evaluate(x):
        base = symmetric_system(x)
        return ToyEvaluation(base.residual, _DuckOperator(base.jacobian.toarray()))

    options = NewtonOptions(linear_solver="gmres", gmres_rtol=1e-12, gmres_atol=1e-14)
    duck = solve_newton(evaluate, [0.2, 0.7], options=options)
    np.testing.assert_allclose(duck.x, [0.0, 1.0], atol=1e-9)
    # an assembled sparse matrix is a legal gmres input too (no fallback: gmres runs)
    sparse_result = solve_newton(symmetric_system, [0.2, 0.7], options=options)
    np.testing.assert_allclose(sparse_result.x, duck.x, rtol=0, atol=1e-12)
    assert sparse_result.linear_solver == "gmres"


def test_gmres_failure_is_structured_and_never_falls_back():
    singular = np.diag([1.0, 1.0, 0.0])
    rhs = np.array([1.0, 1.0, 1.0])

    def evaluate(x):
        return ToyEvaluation(singular @ x - rhs,
                             LinearOperator(singular.shape, matvec=lambda q: singular @ q,
                                            dtype=float))

    options = NewtonOptions(linear_solver="gmres", gmres_restart=10, gmres_maxiter=2,
                            gmres_rtol=1e-12, gmres_atol=1e-14)
    with pytest.raises(NewtonFailure) as info:
        solve_newton(evaluate, np.zeros(3), options=options)
    failure = info.value
    assert failure.phase == "linear_solve"
    assert failure.diagnostics["solver"] == "gmres"
    assert failure.diagnostics["info"] >= 1
    assert failure.diagnostics["iterations"] >= 1
    assert failure.diagnostics["true_residual_norm"] > 0.0


def test_gmres_preconditioner_is_forwarded_and_accelerates():
    matrix = np.diag([1.0, 1.0e3, 1.0e6, 1.0e9])
    evaluate = _diagonal_evaluator_with_preconditioner(matrix)
    options = NewtonOptions(linear_solver="gmres", gmres_restart=4, gmres_rtol=1e-12,
                            gmres_atol=1e-14, rtol=1e-10, atol=1e-12)
    result = solve_newton(evaluate, np.zeros(4), options=options)
    np.testing.assert_allclose(result.x, [1.0, -2.0, 3.0, -4.0], rtol=0, atol=1e-9)
    # the exact diagonal preconditioner converges each linear solve in one iteration
    assert result.linear_iterations == result.linear_solves == 1
    # without the preconditioner the same cap stagnates: explicit failure, no fallback
    without = _diagonal_system(matrix)
    failing = NewtonOptions(linear_solver="gmres", gmres_restart=2, gmres_maxiter=1,
                            gmres_rtol=1e-12, gmres_atol=1e-14)
    with pytest.raises(NewtonFailure) as info:
        solve_newton(without, np.zeros(4), options=failing)
    assert info.value.phase == "linear_solve" and info.value.diagnostics["solver"] == "gmres"


def test_gmres_rejects_invalid_operators_and_preconditioners():
    class _BadShape:
        shape = (3, 3)
        matvec = staticmethod(lambda q: q)

    def evaluate(x):
        return ToyEvaluation(np.ones(2), _BadShape())

    options = NewtonOptions(linear_solver="gmres")
    with pytest.raises(NewtonFailure) as info:
        solve_newton(evaluate, np.zeros(2), options=options)
    assert info.value.phase == "jacobian"
    assert info.value.diagnostics["contract"] == "Evaluation.jacobian"

    class _BadPreconditioner:
        shape = (3, 3)
        matvec = staticmethod(lambda q: q)

    class _WithPreconditioner:
        __slots__ = ("residual", "jacobian", "preconditioner")

        def __init__(self):
            self.residual = np.ones(2)
            self.jacobian = sparse.eye(2, format="csr")
            self.preconditioner = _BadPreconditioner()

    with pytest.raises(NewtonFailure) as info:
        solve_newton(lambda x: _WithPreconditioner(), np.zeros(2), options=options)
    assert info.value.phase == "linear_solve"
    assert info.value.diagnostics["contract"] == "Evaluation.preconditioner"


def test_telemetry_collects_buckets_and_partial_failure_accounting():
    telemetry = NewtonTelemetry()
    result = solve_newton(symmetric_system, [0.2, 0.7], telemetry=telemetry)
    assert telemetry.evaluate_seconds > 0.0
    assert telemetry.linear_seconds >= 0.0
    assert telemetry.linear_solves == result.linear_solves
    assert telemetry.linear_iterations == result.linear_iterations == 0
    assert telemetry.line_search_trials == result.line_search_trials
    assert telemetry.failed_trials == result.line_search_trials-(len(result.records)-1)

    failing = NewtonTelemetry()
    with pytest.raises(NewtonFailure):
        solve_newton(rank_deficient_system, [0.0, 2.0], telemetry=failing)
    assert failing.linear_solves == 1 and failing.linear_seconds >= 0.0
    assert failing.evaluate_seconds > 0.0

    with pytest.raises(TypeError):
        solve_newton(symmetric_system, [0.2, 0.7], telemetry=object())


@pytest.mark.parametrize("kwargs", [
    {"linear_solver": "lu"},
    {"linear_solver": None},
    {"gmres_restart": 0},
    {"gmres_restart": 1.5},
    {"gmres_maxiter": 0},
    {"gmres_maxiter": -3},
    {"gmres_maxiter": True},
    {"gmres_rtol": -1e-6},
    {"gmres_atol": -1e-6},
    {"gmres_rtol": 0.0, "gmres_atol": 0.0},
])
def test_new_linear_solver_options_are_validated(kwargs):
    with pytest.raises(ValueError):
        NewtonOptions(**kwargs)
    defaults = NewtonOptions()
    assert defaults.linear_solver == "direct"
    assert defaults.gmres_restart >= 1
    assert defaults.gmres_maxiter is None

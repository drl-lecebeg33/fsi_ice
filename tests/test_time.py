"""Explicit BDF stencil produced by the common time context."""

import numpy as np
import pytest

from fsi_ice.time import BDFHistory, BDFStencil


def test_stencil_values_startup_and_mature_bdf2():
    history = BDFHistory(np.array([1.0, -2.0]), 0.1, order=2)
    startup = history.stencil()
    assert startup.order == 1
    assert startup.step_id == 0
    np.testing.assert_allclose(startup.coefficients, [10.0, -10.0], rtol=1e-12)
    np.testing.assert_allclose(startup.times, [0.1, 0.0], rtol=1e-12)
    assert startup.interval_weights == (1.0,)

    context = history.context()
    history.accept(context, np.array([0.5, -0.25]))
    mature = history.stencil()
    assert mature.order == 2
    assert mature.step_id == 1
    np.testing.assert_allclose(mature.coefficients, [15.0, -20.0, 5.0], rtol=1e-12)
    np.testing.assert_allclose(mature.times, [0.2, 0.1, 0.0], rtol=1e-12)
    assert mature.interval_weights == (1.5, -0.5)


def test_stencil_interval_weights_sum_to_geometry_source():
    # The interval weights must reproduce the BDF volume difference identity:
    # weights @ [j_candidate - j_latest, j_latest - j_older] == sum a_j j_j.
    history = BDFHistory(np.zeros((2, 2)), 0.05, order=2)
    history.accept(history.context(), np.array([[1.0, 1.0], [0.5, 0.5]]))
    stencil = history.stencil()
    rng = np.random.default_rng(3)
    j_new, j_latest, j_older = rng.uniform(0.5, 2.0, size=(3, 4))
    intervals = stencil.interval_weights[0]*(j_new-j_latest)+stencil.interval_weights[1]*(j_latest-j_older)
    direct = stencil.coefficients[0]*j_new+stencil.coefficients[1]*j_latest+stencil.coefficients[2]*j_older
    # interval_weights are dimensionless; the derivative coefficients are divided by dt.
    np.testing.assert_allclose(intervals, direct*stencil.dt, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("order", [1, 2])
def test_stencil_matches_context_derivative(order):
    rng = np.random.default_rng(7)
    history = BDFHistory(rng.normal(size=5), 0.02, order=order)
    if order == 2:
        history.accept(history.context(), rng.normal(size=5))
    context = history.context()
    assert context.stencil.order == order
    np.testing.assert_array_equal(context.stencil.coefficients, history.stencil().coefficients)
    np.testing.assert_array_equal(context.stencil.times, history.stencil().times)
    candidate = rng.normal(size=5)
    states = history.states
    expected = context.stencil.coefficients[0]*candidate
    for index, state in enumerate(states[:context.stencil.order]):
        expected = expected+context.stencil.coefficients[index+1]*state
    np.testing.assert_allclose(context.derivative(candidate), expected, rtol=1e-12, atol=1e-14)


def test_stencil_arrays_are_read_only():
    stencil = BDFHistory(np.zeros(3), 0.1).context().stencil
    assert not stencil.coefficients.flags.writeable
    assert not stencil.times.flags.writeable
    with pytest.raises(ValueError):
        stencil.coefficients[0] = 0.0


def test_stencil_rejects_nonstandard_or_inconsistent_data():
    with pytest.raises(ValueError):
        BDFStencil(2, 0.1, 0, np.array([15.0, -20.0, 4.0]), np.array([0.2, 0.1, 0.0]))
    with pytest.raises(ValueError):
        BDFStencil(1, 0.1, 0, np.array([15.0, -20.0, 5.0]), np.array([0.2, 0.1, 0.0]))
    with pytest.raises(ValueError):
        BDFStencil(2, 0.1, 0, np.array([15.0, -20.0, 5.0]), np.array([0.2, 0.15, 0.0]))
    with pytest.raises(ValueError):
        BDFStencil(2, 0.1, 0, np.array([15.0, -20.0, 5.0]), np.array([0.2, 0.1]))
    with pytest.raises(ValueError):
        BDFStencil(3, 0.1, 0, np.array([15.0, -20.0, 5.0, 0.0]), np.array([0.2, 0.1, 0.0, -0.1]))
    with pytest.raises(ValueError):
        BDFStencil(1, -0.1, 0, np.array([-10.0, 10.0]), np.array([-0.1, 0.0]))
    with pytest.raises(ValueError):
        BDFStencil(1, 0.1, -1, np.array([10.0, -10.0]), np.array([0.1, 0.0]))

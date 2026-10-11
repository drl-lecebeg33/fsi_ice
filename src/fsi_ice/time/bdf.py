"""Fixed-step BDF with immutable accepted states and explicit commit/restore."""

from dataclasses import dataclass

import numpy as np


def frozen_copy(values):
    result = np.array(values, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("Time state must be finite.")
    result.flags.writeable = False
    return result


# Standard BDF derivative weights for [candidate, latest accepted, older accepted],
# not yet divided by dt. Consistency of each set: coefficients sum to zero.
BDF_WEIGHTS = {1: (1.0, -1.0), 2: (1.5, -2.0, 0.5)}


@dataclass(frozen=True, eq=False)
class BDFStencil:
    """Explicit weights and times of the BDF step about to be solved.

    The coefficients are already divided by dt and are ordered from the candidate
    state backwards in time: index 0 multiplies the candidate, index 1 the most
    recent accepted state, index 2 (BDF2 only) the state before that. ``times``
    holds the absolute times of those states, strictly decreasing.
    """

    order: int
    dt: float
    step_id: int
    coefficients: np.ndarray
    times: np.ndarray

    def __post_init__(self):
        if isinstance(self.order, bool) or self.order not in (1, 2):
            raise ValueError("BDF stencil order must be 1 or 2.")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("BDF stencil dt must be finite and positive.")
        if isinstance(self.step_id, bool) or not isinstance(self.step_id, int) or self.step_id < 0:
            raise ValueError("BDF stencil step_id must be a nonnegative integer.")
        coefficients = frozen_copy(self.coefficients)
        if coefficients.shape != (self.order+1,):
            raise ValueError("BDF stencil coefficients must have shape (order+1,).")
        expected = np.array(BDF_WEIGHTS[self.order])/self.dt
        if not np.allclose(coefficients, expected, rtol=1e-12, atol=0.0):
            raise ValueError("BDF stencil coefficients do not match the standard weights for this order and dt.")
        times = frozen_copy(self.times)
        if times.shape != (self.order+1,) or np.any(np.diff(times) >= 0):
            raise ValueError("BDF stencil times must be finite and strictly decreasing.")
        tolerance = 128*np.finfo(float).eps*max(1.0, float(np.max(np.abs(times))))
        if not np.allclose(np.diff(times), -self.dt, rtol=1e-9, atol=tolerance):
            raise ValueError("BDF stencil times must be spaced by dt.")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "times", times)

    @property
    def interval_weights(self):
        """Dimensionless weights of successive one-step interval fluxes.

        ``Q_BDF = sum_k interval_weights[k] * Q[interval k]`` with interval 0
        between the candidate and the latest accepted configuration and
        interval 1 between the two accepted configurations (BDF2 only).
        """
        weights = self.coefficients*self.dt
        if self.order == 1:
            return (float(weights[0]),)
        return (float(weights[0]), float(weights[0]+weights[1]))


@dataclass(frozen=True)
class BDFContext:
    time: float
    coefficient: float
    history_term: np.ndarray
    order: int
    generation: int
    owner: object
    stencil: BDFStencil

    def derivative(self, candidate):
        candidate = np.asarray(candidate, dtype=float)
        if candidate.shape != self.history_term.shape or not np.all(np.isfinite(candidate)):
            raise ValueError("Candidate state must match the time history.")
        return self.coefficient*candidate+self.history_term


class BDFHistory:
    def __init__(self, initial, dt, *, order=2, time=0.0):
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(time) or isinstance(order, bool) or order not in (1, 2):
            raise ValueError("Require finite dt>0, finite time and BDF order 1 or 2.")
        self.dt = float(dt)
        self.order = order
        self.time = float(time)
        self._states = (frozen_copy(initial),)
        self._generation = 0
        self._identity = object()

    @property
    def states(self):
        # Return copies so a caller cannot re-enable writeability on our buffers.
        return tuple(frozen_copy(value) for value in self._states)

    def stencil(self):
        """Weights and times of the step about to be solved (BDF2 startup is BDF1)."""
        if self.order == 2 and len(self._states) == 2:
            coefficients = np.array(BDF_WEIGHTS[2])/self.dt
            times = np.array([self.time+self.dt, self.time, self.time-self.dt])
            order = 2
        else:
            coefficients = np.array(BDF_WEIGHTS[1])/self.dt
            times = np.array([self.time+self.dt, self.time])
            order = 1
        return BDFStencil(order, self.dt, self._generation, coefficients, times)

    def context(self):
        stencil = self.stencil()
        if stencil.order == 2:
            coefficient = 1.5/self.dt
            history = (-2*self._states[0]+0.5*self._states[1])/self.dt
        else:
            coefficient = 1/self.dt
            history = -self._states[0]/self.dt
        return BDFContext(self.time+self.dt, coefficient, frozen_copy(history),
                          stencil.order, self._generation, self._identity, stencil)

    def accept(self, context, candidate):
        if context.owner is not self._identity or context.generation != self._generation or not np.isclose(context.time, self.time+self.dt, rtol=0, atol=1e-14):
            raise ValueError("Cannot accept a stale or unrelated time context.")
        context.derivative(candidate)
        self._states = (frozen_copy(candidate), self._states[0])
        self.time = context.time
        self._generation += 1

    def snapshot(self):
        return self.time, self._generation, self.states

    def restore(self, snapshot):
        time, generation, states = snapshot
        if not np.isfinite(time) or not isinstance(generation, int) or len(states) not in (1, 2):
            raise ValueError("Invalid time history snapshot.")
        copied = tuple(frozen_copy(state) for state in states)
        if any(state.shape != self._states[0].shape for state in copied):
            raise ValueError("Snapshot has an incompatible shape.")
        self.time, self._generation, self._states = time, generation, copied

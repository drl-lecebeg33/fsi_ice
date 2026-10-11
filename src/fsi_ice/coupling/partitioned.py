"""Dirichlet--Neumann iteration with fixed or scaled Aitken relaxation.

Callbacks evaluate candidates against a fixed accepted history. They must not
accept time histories or publish states. The outer full-field transaction owns
the eventual acceptance. See docs/development/t18_plan.md for norm conventions.
"""

from dataclasses import dataclass

import numpy as np


def _vector(value, name, shape=None):
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 1 or not array.size or (shape is not None and array.shape != shape):
        raise ValueError(f"{name} must be a nonempty vector" +
                         (f" of shape {shape}." if shape is not None else "."))
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CouplingOptions:
    """Dimensionless tolerances; max_iterations counts combined evaluations."""

    relaxation: float = 0.5
    absolute_tolerance: float = 1e-11
    relative_tolerance: float = 1e-9
    max_iterations: int = 150
    case_name: str = "partitioned"
    method: str = "fixed"
    minimum_relaxation: float = 0.05
    maximum_relaxation: float = 1.0
    aitken_denominator_tolerance: float = 1e-14

    def __post_init__(self):
        extra_scalars = ("minimum_relaxation", "maximum_relaxation", "aitken_denominator_tolerance")
        for name in ("relaxation", "absolute_tolerance", "relative_tolerance", *extra_scalars):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or \
                    (name in extra_scalars and isinstance(value, (str, bytes, complex, np.complexfloating))):
                raise ValueError(f"{name} must be a finite real scalar.")
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"{name} must be a finite real scalar.") from None
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if not 0 < self.relaxation <= 1:
            raise ValueError("relaxation must lie in (0,1].")
        if min(self.absolute_tolerance, self.relative_tolerance) < 0 or \
                self.absolute_tolerance + self.relative_tolerance == 0:
            raise ValueError("Tolerances must be nonnegative and not both zero.")
        if isinstance(self.max_iterations, (bool, np.bool_)) or \
                not isinstance(self.max_iterations, (int, np.integer)) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer.")
        if not isinstance(self.case_name, str) or not self.case_name.strip():
            raise ValueError("case_name must be a nonempty string.")
        if not isinstance(self.method, str) or self.method not in ("fixed", "aitken"):
            raise ValueError("method must be 'fixed' or 'aitken'.")
        if not 0 < self.minimum_relaxation <= self.maximum_relaxation:
            raise ValueError("Relaxation bounds must satisfy 0 < minimum_relaxation <= maximum_relaxation.")
        if self.aitken_denominator_tolerance <= 0:
            raise ValueError("aitken_denominator_tolerance must be strictly positive.")
        if self.method == "aitken" and not self.minimum_relaxation <= self.relaxation <= self.maximum_relaxation:
            raise ValueError("Aitken requires minimum_relaxation <= relaxation <= maximum_relaxation.")


@dataclass(frozen=True)
class PartitionedEvaluation:
    """G(x) and the single-field states computed from the same input x.

    fluid, load and solid are opaque physical result objects. The returned
    target is copied by the controller. It never overwrites x with G(x) at
    convergence: fluid data were computed using x, with the recorded gap to G(x).
    """

    target: np.ndarray
    fluid: object
    load: object
    solid: object


class _StageFailure(RuntimeError):
    def __init__(self, stage, cause):
        self.stage = stage
        super().__init__(str(cause))


@dataclass(frozen=True)
class DirichletNeumannMap:
    """Adapter: fluid_solve(x) -> load_transfer(fluid) -> solid_solve(load).

    solid_trace(solid) extracts the interface vector in the exact ordering and
    units of x. Geometry and BDF contexts belong to the caller's candidate
    workspace. This adapter reuses solvers rather than defining new physics.
    """

    fluid_solve: object
    load_transfer: object
    solid_solve: object
    solid_trace: object

    def __post_init__(self):
        for name in ("fluid_solve", "load_transfer", "solid_solve", "solid_trace"):
            if not callable(getattr(self, name)):
                raise TypeError(f"DirichletNeumannMap.{name} must be callable.")

    def __call__(self, interface_values):
        stage = "fluid"
        try:
            fluid = self.fluid_solve(interface_values)
            stage = "load"
            load = self.load_transfer(fluid)
            stage = "solid"
            solid = self.solid_solve(load)
            stage = "trace"
            target = self.solid_trace(solid)
        except Exception as error:
            raise _StageFailure(stage, error) from error
        return PartitionedEvaluation(target, fluid, load, solid)


@dataclass(frozen=True)
class CouplingIteration:
    iteration: int
    residual_norm: float
    threshold: float
    relaxation: float
    residual: np.ndarray
    relaxation_reason: str = "fixed"


@dataclass(frozen=True)
class CouplingResult:
    interface_values: np.ndarray
    evaluation: PartitionedEvaluation
    history: tuple[CouplingIteration, ...]
    reason: str = "interface_residual"


class CouplingFailure(RuntimeError):
    """Terminal candidate failure, with evidence available to a transaction."""

    def __init__(self, case_name, iteration, stage, reason, values, history):
        self.case_name = case_name
        self.iteration = iteration
        self.stage = stage
        self.reason = reason
        self.interface_values = _vector(values, "failed interface input")
        self.history = tuple(history)
        residual = f"; residual={history[-1].residual_norm:.3e}" if history else ""
        super().__init__(f"{case_name}: coupling iteration {iteration}, stage {stage}: {reason}{residual}")


def solve_dirichlet_neumann(evaluate, initial, *, scales, options=None):
    """Evaluate G(x) in order, test its true gap, then apply x += omega*(G(x)-x).

    scales is a positive physical scale vector matching initial. Stop when
    ||(G(x)-x)/scales||_inf <= atol + rtol*max(||x/scales||_inf, ||G(x)/scales||_inf).
    Input arrays are never mutated, including when a callback re-enables writes
    on its private input. Exhaustion and field exceptions never return success.
    Aitken inner products use r/scales. A relatively degenerate difference keeps
    the last coefficient; a non-finite calculation fails with diagnostics.
    History relaxation describes the outgoing update, except terminal rows,
    which retain the last coefficient and never apply an unassessed update.
    """
    if not callable(evaluate):
        raise TypeError("evaluate must be callable.")
    options = CouplingOptions() if options is None else options
    if not isinstance(options, CouplingOptions):
        raise TypeError("options must be CouplingOptions.")
    values = _vector(initial, "initial")
    scales = _vector(scales, "scales", values.shape)
    if np.any(scales <= 0):
        raise ValueError("scales must be strictly positive.")
    history = []
    previous_scaled_residual = None
    relaxation = options.relaxation
    for iteration in range(1, options.max_iterations + 1):
        stage = "evaluate"
        try:
            evaluated = evaluate(_vector(values, "callback input"))
            stage = "residual"
            if not isinstance(evaluated, PartitionedEvaluation):
                raise TypeError("evaluate must return PartitionedEvaluation.")
            target = _vector(evaluated.target, "interface target", values.shape)
            # Reject arithmetic overflow even when every unscaled input is finite.
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                residual = _vector(target - values, "interface residual", values.shape)
                norm = float(np.max(np.abs(residual / scales)))
                magnitude = max(float(np.max(np.abs(values / scales))),
                                float(np.max(np.abs(target / scales))))
                threshold = options.absolute_tolerance + options.relative_tolerance*magnitude
            if not np.isfinite(threshold):
                raise ValueError("Coupling convergence threshold overflowed.")
        except Exception as error:
            phase = error.stage if isinstance(error, _StageFailure) else stage
            raise CouplingFailure(options.case_name, iteration, phase, str(error), values, history) from error
        if norm <= threshold:
            history.append(CouplingIteration(iteration, norm, threshold, relaxation, residual, "converged"))
            evaluation = PartitionedEvaluation(target, evaluated.fluid, evaluated.load, evaluated.solid)
            return CouplingResult(_vector(values, "converged input"), evaluation, tuple(history))
        if iteration == options.max_iterations:
            history.append(CouplingIteration(iteration, norm, threshold, relaxation, residual, "max_iterations"))
            raise CouplingFailure(options.case_name, iteration, "convergence", "maximum iterations exceeded",
                                  values, history)
        relaxation_reason = "fixed"
        if options.method == "aitken":
            relaxation_reason = "initial"
            scaled_residual = residual / scales
            if previous_scaled_residual is not None:
                # Relative denominator test in the same dimensionless inner
                # product as the coefficient; independent of component units.
                with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                    delta = scaled_residual - previous_scaled_residual
                    denominator = float(delta @ delta)
                    numerator = float(previous_scaled_residual @ delta)
                    residual_size = max(float(previous_scaled_residual @ previous_scaled_residual),
                                        float(scaled_residual @ scaled_residual))
                    finite_products = np.all(np.isfinite([denominator, numerator, residual_size]))
                    degenerate = finite_products and denominator <= options.aitken_denominator_tolerance*residual_size
                    coefficient = relaxation if degenerate or not finite_products else \
                        -relaxation * (numerator / denominator)
                if not finite_products or not np.isfinite(coefficient):
                    history.append(CouplingIteration(iteration, norm, threshold, relaxation, residual,
                                                     "coefficient_nonfinite"))
                    raise CouplingFailure(options.case_name, iteration, "relaxation",
                                          "Aitken coefficient arithmetic is non-finite", values, history)
                if degenerate:
                    relaxation_reason = "denominator_degenerate"
                else:
                    relaxation = float(np.clip(coefficient, options.minimum_relaxation,
                                               options.maximum_relaxation))
                    relaxation_reason = "clipped" if relaxation != coefficient else "aitken"
            previous_scaled_residual = scaled_residual
        history.append(CouplingIteration(iteration, norm, threshold, relaxation, residual, relaxation_reason))
        try:
            with np.errstate(over="raise", invalid="raise"):
                next_values = _vector(values + relaxation*residual, "relaxed interface input")
        except Exception as error:
            raise CouplingFailure(options.case_name, iteration, "relaxation", str(error), values, history) from error
        values = next_values

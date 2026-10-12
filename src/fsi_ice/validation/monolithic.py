"""Direction-derivative checks of the monolithic residual and Jacobian.

This module is the reusable verification entry point requested by T21 and
reused by the later T23 consistency study.  It never touches solver state: the
central differences only call
:func:`fsi_ice.coupling.monolithic.monolithic_residual` and the analytic
response comes from ``monolithic_jvp`` (matrix free) or from an assembled
Jacobian.

Conventions (frozen with the T21 task card)

* a *physical* direction ``direction`` is normalised by the per-variable
  physical scales ``variable_scales`` (elementwise) before differencing, so the
  same scan works for velocity, pressure, displacement and nodal-load blocks;
* errors are measured after dividing each residual row by its per-block
  residual scale ``residual_scales`` and taking the 2-norm;
* ``relative_error(h) = ||(fd_h - analytic)/S_R|| / ||analytic/S_R||``; when the
  analytic response is numerically zero the absolute (scaled) error is the
  criterion, reported together with ``degenerate=True``;
* the scan records the truncation-then-rounding trend of the central
  differences; the best (smallest) relative error must not exceed the frozen
  ``spec.md`` threshold of ``1e-5`` for non-degenerate directions.  A
  well-conditioned problem needs steps below ``1e-7`` before rounding dominates,
  so the trend scan uses :data:`EXTENDED_STEPS` while the required gate applies
  to the whole set.
"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from fsi_ice.coupling.monolithic import BLOCK_NAMES, monolithic_jvp, monolithic_residual

#: Central-difference steps of the frozen perturbation scan (m3/T21: 1e-2 ... 1e-7).
DEFAULT_STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)

#: Steps used to exhibit the truncation-then-rounding trend (superset of the
#: frozen range; the required 1e-5 gate holds over the whole set).
EXTENDED_STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)

#: Frozen acceptance threshold of ``spec.md`` section 5.2 / ``m3_freeze.md`` 6.
RELATIVE_DERIVATIVE_LIMIT = 1e-5

#: Rectangular Jacobian blocks whose omission must be detectable (21-04 audit).
#: Rows and columns are block names or contiguous groups of ``BLOCK_NAMES``.
BLOCK_RECTANGLES = MappingProxyType({
    "fluid_geometry_j_um_j_pm": ("u_f,p_f", "d_m"),
    "load_fluid_l_u_l_p": ("g", "u_f,p_f"),
    "load_mesh_l_dm": ("g", "d_m"),
    "load_solid_l_ds": ("g", "d_s"),
    "solid_load_s_h_t": ("v_s", "g"),
    "mesh_solid_j_m_ds": ("d_m", "d_s"),
    "solid_dynamics_j_sd": ("d_s", "v_s"),
    "flow_pressure_j_pu": ("p_f", "u_f"),
})


def _block_slice(layout, names):
    if isinstance(names, str):
        names = tuple(part for part in names.split(",") if part)
    indices = [BLOCK_NAMES.index(name) for name in names]
    if sorted(indices) != list(range(min(indices), max(indices)+1)):
        raise ValueError(f"Block group {names} must be contiguous in {BLOCK_NAMES}.")
    return slice(layout.starts[min(indices)], layout.starts[max(indices)]+layout.sizes[max(indices)])


def block_scale_vector(layout, per_block):
    """Full-length scale vector from a ``{block: scalar or array}`` mapping."""
    vector = np.empty(layout.n_total)
    for name in BLOCK_NAMES:
        if name not in per_block:
            raise ValueError(f"Missing scale for block {name!r}.")
        value = np.asarray(per_block[name], dtype=float)
        if value.ndim == 0:
            vector[layout.slice_of(name)] = float(value)
        else:
            if value.shape != layout.shape_of(name):
                raise ValueError(f"Scale for {name!r} must have shape {layout.shape_of(name)}, got {value.shape}.")
            vector[layout.slice_of(name)] = value.ravel()
    if not np.all(np.isfinite(vector)) or np.any(vector <= 0):
        raise ValueError("All per-variable scales must be finite and strictly positive.")
    return vector


@dataclass(frozen=True)
class DirectionalDerivativeResult:
    """Central-difference scan of one direction against the analytic derivative."""

    steps: tuple
    relative_errors: tuple
    absolute_errors: tuple
    analytic_norm: float
    degenerate: bool
    best_relative_error: float
    best_relative_step: float
    best_absolute_error: float
    best_absolute_step: float
    trend: str

    def summary(self):
        return {
            "steps": list(self.steps),
            "relative_errors": list(self.relative_errors),
            "absolute_errors": list(self.absolute_errors),
            "analytic_norm": self.analytic_norm,
            "degenerate": self.degenerate,
            "best_relative_error": self.best_relative_error,
            "best_relative_step": self.best_relative_step,
            "best_absolute_error": self.best_absolute_error,
            "best_absolute_step": self.best_absolute_step,
            "trend": self.trend,
        }


def _trend_of(errors):
    values = np.asarray(errors, dtype=float)
    best = int(np.argmin(values))
    if 0 < best < len(values)-1 and values[0] > values[best] and values[-1] > values[best]:
        return "decreasing_then_increasing"
    return "not_resolved"


def _validated_inputs(x, direction, context, variable_scales, residual_scales):
    size = context.layout.n_total
    scales = np.asarray(variable_scales, dtype=float)
    residual_scale = np.asarray(residual_scales, dtype=float)
    for name, values in (("variable_scales", scales), ("residual_scales", residual_scale)):
        if values.shape != (size,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{name} must be a finite positive ({size},) vector.")
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (size,) or not np.all(np.isfinite(direction)):
        raise ValueError(f"direction must be a finite ({size},) vector.")
    if not np.any(direction):
        raise ValueError("direction must be nonzero.")
    return direction/scales, residual_scale


def central_difference_vectors(x, direction, *, context, variable_scales, steps=DEFAULT_STEPS):
    """Central differences ``(R(x+hq) - R(x-hq)) / 2h`` for every step.

    ``q`` is the physical ``direction`` divided by ``variable_scales``.  The
    residual evaluations never modify ``x`` or the context, so the vectors can
    be shared by several analytic variants (used by :func:`completeness_audit`).
    """
    q, _ = _validated_inputs(x, direction, context, variable_scales, np.ones(context.layout.n_total))
    vectors = []
    for step in steps:
        step = float(step)
        if not np.isfinite(step) or step <= 0:
            raise ValueError("All scan steps must be finite and positive.")
        plus = monolithic_residual(x+step*q, context=context)
        minus = monolithic_residual(x-step*q, context=context)
        vectors.append(np.asarray((plus-minus)/(2*step), dtype=float))
    return q, vectors


def _scan_from_vectors(vectors, analytic, residual_scale, steps, degenerate_tolerance):
    analytic = np.asarray(analytic, dtype=float)
    analytic_norm = float(np.linalg.norm(analytic/residual_scale))
    degenerate = analytic_norm <= degenerate_tolerance
    relative, absolute = [], []
    for vector in vectors:
        absolute_error = float(np.linalg.norm((vector-analytic)/residual_scale))
        absolute.append(absolute_error)
        relative.append(float("inf") if degenerate else absolute_error/analytic_norm)
    best_absolute = int(np.argmin(absolute))
    best_relative = best_absolute if degenerate else int(np.argmin(relative))
    return DirectionalDerivativeResult(
        steps=tuple(float(step) for step in steps),
        relative_errors=tuple(relative),
        absolute_errors=tuple(absolute),
        analytic_norm=analytic_norm,
        degenerate=degenerate,
        best_relative_error=relative[best_relative],
        best_relative_step=float(steps[best_relative]),
        best_absolute_error=absolute[best_absolute],
        best_absolute_step=float(steps[best_absolute]),
        trend=_trend_of(absolute),
    )


def directional_derivative_scan(x, direction, *, context, variable_scales, residual_scales,
                               steps=DEFAULT_STEPS, analytic=None, degenerate_tolerance=1e-12):
    """Scan central differences of ``direction`` against the analytic derivative.

    ``direction`` is given in physical units and normalised by
    ``variable_scales``; ``residual_scales`` normalises the residual rows.
    ``analytic`` overrides the matrix-free ``monolithic_jvp`` result (e.g. to
    audit an assembled Jacobian).  Nothing is modified; repeated calls with the
    same inputs give identical numbers.
    """
    q, vectors = central_difference_vectors(x, direction, context=context,
                                            variable_scales=variable_scales, steps=steps)
    if analytic is None:
        analytic = monolithic_jvp(x, q, context=context)
    residual_scale = np.asarray(residual_scales, dtype=float)
    _validated_inputs(x, direction, context, variable_scales, residual_scale)
    return _scan_from_vectors(vectors, analytic, residual_scale, steps, degenerate_tolerance)


def jacobian_jvp_gap(x, direction, *, context, jacobian=None, variable_scales=None, residual_scales=None):
    """Agreement of ``J @ q`` (assembled) with the matrix-free ``monolithic_jvp``.

    Returns ``(max_abs_gap, scaled_relative_gap)``; ``q = direction`` when no
    variable scales are given, otherwise ``q = direction/variable_scales``.
    """
    from fsi_ice.coupling.monolithic import monolithic_jacobian

    q = np.asarray(direction, dtype=float)
    if variable_scales is not None:
        q = q/np.asarray(variable_scales, dtype=float)
    matrix = monolithic_jacobian(x, context=context) if jacobian is None else jacobian
    assembled = np.asarray(matrix @ q).ravel()
    free = np.asarray(monolithic_jvp(x, q, context=context)).ravel()
    gap = assembled-free
    scale = np.ones_like(gap) if residual_scales is None else np.asarray(residual_scales, dtype=float)
    reference = max(float(np.linalg.norm(free/scale)), np.finfo(float).tiny)
    return float(np.max(np.abs(gap))), float(np.linalg.norm(gap/scale)/reference)


def completeness_audit(x, direction, *, context, jacobian=None, variable_scales, residual_scales,
                       steps=DEFAULT_STEPS):
    """Drop-one-block completeness audit of the assembled Jacobian.

    The full Jacobian must reproduce the central differences within
    ``RELATIVE_DERIVATIVE_LIMIT``; removing one coupling rectangle must push the
    error above that limit for a direction that exercises the block, otherwise
    the audit reports it as not detected.  Returns ``{variant: scan}`` with
    ``"full"`` first.
    """
    from fsi_ice.coupling.monolithic import monolithic_jacobian

    layout = context.layout
    matrix = monolithic_jacobian(x, context=context) if jacobian is None else jacobian
    q, vectors = central_difference_vectors(x, direction, context=context,
                                            variable_scales=variable_scales, steps=steps)
    residual_scale = np.asarray(residual_scales, dtype=float)
    _validated_inputs(x, direction, context, variable_scales, residual_scale)
    reports = {"full": _scan_from_vectors(vectors, matrix @ q, residual_scale, steps, 0.0)}
    for name, (rows, columns) in BLOCK_RECTANGLES.items():
        dropped = drop_jacobian_block(matrix, _block_slice(layout, rows), _block_slice(layout, columns))
        reports[name] = _scan_from_vectors(vectors, dropped @ q, residual_scale, steps, 0.0)
    return reports


def drop_jacobian_block(matrix, row_slice, column_slice):
    """Zero a rectangular block of a CSR matrix (copy-on-write)."""
    result = matrix.tocsr(copy=True)
    row_mask = np.zeros(result.shape[0], dtype=bool)
    row_mask[np.arange(result.shape[0])[row_slice]] = True
    row_of = np.repeat(np.arange(result.shape[0]), np.diff(result.indptr))
    inside = row_mask[row_of] & (result.indices >= column_slice.start) & (result.indices < column_slice.stop)
    if np.any(inside):
        result.data[inside] = 0.0
        result.eliminate_zeros()
    return result

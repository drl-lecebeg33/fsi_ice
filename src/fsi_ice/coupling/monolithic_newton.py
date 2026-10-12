"""Monolithic Newton integration and steady marching (M5/T22, A6).

This module binds the frozen T21 evaluation contract
(:func:`fsi_ice.coupling.monolithic.evaluate_monolithic`) to the generic damped
Newton prototype (:mod:`fsi_ice.coupling.newton`) and adds the pseudo-time
march that drives the transient BDF step residual to a steady fixed point.

Adapter and tangent routes
--------------------------

``solve_monolithic_step`` wraps ``MonolithicContext`` and a candidate ``x0``
into the ``evaluate(x) -> Evaluation`` protocol of :func:`solve_newton`:

* every line-search trial is evaluated with ``mode="residual"``; a ``None``
  Jacobian in that mode is normal and never raises;
* the tangent of an accepted iterate is materialized lazily when
  ``solve_newton`` reads the evaluation's ``jacobian`` attribute.  With
  ``jacobian_mode="assembled"`` (default) that is one additional
  ``mode="assembled"`` evaluation at the same ``x``; with
  ``jacobian_mode="jvp"`` it is a matrix-free ``LinearOperator`` whose
  ``matvec`` calls ``mode="jvp"`` once per direction.  The route is explicit
  configuration, recorded in every result, counted separately and never
  silently switched; an unavailable or failing route raises
  :class:`MonolithicNewtonFailure` instead of falling back.

``preconditioner`` is an optional callable ``f(x) -> operator`` used only by the
matrix-free route; the returned object is forwarded to GMRES as ``M``.  T22
ships no preconditioner for the real FSI problem (see the validation record):
unpreconditioned GMRES is retained as an explicit, counted option.

Interface compatibility
-----------------------

The T21 residual domain requires ``d_m|Gamma = H_d d_s|Gamma`` (one-to-one
matched P1 trace, ``H_d = I``).  Constraint rows are linear, so a Newton trial
``x0 + alpha d`` keeps a feasible ``x0`` feasible for every ``alpha``; an
infeasible ``x0`` would leave the domain of the candidate geometry.  The step
driver therefore checks the gap before solving (``compatibility="require"``,
default: a violation raises a structured failure naming the constraint) or,
when explicitly configured with ``compatibility="project"``, projects the mesh
trace of ``x0`` onto the solid trace once
(:func:`project_interface_compatibility`).  The gap before the policy is
recorded on the result and on failures.

Steady march
------------

``march_to_steady`` starts from the latest accepted state of the template
context, builds a fresh ``BDFHistory`` per field and marches pseudo-time steps
of constant ``dt`` (BDF1 startup for BDF2, as in :mod:`fsi_ice.time.bdf`).
Each converged step is accepted as the next history entry
(``generation``/``step_id`` increment by one) and a new context is built with
``candidate_id = (template.candidate_id, "march", step)``; the template context
and its arrays are never modified.

The stopping criterion is the scaled BDF time term of the converged step,

    ``max_block max_i |a_0 x_i + sum_{j>=1} a_j x_i^{k-j}| / S_i <= steady_tolerance``

over the state blocks ``u_f``, ``d_s`` and ``v_s`` (the only blocks with a time
derivative in the monolithic residual); ``a_j`` are the context stencil
coefficients and ``x^{k-j}`` the accepted history latest first.  For BDF1 this
is exactly ``a_0 |x^k - x^{k-1}|``; for BDF2 with equal last two accepted
states it reduces to the same expression.  At a genuine fixed point the BDF
time term cancels (``sum_j a_j = 0``) and the step residual is the steady
discrete equation.  ``steady_tolerance`` is dimensionless per block (the block
term is divided by the block's physical scale ``S``; numerically it has units
of 1/s, i.e. a relative state-change rate).  The march stops with
``reason="steady"``, or with ``reason="max_steps"`` after ``max_steps``
converged steps (``steady_reached=False``); Newton failures propagate as
:class:`MonolithicNewtonFailure` with the completed step records, counters and
timings attached.

Cost accounting
---------------

``MonolithicCounters`` uses the frozen T23 field names (``steps``,
``newton_iterations``, ``evaluations``, ``linear_solves``, ``linear_iterations``,
``line_search_trials``, ``failed_trials``, ``failed_steps``) plus the
per-mode breakdown.  ``linear_iterations`` is 0 on the assembled route (a
direct factorization has no Krylov iterations; the result's
``linear_solver="splu"`` marks this), and counts GMRES inner iterations on the
matrix-free route (``linear_solver="gmres"``).  ``evaluations`` counts every
T21 evaluation call, split into ``residual_evaluations``,
``assembled_evaluations`` and ``jvp_evaluations`` (``jvp_directions`` counts
the directions submitted to ``mode="jvp"``); the diagnostic final residual is
one more residual evaluation and is included.  Timing buckets are wall-clock
seconds from ``time.perf_counter``: ``evaluate_seconds`` (inside every
evaluation call), ``linear_seconds`` (inside ``splu``/GMRES; a matrix-free
matvec is counted in both buckets), ``newton_seconds`` (the ``solve_newton``
call) and ``total_seconds`` (the whole step or march).  T23 §4.1 aggregation
conventions apply.

Every returned object is read-only: ``x``, ``residual`` and ``scales`` are
frozen copies, iteration and step records are tuples and mappings are exposed
through ``MappingProxyType``.

中文说明
--------

本模块把 T21 的整体耦合求值契约（:func:`fsi_ice.coupling.monolithic.evaluate_monolithic`）
接到通用阻尼 Newton 原型（:mod:`fsi_ice.coupling.newton`）上，并加上把瞬态 BDF 步残差推到
定常不动点的"伪时间推进"。

关键约定：

* **切线路线显式选择**：``jacobian_mode`` 取 ``"assembled"``（默认，接受点额外一次
  ``mode="assembled"`` 求值得到完整 Jacobian，用 splu 直接分解）或 ``"jvp"``（矩阵无关
  ``LinearOperator``，每次矩阵-向量乘调一次 ``mode="jvp"``，用 GMRES）。路线绝不隐式切换、
  也不回退，不可用或失败一律抛 :class:`MonolithicNewtonFailure`。
* **界面相容性**：T21 残差域要求 ``d_m|Gamma = H_d d_s|Gamma``（一一参考匹配，``H_d = I``）。
  约束行线性，故可行 ``x0`` 经任意 Newton 步仍可行。``solve_monolithic_step`` 在求解前检查
  间隙：``compatibility="require"``（默认）违反即结构化报错；``compatibility="project"`` 则把
  网格迹投影到固体迹（:func:`project_interface_compatibility`）。策略前的间隙记录在结果与失败
  上。
* **伪时间定常推进**：:func:`march_to_steady` 从模板上下文的最近已接受状态出发，为各场新建
  ``BDFHistory``，用恒定 ``dt`` 逐步推进（``order=2`` 用 BDF1 启动）。每步收敛后接受为下一条
  历史，``generation``/``step_id`` 加一，并以
  ``candidate_id = (template.candidate_id, "march", step)`` 建新上下文；模板及其数组不被修改。
  停机判据是收敛步的缩放 BDF 时间项：``max_block max_i |a_0 x_i + sum_{j>=1} a_j x_i^{k-j}| / S_i
  <= steady_tolerance``，只对含时间导数的块 ``u_f``、``d_s``、``v_s`` 取，``S`` 为该块物理尺度
  （无量纲，数值上量纲为 1/s 的"相对状态变化率"）。在真不动点处 BDF 时间项相消
  （``sum_j a_j = 0``），残差即为定常离散方程。
* **代价统计**：:class:`MonolithicCounters` 用 T23 冻结字段名并附各模式细分；
  ``linear_iterations`` 在直接分解路线上为 0、在 GMRES 路线上计入内迭代；所有返回对象只读。
"""

# 说明：以上为模块 docstring 的一部分（因此 "from __future__" 仍是其后第一条语句）。
from __future__ import annotations

from dataclasses import dataclass, fields
from time import perf_counter
from types import MappingProxyType

import numpy as np
from scipy.sparse.linalg import LinearOperator

from fsi_ice.ale import build_ale_geometry
from fsi_ice.coupling.monolithic import (BLOCK_NAMES, MonolithicBase, MonolithicContext,
                                         evaluate_monolithic)
from fsi_ice.coupling.newton import (NewtonFailure, NewtonOptions, NewtonTelemetry, solve_newton)
from fsi_ice.time import BDFHistory

__all__ = [
    "JACOBIAN_MODES",
    "MonolithicCounters",
    "MonolithicMarchResult",
    "MonolithicMarchStep",
    "MonolithicNewtonFailure",
    "MonolithicStepResult",
    "interface_compatibility_gap",
    "march_to_steady",
    "project_interface_compatibility",
    "scaled_time_terms",
    "solve_monolithic_step",
]

#: Explicit tangent routes of the monolithic adapter; no other value is accepted.
# 中文：切线路线仅此两种，显式选择、不隐式推断，也不回退。
JACOBIAN_MODES = ("assembled", "jvp")

#: State blocks that carry a BDF time derivative in the monolithic residual.
# 中文：整体残差中带 BDF 时间导数的状态块（定常判据只对这些块取时间项）。
TIME_TERM_BLOCKS = ("u_f", "d_s", "v_s")

#: Compatibility policies of the step driver.
# 中文：步进驱动的界面相容性策略：require=违反即报错，project=先投影网格迹到固体迹。
COMPATIBILITY_POLICIES = ("require", "project")


@dataclass(frozen=True)
class MonolithicCounters:
    """Work counters of one step or one march (T23 §4.1 field names first).

    ``steps`` is the number of completed march steps (1 for a single step);
    ``newton_iterations`` counts accepted Newton iterations excluding the start
    record; ``evaluations`` counts every monolithic evaluation call and is
    broken down by mode; ``linear_iterations`` are GMRES inner iterations (0
    with the direct factorization, marked by the result's ``linear_solver``);
    ``failed_trials`` are rejected line-search trials and ``failed_steps`` the
    failed steps (1 when a step failed, 0 otherwise).

    中文：一步或一次推进的工作量计数器（先按 T23 §4.1 冻结字段名）。``steps`` 为完成的推进
    步数（单步求解为 1）；``newton_iterations`` 为不含起始记录的已接受 Newton 迭代数；
    ``evaluations`` 为全部整体求值调用次数并细分为 residual/assembled/jvp；``jvp_directions``
    为提交给 ``mode="jvp"`` 的方向数；``linear_iterations`` 为 GMRES 内迭代数（直接分解路线为 0，
    由结果的 ``linear_solver`` 标记）；``failed_trials`` 为被拒绝的线搜索试验，``failed_steps``
    为失败步数（失败为 1，否则 0）。``as_dict`` 返回普通整数字典。
    """

    steps: int = 0
    newton_iterations: int = 0
    evaluations: int = 0
    residual_evaluations: int = 0
    assembled_evaluations: int = 0
    jvp_evaluations: int = 0
    jvp_directions: int = 0
    jacobian_reads: int = 0
    linear_solves: int = 0
    linear_iterations: int = 0
    line_search_trials: int = 0
    failed_trials: int = 0
    failed_steps: int = 0

    def __post_init__(self):
        for entry in fields(self):
            value = getattr(self, entry.name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"MonolithicCounters.{entry.name} must be a nonnegative integer.")

    def __add__(self, other):
        if not isinstance(other, MonolithicCounters):
            return NotImplemented
        return MonolithicCounters(**{entry.name: getattr(self, entry.name)+getattr(other, entry.name)
                                     for entry in fields(self)})

    def as_dict(self):
        return {entry.name: int(getattr(self, entry.name)) for entry in fields(self)}


class _Accounting:
    """Mutable work/timing accumulator of one step; never shared across runs.

    中文：单步（或单次推进）的可变工作量/计时累加器，每次运行独立创建、不跨运行共享；
    ``absorb_telemetry`` 把通用 Newton 原型返回的局部计数与计时并入本累加器。
    """

    def __init__(self):
        for entry in fields(MonolithicCounters):
            setattr(self, entry.name, 0)
        self.evaluate_seconds = 0.0
        self.linear_seconds = 0.0
        self.newton_seconds = 0.0
        self.total_seconds = 0.0

    def counters(self, *, failed_steps=None):
        values = {entry.name: getattr(self, entry.name) for entry in fields(MonolithicCounters)}
        if failed_steps is not None:
            values["failed_steps"] = int(failed_steps)
        return MonolithicCounters(**values)

    def timings(self):
        return MappingProxyType({"evaluate_seconds": float(self.evaluate_seconds),
                                 "linear_seconds": float(self.linear_seconds),
                                 "newton_seconds": float(self.newton_seconds),
                                 "total_seconds": float(self.total_seconds)})

    def absorb_telemetry(self, telemetry):
        """Merge the partial counts/timings of the generic Newton prototype."""
        self.linear_solves += telemetry.linear_solves
        self.linear_iterations += telemetry.linear_iterations
        self.line_search_trials += telemetry.line_search_trials
        self.failed_trials += telemetry.failed_trials
        self.linear_seconds += telemetry.linear_seconds


class MonolithicNewtonFailure(NewtonFailure):
    """``newton.NewtonFailure`` enriched with the T22 context.

    The frozen phase list of the prototype is reused (``evaluate``,
    ``residual``, ``jacobian``, ``linear_solve``, ``line_search``,
    ``convergence``); T22 adds attributes and plain diagnostics: the context
    ``candidate_id`` (as ``repr``), the explicit ``jacobian_mode``, the
    interface compatibility gap, per-block scaled residual norms, the partial
    ``counters``/``timings``, the failing march ``step_index`` and the completed
    ``march_steps`` records.  ``StepTransaction`` forwards ``diagnostics``
    unchanged, so the exception can be raised from a plan callback as is.

    中文：由 ``newton.NewtonFailure`` 派生、附加 T22 上下文的失败异常。沿用原型的冻结阶段
    列表（``evaluate``/``residual``/``jacobian``/``linear_solve``/``line_search``/
    ``convergence``），并加入：上下文 ``candidate_id``（以 ``repr`` 存储）、显式
    ``jacobian_mode``、界面相容间隙、逐块缩放残差范数、部分 ``counters``/``timings``、失败的
    推进 ``step_index`` 与已完成 ``march_steps`` 记录；这些上下文同时并入 ``diagnostics``。
    """

    def __init__(self, *, phase, iteration, reason, residual_norm=None, raw_residual_norm=None,
                 x=None, history=(), diagnostics=None, candidate_id=None, jacobian_mode=None,
                 interface_gap=None, block_norms=None, counters=None, timings=None,
                 step_index=None, march_steps=()):
        merged = dict(diagnostics or {})
        if candidate_id is not None:
            merged["candidate_id"] = repr(candidate_id)
        if jacobian_mode is not None:
            merged["jacobian_mode"] = str(jacobian_mode)
        if interface_gap is not None:
            merged["interface_gap"] = float(interface_gap)
        if block_norms is not None:
            merged["block_norms"] = {str(name): float(value) for name, value in block_norms.items()}
        if counters is not None:
            merged["counters"] = counters.as_dict()
        if timings is not None:
            merged["timings"] = {str(name): float(value) for name, value in timings.items()}
        if step_index is not None:
            merged["step_index"] = int(step_index)
        super().__init__(phase=phase, iteration=iteration, reason=reason,
                         residual_norm=residual_norm, raw_residual_norm=raw_residual_norm,
                         x=x, history=history, diagnostics=merged)
        self.candidate_id = candidate_id
        self.jacobian_mode = jacobian_mode
        self.interface_gap = None if interface_gap is None else float(interface_gap)
        self.block_norms = None if block_norms is None else MappingProxyType(
            {str(name): float(value) for name, value in block_norms.items()})
        self.counters = counters
        self.timings = None if timings is None else MappingProxyType(
            {str(name): float(value) for name, value in timings.items()})
        self.step_index = step_index
        self.march_steps = tuple(march_steps)

    @classmethod
    def from_failure(cls, failure, **extra):
        """Copy a :class:`NewtonFailure` into this class, adding T22 context.

        中文：把通用原型抛出的 :class:`NewtonFailure` 复制成本类，并附加 T22 上下文
        （``extra`` 关键字如 ``candidate_id``/``jacobian_mode``/``counters`` 等）。
        """
        if not isinstance(failure, NewtonFailure):
            raise TypeError("failure must be a NewtonFailure instance.")
        return cls(phase=failure.phase, iteration=failure.iteration, reason=failure.reason,
                   residual_norm=failure.residual_norm, raw_residual_norm=failure.raw_residual_norm,
                   x=failure.x, history=failure.history, diagnostics=failure.diagnostics, **extra)


def _finite_vector(values, size, name):
    """Independent read-only finite 1-D float array or ``ValueError``.

    中文：把输入复制成独立、只读、有限的长度 ``size`` 一维浮点数组，作为不共享内存的防御性
    拷贝；形状不符或含非有限值时抛出 ``ValueError`` 并带上 ``name``。
    """
    try:
        array = np.array(values, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a real numeric one-dimensional array.") from None
    if array.ndim != 1 or array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    array.setflags(write=False)
    return array


def _validated_scales(scales, size):
    # 缩放因子缺省为全 1；必须严格为正，用作收敛判据 atol + rtol*||r0||_S 的分量权重。
    if scales is None:
        scale = np.ones(size)
        scale.setflags(write=False)
        return scale
    scale = _finite_vector(scales, size, "scales")
    if np.any(scale <= 0.0):
        raise ValueError("scales must be strictly positive.")
    return scale


def _validated_x0(x0, context):
    # 校验起点候选 x0：形状须等于布局总长，且必须有限；非有限时以结构化失败抛出。
    try:
        values = np.array(x0, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("x0 must be a real numeric one-dimensional array.") from None
    if values.ndim != 1 or values.shape != (context.layout.n_total,):
        raise ValueError(f"x0 must have shape ({context.layout.n_total},), got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise MonolithicNewtonFailure(
            phase="evaluate", iteration=0, reason="x0 must be finite",
            diagnostics={"contract": "x0", "non_finite_entries":
                         int(np.count_nonzero(~np.isfinite(values)))})
    values.setflags(write=False)
    return values


def _interface_scales(x, context):
    """Reference/current coordinate magnitudes of the wet trace (gap tolerance).

    中文：返回湿界面迹的参考/当前坐标量级（米），供相容间隙容差 ``atol + rtol*scale`` 使用。
    """
    blocks = context.layout.unpack(x)
    d_s = np.asarray(blocks["d_s"]).ravel()[context.topology.solid_dofs]
    d_m = np.asarray(blocks["d_m"]).ravel()[context.layout.mesh_interface_dofs]
    solid = np.asarray(context.topology.X_s_gamma, dtype=float)
    fluid = np.asarray(context.topology.fluid_space.points, dtype=float)[
        context.topology.fluid_vertex_nodes]
    solid_current = solid+d_s.reshape(-1, 2)
    fluid_current = fluid+d_m.reshape(-1, 2)
    if solid_current.size == 0:
        return 0.0
    return float(max(np.max(np.linalg.norm(solid_current, axis=1)),
                     np.max(np.linalg.norm(fluid_current, axis=1))))


def interface_compatibility_gap(x, *, context):
    """Maximum raw violation of ``d_m|Gamma = H_d d_s|Gamma`` at ``x``.

    The T21 wet mesh rows are ``d_m[mesh_interface_dofs] - d_s[solid_dofs]``
    (one-to-one reference matching); the return value is the maximum absolute
    difference over both components, in metres.

    中文：返回 ``x`` 处约束 ``d_m|Gamma = H_d d_s|Gamma`` 的最大原始违反量（米）。对应 T21
    湿网格行 ``d_m[mesh_interface_dofs] - d_s[solid_dofs]``（一一参考匹配），取两分量上绝对
    差的最大值；界面为空时返回 0。
    """
    if not isinstance(context, MonolithicContext):
        raise TypeError("context must be a MonolithicContext.")
    values = _finite_vector(x, context.layout.n_total, "x")
    blocks = context.layout.unpack(values)
    d_m = np.asarray(blocks["d_m"]).ravel()[context.layout.mesh_interface_dofs]
    d_s = np.asarray(blocks["d_s"]).ravel()[context.topology.solid_dofs]
    if d_m.size == 0:
        return 0.0
    return float(np.max(np.abs(d_m-d_s)))


def project_interface_compatibility(x, *, context):
    """Return a compatible copy of ``x``: the mesh trace equals the solid trace.

    This is the explicit projection used when ``compatibility="project"``; the
    projected array is a frozen copy and ``x`` is never modified.

    中文：返回一个满足相容性的 ``x`` 副本：把网格迹（``d_m``）在湿界面处改为固体迹（``d_s``），
    即为 ``compatibility="project"`` 时的显式投影。结果为冻结副本，原 ``x`` 不被修改。
    """
    if not isinstance(context, MonolithicContext):
        raise TypeError("context must be a MonolithicContext.")
    values = _finite_vector(x, context.layout.n_total, "x")
    result = np.array(values, copy=True)
    d_s = result[context.layout.slice_of("d_s")].reshape(context.layout.shape_of("d_s"))
    d_m = result[context.layout.slice_of("d_m")].reshape(context.layout.shape_of("d_m"))
    d_m.ravel()[context.layout.mesh_interface_dofs] = d_s.ravel()[context.topology.solid_dofs]
    result.setflags(write=False)
    return result


def scaled_time_terms(x, *, context, solid_states, fluid_states, scales):
    """Scaled BDF time terms of one stencil at ``x`` (see module docstring).

    ``solid_states``/``fluid_states`` are the accepted histories used by
    ``context.stencil``, latest first with ``stencil.order`` entries.
    Returns a read-only ``{block: value}`` mapping over :data:`TIME_TERM_BLOCKS`.

    中文：给出单个 BDF 模板在 ``x`` 处的缩放时间项（定常判据，见模块 docstring）。对
    ``u_f``/``d_s``/``v_s`` 计算 ``|a_0 x + sum_{j>=1} a_j x^{k-j}|`` 逐分量除以缩放向量后的
    最大绝对值，返回只读的 ``{block: 值}`` 映射；``solid_states``/``fluid_states`` 是模板所用、
    "最近优先"的已接受历史（含 ``stencil.order`` 个条目）。
    """
    if not isinstance(context, MonolithicContext):
        raise TypeError("context must be a MonolithicContext.")
    values = _finite_vector(x, context.layout.n_total, "x")
    layout = context.layout
    stencil = context.stencil
    solid_states = tuple(solid_states)
    fluid_states = tuple(fluid_states)
    if len(solid_states) != stencil.order or len(fluid_states) != stencil.order:
        raise ValueError(f"histories must have stencil.order={stencil.order} entries.")
    scale = _validated_scales(scales, layout.n_total)
    coefficients = np.asarray(stencil.coefficients, dtype=float)
    blocks = layout.unpack(values)
    # a_0 项：当前状态乘以 stencil.coefficients[0]；后续按 j>=1 叠加权重 a_j 的历史状态。
    terms = {"u_f": coefficients[0]*np.asarray(blocks["u_f"]).ravel(),
             "d_s": coefficients[0]*np.asarray(blocks["d_s"]).ravel(),
             "v_s": coefficients[0]*np.asarray(blocks["v_s"]).ravel()}
    for index, ((d_state, v_state), u_state) in enumerate(zip(solid_states, fluid_states)):
        weight = coefficients[index+1]
        terms["d_s"] = terms["d_s"]+weight*np.asarray(d_state).ravel()
        terms["v_s"] = terms["v_s"]+weight*np.asarray(v_state).ravel()
        terms["u_f"] = terms["u_f"]+weight*np.asarray(u_state).ravel()
    return MappingProxyType({
        name: float(np.max(np.abs(terms[name]/scale[layout.slice_of(name)])))
        for name in TIME_TERM_BLOCKS})


class _MonolithicEvaluation:
    """One lazily evaluated candidate state (residual first, tangent on demand).

    中文：一次候选状态的惰性求值对象——先按需算残差，切线（``jacobian``）与预条件子只在被读取
    时才物化，同一 ``x`` 上重复读取复用缓存；适配器不持有求解器状态。
    """

    __slots__ = ("_adapter", "_x", "_residual", "_jacobian", "_preconditioner")

    def __init__(self, adapter, x):
        self._adapter = adapter
        self._x = x
        self._residual = None
        self._jacobian = None
        self._preconditioner = None

    @property
    def residual(self):
        if self._residual is None:
            self._residual = self._adapter.residual(self._x)
        return self._residual

    @property
    def jacobian(self):
        if self._jacobian is None:
            self._jacobian = self._adapter.jacobian(self._x)
        return self._jacobian

    @property
    def preconditioner(self):
        if self._preconditioner is None and self._adapter.uses_preconditioner:
            self._preconditioner = self._adapter.materialize_preconditioner(self._x)
        return self._preconditioner


class _MonolithicAdapter:
    """Binds ``MonolithicContext`` into the ``evaluate`` protocol of ``solve_newton``.

    T21 evaluations are counted and timed here; the tangent is produced lazily:
    the assembled route performs one extra ``mode="assembled"`` evaluation at
    the accepted point, the matrix-free route hands out a ``LinearOperator``
    that calls ``mode="jvp"`` once per direction.  The adapter owns no solver
    state and never modifies the context, the reference mesh or the base.

    中文：把 :class:`MonolithicContext` 绑定到 ``solve_newton`` 的 ``evaluate`` 协议上。T21
    求值在此计数与计时；切线惰性生成——assembled 路线在已接受点额外做一次
    ``mode="assembled"`` 求值，jvp 路线返回一个每次矩阵-向量乘调一次 ``mode="jvp"`` 的
    ``LinearOperator``。适配器不持有求解器状态，也不修改上下文、参考网格或历史基准。
    """

    def __init__(self, context, *, jacobian_mode, accounting, preconditioner=None):
        self.context = context
        self.jacobian_mode = jacobian_mode
        self.accounting = accounting
        self.preconditioner_factory = preconditioner
        self.uses_preconditioner = preconditioner is not None

    def __call__(self, x):
        return _MonolithicEvaluation(self, x)

    def _evaluate(self, x, *, mode, directions=None):
        started = perf_counter()
        try:
            result = evaluate_monolithic(x, context=self.context, mode=mode, directions=directions)
        except Exception as error:
            phase = "evaluate" if mode == "residual" else "jacobian"
            raise MonolithicNewtonFailure(
                phase=phase, iteration=-1,
                reason=f"the monolithic evaluation raised {type(error).__name__}: {error}",
                diagnostics={"source": "evaluate_monolithic", "mode": mode,
                             "error_type": type(error).__name__, "error": str(error)}) from error
        finally:
            self.accounting.evaluate_seconds += perf_counter()-started
        self.accounting.evaluations += 1
        if mode == "residual":
            self.accounting.residual_evaluations += 1
        elif mode == "assembled":
            self.accounting.assembled_evaluations += 1
        else:
            self.accounting.jvp_evaluations += 1
            values = np.asarray(directions)
            self.accounting.jvp_directions += 1 if values.ndim == 1 else int(values.shape[0])
        return result

    def residual(self, x):
        return self._evaluate(x, mode="residual").residual

    def jacobian(self, x):
        self.accounting.jacobian_reads += 1
        # 路线显式：assembled 物化 CSR 供 splu；jvp 返回矩阵无关 LinearOperator 供 GMRES。
        if self.jacobian_mode == "assembled":
            return self._evaluate(x, mode="assembled").jacobian
        size = self.context.layout.n_total

        def matvec(q):
            return self._jvp(x, q)

        return LinearOperator((size, size), matvec=matvec, dtype=float)

    def _jvp(self, x, q):
        value = np.asarray(q, dtype=float)
        if value.shape != (self.context.layout.n_total,):
            raise MonolithicNewtonFailure(
                phase="jacobian", iteration=-1,
                reason=f"the JVP direction must have shape ({self.context.layout.n_total},), "
                       f"got {value.shape}",
                diagnostics={"contract": "Evaluation.jacobian", "mode": "jvp"})
        result = self._evaluate(x, mode="jvp", directions=value)
        return np.array(result.jvp_values, dtype=float, copy=True)

    def materialize_preconditioner(self, x):
        try:
            return self.preconditioner_factory(x)
        except Exception as error:
            raise MonolithicNewtonFailure(
                phase="linear_solve", iteration=-1,
                reason=f"the preconditioner factory raised {type(error).__name__}: {error}",
                diagnostics={"contract": "preconditioner", "error_type": type(error).__name__,
                             "error": str(error)}) from error


def _block_norms(x, residual, layout, scales):
    """``max_i |R_b,i| / S_b,i`` per block (the convergence-criterion pieces).

    中文：逐块计算缩放残差最大值 ``max_i |R_b,i| / S_b,i``，是收敛判据的组成部分。
    """
    return MappingProxyType({
        name: float(np.max(np.abs(np.asarray(residual)[layout.slice_of(name)]
                                  / scales[layout.slice_of(name)])))
        for name in BLOCK_NAMES})


def _solver_name(jacobian_mode):
    # 直接分解路线记 splu，矩阵无关路线记 gmres（供结果中 linear_solver 字段）。
    return "splu" if jacobian_mode == "assembled" else "gmres"


@dataclass(frozen=True)
class MonolithicStepResult:
    """Read-only outcome of one converged monolithic Newton step.

    ``x``, ``residual`` and ``scales`` are frozen copies; ``records`` holds the
    accepted :class:`~fsi_ice.coupling.newton.NewtonIterationRecord` objects
    (iteration 0 is the start).  ``threshold`` is the declared convergence
    threshold of the solve, ``block_norms`` the per-block scaled residual
    maxima at the accepted point and ``interface_gap`` the raw violation of
    ``d_m|Gamma = H_d d_s|Gamma`` there.  ``linear_solver`` is ``"splu"``
    (assembled route; ``linear_iterations`` is then 0 by definition) or
    ``"gmres"`` (matrix-free route).

    中文：一次收敛的整体 Newton 步的只读结果。``x``/``residual``/``scales`` 为冻结副本；
    ``records`` 存放已接受的 :class:`~fsi_ice.coupling.newton.NewtonIterationRecord`（第 0 条为
    起始记录）。``threshold`` 为本次求解声明的收敛阈值，``block_norms`` 为接受点的逐块缩放残差
    最大值，``interface_gap`` 为该点 ``d_m|Gamma = H_d d_s|Gamma`` 的原始违反量。
    ``linear_solver`` 为 ``"splu"``（装配路线，``linear_iterations`` 恒为 0）或 ``"gmres"``
    （矩阵无关路线）。
    """

    success: bool
    x: np.ndarray
    reason: str
    residual_norm: float
    raw_residual_norm: float
    threshold: float
    residual: np.ndarray
    residual_history: tuple
    raw_residual_history: tuple
    records: tuple
    counters: MonolithicCounters
    timings: object
    jacobian_mode: str
    linear_solver: str
    candidate_id: object
    scales: np.ndarray
    block_norms: object
    interface_gap: float
    compatibility_gap: float
    compatibility_policy: str

    def __post_init__(self):
        x = np.array(self.x, dtype=float, copy=True)
        x.setflags(write=False)
        residual = np.array(self.residual, dtype=float, copy=True)
        residual.setflags(write=False)
        scales = np.array(self.scales, dtype=float, copy=True)
        scales.setflags(write=False)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "residual_history",
                           tuple(float(value) for value in self.residual_history))
        object.__setattr__(self, "raw_residual_history",
                           tuple(float(value) for value in self.raw_residual_history))
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "timings", MappingProxyType(dict(self.timings)))
        object.__setattr__(self, "block_norms", MappingProxyType(dict(self.block_norms)))

    @property
    def newton_iterations(self):
        return max(0, len(self.records)-1)

    @property
    def uses_krylov(self):
        """False on the assembled route: a direct factorization has no Krylov iterations.

        中文：装配（直接分解）路线返回 False（无 Krylov 迭代）；仅 GMRES 路线返回 True。
        """
        return self.linear_solver == "gmres"


def _validated_context(context):
    if not isinstance(context, MonolithicContext):
        raise TypeError("context must be a MonolithicContext.")
    return context


def _validated_mode(jacobian_mode):
    if jacobian_mode not in JACOBIAN_MODES:
        raise ValueError(f"jacobian_mode must be one of {JACOBIAN_MODES}, got {jacobian_mode!r}.")
    return jacobian_mode


def _validated_policy(compatibility):
    if compatibility not in COMPATIBILITY_POLICIES:
        raise ValueError(f"compatibility must be one of {COMPATIBILITY_POLICIES}, "
                         f"got {compatibility!r}.")
    return compatibility


def _compatibility_tolerance(x, context):
    """Gap tolerance of the candidate interface geometry check (same atol/rtol form).

    The driver evaluates the form on the current wet-trace coordinates; the
    candidate evaluation repeats the check on the full candidate interface, so
    a borderline state still fails structurally inside ``evaluate_monolithic``.

    中文：候选界面几何检查所用的间隙容差，形式与 ``topology.atol + topology.rtol*scale``
    相同（``scale`` 取当前湿迹坐标量级）。驱动在当前湿迹坐标上先判一次；候选求值会在完整候选
    界面上再判一次，故临界状态仍会在 :func:`evaluate_monolithic` 内以结构化方式失败。
    """
    scale = _interface_scales(x, context)
    return float(context.topology.atol+context.topology.rtol*scale)


def solve_monolithic_step(context, x0, *, options=None, scales=None, jacobian_mode="assembled",
                          compatibility="require", preconditioner=None):
    """One damped monolithic Newton solve of ``R(x) = 0`` at ``context``.

    ``x0`` is the starting candidate (a frozen copy is taken).  ``options`` is a
    :class:`~fsi_ice.coupling.newton.NewtonOptions`; ``scales`` the per-component
    physical scale vector used by the ``atol + rtol*||r0||_S`` criterion (all
    ones when omitted).  ``jacobian_mode`` explicitly selects the tangent route
    (:data:`JACOBIAN_MODES`); no route is inferred and no fallback happens.
    ``preconditioner``, if given, must be a callable ``f(x) -> operator`` used
    only by the ``"jvp"`` route (a ``ValueError`` is raised otherwise).

    Returns a read-only :class:`MonolithicStepResult`; every failure raises
    :class:`MonolithicNewtonFailure` with phase, iteration, normalized and raw
    norms and complete diagnostics (partial counters and timings included).

    中文：在 ``context`` 处对 ``R(x) = 0`` 做一次阻尼整体 Newton 求解。``x0`` 为起点（会取
    冻结副本）；``options`` 为 :class:`~fsi_ice.coupling.newton.NewtonOptions`；``scales`` 为
    各分量物理尺度向量（收敛判据 ``atol + rtol*||r0||_S`` 用，缺省全 1）；``jacobian_mode``
    显式选择切线路线（:data:`JACOBIAN_MODES`，不推断、不回退）；``preconditioner``（可选）须为
    ``f(x) -> operator``，且仅 ``"jvp"`` 路线可用（否则抛 ``ValueError``）。返回只读
    :class:`MonolithicStepResult`；任何失败都以 :class:`MonolithicNewtonFailure` 抛出，携带
    阶段、迭代号、归一化/原始范数与完整诊断（含部分计数与计时）。
    """
    context = _validated_context(context)
    jacobian_mode = _validated_mode(jacobian_mode)
    compatibility = _validated_policy(compatibility)
    if preconditioner is not None and not callable(preconditioner):
        raise TypeError("preconditioner must be a callable f(x) -> operator or None.")
    if preconditioner is not None and jacobian_mode != "jvp":
        raise ValueError("preconditioner is only used with jacobian_mode='jvp'.")
    options = NewtonOptions() if options is None else options
    if not isinstance(options, NewtonOptions):
        raise TypeError("options must be a NewtonOptions instance.")
    scale = _validated_scales(scales, context.layout.n_total)
    started = perf_counter()
    accounting = _Accounting()
    try:
        values = _validated_x0(x0, context)
    except MonolithicNewtonFailure as failure:
        raise MonolithicNewtonFailure.from_failure(
            failure, candidate_id=context.candidate_id, jacobian_mode=jacobian_mode,
            counters=accounting.counters(failed_steps=1), timings=accounting.timings()) from failure
    gap_before = interface_compatibility_gap(values, context=context)
    gap_tolerance = _compatibility_tolerance(values, context)
    # 相容性策略：project 先把网格迹投影到固体迹；require 则违反即结构化报错。
    if compatibility == "project":
        values = project_interface_compatibility(values, context=context)
    elif gap_before > gap_tolerance:
        raise MonolithicNewtonFailure(
            phase="residual", iteration=0,
            reason="the initial state violates the interface compatibility constraint "
                   "d_m|Gamma = H_d d_s|Gamma",
            x=values, diagnostics={"requirement": "d_m|Gamma = H_d d_s|Gamma",
                                   "policy": compatibility,
                                   "interface_gap": gap_before,
                                   "gap_tolerance": gap_tolerance},
            candidate_id=context.candidate_id, jacobian_mode=jacobian_mode,
            interface_gap=gap_before, counters=accounting.counters(failed_steps=1),
            timings=accounting.timings())

    adapter = _MonolithicAdapter(context, jacobian_mode=jacobian_mode, accounting=accounting,
                                 preconditioner=preconditioner)
    telemetry = NewtonTelemetry()
    newton_started = perf_counter()
    try:
        outcome = solve_newton(adapter, values, options=options, scales=scale, telemetry=telemetry)
    except NewtonFailure as failure:
        accounting.newton_seconds = perf_counter()-newton_started
        accounting.total_seconds = perf_counter()-started
        accounting.absorb_telemetry(telemetry)
        accounting.newton_iterations = max(0, len(failure.history)-1)
        gap = None if failure.x is None else interface_compatibility_gap(
            failure.x, context=context)
        raise MonolithicNewtonFailure.from_failure(
            failure, candidate_id=context.candidate_id, jacobian_mode=jacobian_mode,
            interface_gap=gap, counters=accounting.counters(failed_steps=1),
            timings=accounting.timings()) from failure
    accounting.newton_seconds = perf_counter()-newton_started
    accounting.absorb_telemetry(telemetry)
    accounting.newton_iterations = max(0, len(outcome.records)-1)
    accounting.steps = 1

    # One diagnostic residual evaluation at the accepted point provides the
    # final residual vector, the per-block scaled norms and the interface gap.
    residual = np.array(adapter.residual(outcome.x), copy=True)
    block_norms = _block_norms(outcome.x, residual, context.layout, scale)
    gap = interface_compatibility_gap(outcome.x, context=context)
    accounting.total_seconds = perf_counter()-started
    return MonolithicStepResult(
        success=True, x=outcome.x, reason=outcome.reason, residual_norm=outcome.residual_norm,
        raw_residual_norm=outcome.raw_residual_norm, threshold=outcome.records[-1].threshold,
        residual=residual, residual_history=outcome.residual_history,
        raw_residual_history=outcome.raw_residual_history, records=outcome.records,
        counters=accounting.counters(), timings=accounting.timings(),
        jacobian_mode=jacobian_mode, linear_solver=_solver_name(jacobian_mode),
        candidate_id=context.candidate_id, scales=scale, block_norms=block_norms,
        interface_gap=gap, compatibility_gap=gap_before, compatibility_policy=compatibility)


@dataclass(frozen=True)
class MonolithicMarchStep:
    """Read-only record of one accepted march step.

    ``x`` is the frozen converged state accepted into the history,
    ``time_term``/``time_terms`` the scaled BDF time terms of that state and
    ``counters``/``timings`` the deltas of this step only.

    中文：一次被接受的推进步的只读记录。``x`` 是存入历史的冻结收敛状态；
    ``time_term``/``time_terms`` 是该状态缩放后的 BDF 时间项；``counters``/``timings`` 只含
    本步增量。``generation``/``order``/``dt`` 记录该步的 BDF 代数、阶数与时间步长。
    """

    index: int
    generation: int
    order: int
    dt: float
    newton_iterations: int
    residual_norm: float
    threshold: float
    time_term: float
    time_terms: object
    interface_gap: float
    x: np.ndarray
    counters: MonolithicCounters
    timings: object

    def __post_init__(self):
        x = np.array(self.x, dtype=float, copy=True)
        x.setflags(write=False)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "time_terms", MappingProxyType(dict(self.time_terms)))
        object.__setattr__(self, "timings", MappingProxyType(dict(self.timings)))


@dataclass(frozen=True)
class MonolithicMarchResult:
    """Read-only outcome of a steady march.

    ``success`` equals ``steady_reached``: the march stops with
    ``reason="steady"`` when the scaled BDF time term is below
    ``steady_tolerance``, or with ``reason="max_steps"`` after ``max_steps``
    converged steps.  ``x`` and ``scales`` are frozen copies; ``steps`` holds
    one :class:`MonolithicMarchStep` per completed step.  Counter and timing
    mappings aggregate the steps (``total_seconds`` is the march wall clock).

    中文：一次定常推进的只读结果。``success`` 等于 ``steady_reached``：缩放 BDF 时间项低于
    ``steady_tolerance`` 时以 ``reason="steady"`` 停止，或走满 ``max_steps`` 个收敛步（
    ``steady_reached=False``，``reason="max_steps"``）。``x``/``scales`` 为冻结副本；``steps``
    每个完成步一条 :class:`MonolithicMarchStep`。计数与计时映射对步做聚合（``total_seconds``
    为推进墙钟时间）。
    """

    success: bool
    steady_reached: bool
    x: np.ndarray
    reason: str
    dt: float
    order: int
    steady_tolerance: float
    final_time_term: float
    final_time_terms: object
    steps: tuple
    counters: MonolithicCounters
    timings: object
    jacobian_mode: str
    linear_solver: str
    candidate_id: object
    scales: np.ndarray

    def __post_init__(self):
        x = np.array(self.x, dtype=float, copy=True)
        x.setflags(write=False)
        scales = np.array(self.scales, dtype=float, copy=True)
        scales.setflags(write=False)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "final_time_terms", MappingProxyType(dict(self.final_time_terms)))
        object.__setattr__(self, "timings", MappingProxyType(dict(self.timings)))

    @property
    def uses_krylov(self):
        """True on the matrix-free (GMRES) route.

        中文：矩阵无关（GMRES）路线返回 True，装配（直接分解）路线返回 False。
        """
        return self.linear_solver == "gmres"


def _march_timings(step_timings, total_seconds):
    """Aggregate step timing buckets; ``total_seconds`` is the march wall clock.

    中文：对逐步计时桶（``evaluate_seconds``/``linear_seconds``/``newton_seconds``）求和，
    并把 ``total_seconds`` 设为整个推进的墙钟时间。
    """
    keys = ("evaluate_seconds", "linear_seconds", "newton_seconds")
    result = {key: float(sum(entry[key] for entry in step_timings)) for key in keys}
    result["total_seconds"] = float(total_seconds)
    return MappingProxyType(result)


def march_to_steady(context, x0, *, dt, order=2, max_steps=50, steady_tolerance=1e-8,
                    options=None, scales=None, jacobian_mode="assembled",
                    compatibility="require", preconditioner=None):
    """March constant-``dt`` BDF steps from an initial history to steady state.

    The initial accepted state is the latest entry of ``context.base`` (solid
    ``(d_s, v_s)`` pair, fluid ``u_f`` and accepted geometry); fresh
    ``BDFHistory`` objects restart the generation numbering at 0 and increment
    it once per accepted step (``order=2`` uses the BDF1 startup step of
    :class:`~fsi_ice.time.bdf.BDFHistory`).  ``x0`` is the first Newton start
    point; later steps start from the previous converged solution.

    A step is a :func:`solve_monolithic_step` call whose converged state is
    accepted as the next history entry (candidate geometry rebuilt from the
    converged ``d_m``).  The stopping criterion is the scaled BDF time term of
    the converged step (see the module docstring) against ``steady_tolerance``;
    ``max_steps`` bounds the number of steps.  A Newton failure propagates as
    :class:`MonolithicNewtonFailure` carrying the completed
    :class:`MonolithicMarchStep` records and the aggregated counters.  The
    input context and ``x0`` are never modified.

    中文：从初始历史出发，用恒定 ``dt`` 的 BDF 步推进到定常。初始已接受状态取
    ``context.base`` 的最近条目（固体 ``(d_s, v_s)``、流体 ``u_f`` 与已接受几何）；新建的
    ``BDFHistory`` 把代数编号从 0 重新开始，每接受一步加一（``order=2`` 用
    :class:`~fsi_ice.time.bdf.BDFHistory` 的 BDF1 启动步）。``x0`` 作为第一步的 Newton 起点，
    之后每步从上一步收敛解出发。每步是一次 :func:`solve_monolithic_step`，其收敛状态被接受为
    下一条历史（并由收敛 ``d_m`` 重建候选几何）。停机判据为收敛步的缩放 BDF 时间项（见模块
    docstring）与 ``steady_tolerance`` 比较，``max_steps`` 限制步数。Newton 失败以
    :class:`MonolithicNewtonFailure` 抛出，携带已完成的 :class:`MonolithicMarchStep` 记录与
    聚合计数。输入上下文与 ``x0`` 均不被修改。
    """
    context = _validated_context(context)
    jacobian_mode = _validated_mode(jacobian_mode)
    compatibility = _validated_policy(compatibility)
    if preconditioner is not None and not callable(preconditioner):
        raise TypeError("preconditioner must be a callable f(x) -> operator or None.")
    if preconditioner is not None and jacobian_mode != "jvp":
        raise ValueError("preconditioner is only used with jacobian_mode='jvp'.")
    if isinstance(dt, (bool, np.bool_)) or not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a finite positive number.")
    if isinstance(order, (bool, np.bool_)) or order not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    if isinstance(max_steps, (bool, np.bool_)) or not isinstance(max_steps, (int, np.integer)) \
            or max_steps < 1:
        raise ValueError("max_steps must be a positive integer.")
    if isinstance(steady_tolerance, (bool, np.bool_)) or not np.isscalar(steady_tolerance) \
            or not np.isfinite(steady_tolerance) or steady_tolerance <= 0:
        raise ValueError("steady_tolerance must be a finite positive number.")
    options = NewtonOptions() if options is None else options
    if not isinstance(options, NewtonOptions):
        raise TypeError("options must be a NewtonOptions instance.")
    values = _validated_x0(x0, context)
    scale = _validated_scales(scales, context.layout.n_total)

    base = context.base
    reference_mesh = base.reference_mesh
    initial_displacement, initial_velocity = base.solid_states[0]
    solid_history = BDFHistory(np.stack((initial_displacement, initial_velocity)), dt, order=order)
    fluid_history = BDFHistory(base.fluid_states[0], dt, order=order)
    geometries = [base.accepted_geometries[0]]

    started = perf_counter()
    steps = []
    counters = MonolithicCounters()
    step_timings = []
    final_terms = None
    final_term = float("inf")
    x = np.array(values, copy=True)
    for index in range(int(max_steps)):
        stencil = solid_history.stencil()
        # 固体与流体历史必须同步推进：两者的 step_id 若不一致说明代数错位，直接报错。
        if fluid_history.stencil().step_id != stencil.step_id:
            raise RuntimeError("march histories lost generation synchronisation.")
        solid_states = tuple((state[0], state[1])
                             for state in solid_history.states[:stencil.order])
        fluid_states = tuple(fluid_history.states[:stencil.order])
        step_base = MonolithicBase(
            reference_mesh=reference_mesh, material=base.material,
            fluid_density=base.fluid_density, fluid_viscosity=base.fluid_viscosity,
            solid_states=solid_states, fluid_states=fluid_states,
            accepted_geometries=tuple(geometries[:stencil.order]),
            generation=stencil.step_id, solid_body_force=base.solid_body_force,
            fluid_body_force=base.fluid_body_force,
            fluid_boundary_velocity=base.fluid_boundary_velocity,
            mesh_boundary_displacement=base.mesh_boundary_displacement,
            solid_boundary_displacement=base.solid_boundary_displacement,
            solid_boundary_velocity=base.solid_boundary_velocity,
            quality_limits=base.quality_limits)
        step_context = MonolithicContext(
            context.layout, step_base, stencil, context.topology, context.harmonic,
            candidate_id=(context.candidate_id, "march", index),
            thickness=context.thickness, pressure_constraint=context.pressure_constraint)
        try:
            outcome = solve_monolithic_step(
                step_context, x, options=options, scales=scale, jacobian_mode=jacobian_mode,
                compatibility=compatibility, preconditioner=preconditioner)
        except NewtonFailure as failure:
            extra_counters = getattr(failure, "counters", None)
            if extra_counters is None:
                extra_counters = MonolithicCounters(failed_steps=1)
            failing_timings = list(step_timings)
            if getattr(failure, "timings", None) is not None:
                failing_timings.append(dict(failure.timings))
            raise MonolithicNewtonFailure.from_failure(
                failure, step_index=index, march_steps=tuple(steps),
                candidate_id=(context.candidate_id, "march", index),
                jacobian_mode=jacobian_mode,
                counters=counters+extra_counters,
                timings=_march_timings(failing_timings, perf_counter()-started)) from failure

        blocks = step_context.layout.unpack(outcome.x)
        # 用收敛状态评估缩放 BDF 时间项（定常判据），只对 u_f/d_s/v_s 取。
        terms = scaled_time_terms(outcome.x, context=step_context, solid_states=solid_states,
                                  fluid_states=fluid_states, scales=scale)
        final_terms = terms
        final_term = max(terms.values())
        record = MonolithicMarchStep(
            index=index, generation=stencil.step_id, order=stencil.order, dt=float(dt),
            newton_iterations=outcome.newton_iterations, residual_norm=outcome.residual_norm,
            threshold=outcome.threshold, time_term=final_term, time_terms=terms,
            interface_gap=outcome.interface_gap, x=outcome.x, counters=outcome.counters,
            timings=outcome.timings)
        steps.append(record)
        counters = counters+outcome.counters
        step_timings.append(dict(outcome.timings))

        # Accept the converged state as the next history entry and rebuild the
        # accepted candidate geometry from its mesh displacement.
        solid_history.accept(solid_history.context(),
                             np.stack((np.asarray(blocks["d_s"]), np.asarray(blocks["v_s"]))))
        fluid_history.accept(fluid_history.context(), np.asarray(blocks["u_f"]))
        geometry = build_ale_geometry(
            reference_mesh, np.asarray(blocks["d_m"]), quality_limits=base.quality_limits,
            candidate_id=("accepted", context.candidate_id, index))
        # 只保留最近 stencil.order 个已接受几何（最新在列表首），供下一步的 BDF 历史使用。
        geometries.insert(0, geometry)
        del geometries[2:]

        # 停机判据：缩放 BDF 时间项低于容差即视为到达定常不动点。
        if final_term <= steady_tolerance:
            return MonolithicMarchResult(
                success=True, steady_reached=True, x=outcome.x, reason="steady", dt=float(dt),
                order=int(order), steady_tolerance=float(steady_tolerance),
                final_time_term=final_term, final_time_terms=terms, steps=tuple(steps),
                counters=counters, timings=_march_timings(step_timings, perf_counter()-started),
                jacobian_mode=jacobian_mode, linear_solver=_solver_name(jacobian_mode),
                candidate_id=context.candidate_id, scales=scale)
        x = np.array(outcome.x, copy=True)

    return MonolithicMarchResult(
        success=False, steady_reached=False, x=x, reason="max_steps", dt=float(dt),
        order=int(order), steady_tolerance=float(steady_tolerance),
        final_time_term=final_term,
        final_time_terms=final_terms if final_terms is not None else MappingProxyType({}),
        steps=tuple(steps), counters=counters,
        timings=_march_timings(step_timings, perf_counter()-started),
        jacobian_mode=jacobian_mode, linear_solver=_solver_name(jacobian_mode),
        candidate_id=context.candidate_id, scales=scale)

"""Generic damped Newton solver for the M5/T22 monolithic coupling prototype.

The solver consumes only a pure evaluation contract, which T21 provides:

    evaluate(x) -> Evaluation
    Evaluation.residual : one-dimensional float array r(x) with shape (n,)
    Evaluation.jacobian : scipy sparse Jacobian dr/dx with shape (n, n)

It deliberately does not import ``fsi_ice.coupling.monolithic``; a T21
evaluator satisfies the structural protocol directly.  At T22 integration the
full entry point is bound as, for example,
``lambda x: evaluate_monolithic(x, context=attempt_context, mode="full")``.

The solver owns every array it reads: ``x0`` is copied, the evaluator receives
a fresh copy per call, and returned residuals/Jacobians are copied before use,
so callers and evaluators cannot observe or cause in-place mutation.  Failures
raise :class:`NewtonFailure` with phase, iteration, residual norm and frozen
diagnostics; ``StepTransaction`` already forwards an exception's ``diagnostics``
attribute into its failure audit, so the exception can be raised from inside a
plan callback unchanged.

The linear tangent may be assembled (sparse/dense matrix, direct ``splu``
factorization) or matrix free: with ``NewtonOptions(linear_solver="gmres")`` the
evaluation may expose ``jacobian`` as any object accepted by
``scipy.sparse.linalg.aslinearoperator`` (a ``LinearOperator`` or an object with
``shape`` and ``matvec``), and the direction is computed with GMRES.  An optional
``preconditioner`` attribute on the same evaluation (again any object accepted
by ``aslinearoperator``) is passed to GMRES as ``M``.  The choice is explicit
configuration: a failed or unavailable matrix-free solve raises
``NewtonFailure(phase="linear_solve")`` and never falls back to another
algorithm.  ``NewtonTelemetry`` additionally collects wall-clock buckets and
Krylov iteration counts even when a solve fails part way.

Design and the explicit T22 integration plan live in
``docs/development/m5_t22_plan.md``.  This module is a generic prototype: it is
verified against toy systems only and is not FSI validation.

中文说明：本模块提供 M5/T22 单体耦合原型所用的通用阻尼 Newton 求解器，只
依赖纯求值契约 evaluate(x) -> Evaluation（残差 r(x) 与其切向导数 dr/dx），
不导入 ``fsi_ice.coupling.monolithic``。关键约定：收敛判据用逐分量归一化无穷
范数 max_i |r_i|/scales_i，其中 scales 为与未知量同单位的物理尺度向量；线搜索
以原始残差二范数作 Armijo 回退，收敛仍看归一化范数；线性切向可选用直接分解
（splu）或矩阵自由 GMRES，一旦失败即抛 NewtonFailure，绝不回退到其它算法。
求解器复制每次读入的数组，故调用方与求值器都无法观察或造成原地修改。
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import aslinearoperator, gmres, splu

__all__ = [
    "Evaluation",
    "Evaluator",
    "NewtonFailure",
    "NewtonIterationRecord",
    "NewtonOptions",
    "NewtonResult",
    "NewtonTelemetry",
    "solve_newton",
]


@runtime_checkable
class Evaluation(Protocol):
    """Contract of one pure evaluation: the residual and its tangent.

    ``residual`` must be a one-dimensional array of length ``n``; ``jacobian``
    must be square with shape ``(n, n)``.  The frozen T21 contract delivers an
    assembled Jacobian as a scipy CSR matrix; for the matrix-free path
    (``linear_solver="gmres"``) any object accepted by
    ``scipy.sparse.linalg.aslinearoperator`` is allowed (a ``LinearOperator`` or
    an object with ``shape`` and ``matvec``).  The optional ``preconditioner``
    attribute, when present, must be another such object.  All values the solver
    reads are copied, so an evaluator may safely reuse its buffers.

    中文：一次纯求值的契约，给出残差与其切线。``residual`` 必须是一维数组、
    长度为 n；``jacobian`` 必须为 n×n 方阵（组装路径为 scipy CSR）。矩阵自由
    路径（``linear_solver="gmres"``）允许任何可被 ``aslinearoperator`` 接受的
    对象（``LinearOperator`` 或含 ``shape`` 与 ``matvec`` 的对象）；可选
    ``preconditioner`` 需为同类对象。求解器会复制所有读入值，求值器可安全复用
    自身缓冲区。
    """

    @property
    def residual(self) -> np.ndarray:
        """Residual vector r(x); shape (n,).

        中文：残差向量 r(x)，形状 (n,)。
        """

    @property
    def jacobian(self):
        """Derivative dr/dx as a sparse matrix or a matrix-free operator; shape (n, n).

        中文：切向导数 dr/dx，可为稀疏矩阵或矩阵自由算子，形状 (n, n)。
        """


class Evaluator(Protocol):
    """Pure callable ``evaluate(x) -> Evaluation`` (duck typing).

    中文：纯可调用对象 ``evaluate(x) -> Evaluation``，通过鸭子类型满足。
    """

    def __call__(self, x: np.ndarray) -> Evaluation:
        """Return a fresh evaluation at ``x``; must not mutate ``x`` or state.

        中文：在 ``x`` 处返回一次新的求值；不得修改 ``x`` 或自身状态。
        """


def _real_scalar(value, name):
    """Finite real scalar, rejecting booleans, strings and complex values.

    中文：校验并返回有限实数标量；拒绝布尔、字符串与复数（这些易被误当成合法
    数值配置）。
    """
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or \
            isinstance(value, (str, bytes, complex, np.complexfloating)):
        raise ValueError(f"{name} must be a finite real scalar.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite real scalar.") from None
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _own_vector(values, name, *, size=None):
    """Independent read-only finite 1-D float array.

    中文：把输入拷贝为独立的一维只读有限浮点数组，可选校验长度 ``size``；用于
    求解器对外部传入向量的所有权保护。
    """
    try:
        array = np.array(values, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a real numeric one-dimensional array.") from None
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    if size is not None and array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    array.setflags(write=False)
    return array


def _own_residual(values, name, size):
    """Independent read-only residual copy; finiteness is checked by callers.

    中文：把残差拷贝为独立只读数组并校验形状 (size,)；有限性由调用方另行检查
    （此处不查，便于把 ``inf``/``nan`` 记录进线搜索诊断）。
    """
    try:
        array = np.array(values, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} is not a real numeric array.") from None
    if array.ndim != 1 or array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    array.setflags(write=False)
    return array


def _own_jacobian(value, name, size):
    """Independent CSR copy of a square real matrix; dense input is allowed.

    中文：把切线矩阵拷贝为独立的 CSR（压缩稀疏行）方阵，接受稀疏或稠密输入；
    拒绝复数值、非 (size, size) 形状与非有限元素。
    """
    if value is None:
        raise ValueError(f"{name} is missing: the evaluation exposes no jacobian.")
    try:
        if sparse.issparse(value):
            if np.issubdtype(value.dtype, np.complexfloating):
                raise ValueError(f"{name} must be real.")
            matrix = value.tocsr(copy=True)
            matrix = matrix.astype(float, copy=False)
        else:
            raw = np.asarray(value)
            if np.issubdtype(raw.dtype, np.complexfloating):
                raise ValueError(f"{name} must be real.")
            matrix = sparse.csr_matrix(np.array(value, dtype=float, copy=True))
    except ValueError:
        raise
    except (TypeError, OverflowError) as error:
        raise ValueError(f"{name} is not a real numeric matrix: {error}") from None
    if matrix.ndim != 2 or matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}), got {matrix.shape}.")
    matrix.sum_duplicates()
    matrix.sort_indices()
    if matrix.data.size and not np.all(np.isfinite(matrix.data)):
        raise ValueError(f"{name} contains non-finite entries.")
    return matrix


class NewtonFailure(RuntimeError):
    """Terminal Newton failure with phase, iteration and frozen diagnostics.

    ``phase`` is one of ``evaluate``, ``residual``, ``jacobian``,
    ``linear_solve``, ``line_search`` or ``convergence``.  ``x`` is a read-only
    copy of the last accepted iterate (or ``None`` if none was recorded) and
    ``history`` holds the accepted iteration records produced so far.
    ``diagnostics`` is a mapping of plain values suitable for the
    ``StepTransaction`` failure audit.

    中文：Newton 的终止性失败结果，携带阶段、迭代号与冻结诊断。``phase`` 取
    ``evaluate``/``residual``/``jacobian``/``linear_solve``/``line_search``/
    ``convergence``；``x`` 为最后被接受迭代点的只读拷贝（无记录时为 ``None``），
    ``history`` 为迄今已接受的迭代记录；``diagnostics`` 适合交给
    ``StepTransaction`` 失败审计。
    """

    def __init__(self, *, phase, iteration, reason, residual_norm=None,
                 raw_residual_norm=None, x=None, history=(), diagnostics=None):
        self.phase = str(phase)
        self.iteration = int(iteration)
        self.reason = str(reason)
        self.residual_norm = None if residual_norm is None else float(residual_norm)
        self.raw_residual_norm = None if raw_residual_norm is None else float(raw_residual_norm)
        self.x = None if x is None else _own_vector(x, "failure x")
        self.history = tuple(history)
        self.diagnostics = MappingProxyType(dict(diagnostics or {}))
        detail = "" if self.residual_norm is None else f"; ||r|| = {self.residual_norm:.3e}"
        super().__init__(
            f"Newton failure at phase {self.phase!r}, iteration {self.iteration}: {reason}{detail}")


@dataclass(frozen=True)
class NewtonOptions:
    """Solver controls; tolerances are dimensionless.

    The convergence threshold is ``atol + rtol * ||r0||`` in the per-component
    normalized infinity norm, where ``r0`` is the residual at the starting
    point of this solve.  Backtracking halves ``alpha`` from 1 with at most
    ``max_backtracks`` rejected trials per Newton step and never attempts a
    damping below ``minimum_damping``.

    ``linear_solver="direct"`` (default) factorizes the assembled ``jacobian``
    with ``splu``.  ``linear_solver="gmres"`` uses the matrix-free path: the
    evaluation's ``jacobian`` (and optional ``preconditioner``) is passed to
    ``scipy.sparse.linalg.gmres`` with ``restart``, ``maxiter`` (restart-cycle
    cap, ``None`` uses the scipy default) and the ``gmres_rtol``/``gmres_atol``
    convergence test ``||b - A x|| <= max(gmres_rtol*||b||, gmres_atol)``.

    中文：求解器控制参数，各容差均为无量纲量。收敛阈值按逐分量归一化无穷范数
    取 ``atol + rtol * ||r0||``，其中 r0 为本次求解起点的残差。回溯线搜索从
    alpha=1 起按 ``backtrack_factor`` 折半，每步 Newton 至多拒绝
    ``max_backtracks`` 次，且阻尼不会低于 ``minimum_damping``。
    ``linear_solver="direct"``（默认）对组装后的 jacobian 做 splu 分解；
    ``"gmres"`` 走矩阵自由路径，把求值给出的 jacobian（及可选 preconditioner）
    交给 scipy 的 gmres。
    """

    max_iterations: int = 50  # Newton 主迭代上限
    rtol: float = 1e-10  # 相对容差：阈值中与起点归一化范数相乘的系数
    atol: float = 1e-12  # 绝对容差：与尺度无关的收敛下限
    line_search: bool = True  # 是否启用 Armijo 回溯
    max_backtracks: int = 30  # 每步最多拒绝的回溯试探次数
    backtrack_factor: float = 0.5  # 每次回溯的阻尼折半因子，须在 (0,1) 内
    armijo_slope: float = 1e-4  # Armijo 充分下降斜率，须在 (0,1) 内
    minimum_damping: float = 2.0 ** -24  # 阻尼下限，低于此值即判定回溯失败
    linear_solver: str = "direct"  # "direct"=splu 直接分解；"gmres"=矩阵自由
    gmres_restart: int = 100  # GMRES 重启周期长度
    gmres_maxiter: int | None = None  # 重启循环上限；None 用 scipy 默认
    gmres_rtol: float = 1e-10  # GMRES 相对收敛容差
    gmres_atol: float = 1e-12  # GMRES 绝对收敛容差

    def __post_init__(self):
        for name in ("rtol", "atol", "backtrack_factor", "armijo_slope", "minimum_damping",
                     "gmres_rtol", "gmres_atol"):
            object.__setattr__(self, name, _real_scalar(getattr(self, name), name))
        for name in ("max_iterations", "max_backtracks", "gmres_restart"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer.")
            object.__setattr__(self, name, int(value))
        if self.gmres_maxiter is not None:
            value = self.gmres_maxiter
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError("gmres_maxiter must be a positive integer or None.")
            object.__setattr__(self, "gmres_maxiter", int(value))
        if not isinstance(self.line_search, (bool, np.bool_)):
            raise ValueError("line_search must be a boolean.")
        object.__setattr__(self, "line_search", bool(self.line_search))
        if not isinstance(self.linear_solver, str) or self.linear_solver not in ("direct", "gmres"):
            raise ValueError("linear_solver must be 'direct' or 'gmres'.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        if self.max_backtracks < 0:
            raise ValueError("max_backtracks must be nonnegative.")
        if self.gmres_restart < 1:
            raise ValueError("gmres_restart must be at least 1.")
        if min(self.atol, self.rtol) < 0.0 or self.atol + self.rtol == 0.0:
            raise ValueError("atol and rtol must be nonnegative and not both zero.")
        if self.gmres_rtol < 0.0 or self.gmres_atol < 0.0 or self.gmres_rtol + self.gmres_atol == 0.0:
            raise ValueError("gmres_rtol and gmres_atol must be nonnegative and not both zero.")
        if not 0.0 < self.backtrack_factor < 1.0:
            raise ValueError("backtrack_factor must lie in (0, 1).")
        if not 0.0 < self.armijo_slope < 1.0:
            raise ValueError("armijo_slope must lie in (0, 1).")
        if not 0.0 < self.minimum_damping <= 1.0:
            raise ValueError("minimum_damping must lie in (0, 1].")


@dataclass
class NewtonTelemetry:
    """Mutable per-solve accounting sink (never shared across solves).

    ``evaluate_seconds`` and ``linear_seconds`` accumulate wall-clock time
    (``perf_counter``) inside evaluator calls and linear solves respectively;
    the evaluator time of a matrix-free matvec is counted in both buckets (the
    matvec runs inside the linear solve), which is the documented aggregation
    convention.  ``linear_solves``, ``linear_iterations`` (GMRES inner
    iterations; 0 for a direct factorization), ``line_search_trials`` and
    ``failed_trials`` are updated as the solve proceeds, so a caller that
    catches a :class:`NewtonFailure` still sees the partial accounting.

    中文：单次求解的可变计量汇（绝不跨求解共享）。``evaluate_seconds`` 与
    ``linear_seconds`` 分别累计求值调用与线性求解内的墙钟时间（perf_counter）；
    矩阵自由 matvec 的求值时间会同时计入两个桶（其运行在线性求解内），这是既定
    的归并约定。各类计数在求解过程中即时更新，因此即便捕获到
    :class:`NewtonFailure`，也能看到部分计量。
    """

    evaluate_seconds: float = 0.0  # 求值调用累计墙钟秒数
    linear_seconds: float = 0.0
    linear_solves: int = 0
    linear_iterations: int = 0
    line_search_trials: int = 0
    failed_trials: int = 0


@dataclass(frozen=True)
class NewtonIterationRecord:
    """Read-only record of one accepted iterate (iteration 0 is the start).

    ``trial_residual_norms`` lists the normalized infinity norm of every
    line-search trial of the step that led to this iterate, in order, with
    rejected non-finite trials recorded as ``inf``.  ``trial_merit_norms``
    records the raw two-norm merit used for the Armijo test of the same trials.
    ``damping`` and ``step_norm`` are ``None`` for the start record.

    中文：单条被接受迭代的只读记录（第 0 次为起点）。``trial_residual_norms``
    依次列出该步线搜索每个试探点的归一化无穷范数，非有限的被拒试探记为 ``inf``；
    ``trial_merit_norms`` 记录同一批试探用于 Armijo 测试的原始二范数 merit。
    起点记录中 ``damping`` 与 ``step_norm`` 为 ``None``。
    """

    iteration: int  # 迭代号；0 表示起点
    residual_norm: float
    raw_residual_norm: float
    threshold: float
    damping: float | None
    trial_count: int
    trial_residual_norms: tuple
    trial_merit_norms: tuple
    step_norm: float | None

    def __post_init__(self):
        object.__setattr__(self, "trial_residual_norms",
                           tuple(float(value) for value in self.trial_residual_norms))
        object.__setattr__(self, "trial_merit_norms",
                           tuple(float(value) for value in self.trial_merit_norms))


@dataclass(frozen=True)
class NewtonResult:
    """Read-only outcome of a successful solve.

    ``success`` is always ``True`` for a returned result: non-convergence
    raises :class:`NewtonFailure` instead.  ``x`` is a read-only copy;
    ``residual_history`` and ``raw_residual_history`` hold the normalized and
    raw infinity norms per accepted iterate (iteration 0 first).  ``records``
    holds the same iterates in full detail.  The counters support the M5 cost
    accounting: evaluator calls, linear solves, line-search trials and, for the
    matrix-free path, Krylov iterations (``linear_iterations`` is 0 when the
    direct factorization is used; ``linear_solver`` names the chosen algorithm).

    中文：一次成功求解的只读结果。返回的结果 ``success`` 恒为 ``True``（未收敛
    会抛 :class:`NewtonFailure`）。``x`` 为只读拷贝；``residual_history`` 与
    ``raw_residual_history`` 按迭代（第 0 次在前）保存归一化与原始无穷范数；
    ``records`` 保存同一批迭代的完整细节。计数器用于 M5 成本核算：求值次数、
    线性求解次数、线搜索试探次数，以及矩阵自由路径的 Krylov 迭代数（用直接分解
    时 ``linear_iterations`` 为 0）。
    """

    success: bool  # 恒为 True：返回即收敛
    x: np.ndarray
    reason: str
    residual_norm: float
    raw_residual_norm: float
    residual_history: tuple
    raw_residual_history: tuple
    records: tuple
    evaluations: int
    linear_solves: int
    line_search_trials: int
    linear_iterations: int = 0
    linear_solver: str = "direct"

    def __post_init__(self):
        x = np.array(self.x, dtype=float, copy=True)
        x.setflags(write=False)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "residual_history",
                           tuple(float(value) for value in self.residual_history))
        object.__setattr__(self, "raw_residual_history",
                           tuple(float(value) for value in self.raw_residual_history))
        object.__setattr__(self, "records", tuple(self.records))


def _set_failure_iteration(failure, iteration):
    """Attribute an evaluation-raised failure to the current Newton iteration.

    The T22 monolithic adapter raises its failures while an evaluation property
    is read (``residual``, ``jacobian``), a moment at which the adapter cannot
    know the Newton iteration; it marks them with ``iteration=-1`` and the
    session fills the iteration in when it catches them.

    中文：把求值阶段抛出的失败归因到当前 Newton 迭代。T22 单体适配器在读取
    ``residual``/``jacobian`` 属性时抛错，此时它尚不知迭代号，故先以
    ``iteration=-1`` 标记，由本 session 在捕获时补填真实迭代号并刷新异常消息。
    """
    if failure.iteration < 0:
        failure.iteration = int(iteration)
        detail = "" if failure.residual_norm is None else f"; ||r|| = {failure.residual_norm:.3e}"
        failure.args = (f"Newton failure at phase {failure.phase!r}, iteration {failure.iteration}: "
                        f"{failure.reason}{detail}",)
    return failure


class _NewtonSession:
    """Mutable state of one solve; never shared between calls.

    中文：单次求解的可变状态；绝不在调用间共享。持有当前迭代点、尺度向量、
    残差/切线缓存，以及迭代记录与各类计数器。
    """

    def __init__(self, evaluate, x0, scales, options, telemetry=None):
        self.evaluate = evaluate
        self.options = options
        self.telemetry = telemetry
        self.x = x0
        self.scales = scales
        self.n = int(x0.shape[0])
        self.records = []
        self.evaluations = 0
        self.linear_solves = 0
        self.linear_iterations = 0
        self.line_search_trials = 0
        self.residual = None
        self.jacobian = None
        self.preconditioner = None
        self.norm = None
        self.raw_norm = None
        self.merit = None
        self.threshold = None

    # ------------------------------------------------------------- plumbing
    def failure(self, phase, iteration, reason, *, diagnostics=None,
                residual_norm=None, raw_residual_norm=None, x=None):
        return NewtonFailure(
            phase=phase, iteration=iteration, reason=reason,
            residual_norm=self.norm if residual_norm is None else residual_norm,
            raw_residual_norm=self.raw_norm if raw_residual_norm is None else raw_residual_norm,
            x=self.x if x is None else x,
            history=self.records, diagnostics=diagnostics)

    def _call(self, point, iteration):
        self.evaluations += 1
        callback_input = np.array(point, dtype=float, copy=True)  # 传给求值器的始终是全新副本
        started = perf_counter()
        try:
            result = self.evaluate(callback_input)
        except Exception as error:
            raise self.failure(
                "evaluate", iteration,
                f"the evaluator raised {type(error).__name__}: {error}",
                diagnostics={"error_type": type(error).__name__, "error": str(error)}) from error
        finally:
            if self.telemetry is not None:
                self.telemetry.evaluate_seconds += perf_counter()-started
        if result is None:
            raise self.failure("evaluate", iteration, "the evaluator returned None")
        return result

    def _read_residual(self, result, iteration):
        try:
            value = getattr(result, "residual", None)
        except NewtonFailure as failure:
            raise _set_failure_iteration(failure, iteration)
        except Exception as error:
            raise self.failure(
                "evaluate", iteration,
                f"reading the residual raised {type(error).__name__}: {error}",
                diagnostics={"error_type": type(error).__name__, "error": str(error)}) from error
        if value is None:
            raise self.failure("residual", iteration,
                               "the evaluation does not expose a residual",
                               diagnostics={"contract": "Evaluation.residual"})
        try:
            return _own_residual(value, "evaluate(x).residual", self.n)
        except ValueError as error:
            raise self.failure("residual", iteration, str(error),
                               diagnostics={"contract": "Evaluation.residual"}) from error

    def _read_jacobian(self, result, iteration):
        try:
            value = getattr(result, "jacobian", None)
        except NewtonFailure as failure:
            raise _set_failure_iteration(failure, iteration)
        except Exception as error:
            raise self.failure(
                "jacobian", iteration,
                f"reading the jacobian raised {type(error).__name__}: {error}",
                diagnostics={"error_type": type(error).__name__, "error": str(error)}) from error
        if value is None:
            raise self.failure("jacobian", iteration,
                               "the evaluation does not expose a jacobian",
                               diagnostics={"contract": "Evaluation.jacobian"})
        try:
            return _own_jacobian(value, "evaluate(x).jacobian", self.n)
        except ValueError as error:
            raise self.failure("jacobian", iteration, str(error),
                               diagnostics={"contract": "Evaluation.jacobian"}) from error

    def _read_operator(self, result, iteration):
        """Matrix-free tangent (and optional preconditioner) for the GMRES path.

        中文：为 GMRES 路径读取矩阵自由切向（及可选预条件子），用
        ``aslinearoperator`` 归一化为带 shape/matvec 的线性算子，并校验形状为
        (n, n)。
        """
        try:
            value = getattr(result, "jacobian", None)
        except NewtonFailure as failure:
            raise _set_failure_iteration(failure, iteration)
        except Exception as error:
            raise self.failure(
                "jacobian", iteration,
                f"reading the jacobian raised {type(error).__name__}: {error}",
                diagnostics={"error_type": type(error).__name__, "error": str(error)}) from error
        if value is None:
            raise self.failure("jacobian", iteration,
                               "the evaluation does not expose a jacobian or linear operator",
                               diagnostics={"contract": "Evaluation.jacobian"})
        try:
            operator = aslinearoperator(value)
        except Exception as error:
            raise self.failure(
                "jacobian", iteration,
                f"the jacobian is neither an assembled matrix nor a linear operator: "
                f"{type(error).__name__}: {error}",
                diagnostics={"contract": "Evaluation.jacobian",
                             "error_type": type(error).__name__, "error": str(error)}) from error
        if operator.shape != (self.n, self.n):
            raise self.failure("jacobian", iteration,
                               f"the tangent operator must have shape ({self.n}, {self.n}), "
                               f"got {operator.shape}",
                               diagnostics={"contract": "Evaluation.jacobian"})
        try:
            preconditioner = getattr(result, "preconditioner", None)
        except NewtonFailure as failure:
            raise _set_failure_iteration(failure, iteration)
        except Exception as error:
            raise self.failure(
                "linear_solve", iteration,
                f"reading the preconditioner raised {type(error).__name__}: {error}",
                diagnostics={"error_type": type(error).__name__, "error": str(error)}) from error
        if preconditioner is not None:
            try:
                preconditioner = aslinearoperator(preconditioner)
            except Exception as error:
                raise self.failure(
                    "linear_solve", iteration,
                    f"the preconditioner is neither an assembled matrix nor a linear operator: "
                    f"{type(error).__name__}: {error}",
                    diagnostics={"contract": "Evaluation.preconditioner",
                                 "error_type": type(error).__name__, "error": str(error)}) from error
            if preconditioner.shape != (self.n, self.n):
                raise self.failure("linear_solve", iteration,
                                   f"the preconditioner must have shape ({self.n}, {self.n}), "
                                   f"got {preconditioner.shape}",
                                   diagnostics={"contract": "Evaluation.preconditioner"})
        self.preconditioner = preconditioner
        return operator

    def _read_tangent(self, result, iteration):
        if self.options.linear_solver == "direct":
            self.jacobian = self._read_jacobian(result, iteration)
        else:
            self.jacobian = self._read_operator(result, iteration)

    def _measures(self, residual):
        """Return (normalized infinity norm, raw infinity norm, raw two-norm)."""
        raw = float(np.max(np.abs(residual)))  # 原始无穷范数 max_i |r_i|
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            normalized = np.abs(residual) / self.scales  # 逐分量按物理尺度归一化
            square_sum = float(np.sum(residual * residual))
        if not np.all(np.isfinite(normalized)):
            scaled = float("inf")  # 任一归一化分量非有限则归一化范数记 inf
        else:
            scaled = float(np.max(normalized))
        # 收敛用归一化无穷范数，线搜索 merit 用原始残差二范数 sqrt(sum r_i^2)
        merit = float("inf") if not np.isfinite(square_sum) or square_sum < 0.0 \
            else float(np.sqrt(square_sum))
        return scaled, raw, merit

    def _record(self, iteration, *, damping, trial_norms, trial_merits, step_norm):
        return NewtonIterationRecord(
            iteration=iteration, residual_norm=self.norm, raw_residual_norm=self.raw_norm,
            threshold=self.threshold, damping=damping, trial_count=len(trial_norms),
            trial_residual_norms=tuple(trial_norms), trial_merit_norms=tuple(trial_merits),
            step_norm=step_norm)

    def _result(self, reason):
        return NewtonResult(
            success=True, x=self.x, reason=reason, residual_norm=self.norm,
            raw_residual_norm=self.raw_norm,
            residual_history=tuple(row.residual_norm for row in self.records),
            raw_residual_history=tuple(row.raw_residual_norm for row in self.records),
            records=tuple(self.records), evaluations=self.evaluations,
            linear_solves=self.linear_solves, line_search_trials=self.line_search_trials,
            linear_iterations=self.linear_iterations,
            linear_solver=self.options.linear_solver)

    # ------------------------------------------------------------ algorithm
    def _direct_direction(self, iteration):
        try:
            # 解 J·d = -r 得 Newton 方向 d（LU 分解，csc 仅转换格式不改数值）
            factorization = splu(sparse.csc_matrix(self.jacobian))
            direction = np.asarray(factorization.solve(-self.residual), dtype=float)
        except Exception as error:
            raise self.failure(
                "linear_solve", iteration,
                f"Jacobian factorization or solve failed: {type(error).__name__}: {error}",
                diagnostics={"solver": "splu", "nnz": int(self.jacobian.nnz),
                             "error_type": type(error).__name__, "error": str(error)}) from error
        return direction

    def _gmres_direction(self, iteration):
        options = self.options
        inner_iterations = [0]

        def count_iteration(residual_norm):
            # gmres 每完成一次内迭代触发一次回调；用单元素列表跨作用域累加计数
            inner_iterations[0] += 1
            self.linear_iterations += 1
            if self.telemetry is not None:
                self.telemetry.linear_iterations += 1

        try:
            direction, info = gmres(
                self.jacobian, -self.residual, M=self.preconditioner,
                rtol=options.gmres_rtol, atol=options.gmres_atol,
                restart=options.gmres_restart, maxiter=options.gmres_maxiter,
                callback=count_iteration, callback_type="pr_norm")
        except NewtonFailure as failure:
            raise _set_failure_iteration(failure, iteration)
        except Exception as error:
            raise self.failure(
                "linear_solve", iteration,
                f"GMRES failed: {type(error).__name__}: {error}",
                diagnostics={"solver": "gmres", "iterations": inner_iterations[0],
                             "restart": options.gmres_restart, "maxiter": options.gmres_maxiter,
                             "rtol": options.gmres_rtol, "atol": options.gmres_atol,
                             "error_type": type(error).__name__, "error": str(error)}) from error
        if info != 0:
            try:
                # info!=0 表示 GMRES 未达容差；显式算真实残差 ||r + J·d||（因 b = -r）
                true_residual = float(np.linalg.norm(
                    self.residual+self.jacobian.matvec(np.asarray(direction, dtype=float))))
            except Exception:
                true_residual = float("nan")
            raise self.failure(
                "linear_solve", iteration,
                "GMRES did not converge to the requested tolerance",
                diagnostics={"solver": "gmres", "iterations": inner_iterations[0],
                             "info": int(info), "restart": options.gmres_restart,
                             "maxiter": options.gmres_maxiter, "rtol": options.gmres_rtol,
                             "atol": options.gmres_atol,
                             "true_residual_norm": true_residual})
        return np.asarray(direction, dtype=float), inner_iterations[0]

    def _newton_direction(self, iteration):
        self.linear_solves += 1
        if self.telemetry is not None:
            self.telemetry.linear_solves += 1
        started = perf_counter()
        try:
            if self.options.linear_solver == "direct":
                direction = self._direct_direction(iteration)
                iterations = 0
                solver = "splu"
            else:
                direction, iterations = self._gmres_direction(iteration)
                solver = "gmres"
        finally:
            if self.telemetry is not None:
                self.telemetry.linear_seconds += perf_counter()-started
        if direction.shape != (self.n,) or not np.all(np.isfinite(direction)):
            # 方向形状不符或含非有限分量：判为线性求解失败并统计非有限分量数
            entries = -1 if direction.ndim != 1 else int(np.count_nonzero(~np.isfinite(direction)))
            raise self.failure("linear_solve", iteration, "the Newton direction is not finite",
                               diagnostics={"solver": solver,
                                            "iterations": iterations,
                                            "non_finite_entries": entries})
        return direction, float(np.max(np.abs(direction)))

    def _line_search(self, direction, step_norm, iteration):
        # Armijo merit: the raw residual two-norm.  Along the Newton direction
        # r(x + alpha*d) shrinks like (1 - alpha) r for small alpha, so this
        # merit always admits a small enough step.  A component-scaled norm is
        # deliberately NOT used here: when physical scales differ by orders of
        # magnitude it can reject a full step that drives the dominant
        # components down quadratically while a tiny component crosses zero
        # (regression: test_scales_preserve_nonlinear_solution crept for 50+
        # iterations before this choice).  Convergence still uses the
        # per-component normalized infinity norm.
        #
        # 中文：merit 取原始残差二范数。沿 Newton 方向，小 alpha 下
        # r(x+alpha*d) ≈ (1-alpha) r，故该 merit 总能接受足够小的步长。此处
        # 刻意不用分量归一化范数作 merit：当物理尺度相差数量级时，它可能拒绝
        # 一个能让主导分量二次收敛、而极小分量穿越零点的整步（回归案例：
        # test_scales_preserve_nonlinear_solution 曾因此爬行 50+ 迭代）。收敛判定
        # 仍用逐分量归一化无穷范数。
        options = self.options
        current = self.merit
        trial_norms = []
        trial_merits = []
        alpha = 1.0
        for _ in range(options.max_backtracks + 1):
            with np.errstate(over="ignore", invalid="ignore"):
                point = self.x + alpha * direction
            evaluation = None
            residual = None
            trial_norm, raw_norm, trial_merit = float("inf"), float("nan"), float("inf")
            if np.all(np.isfinite(point)):
                evaluation = self._call(point, iteration)
                residual = self._read_residual(evaluation, iteration)
                if np.all(np.isfinite(residual)):
                    trial_norm, raw_norm, trial_merit = self._measures(residual)
            self.line_search_trials += 1
            if self.telemetry is not None:
                self.telemetry.line_search_trials += 1
            trial_norms.append(trial_norm)
            trial_merits.append(trial_merit)
            # Armijo 充分下降：merit(alpha) <= (1 - c*alpha) * merit_current
            acceptable = np.isfinite(trial_merit) and (
                not options.line_search
                or trial_merit <= (1.0 - options.armijo_slope * alpha) * current)
            if acceptable:
                point.setflags(write=False)
                return (point, evaluation, residual, trial_norm, raw_norm, trial_merit,
                        alpha, trial_norms, trial_merits)
            if self.telemetry is not None:
                self.telemetry.failed_trials += 1
            if not options.line_search:
                break
            alpha *= options.backtrack_factor
            if alpha < options.minimum_damping:
                break
        raise self.failure(
            "line_search", iteration,
            "backtracking failed to find a finite step with sufficient decrease",
            diagnostics={"trial_residual_norms": tuple(trial_norms),
                         "trial_merit_norms": tuple(trial_merits),
                         "current_merit": current, "minimum_damping": options.minimum_damping,
                         "full_step_norm": step_norm})

    def run(self):
        options = self.options
        result = self._call(self.x, 0)  # 迭代 0：在起点 x0 处求值
        self.residual = self._read_residual(result, 0)
        if not np.all(np.isfinite(self.residual)):
            raise self.failure(
                "residual", 0, "the initial residual is not finite",
                residual_norm=float("inf"), raw_residual_norm=float("nan"),
                diagnostics={"non_finite_entries":
                             int(np.count_nonzero(~np.isfinite(self.residual)))})
        self.norm, self.raw_norm, self.merit = self._measures(self.residual)
        if not np.isfinite(self.norm):
            raise self.failure("residual", 0, "the normalized initial residual norm is not finite",
                               residual_norm=float("inf"), raw_residual_norm=self.raw_norm,
                               diagnostics={"scaled_norm_is_finite": False})
        self.threshold = options.atol + options.rtol * self.norm  # 阈值 = atol + rtol*||r0||
        if not np.isfinite(self.threshold):
            raise self.failure("residual", 0, "the convergence threshold is not finite",
                               diagnostics={"atol": options.atol, "rtol": options.rtol})
        self.records.append(self._record(0, damping=None, trial_norms=(), trial_merits=(),
                                         step_norm=None))
        if self.norm <= self.threshold:
            return self._result("converged")  # 起点即满足阈值，无需迭代
        self._read_tangent(result, 0)  # 读第 0 步切线，供首次 Newton 方向使用
        for iteration in range(1, options.max_iterations + 1):
            direction, step_norm = self._newton_direction(iteration)
            point, result, residual, trial_norm, raw_norm, trial_merit, damping, \
                trial_norms, trial_merits = self._line_search(direction, step_norm, iteration)
            self.x = point  # 只有通过线搜索的试探点才被接受为当前迭代
            self.residual = residual
            self.norm = trial_norm
            self.raw_norm = raw_norm
            self.merit = trial_merit
            self.records.append(self._record(iteration, damping=damping,
                                             trial_norms=trial_norms, trial_merits=trial_merits,
                                             step_norm=step_norm))
            if self.norm <= self.threshold:
                return self._result("converged")
            self._read_tangent(result, iteration)
        raise self.failure(
            "convergence", options.max_iterations,
            "maximum iterations exceeded without convergence",
            diagnostics={"max_iterations": options.max_iterations, "threshold": self.threshold})


def solve_newton(evaluate, x0, *, options=None, scales=None, telemetry=None):
    """Solve ``r(x) = 0`` from ``x0`` with a damped Newton iteration.

    ``evaluate`` must be a pure callable ``evaluate(x) -> Evaluation`` exposing
    the residual and tangent of the system (see :class:`Evaluation`).  It
    receives a fresh copy of each trial point, so it cannot mutate solver state;
    the solver copies everything it reads, so evaluator buffers may be reused.

    ``scales`` is an optional strictly positive per-component physical scale
    vector.  Convergence is tested on ``max_i |r_i| / scales_i <= atol +
    rtol * ||r0||`` where ``||r0||`` is the same normalized norm at ``x0``;
    both raw and normalized norms are recorded.

    ``options.linear_solver`` selects the explicit tangent route:
    ``"direct"`` (default) factorizes the assembled ``jacobian`` with ``splu``,
    ``"gmres"`` treats ``jacobian`` as a matrix-free linear operator (an
    optional ``preconditioner`` attribute is forwarded to GMRES).  A failed
    matrix-free solve raises with ``phase="linear_solve"``; there is no silent
    fallback.

    ``telemetry`` is an optional :class:`NewtonTelemetry` sink that collects
    wall-clock buckets, Krylov iterations and trial counts, including the
    partial accounting of a failed solve.

    Returns a read-only :class:`NewtonResult` on convergence.  Every other
    outcome raises :class:`NewtonFailure` with phase, iteration, residual norm
    and diagnostics; the input arrays are never modified.

    中文：从 ``x0`` 出发用阻尼 Newton 迭代求解 ``r(x) = 0``。``evaluate`` 需为
    纯可调用 ``evaluate(x) -> Evaluation``，每次收到试探点的全新拷贝，故不会
    改动求解器状态；求解器复制所有读入值，求值器可复用缓冲区。``scales`` 为
    可选的逐分量正物理尺度（单位与未知量一致），收敛判据为
    ``max_i |r_i|/scales_i <= atol + rtol*||r0||``，同时记录原始与归一化范数。
    ``options.linear_solver`` 选择切线求解路径（direct=splu，gmres=矩阵自由），
    矩阵自由求解失败即抛 ``phase="linear_solve"``，无静默回退。``telemetry`` 为
    可选的 :class:`NewtonTelemetry` 计量汇。收敛返回只读
    :class:`NewtonResult`，其它情况抛 :class:`NewtonFailure`；输入数组永不被修改。
    """
    if not callable(evaluate):
        raise TypeError("evaluate must be callable.")
    options = NewtonOptions() if options is None else options
    if not isinstance(options, NewtonOptions):
        raise TypeError("options must be a NewtonOptions instance.")
    if telemetry is not None and not isinstance(telemetry, NewtonTelemetry):
        raise TypeError("telemetry must be a NewtonTelemetry instance or None.")
    initial = _own_vector(x0, "x0")
    if scales is None:
        scale = np.ones(initial.shape[0])  # 未给尺度时视为单位尺度（无归一化）
        scale.setflags(write=False)
    else:
        scale = _own_vector(scales, "scales", size=initial.shape[0])
        if np.any(scale <= 0.0):
            raise ValueError("scales must be strictly positive.")
    return _NewtonSession(evaluate, initial, scale, options, telemetry).run()

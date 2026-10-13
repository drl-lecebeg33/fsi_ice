"""Dirichlet--Neumann iteration with fixed, scaled Aitken or IQN-ILS updates.

Callbacks evaluate candidates against a fixed accepted history. They must not
accept time histories or publish states. The outer full-field transaction owns
the eventual acceptance. See docs/development/t18_plan.md for norm conventions.

中文说明：本模块实现 Dirichlet--Neumann 分区耦合的定常不动点迭代，支持三种
界面松弛控制：``fixed``（固定松弛）、``aitken``（标量 Aitken 动态松弛）与
``iqn``（IQN-ILS 拟牛顿最小二乘松弛）。迭代每步对一个候选界面量 x 求解
G(x)（流体→载荷→固体→界面迹），以界面残差 r=G(x)-x 判敛，再按所选方法更新。
关键约定：残差与尺度均逐分量归一化（无量纲），收敛用无穷范数；回调只针对固定的
已接受历史求值，绝不接受时间历史或发布状态，最终接受由外层全场事务统一负责。
"""

from dataclasses import dataclass

import numpy as np


# 中文：把输入拷贝为独立只读的一维有限浮点向量，可选校验形状；用于保护所有
# 外部传入的界面量/尺度向量不被原地修改。
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
    """Dimensionless tolerances; max_iterations counts combined evaluations.

    中文：耦合迭代的控制参数，各容差均为无量纲量；``max_iterations`` 统计的是
    合并求值次数（一次候选 G(x) 求值算一次迭代）。``method`` 选择界面松弛方式。
    """

    relaxation: float = 0.5  # 固定/初始界面松弛因子 omega，须在 (0,1]
    absolute_tolerance: float = 1e-11  # 绝对收敛容差（作用于归一化残差）
    relative_tolerance: float = 1e-9  # 相对收敛容差（乘以界面量级）
    max_iterations: int = 150  # 最大耦合迭代数（含合并求值次数）
    case_name: str = "partitioned"  # 用例名，用于失败消息
    method: str = "fixed"  # 界面松弛方法："fixed" / "aitken" / "iqn"
    minimum_relaxation: float = 0.05  # Aitken 系数裁剪下界
    maximum_relaxation: float = 1.0  # Aitken 系数裁剪上界
    aitken_denominator_tolerance: float = 1e-14  # Aitken 分母退化判据（相对阈值）
    iqn_initial_relaxation: float = 1.0  # IQN 初始松弛 omega_0，须在 (0,1]
    iqn_alpha_max: float = 10.0  # |alpha| 上限，超出即触发 IQN 重启
    iqn_history_size: int = 0  # 保存的 (s,y) 对数上限；0 表示无界
    iqn_restart_tolerance: float = 1e-2  # LS 相对残差重启阈值

    def __post_init__(self):
        base_scalars = ("relaxation", "absolute_tolerance", "relative_tolerance",
                        "minimum_relaxation", "maximum_relaxation", "aitken_denominator_tolerance")
        iqn_scalars = ("iqn_initial_relaxation", "iqn_alpha_max", "iqn_restart_tolerance")
        for name in (*base_scalars, *iqn_scalars):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or \
                    isinstance(value, (str, bytes, complex, np.complexfloating)):
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
        if not isinstance(self.method, str) or self.method not in ("fixed", "aitken", "iqn"):
            raise ValueError("method must be 'fixed', 'aitken' or 'iqn'.")
        if not 0 < self.minimum_relaxation <= self.maximum_relaxation:
            raise ValueError("Relaxation bounds must satisfy 0 < minimum_relaxation <= maximum_relaxation.")
        if self.aitken_denominator_tolerance <= 0:
            raise ValueError("aitken_denominator_tolerance must be strictly positive.")
        if self.method == "aitken" and not self.minimum_relaxation <= self.relaxation <= self.maximum_relaxation:
            raise ValueError("Aitken requires minimum_relaxation <= relaxation <= maximum_relaxation.")
        if not 0 < self.iqn_initial_relaxation <= 1:
            raise ValueError("iqn_initial_relaxation must lie in (0,1].")
        if self.iqn_alpha_max <= 0:
            raise ValueError("iqn_alpha_max must be strictly positive.")
        if isinstance(self.iqn_history_size, (bool, np.bool_)) or \
                not isinstance(self.iqn_history_size, (int, np.integer)) or self.iqn_history_size < 0:
            raise ValueError("iqn_history_size must be a nonnegative integer.")
        if self.iqn_restart_tolerance <= 0:
            raise ValueError("iqn_restart_tolerance must be strictly positive.")


@dataclass(frozen=True)
class PartitionedEvaluation:
    """G(x) and the single-field states computed from the same input x.

    fluid, load and solid are opaque physical result objects. The returned
    target is copied by the controller. It never overwrites x with G(x) at
    convergence: fluid data were computed using x, with the recorded gap to G(x).

    中文：一次分区求值的结果，含界面映象 G(x) 与由同一输入 x 算出的各单场状态。
    ``fluid``/``load``/``solid`` 为不透明的物理结果对象；``target`` 由控制器拷贝。
    收敛时不会用 G(x) 覆盖 x：流体数据是用 x 算出的，二者之差（gap）会被记录。
    """

    target: np.ndarray  # 界面映象 G(x)，与 x 同维、同单位
    fluid: object  # 流体解（不透明）
    load: object  # 界面载荷传递结果（不透明）
    solid: object  # 固体解（不透明）


class _StageFailure(RuntimeError):
    """中文：分区单场求解的失败包装，记录出错阶段名（fluid/load/solid/trace）
    及其原因，供外层归纳为带阶段的 CouplingFailure。"""

    def __init__(self, stage, cause):
        self.stage = stage  # 出错阶段名，由 DirichletNeumannMap.__call__ 逐步赋值
        super().__init__(str(cause))


@dataclass(frozen=True)
class DirichletNeumannMap:
    """Adapter: fluid_solve(x) -> load_transfer(fluid) -> solid_solve(load).

    solid_trace(solid) extracts the interface vector in the exact ordering and
    units of x. Geometry and BDF contexts belong to the caller's candidate
    workspace. This adapter reuses solvers rather than defining new physics.

    中文：适配器，按 fluid_solve(x) -> load_transfer(fluid) -> solid_solve(load)
    顺序求值界面映象。``solid_trace(solid)`` 抽取与 x 完全同序、同单位的界面向量。
    几何与 BDF 上下文由调用方的候选工作区提供；本适配器只复用既有求解器，不定义
    新物理。
    """

    fluid_solve: object  # 由界面位移求解流体，返回不透明流体解
    load_transfer: object  # 把流体解转换为界面载荷
    solid_solve: object  # 由界面载荷求解固体
    solid_trace: object  # 由固体解抽取界面向量（与 x 同序同单位）

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
    """中文：一条耦合迭代的只读记录，含迭代号、归一化界面残差范数、当步收敛阈值、
    实际使用的松弛因子及松弛依据（fixed/aitken/iqn/initial/converged 等）。"""

    iteration: int  # 迭代号（从 1 起）
    residual_norm: float  # 本步归一化界面残差无穷范数 max_i |r_i|/scales_i
    threshold: float  # 本步收敛阈值 atol + rtol*界面量级
    relaxation: float  # 本步实际使用的松弛因子（或 IQN 的 alpha 系数）
    residual: np.ndarray  # 界面残差 r = G(x) - x（原始、未归一化）
    relaxation_reason: str = "fixed"  # 松弛因子的来源说明


def _iqn_max_alpha(alpha):
    """Maximum absolute alpha coefficient, or ``-∞`` for an empty vector."""

    # 中文：返回 |alpha| 的最大值；空向量的上界取 -inf。
    if alpha.size == 0:
        return -np.inf
    return float(np.max(np.abs(alpha)))


def _iqn_solve(V, rhs):
    """Return ``(alpha, used_columns, ls_residual_norm)`` for the LS problem
    V·α = rhs. ``used_columns`` is the LS-reported numerical rank, which may
    be less than the column count when V is rank-deficient.
    """

    # 中文：求解最小二乘问题 V·alpha = rhs，返回 (alpha, 数值秩, LS 残差范数)。
    # 当 V 秩亏时，``used_columns``（数值秩）会小于列数，可用于触发 IQN 重启。
    alpha, residuals, rank, _ = np.linalg.lstsq(V, rhs, rcond=None)
    if residuals.size:
        ls_residual = float(np.sqrt(residuals[0]))
    else:
        ls_residual = float(np.linalg.norm(V @ alpha - rhs))
    return alpha, int(rank), ls_residual


@dataclass(frozen=True)
class CouplingResult:
    """中文：分区耦合收敛结果，含收敛的界面量、对应的 PartitionedEvaluation、
    完整迭代历史与收敛原因。"""

    interface_values: np.ndarray  # 收敛的界面量 x（只读拷贝）
    evaluation: PartitionedEvaluation  # 在 x 处求得的 G(x) 与各单场状态
    history: tuple[CouplingIteration, ...]  # 全部迭代记录
    reason: str = "interface_residual"  # 收敛原因


class CouplingFailure(RuntimeError):
    """Terminal candidate failure, with evidence available to a transaction.

    中文：候选求解的终止性失败，携带可供事务使用的证据：用例名、迭代号、阶段、
    原因、失败时的界面量与已产生的迭代历史。
    """

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

    For ``method='iqn'``, V contains residual differences y_j, W contains
    output differences s_j+y_j, and (V/scales)*alpha = -r/scales is fitted
    in least squares. The update is x_new = G(x) + W*alpha, the standard
    IQN-ILS inverse-residual secant update. Iteration 1 has no history and
    falls back to ω_0·r_k. Restart clears the stored pairs and falls back to
    the same ω_0·r_k step. Restart fires when LS reports a numerically
    rank-deficient V, any α exceeds ``iqn_alpha_max``, or the relative LS
    residual exceeds ``iqn_restart_tolerance``. The capped history length is
    ``iqn_history_size`` (0 = unbounded, capped at ``max_iterations−1``).

    中文：按顺序求值 G(x)、检验其与 x 的真实间隙，再施加松弛更新
    x += omega*(G(x)-x)。``scales`` 为与 ``initial`` 同形的正物理尺度向量。
    收敛判据：||(G(x)-x)/scales||_inf <= atol + rtol*max(||x/scales||_inf,
    ||G(x)/scales||_inf)。输入数组永不被修改（即便回调重新打开其私有输入的
    可写标志）。迭代耗尽或单场异常绝不返回成功。Aitken 内积用 r/scales 计算；
    相对退化时沿用上一系数；非有限计算则以诊断信息报错。历史中的 relaxation
    描述外出更新，终止行除外——它保留上一系数且不施加未评估的更新。
    IQN-ILS（``method='iqn'``）更新为 x_{k+1} = x_k − V_k·alpha_k，其中 V_k 的
    列 v_j = s_j + omega_0*y_j，alpha_k 最小二乘解 V_k·alpha = −r_k，
    omega_0 = ``iqn_initial_relaxation``；首步无历史时退回 omega_0*r_k，重启时
    清空历史并同样退回该步。重启条件：V 数值秩亏、任一 |alpha| 超过
    ``iqn_alpha_max``，或 LS 相对残差超过 ``iqn_restart_tolerance``。
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
    previous_scaled_residual = None  # 上一步归一化残差 r/scales，供 Aitken 差分使用
    relaxation = options.relaxation
    iqn_omega0 = options.iqn_initial_relaxation
    # 中文：待提升的 IQN 增量对 (s_j, r_j)：s_j 为上一步位移增量，r_j 为对应残差。
    iqn_pending = None  # (s_j, r_j) for the just-applied update at iteration j
    # 中文：已提升的 (s_j, y_j) 对列表，y_j = r_{j+1} - r_j；r_{j+1} 已知后入列。
    iqn_history = []  # list of (s_j, y_j) promoted once r_{j+1} is known.
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
                # 归一化残差无穷范数：逐分量除以物理尺度，量纲一致
                norm = float(np.max(np.abs(residual / scales)))
                # 界面量级：取 x 与 G(x) 归一化后较大者，用于相对容差
                magnitude = max(float(np.max(np.abs(values / scales))),
                                float(np.max(np.abs(target / scales))))
                # 收敛阈值 = atol + rtol*界面量级
                threshold = options.absolute_tolerance + options.relative_tolerance*magnitude
            if not np.isfinite(threshold):
                raise ValueError("Coupling convergence threshold overflowed.")
        except Exception as error:
            phase = error.stage if isinstance(error, _StageFailure) else stage
            raise CouplingFailure(options.case_name, iteration, phase, str(error), values, history) from error
        if norm <= threshold:
            # 达到界面残差阈值：记录收敛行并返回当前 x 及对应求值
            history.append(CouplingIteration(iteration, norm, threshold, relaxation, residual, "converged"))
            evaluation = PartitionedEvaluation(target, evaluated.fluid, evaluated.load, evaluated.solid)
            return CouplingResult(_vector(values, "converged input"), evaluation, tuple(history))
        if iteration == options.max_iterations:
            # 最后一次迭代仍未收敛：记录后以 convergence 阶段终止
            history.append(CouplingIteration(iteration, norm, threshold, relaxation, residual, "max_iterations"))
            raise CouplingFailure(options.case_name, iteration, "convergence", "maximum iterations exceeded",
                                  values, history)
        # Promote the pending pair into history once both (s_j, r_j) and r_{j+1}
        # are available. y_j = r_{j+1} − r_j.
        # 中文：当 (s_j, r_j) 与 r_{j+1} 都具备时把待提升对提升进历史，
        # y_j = r_{j+1} - r_j；仅在 IQN 方法且 y_j 有限时保留。
        if iqn_pending is not None:
            s_j, r_j = iqn_pending
            iqn_pending = None
            if options.method == "iqn":
                y_j = residual - r_j
                if np.all(np.isfinite(y_j)):
                    iqn_history.append((s_j, y_j))
                    if options.iqn_history_size and len(iqn_history) > options.iqn_history_size:
                        iqn_history = iqn_history[-options.iqn_history_size:]
        relaxation_reason = "fixed"
        # 中文：若非 None，则用它替代常规 omega*residual 更新（IQN 路径）。
        applied_update = None  # If not None, overrides omega*(residual).
        if options.method == "aitken":
            relaxation_reason = "initial"
            scaled_residual = residual / scales  # 当前归一化残差 r/scales
            if previous_scaled_residual is not None:
                # Relative denominator test in the same dimensionless inner
                # product as the coefficient; independent of component units.
                # 中文：退化判据在系数所用的同一无量纲内积空间中做，与分量单位无关。
                with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                    delta = scaled_residual - previous_scaled_residual  # 相邻归一化残差之差
                    denominator = float(delta @ delta)  # ||delta||^2
                    numerator = float(previous_scaled_residual @ delta)  # prev·delta
                    residual_size = max(float(previous_scaled_residual @ previous_scaled_residual),
                                        float(scaled_residual @ scaled_residual))  # max(||prev||^2,||r||^2)
                    finite_products = np.all(np.isfinite([denominator, numerator, residual_size]))
                    # 分母相对退化：delta 太小（相对残差量级），沿用上一系数
                    degenerate = finite_products and denominator <= options.aitken_denominator_tolerance*residual_size
                    # 标量 Aitken 系数 omega_new = -omega*(prev·delta)/(delta·delta)
                    coefficient = relaxation if degenerate or not finite_products else \
                        -relaxation * (numerator / denominator)
                if not finite_products or not np.isfinite(coefficient):
                    # 系数运算非有限：记录后以 relaxation 阶段终止
                    history.append(CouplingIteration(iteration, norm, threshold, relaxation, residual,
                                                     "coefficient_nonfinite"))
                    raise CouplingFailure(options.case_name, iteration, "relaxation",
                                          "Aitken coefficient arithmetic is non-finite", values, history)
                if degenerate:
                    relaxation_reason = "denominator_degenerate"  # 退化：沿用上一系数
                else:
                    # 把系数裁剪到 [minimum, maximum]，越界记为 clipped
                    relaxation = float(np.clip(coefficient, options.minimum_relaxation,
                                               options.maximum_relaxation))
                    relaxation_reason = "clipped" if relaxation != coefficient else "aitken"
            previous_scaled_residual = scaled_residual  # 供下次差分
        elif options.method == "iqn":
            if not iqn_history:
                # 首步无历史：退回 omega_0*r_k 常规松弛步
                relaxation = iqn_omega0
                relaxation_reason = "iqn_initial"
            else:
                # V 的列 y_j = F(x_j) - F(x_{k-1}) 为残差差（按 scales 无量纲化），
                # W 的列 s_j + y_j 为对应的求解器输出差；最小二乘解 V·alpha = -r_k/scales。
                try:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        V = np.column_stack([y_j for _, y_j in iqn_history]) / scales[:, None]
                        W = np.column_stack([s_j+y_j for s_j, y_j in iqn_history])
                        scaled_residual = residual/scales
                        alpha, used, ls_residual = _iqn_solve(V, -scaled_residual)
                except (np.linalg.LinAlgError, ValueError, FloatingPointError) as error:
                    history.append(CouplingIteration(iteration, norm, threshold, iqn_omega0, residual,
                                                     "iqn_restart"))
                    raise CouplingFailure(options.case_name, iteration, "relaxation",
                                          f"IQN-ILS linear solve failed: {error}", values, history) from error
                alpha_inf = _iqn_max_alpha(alpha)
                # 相对尺度：以无量纲化残差的二范数与机器最小值兜底，避免除零
                reference_scale = max(float(np.linalg.norm(scaled_residual)), np.finfo(float).tiny)
                # 重启判据：V 秩亏、|alpha| 过大，或 LS 相对残差超过阈值
                restart = (used < len(iqn_history)
                           or not np.isfinite(alpha_inf)
                           or not np.isfinite(ls_residual)
                           or alpha_inf > options.iqn_alpha_max
                           or (ls_residual > 0
                               and ls_residual / reference_scale > options.iqn_restart_tolerance))
                if restart:
                    relaxation = iqn_omega0  # 清空历史并退回 omega_0*r_k
                    relaxation_reason = "iqn_restart"
                    iqn_history = []
                else:
                    relaxation = alpha_inf  # 记录 |alpha|_inf 作为本步"松弛量"
                    relaxation_reason = "iqn"
                    with np.errstate(over="ignore", invalid="ignore"):
                        applied_update = residual + W @ alpha  # x_new = x + r + W·alpha
        history.append(CouplingIteration(iteration, norm, threshold, relaxation, residual, relaxation_reason))
        try:
            with np.errstate(over="raise", invalid="raise"):
                if applied_update is None:
                    # 常规松弛：x_{k+1} = x_k + omega*r_k
                    next_values = _vector(values + relaxation*residual, "relaxed interface input")
                else:
                    # IQN：x_new = G(x) + W·alpha = x + r + W·alpha（标准 IQN-ILS）
                    next_values = _vector(values + applied_update, "IQN-ILS interface input")
        except Exception as error:
            raise CouplingFailure(options.case_name, iteration, "relaxation", str(error), values, history) from error
        if options.method == "iqn":
            # Store (s, r) for the just-applied update; the next iteration forms
            # y = r_{next} − r once r_{next} is computed and promotes the pair.
            # 中文：记录本步位移增量 s 与残差 r，待下次算出 r_{next} 后提升为 (s, y)。
            iqn_pending = (next_values - values, residual)
        values = next_values

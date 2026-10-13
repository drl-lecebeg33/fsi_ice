"""Fixed-step BDF with immutable accepted states and explicit commit/restore.

中文说明：本模块实现定步长 BDF1/BDF2 时间模板与历史管理。``BDFStencil`` 给出待求解步
的显式权重与时间点，``BDFContext`` 固定某一步的系数与历史项，``BDFHistory`` 保存最近
1~2 层接受状态并用 generation 记录代次。关键约定：权重均已除以 dt 并按“候选态到更早
接受态”排列；BDF2 在只有一层历史时自动降级为 BDF1 启动步；接受与恢复都做深复制并置
只读，接受时用 owner/generation/time 校验拒绝过期或外来上下文，以防 ABA 重放。
"""

from dataclasses import dataclass

import numpy as np


def frozen_copy(values):
    """Deep-copy ``values`` into an independent read-only finite float array.

    中文：把输入深复制为独立、只读且有限的浮点数组，供内部保存已接受状态；这样即便
    调用方重新开启写权限，也无法原地改写内部缓冲。
    """
    result = np.array(values, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("Time state must be finite.")
    result.flags.writeable = False  # 中文：关闭写权限，防止调用方原地改写内部缓冲。
    return result


# Standard BDF derivative weights for [candidate, latest accepted, older accepted],
# not yet divided by dt. Consistency of each set: coefficients sum to zero.
# 中文：标准 BDF 导数权重，按 [候选态, 最近接受态, 更早接受态] 排列，尚未除以 dt；
# 每组系数之和为零以保证时间一致性（BDF1: 1-1=0，BDF2: 1.5-2+0.5=0）。
BDF_WEIGHTS = {1: (1.0, -1.0), 2: (1.5, -2.0, 0.5)}


@dataclass(frozen=True, eq=False)
class BDFStencil:
    """Explicit weights and times of the BDF step about to be solved.

    The coefficients are already divided by dt and are ordered from the candidate
    state backwards in time: index 0 multiplies the candidate, index 1 the most
    recent accepted state, index 2 (BDF2 only) the state before that. ``times``
    holds the absolute times of those states, strictly decreasing.

    中文：描述待求解 BDF 步的权重与时间。coefficients 已除以 dt，index 0 乘候选态，
    index 1 乘最近接受态，index 2（仅 BDF2）乘更早接受态；times 为这些状态的绝对时间，
    严格递减且相邻间隔恰好为 dt。构造时会校验权重与标准权重一致、时间按 dt 间隔。
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

        中文：把已除 dt 的系数还原为无量纲的逐区间通量权重，用于对逐区间通量加权求和；
        BDF1 只返回第一个区间权重，BDF2 的第二个区间权重由前两个系数相加得到。
        """
        weights = self.coefficients*self.dt
        if self.order == 1:
            return (float(weights[0]),)
        return (float(weights[0]), float(weights[0]+weights[1]))


@dataclass(frozen=True)
class BDFContext:
    """One pending-step context: coefficient, history term, order and provenance.

    中文：一次待求解步的只读上下文，记录时间、系数、历史项、阶数、代次与所有者标识。
    ``derivative`` 计算 ``coefficient*candidate + history_term``，即已除以 dt 的 BDF 时间
    导数（单位：状态量每单位时间）。owner/generation 用于在接受时拒绝过期或外来上下文。
    """

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
    """Fixed-step BDF history of one field: 1-2 accepted layers, time and generation.

    中文：单个场的定步长 BDF 历史。内部保存最近 1~2 层已接受状态、当前时间与代次；
    ``snapshot``/``restore`` 用于保存与恢复，``stencil``/``context`` 生成下一步模板，
    ``accept`` 用私有 context 提交一层新状态并自增 generation。公开读取返回只读副本。
    """

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
        # 中文：BDF2 需要两层历史；启动阶段只有一层时自动退化为 BDF1。
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
        # 中文：系数与历史项均已除以 dt；BDF2 用最近两层状态拼历史项，BDF1 只用最新层。
        if stencil.order == 2:
            coefficient = 1.5/self.dt
            history = (-2*self._states[0]+0.5*self._states[1])/self.dt
        else:
            coefficient = 1/self.dt
            history = -self._states[0]/self.dt
        return BDFContext(self.time+self.dt, coefficient, frozen_copy(history),
                          stencil.order, self._generation, self._identity, stencil)

    def accept(self, context, candidate):
        # 中文：只接受本历史“当前代次 + 时间恰为下一步”的 context，拒绝过期/外来上下文（防 ABA）。
        if context.owner is not self._identity or context.generation != self._generation or not np.isclose(context.time, self.time+self.dt, rtol=0, atol=1e-14):
            raise ValueError("Cannot accept a stale or unrelated time context.")
        context.derivative(candidate)  # 中文：顺带校验 candidate 的形状与有限性。
        # 中文：新候选成为最新层，旧最新层保留，供 BDF2 使用（最多保留两层）。
        self._states = (frozen_copy(candidate), self._states[0])
        self.time = context.time
        self._generation += 1  # 中文：代次自增，使此前取出的 context 全部失效。

    def snapshot(self):
        """Return ``(time, generation, states)``; states are read-only copies.

        中文：返回当前时间、代次与只读状态副本，用于保存或复制该历史。
        """
        return self.time, self._generation, self.states

    def restore(self, snapshot):
        """Restore ``(time, generation, states)`` from ``snapshot`` onto this history.

        中文：用快照覆盖本历史的 time/generation/states；状态深复制并置只读，同时校验
        层数（1 或 2）与形状与当前历史兼容，避免恢复出不可用的历史。
        """
        time, generation, states = snapshot
        if not np.isfinite(time) or not isinstance(generation, int) or len(states) not in (1, 2):
            raise ValueError("Invalid time history snapshot.")
        copied = tuple(frozen_copy(state) for state in states)
        if any(state.shape != self._states[0].shape for state in copied):
            raise ValueError("Snapshot has an incompatible shape.")
        self.time, self._generation, self._states = time, generation, copied

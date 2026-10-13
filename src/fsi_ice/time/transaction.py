"""Full-field step transactions: accepted root, private workspace, atomic publish.

Design follows ``docs/development/validation_plan.md`` section 3 and
``docs/development/m3_freeze.md`` section 4.2:

* :class:`AcceptedState` is the single accepted root of a whole-field state:
  global time, fixed ``dt``/``order``, accepted step/epoch, the solid ``d/v``
  history, the fluid ``u``/``p`` with its velocity history, the mesh
  displacement ``a``/current ``x``/velocity ``w`` with history, the accepted
  interface motion/load, predictor values and successful monitoring rows.
  Every internal array is an independent deep copy; public readers receive
  read-only copies and accepted histories are never modified in place.
* :class:`StepTransaction` runs one attempt as
  ``begin -> evaluate -> validate -> prepare -> publish | abort`` on private
  work areas (three fresh ``BDFHistory`` objects plus candidate arrays that
  share no mutable storage with the root).  Any failure is terminal: the
  attempt is aborted, an audit record is appended next to the root (never
  inside it) and a new attempt with a fresh owner/``attempt_id`` is required,
  so old contexts or snapshots cannot be replayed (anti-ABA).
* ``prepare`` performs every fallible operation, including the three private
  ``BDFHistory.accept`` calls and the frozen monitoring row, before
  ``publish`` replaces the live root reference exactly once.  IO failures
  after publish are tracked as ``output_pending`` and rewritten idempotently
  per accepted epoch; the committed physical step is never rolled back.

:class:`StateCoordinator` owns the live root reference, grants the single
write permission at ``begin`` and serialises ``publish``/``load``/``restore``
and output bookkeeping on one lock.

中文说明：本模块实现“全场单步事务”。``AcceptedState`` 是全场唯一被接受的根状态：
内部数组均为独立深复制，公开读取返回只读副本，已接受历史永不原地修改。
``StepTransaction`` 以 ``begin -> evaluate -> validate -> prepare -> publish|abort``
在私有工作区推进单步（三个全新的 ``BDFHistory`` 加候选数组，与根状态不共享可变存储）。
任何失败都是终态的：本次尝试作废，在根状态之外追加失败审计记录，必须用全新的
owner/``attempt_id`` 重开尝试，使旧 context 或快照无法重放（防 ABA）。``prepare`` 执行
全部可能失败的操作（含三次私有 ``BDFHistory.accept`` 与冻结监测行），``publish`` 仅替换
一次活动根引用；发布后的 IO 失败记为 ``output_pending``，按接受 epoch 幂等重写，绝不
回滚已提交的物理步。``StateCoordinator`` 持有活动根引用，在 ``begin`` 授予唯一写权限，
并在单把锁上串行化 ``publish``/``load``/``restore`` 与输出记账。
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from fsi_ice.time.bdf import BDFContext, BDFHistory

__all__ = [
    "AcceptedState",
    "FailureRecord",
    "OutputRecord",
    "PendingOutput",
    "StateCoordinator",
    "StateSnapshot",
    "StepConflict",
    "StepPlan",
    "StepTransaction",
    "StepView",
    "TransactionError",
]

_FIELDS = ("solid", "fluid", "mesh")
_HOOKS = ("after_solid_accept", "after_fluid_accept", "after_mesh_accept", "before_freeze")
_CONFIRM_STAGES = ("geometry", "solid", "fluid", "interface")
# Globally monotonic attempt serial: every transaction gets a fresh id, so
# geometry revisions ``(attempt_id, candidate)`` are unique process-wide and
# never reused even by a retried terminal attempt.
# 中文：进程级单调递增的尝试序号，保证每次尝试的 id 唯一；几何版本
# ``(attempt_id, candidate)`` 因此全局唯一，重试的终态尝试也不会复用旧 id（防 ABA）。
_ATTEMPT_SERIAL = itertools.count(1)


class TransactionError(RuntimeError):
    """Lifecycle violation: wrong stage order or reuse of a terminal attempt.

    中文：事务生命周期违规，例如阶段顺序错误，或复用一个已终态（committed/aborted）的尝试。
    """


class StepConflict(TransactionError):
    """Publish conflict: the live accepted root no longer matches this base.

    中文：发布冲突，说明活动根状态已不再等于本事务的 base（身份或 epoch 已变），拒绝提交。
    """


def _as_int(value, name, *, minimum=0):
    """Validate and normalise an integer, rejecting bools as integers.

    中文：校验并规范化整数参数，显式排除 bool，并要求不小于 ``minimum``。
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


def _as_float(value, name, *, positive=False, nonnegative=False):
    """Validate and normalise a finite real number with optional sign bounds.

    中文：校验并规范化有限实数，可按需要求为正或非负；bool 视为非法输入。
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real number, got {value!r}.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive, got {number!r}.")
    if nonnegative and number < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {number!r}.")
    return number


def _own_array(values, shape, name):
    """Deep-copy ``values`` into an independent read-only finite float array.

    中文：把 ``values`` 深复制为独立、只读且有限的浮点数组（可选校验形状）；返回副本与
    入参不共享存储，因此外部改动无法影响内部已接受状态。
    """
    try:
        array = np.array(values, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real numeric array: {error}") from None
    if shape is not None and array.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    array.setflags(write=False)
    return array


def _expose_array(values):
    """Fresh read-only copy handed to callers so mutations cannot leak back.

    中文：交给调用方的是全新的只读副本，即便调用方重新开启写权限也无法回改内部缓冲。
    """
    copy = np.array(values, dtype=float, copy=True)
    copy.setflags(write=False)
    return copy


def _own_mapping(values, name):
    """Deep-copy a mapping of arrays into a read-only mapping of read-only arrays.

    中文：把映射中每个值深复制为只读、至少一维且有限的数组，并用只读代理包住整个映射；
    ``None`` 视为空映射。用于接口量/预报量等按名存放的数组集合。
    """
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping of arrays.")
    owned = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be nonempty strings.")
        array = np.array(np.asarray(value, dtype=float), copy=True)
        if array.ndim < 1:
            raise ValueError(f"{name}[{key!r}] must be at least one-dimensional.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name}[{key!r}] must be finite.")
        array.setflags(write=False)
        owned[key] = array
    return MappingProxyType(owned)


def _expose_mapping(values):
    """Read-only mapping of fresh read-only array copies for external readers.

    中文：返回由全新只读数组副本构成的只读映射，供外部安全读取。
    """
    return MappingProxyType({key: _expose_array(value) for key, value in values.items()})


def _freeze_row(value, path="row"):
    """Deep-freeze one serializable monitoring value.

    中文：把一行监测数据递归深冻结：映射转只读代理、list/tuple 转 tuple、ndarray 转只读
    副本、numpy 标量转 Python 标量；遇到不可序列化的类型抛 ``ValueError``。
    """
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_row(item, f"{path}[{key!r}]")
                                 for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_row(item, f"{path}[{index}]") for index, item in enumerate(value))
    if isinstance(value, np.ndarray):
        return _expose_array(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    raise ValueError(f"{path} contains a non-serializable value of type {type(value).__name__}.")


def _copy_row(value):
    """Deep-copy a frozen monitoring row back into plain mutable containers.

    中文：把已冻结的监测行递归复制为普通容器（映射转 dict、只读数组转只读副本），
    便于交给序列化器而不暴露内部存储。
    """
    if isinstance(value, Mapping):
        return {key: _copy_row(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_row(item) for item in value)
    if isinstance(value, np.ndarray):
        return _expose_array(value)
    return value


def _freeze_diagnostics(value):
    """Freeze a failure's diagnostics into a read-only mapping.

    中文：把失败诊断信息冻结为只读映射；非映射的裸值统一包进 ``{"detail": ...}``。
    """
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        return MappingProxyType({"detail": _freeze_row(value, "diagnostics")})
    return MappingProxyType({str(key): _freeze_row(item, f"diagnostics[{key!r}]")
                             for key, item in value.items()})


def _shape(value, name):
    """Normalise an extent sequence into a tuple of nonnegative integers.

    中文：把形状序列规范为整数元组，拒绝负数维度，用于接口/预报字段的形状声明。
    """
    try:
        shape = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a tuple of integers, got {value!r}.") from None
    if any(item < 0 for item in shape):
        raise ValueError(f"{name} must not contain negative extents, got {shape}.")
    return shape


def _rebuild_history(states, dt, order, time, generation):
    """Fresh ``BDFHistory`` with copied layers and the preserved generation.

    中文：用拷贝出的历史层与原有 generation 重建一个全新的 ``BDFHistory``；先按最新层
    构造，再用 ``restore`` 覆写完整快照，使副本与源历史互不影响。
    """
    history = BDFHistory(states[-1], dt, order=order, time=time)
    history.restore((time, generation, tuple(states)))
    return history


def _probe_generation(history):
    """Read the current generation of a history (snapshot index 1).

    中文：读取某历史当前的 generation（``snapshot`` 的第 2 个元素），用于代次一致性校验。
    """
    return history.snapshot()[1]


class AcceptedState:
    """Immutable whole-field accepted root; only a publish replaces it.

    All constructor inputs are deep-copied into independent read-only arrays.
    Public properties return fresh read-only copies, never the owned buffers.

    中文：全场唯一的、不可变的已接受根状态，只有 ``publish`` 会整体替换它。构造参数一律
    深复制为独立只读数组；对外属性返回全新只读副本而非内部缓冲。字段包含全局时间、
    固定 dt/order、已接受步号/epoch、固体 d/v 历史、流体 u/p 及其速度历史、网格位移 a、
    当前坐标 x 与速度 w 的历史、接口运动/载荷、预报量与成功的监测行。
    """

    __slots__ = ("time", "dt", "order", "step", "epoch", "_solid_history", "_fluid_history",
                 "_mesh_history", "_pressure", "_mesh_current", "_mesh_velocity", "_interface",
                 "_predictor", "_monitoring", "_identity")

    def __init__(self, *, time, dt, order, step, epoch, solid_states, fluid_states, mesh_states,
                 solid_generation, fluid_generation, mesh_generation, pressure, mesh_current,
                 mesh_velocity, interface=None, predictor=None, monitoring=()):
        self.time = _as_float(time, "time")
        self.dt = _as_float(dt, "dt", positive=True)
        order = _as_int(order, "order")
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2.")
        self.order = order
        self.step = _as_int(step, "step")
        self.epoch = _as_int(epoch, "epoch")
        solid_states = tuple(_own_array(state, None, f"solid_states[{index}]")
                             for index, state in enumerate(solid_states))
        fluid_states = tuple(_own_array(state, None, f"fluid_states[{index}]")
                             for index, state in enumerate(fluid_states))
        mesh_states = tuple(_own_array(state, None, f"mesh_states[{index}]")
                            for index, state in enumerate(mesh_states))
        if not solid_states or not fluid_states or not mesh_states:
            raise ValueError("Every field must carry at least one accepted history layer.")
        solid_shape = solid_states[0].shape
        if len(solid_shape) != 3 or solid_shape[0] != 2:
            raise ValueError(f"solid states must have shape (2, n_nodes, 2), got {solid_shape}.")
        if any(state.shape != solid_shape for state in solid_states):
            raise ValueError("All solid history layers must share one shape.")
        fluid_shape = fluid_states[0].shape
        if len(fluid_shape) != 2 or fluid_shape[1] != 2:
            raise ValueError(f"fluid states must have shape (n_P2, 2), got {fluid_shape}.")
        if any(state.shape != fluid_shape for state in fluid_states):
            raise ValueError("All fluid history layers must share one shape.")
        mesh_shape = mesh_states[0].shape
        if len(mesh_shape) != 2 or mesh_shape[1] != 2:
            raise ValueError(f"mesh states must have shape (n_nodes, 2), got {mesh_shape}.")
        if any(state.shape != mesh_shape for state in mesh_states):
            raise ValueError("All mesh history layers must share one shape.")
        layers = {len(solid_states), len(fluid_states), len(mesh_states)}
        if len(layers) != 1:
            raise ValueError("All fields must carry the same number of history layers.")
        layers = layers.pop()
        if not 1 <= layers <= 2:
            raise ValueError(f"History layer count {layers} must be 1 or 2.")
        self._solid_history = _rebuild_history(solid_states, self.dt, self.order, self.time,
                                               _as_int(solid_generation, "solid_generation"))
        self._fluid_history = _rebuild_history(fluid_states, self.dt, self.order, self.time,
                                               _as_int(fluid_generation, "fluid_generation"))
        self._mesh_history = _rebuild_history(mesh_states, self.dt, self.order, self.time,
                                              _as_int(mesh_generation, "mesh_generation"))
        pressure = _own_array(pressure, None, "pressure")
        if pressure.ndim != 1:
            raise ValueError(f"pressure must have shape (n_p,), got {pressure.shape}.")
        self._pressure = pressure
        self._mesh_current = _own_array(mesh_current, mesh_shape, "mesh_current")
        self._mesh_velocity = _own_array(mesh_velocity, mesh_shape, "mesh_velocity")
        self._interface = _own_mapping(interface, "interface")
        self._predictor = _own_mapping(predictor, "predictor")
        self._monitoring = tuple(_freeze_row(row, f"monitoring[{index}]")
                                 for index, row in enumerate(monitoring))
        self._identity = object()

    @classmethod
    def initial(cls, *, solid, velocity, pressure, mesh_displacement, mesh_current, mesh_velocity,
                time=0.0, dt, order=2, interface=None, predictor=None, monitoring=()):
        """Accepted root with one history layer per field at ``time``.

        中文：构造初始根状态：每个场只有一层历史，step/epoch/generation 均为 0，
        ``time`` 默认为 0。固体初始状态形状需为 ``(2, n_nodes, 2)``。
        """
        solid_values = np.asarray(solid, dtype=float)
        if solid_values.ndim != 3 or solid_values.shape[0] != 2:
            raise ValueError("initial solid state must have shape (2, n_nodes, 2).")
        return cls(time=time, dt=dt, order=order, step=0, epoch=0,
                   solid_states=(solid_values,), fluid_states=(np.asarray(velocity, dtype=float),),
                   mesh_states=(np.asarray(mesh_displacement, dtype=float),),
                   solid_generation=0, fluid_generation=0, mesh_generation=0,
                   pressure=pressure, mesh_current=mesh_current, mesh_velocity=mesh_velocity,
                   interface=interface, predictor=predictor, monitoring=monitoring)

    # ------------------------------------------------------------------ reads
    def _history(self, field):
        if field == "solid":
            return self._solid_history
        if field == "fluid":
            return self._fluid_history
        if field == "mesh":
            return self._mesh_history
        raise ValueError(f"Unknown field {field!r}; expected one of {_FIELDS}.")

    @property
    def solid_history(self):
        return tuple(_expose_array(state) for state in self._solid_history.states)

    @property
    def solid_state(self):
        return self.solid_history[0]

    @property
    def fluid_history(self):
        return tuple(_expose_array(state) for state in self._fluid_history.states)

    @property
    def velocity(self):
        return self.fluid_history[0]

    @property
    def mesh_history(self):
        return tuple(_expose_array(state) for state in self._mesh_history.states)

    @property
    def mesh_displacement(self):
        return self.mesh_history[0]

    @property
    def mesh_current(self):
        return _expose_array(self._mesh_current)

    @property
    def mesh_velocity(self):
        return _expose_array(self._mesh_velocity)

    @property
    def pressure(self):
        return _expose_array(self._pressure)

    @property
    def interface(self):
        return _expose_mapping(self._interface)

    @property
    def predictor(self):
        return _expose_mapping(self._predictor)

    @property
    def monitoring(self):
        return tuple(_copy_row(row) for row in self._monitoring)

    @property
    def generations(self):
        return MappingProxyType({field: _probe_generation(self._history(field)) for field in _FIELDS})

    @property
    def shapes(self):
        return MappingProxyType({
            "solid": tuple(self.solid_state.shape),
            "fluid": tuple(self.velocity.shape),
            "mesh": tuple(self.mesh_displacement.shape),
            "pressure": tuple(self._pressure.shape),
        })

    def _layer_counts(self):
        return {field: len(self._history(field).states) for field in _FIELDS}


@dataclass(frozen=True, eq=False)
class StateSnapshot:
    """Recoverable snapshot; only ``StateCoordinator.snapshot`` produces one.

    中文：可恢复的快照，记录根状态的全部信息与来源凭据（coordinator owner、状态 token、
    配置指纹、epoch/step/time 等）。只能由 ``StateCoordinator.snapshot`` 生成、并交由其
    ``restore`` 校验后使用，禁止手工拼装。
    """

    coordinator_owner: object
    state_token: object
    config_fingerprint: str
    epoch: int
    step: int
    time: float
    dt: float
    order: int
    solid_states: tuple
    fluid_states: tuple
    mesh_states: tuple
    solid_generation: int
    fluid_generation: int
    mesh_generation: int
    pressure: np.ndarray
    mesh_current: np.ndarray
    mesh_velocity: np.ndarray
    interface: Mapping
    predictor: Mapping
    monitoring: tuple


@dataclass(frozen=True, eq=False)
class FailureRecord:
    """One aborted attempt; kept outside the accepted root as a separate audit log.

    中文：一次被作废尝试的审计记录，保存在已接受根状态之外（绝不进入物理状态），记录
    attempt_id、base_epoch、目标步/时间、失败阶段与字段、原因及诊断信息。
    """

    attempt_id: int
    base_epoch: int
    target_step: int
    target_time: float
    dt: float
    order: int
    stage: str
    field: object
    reason: str
    diagnostics: Mapping


@dataclass(frozen=True)
class OutputRecord:
    """One successful output write for an accepted epoch (at most one per epoch).

    中文：某个已接受 epoch 的一次成功输出记录，每个 epoch 至多一条，用于幂等判定。
    """

    epoch: int
    step: int
    payload: object


@dataclass(frozen=True)
class PendingOutput:
    """An accepted physical step whose external output write failed and must be replayed.

    中文：已接受物理步的外部输出写出失败、等待重放的记录，保存 epoch、step、payload 与
    失败原因；物理根状态不回滚。
    """

    epoch: int
    step: int
    payload: object
    reason: str


@dataclass(frozen=True, eq=False)
class StepView:
    """Read-only candidate/state snapshot handed to plan callables.

    Arrays are fresh read-only copies, so a plan callable that re-enables
    writeability and mutates them cannot change the transaction workspace or
    the accepted root.  ``geometry`` is an opaque object owned by the plan.

    中文：交给 plan 可调用对象的只读候选/状态视图。其中数组均为全新只读副本，即使 plan
    重新开启写权限并修改，也改不到事务工作区或已接受根状态；``geometry`` 是由 plan 自持的
    不透明对象。``stencil`` 给出本尝试网格位移的 BDF 模板（已合并历史项）。
    """

    stage: str
    attempt_id: int
    base_epoch: int
    target_step: int
    target_time: float
    dt: float
    order: int
    base: AcceptedState
    mesh_context: BDFContext
    solid_context: BDFContext
    fluid_context: BDFContext
    revision: object
    displacement: object
    geometry: object
    mesh_current: object
    mesh_velocity: object
    solid: object
    velocity: object
    pressure: object
    interface: Mapping

    @property
    def stencil(self):
        """Mesh-displacement BDF stencil of this attempt (merged history term)."""
        return self.mesh_context.stencil


@dataclass(frozen=True, eq=False, kw_only=True)
class StepPlan:
    """Physics adapter for one step: candidate builders, solvers and lifecycle hooks.

    Each callable receives one :class:`StepView` and must return plain arrays;
    the transaction deep-copies and validates every returned value, so a plan
    never owns accepted storage.  ``confirms`` entries run right after the
    corresponding ``evaluate`` stage (geometry/solid/fluid/interface) and
    ``checks`` run during the final ``validate`` gate; any exception aborts the
    attempt with the entry name recorded in the failure audit.  ``hooks`` are
    observation/lifecycle seams invoked inside ``prepare`` (after each private
    accept and before freezing the new root); raising there must leave the live
    root untouched, which the fault-injection suite verifies.

    中文：单个物理步的物理适配器，提供候选构造函数、求解器与生命周期钩子。每个可调用
    对象接收一个 ``StepView`` 并返回普通数组，事务会深复制并校验返回值，因此 plan 不持有
    已接受存储。``confirms`` 在对应 evaluate 阶段后立即运行（geometry/solid/fluid/interface），
    ``checks`` 在最终 ``validate`` 门运行；任何异常都会作废本次尝试，并把条目名记入失败审计。
    ``hooks`` 是 ``prepare`` 内部的观察/生命周期接缝（每次私有 accept 之后、冻结新根之前），
    在其中抛错必须保证活动根不变，故障注入测试对此有验证。
    """

    build_displacement: Callable[[StepView], np.ndarray]
    build_geometry: Callable[[StepView], object]
    build_mesh_velocity: Callable[[StepView], np.ndarray]
    build_mesh_current: Callable[[StepView], np.ndarray]
    solve_solid: Callable[[StepView], np.ndarray]
    solve_fluid: Callable[[StepView], tuple]
    build_interface: Callable[[StepView], Mapping] | None = None
    interface_shapes: Mapping = field(default_factory=dict)
    build_predictor: Callable[[StepView], Mapping] | None = None
    predictor_shapes: Mapping = field(default_factory=dict)
    confirms: Mapping = field(default_factory=dict)
    checks: Mapping = field(default_factory=dict)
    build_monitor: Callable[[StepView], Mapping] | None = None
    serialize_monitor: Callable[[Mapping], object] | None = None
    hooks: Mapping = field(default_factory=dict)

    def __post_init__(self):
        for name in ("build_displacement", "build_geometry", "build_mesh_velocity", "build_mesh_current",
                     "solve_solid", "solve_fluid"):
            if not callable(getattr(self, name)):
                raise TypeError(f"StepPlan.{name} must be callable.")
        for name in ("build_interface", "build_predictor", "build_monitor", "serialize_monitor"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"StepPlan.{name} must be callable or None.")
        interface_shapes = MappingProxyType(
            {str(key): _shape(value, f"interface_shapes[{key!r}]")
             for key, value in dict(self.interface_shapes).items()})
        predictor_shapes = MappingProxyType(
            {str(key): _shape(value, f"predictor_shapes[{key!r}]")
             for key, value in dict(self.predictor_shapes).items()})
        confirms = MappingProxyType({str(key): value for key, value in dict(self.confirms).items()})
        checks = dict(self.checks)
        hooks = MappingProxyType({str(key): value for key, value in dict(self.hooks).items()})
        for name, value in confirms.items():
            if name not in _CONFIRM_STAGES:
                raise ValueError(f"Unknown confirm stage {name!r}; allowed {list(_CONFIRM_STAGES)}.")
            if not callable(value):
                raise TypeError(f"confirms[{name!r}] must be callable.")
        for name, value in checks.items():
            if not callable(value):
                raise TypeError(f"checks[{name!r}] must be callable.")
        for name in hooks:
            if name not in _HOOKS:
                raise ValueError(f"Unknown lifecycle hook {name!r}; allowed {list(_HOOKS)}.")
        if self.build_interface is not None and not interface_shapes:
            raise ValueError("interface_shapes is required when build_interface is provided.")
        object.__setattr__(self, "interface_shapes", interface_shapes)
        object.__setattr__(self, "predictor_shapes", predictor_shapes)
        object.__setattr__(self, "confirms", confirms)
        object.__setattr__(self, "checks", MappingProxyType(checks))
        object.__setattr__(self, "hooks", hooks)


class StepTransaction:
    """One whole-field step attempt with a private workspace and terminal states.

    中文：一次全场单步尝试，拥有私有工作区与终态（open/prepared/committed/aborted）。
    构造函数从 base 复制三个场的私有 ``BDFHistory`` 与 context，并校验 plan 与已接受接口
    兼容、三个场历史同步；任意时刻最多只有一个活动写尝试。
    """

    def __init__(self, coordinator, base, plan):
        if not isinstance(base, AcceptedState):
            raise TypeError("base must be an AcceptedState.")
        if not isinstance(plan, StepPlan):
            raise TypeError("plan must be a StepPlan.")
        self._coordinator = coordinator
        self._base = base
        self._plan = plan
        self._attempt_id = coordinator._next_attempt_id()
        self._base_epoch = base.epoch
        self._dt = base.dt
        self._order = base.order
        self._target_time = base.time + base.dt
        self._status = "open"
        self._validated = False
        self._prepared = None
        self._live = None
        self._revision = None
        self._revision_counter = itertools.count(1)
        self._candidates = {}
        self._validation = MappingProxyType({})
        self._monitor_row = None
        self._predictor = MappingProxyType({})
        self._output_payload = None
        self._output_epoch = None
        self._failure = None
        self._base_shapes = dict(base.shapes)
        self._base_generations = dict(base.generations)
        self._base_layers = base._layer_counts()
        # 中文：为每个场建立与根状态不共享存储的私有历史副本，失败回滚不会污染根。
        self._histories = {field: _copy_history(base._history(field)) for field in _FIELDS}
        self._contexts = {field: history.context() for field, history in self._histories.items()}
        self._require_plan_compatible()
        self._require_synchronized()

    # ------------------------------------------------------------- properties
    @property
    def status(self):
        return self._status

    @property
    def attempt_id(self):
        return self._attempt_id

    @property
    def base_epoch(self):
        return self._base_epoch

    @property
    def base(self):
        return self._base

    @property
    def target_time(self):
        return self._target_time

    @property
    def dt(self):
        return self._dt

    @property
    def order(self):
        return self._order

    @property
    def mesh_context(self):
        return self._contexts["mesh"]

    @property
    def solid_context(self):
        return self._contexts["solid"]

    @property
    def fluid_context(self):
        return self._contexts["fluid"]

    @property
    def failure(self):
        return self._failure

    @property
    def validation(self):
        return MappingProxyType(dict(self._validation))

    @property
    def output_payload(self):
        return self._output_payload

    @property
    def output_status(self):
        if self._output_payload is None:
            return "none"
        return self._coordinator.output_state(self._output_epoch)

    # ---------------------------------------------------------------- guarding
    def _require(self, *statuses):
        if self._status not in statuses:
            raise TransactionError(
                f"This attempt is {self._status!r}; only {list(statuses)} operations are allowed.")

    def _require_open(self):
        self._require("open")

    def _require_plan_compatible(self):
        accepted = dict(self._base.interface)
        declared = dict(self._plan.interface_shapes)
        if accepted and self._plan.build_interface is None:
            raise ValueError("The plan drops the accepted interface fields; a fixed M3 configuration "
                             "must keep every accepted interface array.")
        if accepted and set(accepted) != set(declared):
            raise ValueError(f"The plan interface fields {sorted(declared)} do not match the accepted "
                             f"fields {sorted(accepted)}.")
        for key, shape in declared.items():
            if key in accepted and tuple(accepted[key].shape) != tuple(shape):
                raise ValueError(f"interface_shapes[{key!r}]={tuple(shape)} does not match the accepted "
                                 f"shape {tuple(accepted[key].shape)}.")

    def _require_synchronized(self):
        # 中文：校验三个场历史的时间、层数、代次在 begin 期间保持一致，防止半更新状态。
        times = {field: self._histories[field].time for field in _FIELDS}
        if len(set(times.values())) != 1:
            raise ValueError(f"The three field histories are not synchronized: {times}.")
        generations = {field: _probe_generation(self._histories[field]) for field in _FIELDS}
        layers = {field: len(self._histories[field].states) for field in _FIELDS}
        if len(set(layers.values())) != 1:
            raise ValueError(f"The three field histories carry different layer counts: {layers}.")
        for field in _FIELDS:
            expected = (self._base_generations[field], self._base_layers[field])
            actual = (generations[field], layers[field])
            if actual != expected:
                raise ValueError(f"Field {field!r} history changed during begin: {actual} != {expected}.")
        self._base_generations = generations

    def check_context(self, context, field):
        """Reject foreign or stale BDF contexts before any live write.

        A context produced by another attempt, another field or another
        transaction is rejected even when generation and time look identical.

        中文：在任何活动写入之前拒绝外来或过期 BDF context。即便 generation 与时间看起来
        相同，只要 context 来自其它尝试、其它场或其它事务（owner 不同），或 generation 过期、
        时间/阶数不匹配，都会抛错。返回 ``True`` 表示校验通过。
        """
        if field not in _FIELDS:
            raise ValueError(f"Unknown field {field!r}; expected one of {_FIELDS}.")
        self._require("open")
        if not isinstance(context, BDFContext):
            raise TypeError("context must be a BDFContext.")
        history = self._histories[field]
        if context.owner is not history.context().owner:
            raise ValueError(f"The context does not belong to this attempt's {field!r} history owner.")
        if context.generation != _probe_generation(history):
            raise ValueError(f"The context generation {context.generation} is stale for field {field!r}.")
        if not np.isclose(context.time, history.time + self._dt, rtol=0.0, atol=1e-14):
            raise ValueError(f"The context time {context.time!r} does not match field {field!r} target time.")
        if context.order != self._contexts[field].order:
            raise ValueError(f"The context order {context.order} does not match field {field!r}.")
        return True

    # ------------------------------------------------------------ view sharing
    def _view(self, stage):
        candidates = self._candidates

        def show(name):
            value = candidates.get(name)
            return None if value is None else _expose_array(value)

        interface = candidates.get("interface")
        exposed = MappingProxyType({}) if interface is None else _expose_mapping(interface)
        return StepView(
            stage=stage, attempt_id=self._attempt_id, base_epoch=self._base_epoch,
            target_step=self._base.step + 1, target_time=self._target_time, dt=self._dt,
            order=self._order, base=self._base, mesh_context=self._contexts["mesh"],
            solid_context=self._contexts["solid"], fluid_context=self._contexts["fluid"],
            revision=self._revision, displacement=show("mesh_displacement"),
            geometry=candidates.get("geometry"), mesh_current=show("mesh_current"),
            mesh_velocity=show("mesh_velocity"), solid=show("solid"), velocity=show("velocity"),
            pressure=show("pressure"), interface=exposed)

    # ------------------------------------------------------------- stage tools
    def _fail(self, stage, field, reason, diagnostics=None):
        if self._status in ("committed", "aborted"):
            # 中文：已提交/已作废的尝试是终态，失败不可被覆盖，直接返回既有失败记录。
            return self._failure
        record = FailureRecord(
            attempt_id=self._attempt_id, base_epoch=self._base_epoch,
            target_step=self._base.step + 1, target_time=self._target_time,
            dt=self._dt, order=self._order, stage=str(stage), field=None if field is None else str(field),
            reason=str(reason), diagnostics=_freeze_diagnostics(diagnostics))
        self._failure = record
        self._status = "aborted"
        # 中文：作废为终态——清空所有私有准备物与候选，确保无法再从本尝试发布。
        self._prepared = None
        self._validated = False
        self._candidates = {}
        self._revision = None
        self._monitor_row = None
        self._coordinator._record_failure(record)
        self._coordinator._release(self)  # 中文：释放写权限，允许用新尝试重来。
        return record

    def _reject(self, stage, field, message, diagnostics=None):
        self._fail(stage, field, message, diagnostics)
        raise ValueError(message)

    def _call(self, function, stage, field):
        view = self._view(stage)
        try:
            return function(view)
        except Exception as error:
            self._fail(stage, field, f"{type(error).__name__}: {error}", getattr(error, "diagnostics", None))
            raise

    def _confirm(self, stage):
        confirm = self._plan.confirms.get(stage)
        if confirm is not None:
            self._call(confirm, stage, stage)

    def _own_candidate(self, values, shape, stage, field):
        try:
            return _own_array(values, shape, f"{field} candidate")
        except (TypeError, ValueError) as error:
            self._fail(stage, field, str(error))
            raise

    def _drop(self, *names):
        for name in names:
            self._candidates.pop(name, None)
        self._validated = False

    def _hook(self, name, field, history):
        hook = self._plan.hooks.get(name)
        if hook is None:
            return
        info = {"hook": name, "field": field}
        if history is not None:
            time, generation, states = history.snapshot()
            info.update({"time": time, "generation": generation, "layers": len(states)})
        try:
            hook(MappingProxyType(info))
        except Exception as error:
            self._fail("prepare", field or "freeze",
                       f"lifecycle hook {name!r} raised {type(error).__name__}: {error}",
                       getattr(error, "diagnostics", None))
            raise

    # ---------------------------------------------------------------- stages
    def evaluate_geometry(self, displacement=None):
        """Stage 1: candidate mesh displacement, geometry, coordinates and velocity.

        中文：第 1 阶段：生成候选网格位移、几何、当前坐标与网格速度。可选传入
        ``displacement`` 覆盖 ``plan.build_displacement``；本阶段还会丢弃旧的
        solid/fluid/interface 候选，并刷新几何版本身份 ``revision``（attempt_id 与计数器组成）。
        """
        self._require_open()
        if displacement is None:
            displacement = self._call(self._plan.build_displacement, "geometry", "mesh displacement")
        candidate = self._own_candidate(displacement, self._base_shapes["mesh"],
                                        "geometry", "mesh displacement")
        self._candidates["mesh_displacement"] = candidate
        self._drop("solid", "velocity", "pressure", "interface")
        self._revision = (self._attempt_id, next(self._revision_counter))
        geometry = self._call(self._plan.build_geometry, "geometry", "mesh geometry")
        self._candidates["geometry"] = geometry
        mesh_current = self._own_candidate(self._call(self._plan.build_mesh_current, "geometry",
                                                      "mesh coordinates"),
                                           self._base_shapes["mesh"], "geometry", "mesh coordinates")
        self._candidates["mesh_current"] = mesh_current
        mesh_velocity = self._own_candidate(self._call(self._plan.build_mesh_velocity, "geometry",
                                                       "mesh velocity"),
                                            self._base_shapes["mesh"], "geometry", "mesh velocity")
        self._candidates["mesh_velocity"] = mesh_velocity
        self._confirm("geometry")
        return self._revision

    def evaluate_solid(self):
        """Stage 2: solid candidate ``(2, n_s, 2)`` from the private solid context.

        中文：第 2 阶段：用私有固体 context 求解得到形状 ``(2, n_s, 2)`` 的固体候选，
        并作废旧 interface 候选。返回只读副本。
        """
        self._require_open()
        if "geometry" not in self._candidates:
            raise TransactionError("evaluate_geometry must complete before evaluate_solid.")
        candidate = self._own_candidate(self._call(self._plan.solve_solid, "solid", "solid"),
                                        self._base_shapes["solid"], "solid", "solid")
        self._candidates["solid"] = candidate
        self._drop("interface")
        self._confirm("solid")
        return _expose_array(candidate)

    def evaluate_fluid(self):
        """Stage 3: fluid candidate ``(u, p)``; ``u``/``p`` are copied independently.

        中文：第 3 阶段：求解流体得到 ``(u, p)`` 候选；``solve_fluid`` 必须返回二元组，u 与 p
        分别按声明形状深复制为只读数组，并作废旧 interface 候选。返回二者的只读副本。
        """
        self._require_open()
        if "geometry" not in self._candidates:
            raise TransactionError("evaluate_geometry must complete before evaluate_fluid.")
        result = self._call(self._plan.solve_fluid, "fluid", "fluid")
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            self._reject("fluid", "fluid", "solve_fluid must return a (velocity, pressure) pair.")
        velocity = self._own_candidate(result[0], self._base_shapes["fluid"], "fluid", "fluid velocity")
        pressure = self._own_candidate(result[1], self._base_shapes["pressure"], "fluid", "fluid pressure")
        self._candidates["velocity"] = velocity
        self._candidates["pressure"] = pressure
        self._drop("interface")
        self._confirm("fluid")
        return _expose_array(velocity), _expose_array(pressure)

    def evaluate_interface(self):
        """Stage 4: interface motion/load arrays with declared names and shapes.

        中文：第 4 阶段：构造接口运动/载荷数组，字段名与形状必须与 ``plan.interface_shapes``
        完全一致（多、少、错形都报错）；若无 ``build_interface`` 则记为空映射。要求几何、固体、
        流体候选均已就绪。
        """
        self._require_open()
        if self._plan.build_interface is None:
            self._candidates["interface"] = MappingProxyType({})
            return MappingProxyType({})
        if "geometry" not in self._candidates or "solid" not in self._candidates or \
                "velocity" not in self._candidates:
            raise TransactionError("evaluate_interface requires the geometry, solid and fluid candidates.")
        result = self._call(self._plan.build_interface, "interface", "interface")
        if not isinstance(result, Mapping):
            self._reject("interface", "interface", "build_interface must return a mapping of arrays.")
        declared = dict(self._plan.interface_shapes)
        if set(result) != set(declared):
            missing = sorted(set(declared) - set(result))
            unexpected = sorted(set(result) - set(declared))
            self._reject("interface", "interface",
                         f"interface arrays do not match the declared fields; missing={missing}, "
                         f"unexpected={unexpected}.")
        stored = {}
        for key, shape in declared.items():
            try:
                array = np.array(np.asarray(result[key], dtype=float), copy=True)
            except (TypeError, ValueError) as error:
                self._reject("interface", key, f"interface[{key!r}] is not a real numeric array: {error}")
            if array.shape != tuple(shape):
                self._reject("interface", key, f"interface[{key!r}] must have shape {tuple(shape)}, "
                                               f"got {array.shape}.")
            array.setflags(write=False)
            stored[key] = array
        self._candidates["interface"] = MappingProxyType(stored)
        self._confirm("interface")
        return self._expose_interface()

    def _expose_interface(self):
        interface = self._candidates.get("interface")
        return MappingProxyType({}) if interface is None else _expose_mapping(interface)

    def evaluate(self, displacement=None):
        """Run the four evaluation stages in order with the plan's callables.

        中文：依次运行四个评估阶段（几何、固体、流体，以及存在时的接口），返回 ``self``。
        """
        self.evaluate_geometry(displacement)
        self.evaluate_solid()
        self.evaluate_fluid()
        if self._plan.build_interface is not None:
            self.evaluate_interface()
        return self

    # ------------------------------------------------------------- validation
    def _missing_candidates(self):
        missing = []
        for name, shape in self._expected_candidate_shapes().items():
            if name not in self._candidates:
                missing.append(name)
        return missing

    def _expected_candidate_shapes(self):
        expected = {
            "mesh_displacement": self._base_shapes["mesh"],
            "geometry": None,
            "mesh_current": self._base_shapes["mesh"],
            "mesh_velocity": self._base_shapes["mesh"],
            "solid": self._base_shapes["solid"],
            "velocity": self._base_shapes["fluid"],
            "pressure": self._base_shapes["pressure"],
        }
        if self._plan.build_interface is not None:
            expected["interface"] = None
        return expected

    def validate(self):
        """Final gate: re-check every candidate and run the physics checks.

        Rejects non-finite or wrongly shaped values, runs ``plan.checks`` in
        order, then freezes the successful monitoring row and the predictor
        values for the next root.  A failure discards the whole candidate.

        中文：最终校验门。先检查候选是否齐全，再拒绝非有限值，然后按顺序运行 ``plan.checks``，
        最后冻结本步监测行与给下一个根的预报量。任一失败都会丢弃整组候选并作废本次尝试。
        """
        self._require_open()
        missing = self._missing_candidates()
        if missing:
            raise TransactionError(f"evaluate must complete before validate; missing {missing}.")
        for name in ("mesh_displacement", "mesh_current", "mesh_velocity", "solid", "velocity", "pressure"):
            array = self._candidates[name]
            if not np.all(np.isfinite(array)):
                self._reject("validate", name, f"The {name} candidate is not finite.")
        if self._plan.build_interface is not None:
            for key, array in self._candidates["interface"].items():
                if not np.all(np.isfinite(array)):
                    self._reject("validate", key, f"The interface candidate {key!r} is not finite.")
        report = {}
        for name, check in self._plan.checks.items():
            report[name] = self._call(check, "validate", name)
        if self._plan.build_monitor is None:
            row = {}
        else:
            row = self._call(self._plan.build_monitor, "validate", "monitor")
        try:
            self._monitor_row = _freeze_row(row, "monitor")
        except ValueError as error:
            self._fail("validate", "monitor", str(error))
            raise
        if self._plan.build_predictor is None:
            predictor = {
                "mesh_displacement": self._candidates["mesh_displacement"],
                "solid": self._candidates["solid"],
                "velocity": self._candidates["velocity"],
                "pressure": self._candidates["pressure"],
            }
        else:
            predictor = self._call(self._plan.build_predictor, "validate", "predictor")
        self._predictor = self._freeze_predictor(predictor)
        self._validation = MappingProxyType(report)
        self._validated = True
        return self._validation

    def _freeze_predictor(self, values):
        if not isinstance(values, Mapping):
            self._reject("validate", "predictor", "build_predictor must return a mapping of arrays.")
        declared = dict(self._plan.predictor_shapes)
        if declared and set(values) != set(declared):
            self._reject("validate", "predictor",
                         f"predictor fields {sorted(values)} do not match the declared "
                         f"fields {sorted(declared)}.")
        owned = {}
        for key, value in values.items():
            key = str(key)
            shape = declared.get(key)
            if shape is None:
                array = np.asarray(value, dtype=float)
                if array.ndim < 1:
                    self._reject("validate", "predictor", f"predictor[{key!r}] must be an array.")
                shape = array.shape
            try:
                owned[key] = _own_array(value, shape, f"predictor[{key!r}]")
            except (TypeError, ValueError) as error:
                self._fail("validate", "predictor", str(error))
                raise
        return MappingProxyType(owned)

    # ---------------------------------------------------------------- prepare
    def prepare(self):
        """Accept on the three private histories, then freeze the whole next root.

        Every fallible operation happens here; the new root exists only inside
        this attempt until ``publish`` swaps the live reference once.

        中文：在三个私有历史上执行 accept，然后冻结整个下一个根状态。所有可能失败的操作都
        集中在这里；新根在 ``publish`` 替换活动引用之前只存在于本尝试内部。这里还会校验每个
        私有历史恰好前进一代且时间等于目标时间，并生成提交给 ``publish`` 的输出 payload。
        """
        self._require_open()
        if not self._validated:
            raise TransactionError("validate() must complete before prepare().")
        for field in ("solid", "fluid", "mesh"):
            candidate = self._candidates[{"solid": "solid", "fluid": "velocity", "mesh": "mesh_displacement"}[field]]
            history = self._histories[field]
            context = self._contexts[field]
            try:
                history.accept(context, candidate)
            except Exception as error:
                self._fail("prepare", field, f"private accept failed: {type(error).__name__}: {error}")
                raise
            self._hook(f"after_{field}_accept", field, history)
        self._hook("before_freeze", None, None)
        expected_generation = {field: self._base_generations[field] + 1 for field in _FIELDS}
        # BDFHistory always stores two layers after accept; the accepted root keeps
        # one layer for BDF1 and up to two layers for BDF2 (startup then mature).
        # 中文：私有历史 accept 后总有两层；已接受根在 BDF1 下只留一层，在 BDF2 下按启动
        # （一层）到成熟（两层）保留层数。预期每个场的代次都恰好自增 1。
        private_layers = min(2, self._base_layers["solid"] + 1)
        root_layers = 1 if self._order == 1 else private_layers
        for field in _FIELDS:
            history = self._histories[field]
            time, generation, states = history.snapshot()
            if not np.isclose(time, self._target_time, rtol=0.0, atol=1e-14):
                self._reject("prepare", field, f"private {field} history time {time!r} does not match "
                                               f"the target {self._target_time!r}.")
            if generation != expected_generation[field] or len(states) != private_layers:
                self._reject("prepare", field, f"private {field} history did not advance exactly one "
                                               f"step (generation {generation}, layers {len(states)}).")
        payload = None
        if self._plan.serialize_monitor is not None:
            payload = self._call(lambda _view: self._plan.serialize_monitor(_copy_row(self._monitor_row)),
                                 "prepare", "serialize")
        try:
            prepared = AcceptedState(
                time=self._target_time, dt=self._dt, order=self._order, step=self._base.step + 1,
                epoch=self._base.epoch + 1,
                solid_states=self._histories["solid"].states[:root_layers],
                fluid_states=self._histories["fluid"].states[:root_layers],
                mesh_states=self._histories["mesh"].states[:root_layers],
                solid_generation=expected_generation["solid"],
                fluid_generation=expected_generation["fluid"],
                mesh_generation=expected_generation["mesh"],
                pressure=self._candidates["pressure"],
                mesh_current=self._candidates["mesh_current"],
                mesh_velocity=self._candidates["mesh_velocity"],
                interface=self._candidates.get("interface"),
                predictor=self._predictor,
                monitoring=self._base.monitoring + (self._monitor_row,))
        except Exception as error:
            self._fail("prepare", "freeze", f"freezing the next root failed: {type(error).__name__}: {error}")
            raise
        self._prepared = prepared
        self._output_payload = payload
        self._output_epoch = prepared.epoch
        self._status = "prepared"
        return None

    # ---------------------------------------------------------------- publish
    def publish(self):
        """Replace the live accepted root exactly once and mark the attempt committed.

        中文：恰好一次地替换活动已接受根并标记本尝试为 committed。若活动根已不再是本事务的
        base，则抛出 ``StepConflict`` 并作废本次尝试；提交成功后清空私有工作区。
        """
        self._require("prepared")
        try:
            live = self._coordinator._commit(self)
        except StepConflict as error:
            self._fail("publish", None, f"publish conflict: {error}")
            raise
        self._status = "committed"
        # 中文：提交成功后释放私有工作区，不再保留候选与已准备根。
        self._prepared = None
        self._validated = False
        self._candidates = {}
        self._revision = None
        self._live = live
        return live

    def abort(self, *, stage="abort", field=None, reason="aborted by caller", diagnostics=None):
        """Explicit rollback. Committed attempts cannot be aborted.

        中文：显式回滚。已 committed 的尝试不能作废（物理步已成立，抛 ``TransactionError``）；
        已 aborted 的返回 ``False``；否则记录失败并返回 ``True``。
        """
        if self._status == "committed":
            raise TransactionError("A committed attempt cannot be aborted; the physical step stands.")
        if self._status == "aborted":
            return False
        self._fail(stage, field, reason, diagnostics)
        return True

    # ----------------------------------------------------------------- output
    def write_output(self, writer):
        """Idempotent post-publish write; failure is reported as ``output_pending``.

        中文：发布后的幂等写出。无 payload 时返回 ``False``；失败会被记为 ``output_pending``，
        结果与协调器的输出账本一致。
        """
        self._require("committed")
        if self._output_payload is None:
            return False
        return self._coordinator.write_output(self._output_epoch, writer, payload=self._output_payload)


def _copy_history(history):
    time, generation, states = history.snapshot()
    return _rebuild_history(states, history.dt, history.order, time, generation)


def _state_from_snapshot(snapshot):
    return AcceptedState(
        time=snapshot.time, dt=snapshot.dt, order=snapshot.order, step=snapshot.step,
        epoch=snapshot.epoch, solid_states=snapshot.solid_states, fluid_states=snapshot.fluid_states,
        mesh_states=snapshot.mesh_states, solid_generation=snapshot.solid_generation,
        fluid_generation=snapshot.fluid_generation, mesh_generation=snapshot.mesh_generation,
        pressure=snapshot.pressure, mesh_current=snapshot.mesh_current,
        mesh_velocity=snapshot.mesh_velocity, interface=snapshot.interface,
        predictor=snapshot.predictor, monitoring=snapshot.monitoring)


class StateCoordinator:
    """Single-writer holder of the live accepted root.

    ``begin`` grants the one write permission, ``publish``/``load``/``restore``
    and the output ledger are serialized on one lock so readers always observe
    one complete root.  Failure records and output records live next to the
    root, never inside accepted physical state.

    中文：活动已接受根状态的单写者持有者。``begin`` 授予唯一写权限；
    ``publish``/``load``/``restore`` 与输出账本在同一把锁上串行化，读者始终看到完整一致的根。
    失败记录与输出记录存放在根之外（作为审计账本），绝不进入已接受的物理状态。
    """

    def __init__(self, state, *, config_fingerprint=""):
        if not isinstance(state, AcceptedState):
            raise TypeError("state must be an AcceptedState.")
        self._owner = object()
        self._state = state
        self._config_fingerprint = str(config_fingerprint)
        self._lock = threading.RLock()
        self._active = None
        self._failures = []
        self._accepted_steps = {state.epoch: state.step}
        self._outputs = {}
        self._pending = None

    # ------------------------------------------------------------------ reads
    @property
    def state(self):
        with self._lock:
            return self._state

    @property
    def config_fingerprint(self):
        return self._config_fingerprint

    @property
    def active(self):
        with self._lock:
            return self._active

    @property
    def failures(self):
        with self._lock:
            return tuple(self._failures)

    @property
    def output_records(self):
        with self._lock:
            return tuple(self._outputs.values())

    @property
    def pending_output(self):
        with self._lock:
            return self._pending

    # ------------------------------------------------------------- lifecycle
    def begin(self, *, plan, dt=None, order=None):
        """Start the single open attempt from the live accepted root.

        中文：从活动已接受根开启唯一一个 open 尝试。若已有活动尝试则拒绝；可选的 dt/order
        必须与根上固定的值一致（本设计固定步长且固定阶数），否则报错。
        """
        if not isinstance(plan, StepPlan):
            raise TypeError("plan must be a StepPlan.")
        with self._lock:
            if self._active is not None:
                raise TransactionError("Another attempt is already open; a coordinator has one writer.")
            base = self._state
            if dt is not None:
                dt_value = _as_float(dt, "dt", positive=True)
                if dt_value != base.dt:
                    raise ValueError(f"The step size is fixed at dt={base.dt!r}; refusing dt={dt_value!r}.")
            if order is not None:
                order_value = _as_int(order, "order")
                if order_value not in (1, 2) or order_value != base.order:
                    raise ValueError(f"The BDF order is fixed at order={base.order}; refusing order={order_value!r}.")
            transaction = StepTransaction(self, base, plan)
            self._active = transaction
            return transaction

    def _next_attempt_id(self):
        return next(_ATTEMPT_SERIAL)

    def _record_failure(self, record):
        with self._lock:
            self._failures.append(record)

    def _release(self, transaction):
        with self._lock:
            if self._active is transaction:
                self._active = None

    def _commit(self, transaction):
        """Swap the live root under lock after re-checking the write permission and base.

        中文：持锁提交：再次确认本尝试仍持有写权限，且活动根仍是它的 base（身份与 epoch 都
        一致），否则抛 ``StepConflict``（防 ABA/陈旧发布）；随后一次性替换活动根并释放写权限。
        """
        with self._lock:
            if self._active is not transaction:
                raise StepConflict("This attempt no longer holds the coordinator write permission.")
            live = self._state
            # 中文：身份与 epoch 双重校验；不一致说明期间根已变（防 ABA/陈旧发布）。
            if live is not transaction._base or live.epoch != transaction.base_epoch:
                raise StepConflict(
                    f"The live accepted epoch {live.epoch} no longer matches base epoch "
                    f"{transaction.base_epoch}.")
            prepared = transaction._prepared
            if prepared is None:
                raise TransactionError("Nothing was prepared for publication.")
            self._state = prepared
            self._active = None
            self._accepted_steps[prepared.epoch] = prepared.step
            return prepared

    def load(self, state):
        """Adopt an externally recovered root (same fixed dt/order/shapes).

        中文：接纳一个外部恢复的根状态，要求 dt、order 与各场形状都与当前协调器一致，
        并登记其 epoch/step。返回被接纳的根。
        """
        if not isinstance(state, AcceptedState):
            raise TypeError("load() accepts an AcceptedState.")
        with self._lock:
            live = self._state
            if state.dt != live.dt or state.order != live.order:
                raise ValueError(f"The loaded root dt/order {state.dt!r}/{state.order!r} do not match "
                                 f"the coordinator {live.dt!r}/{live.order!r}.")
            if dict(state.shapes) != dict(live.shapes):
                raise ValueError("The loaded root shapes do not match the coordinator field shapes.")
            self._state = state
            self._accepted_steps.setdefault(state.epoch, state.step)
            return state

    def snapshot(self):
        """Capture the current live root; only this coordinator may restore it.

        中文：捕获当前活动根生成 ``StateSnapshot``，并写入本协调器的 owner 标识、根身份 token
        与配置指纹；只有同一协调器才能用其 ``restore`` 恢复。
        """
        with self._lock:
            state = self._state
            return StateSnapshot(
                coordinator_owner=self._owner, state_token=state._identity,
                config_fingerprint=self._config_fingerprint, epoch=state.epoch, step=state.step,
                time=state.time, dt=state.dt, order=state.order, solid_states=state.solid_history,
                fluid_states=state.fluid_history, mesh_states=state.mesh_history,
                solid_generation=state.generations["solid"], fluid_generation=state.generations["fluid"],
                mesh_generation=state.generations["mesh"], pressure=state.pressure,
                mesh_current=state.mesh_current, mesh_velocity=state.mesh_velocity,
                interface=state.interface, predictor=state.predictor, monitoring=state.monitoring)

    def restore(self, snapshot):
        """Replace the live root with a fresh copy of ``snapshot`` (recovery replay).

        Foreign owner, stale root/epoch, wrong config, mismatched dt/order/layers
        or malformed arrays are rejected before any live write.  The restored
        root receives fresh history identities, so old contexts are rejected.

        中文：用 ``snapshot`` 的全新副本替换活动根（恢复重放）。任何活动写入之前先拒绝外来
        owner、陈旧根/epoch、配置不符、dt/order/层数不匹配或畸形数组。恢复出的根拥有全新的
        历史身份，因此此前取出的旧 context 会被拒绝，避免 ABA 重放。
        """
        if not isinstance(snapshot, StateSnapshot):
            raise TypeError("restore() accepts only a StateSnapshot produced by snapshot(); "
                            "arbitrary BDF tuples are not a recovery protocol.")
        with self._lock:
            live = self._state
            if snapshot.coordinator_owner is not self._owner:
                raise ValueError("Foreign snapshot: it was produced by another coordinator.")
            if snapshot.state_token is not live._identity:
                # 中文：快照必须对应当前活动根身份；陈旧快照（根已前进或被替换）直接拒绝。
                raise ValueError("Stale snapshot: the live root advanced or was replaced after it was taken.")
            if snapshot.config_fingerprint != self._config_fingerprint:
                raise ValueError("Snapshot config fingerprint does not match this coordinator.")
            if not np.isfinite(snapshot.dt) or snapshot.dt <= 0:
                raise ValueError(f"Snapshot dt must be finite and positive, got {snapshot.dt!r}.")
            if snapshot.dt != live.dt:
                raise ValueError(f"Snapshot dt {snapshot.dt!r} does not match the fixed dt {live.dt!r}.")
            if snapshot.order != live.order:
                raise ValueError(f"Snapshot order {snapshot.order!r} does not match the fixed "
                                 f"order {live.order!r}.")
            if snapshot.epoch != live.epoch or snapshot.step != live.step:
                raise ValueError(f"Stale epoch: snapshot epoch/step {snapshot.epoch}/{snapshot.step} "
                                 f"does not match the live root {live.epoch}/{live.step}.")
            if snapshot.time != live.time:
                raise ValueError(f"Snapshot time {snapshot.time!r} does not match the live "
                                 f"time {live.time!r}.")
            layers = {len(snapshot.solid_states), len(snapshot.fluid_states), len(snapshot.mesh_states)}
            if len(layers) != 1 or not 1 <= layers.pop() <= 2:
                raise ValueError("Snapshot history layer counts must be 1 or 2 for every field.")
            rebuilt = _state_from_snapshot(snapshot)
            self._state = rebuilt
            return rebuilt

    # ----------------------------------------------------------------- output
    def output_state(self, epoch):
        """``written``/``pending``/``unwritten`` for an accepted epoch.

        中文：返回某个已接受 epoch 的输出状态：已成功写出为 ``written``，等待重放为
        ``pending``，尚未写为 ``unwritten``；未被本协调器接受的 epoch 会报错。
        """
        epoch = _as_int(epoch, "epoch", minimum=1)
        with self._lock:
            if epoch in self._outputs:
                return "written"
            if self._pending is not None and self._pending.epoch == epoch:
                return "pending"
            if epoch > self._state.epoch or epoch not in self._accepted_steps:
                raise ValueError(f"Epoch {epoch} has not been accepted by this coordinator.")
            return "unwritten"

    def write_output(self, epoch, writer, *, payload=None):
        """Write one accepted epoch's output idempotently.

        Returns ``True`` on a completed write, ``False`` when the writer raised
        (the accepted epoch is then kept with ``output_pending``) or when the
        epoch already has its single success record.  The physical root is
        never rolled back and nothing is re-solved.

        中文：幂等地写出某个已接受 epoch 的输出。写入成功返回 ``True``；writer 抛错（该已接受
        epoch 随即记为 ``output_pending``）或该 epoch 已有唯一成功记录时返回 ``False``。
        物理根绝不回滚，也不会重新求解。
        """
        epoch = _as_int(epoch, "epoch", minimum=1)
        if not callable(writer):
            raise TypeError("writer must be callable.")
        with self._lock:
            if epoch not in self._accepted_steps or epoch > self._state.epoch:
                raise ValueError(f"Epoch {epoch} has not been accepted by this coordinator.")
            if epoch in self._outputs:
                # 中文：该 epoch 已有唯一成功记录，幂等返回 False，不重复写出。
                return False
            if payload is None:
                if self._pending is None or self._pending.epoch != epoch:
                    raise ValueError(f"No payload is available for epoch {epoch}.")
                payload = self._pending.payload
            step = self._accepted_steps[epoch]
            try:
                writer(payload)
            except Exception as error:
                reason = f"{type(error).__name__}: {error}"
                # 中文：仅标记为待重放并保留已接受根，绝不回滚物理步。
                self._pending = PendingOutput(epoch, step, payload, reason)
                return False
            self._outputs[epoch] = OutputRecord(epoch, step, payload)
            if self._pending is not None and self._pending.epoch == epoch:
                self._pending = None
            return True

    def flush_output(self, writer):
        """Retry the pending output write; no-op when nothing is pending.

        中文：重试待重放的输出写出；没有待重放项时直接返回 ``False``。持锁读取待写项后在锁外
        调用 ``write_output``，避免写 IO 时长时间持锁。
        """
        with self._lock:
            pending = self._pending
            if pending is None:
                return False
        return self.write_output(pending.epoch, writer, payload=pending.payload)

"""Interface motion and dual load mapping through the fixed reference ``H``.

Mapping accepts plain state arrays only: it never reads, advances or accepts a
time history, and it never modifies its inputs.

本模块通过固定参照界面的耦合矩阵 ``H`` 在固体 P1 迹与流体 P2 迹之间映射运动
与对偶载荷。关键约定：只接受当前状态数组，不读写、不推进时间历史，也不修改
输入；所有向量按交错自由度 ``[x0, y0, x1, y1, ...]`` 排列，位移量单位为长度、
载荷单位为力，``H`` 本身无量纲。
"""

from dataclasses import dataclass

import numpy as np

from fsi_ice.interface.topology import InterfaceTopology


def _read_only(array):
    array.setflags(write=False)
    return array


def _finite_vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array.astype(float, copy=True)


@dataclass(frozen=True)
class InterfaceMotion:
    """Mapped interface motion for one candidate solid state.

    ``vertex_values`` gathers the solid interface P1 trace, ``p2_values`` is
    the fluid P2 trace ``H`` applied to the compact solid trace, and
    ``compact_values``/``solid_values`` keep the interleaved compact and full
    solid vectors for diagnostics.  All arrays are read-only.

    单个候选固体状态下映射得到的界面运动。``vertex_values`` 为收集到的固体界面
    P1 迹 ``(n_sG, 2)``，``p2_values`` 为对压缩固体迹施加 ``H`` 后的流体 P2 迹
    ``(n_fG, 2)``；``compact_values``/``solid_values`` 保留交错压缩向量
    ``(2*n_sG,)`` 与完整固体向量 ``(n_s_nodes, 2)`` 供诊断。位移类量单位为长度，
    速度类量为速度单位；所有数组只读。
    """

    topology_id: str
    solid_values: np.ndarray
    compact_values: np.ndarray
    vertex_values: np.ndarray
    p2_values: np.ndarray


def map_motion(topology, solid_values):
    """Map full solid displacement/velocity ``(n_s_nodes, 2)`` to the interface.

    The output ``vertex_values`` is the gathered solid P1 trace and
    ``p2_values`` is the fluid P2 trace; every fluid mid-edge value is the
    average of its two matched solid endpoints.  Histories are neither read
    nor accepted, and ``solid_values`` is copied before use.

    将完整固体位移/速度 ``(n_s_nodes, 2)`` 映射到界面。输入 ``solid_values`` 为
    长度或速度量，输出 ``vertex_values`` 为收集到的固体 P1 迹、``p2_values`` 为
    流体 P2 迹，每个流体棱中点值等于其两个匹配固体端点的平均值（对应 ``h`` 中
    的 0.5 权重）。不读取历史，使用前复制 ``solid_values``。
    """
    if not isinstance(topology, InterfaceTopology):
        raise TypeError("topology must be an InterfaceTopology.")
    values = np.asarray(solid_values, dtype=float)
    expected = (topology.n_solid_nodes, 2)
    if values.shape != expected:
        raise ValueError(f"solid_values must have shape {expected}, got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("solid_values must be finite.")
    values = np.array(values, dtype=float, copy=True)
    # 按 solid_dofs 抽取固体界面自由度，得到交错压缩迹（作为 H 的输入）。
    compact = np.array(values.ravel()[topology.solid_dofs], copy=True)
    vertex_values = np.array(values[topology.solid_nodes], copy=True)
    # p2_values = H @ compact：棱中点取两端点平均后的流体 P2 迹。
    p2_values = np.asarray(topology.H @ compact).reshape(topology.n_fluid_interface_nodes, 2)
    return InterfaceMotion(topology_id=topology.topology_id, solid_values=_read_only(values),
                           compact_values=_read_only(compact), vertex_values=_read_only(vertex_values),
                           p2_values=_read_only(np.array(p2_values, copy=True)))


def map_load(topology, fluid_loads):
    """Map integrated fluid dual nodal forces to compact solid interface forces.

    ``fluid_loads`` must be the already integrated dual nodal force vector on
    the fluid P2 interface trace, interleaved as ``[x0, y0, x1, y1, ...]`` and
    ordered by ``topology.fluid_p2_nodes``.  It is *not* a nodal traction or
    traction-coefficient vector: the measure, quadrature weights and thickness
    must already be included, otherwise the transferred load silently changes.
    The output is ``H.T @ fluid_loads`` on ``topology.solid_nodes`` in the
    same interleaved convention.

    将已积分的流体对偶节点力映射为压缩的固体界面力。``fluid_loads`` 必须是流体
    P2 界面迹上已积分的对偶节点力向量（单位力），按 ``[x0, y0, ...]`` 交错并按
    ``topology.fluid_p2_nodes`` 排序；它*不是*节点应力或应力系数向量，测度、
    积分权重与厚度必须已经乘入，否则传递的载荷会被静默改变。输出为
    ``H.T @ fluid_loads``，位于 ``topology.solid_nodes``，采用相同交错约定。
    """
    if not isinstance(topology, InterfaceTopology):
        raise TypeError("topology must be an InterfaceTopology.")
    loads = _finite_vector(fluid_loads, 2 * topology.n_fluid_interface_nodes, "fluid_loads")
    return _read_only(np.asarray(topology.H.T @ loads).astype(float, copy=True))


def scatter_solid_load(topology, compact):
    """Scatter a compact solid interface force into the full solid vector.

    Non-interface entries are zero, so the result can be added to the body
    force to form the ``external`` vector consumed by the solid residual.

    将压缩的固体界面力（单位力）散布到完整固体力向量中。非界面分量置零，因此
    结果可直接与体力相加，构成固体残差所需的 ``external`` 向量。
    """
    if not isinstance(topology, InterfaceTopology):
        raise TypeError("topology must be an InterfaceTopology.")
    values = _finite_vector(compact, 2 * topology.n_solid_interface_nodes, "compact")
    full = np.zeros(2 * topology.n_solid_nodes, dtype=float)
    # 直接写入界面自由度行；其余行保持为零，作为载荷向量的零填充。
    full[topology.solid_dofs] = values
    return _read_only(full)

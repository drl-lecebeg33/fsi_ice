"""Monolithic global residual, complete coupling Jacobian and matrix-free JVP.

Unknown ordering
----------------

``x = [u_f, p_f, d_s, v_s, d_m, g]``

==================  =========================================================
``u_f``             fluid P2 velocity ``(n_P2, 2)``
``p_f``             fluid P1 pressure ``(n_P1,)``
``d_s`` / ``v_s``   solid reference-configuration P1 displacement / velocity
                    ``(n_solid_nodes, 2)`` each
``d_m``             fluid reference-grid P1 displacement ``(n_f1, 2)``
``g``               fluid P2 interface-trace dual nodal force in the
                    :func:`~fsi_ice.interface.traction.assemble_interface_load`
                    order ``(n_fG, 2)``; the thickness is already included once
                    and the sign is fluid-on-solid
==================  =========================================================

Every two-dimensional nodal block uses the interleaved
``(node0-x, node0-y, node1-x, ...)`` order of
:func:`~fsi_ice.solid.linear.vector_dofs`.  No block of this module re-implements
single-field physics: the flow blocks call
:func:`~fsi_ice.fluid.ale.ale_navier_stokes_residual` on the candidate geometry
built from ``d_m``, the solid blocks call
:func:`~fsi_ice.solid.stvk.stvk_residual`/:func:`~fsi_ice.solid.stvk.assemble_mass`,
the mesh block reuses :attr:`~fsi_ice.ale.harmonic.HarmonicExtension.stiffness`
and the interface blocks call
:func:`~fsi_ice.interface.traction.assemble_interface_load` and
:func:`~fsi_ice.interface.traction.interface_load_jvp`.  No single-field
``solve`` and no single-field Newton is called; every evaluation is a pure
function of ``x`` and the frozen context.

Residual
--------

``R_u``/``R_p``      uneliminated ALE Navier-Stokes residual on the candidate
                     geometry; non-wet-wall known-velocity rows become
                     ``u_f - u_D`` and the wet fluid P2 trace rows become
                     ``G_f u_f - H G_s v_s``
``R_sd``             ``D_BDF d_s - v_s``
``R_sv``             ``M_s D_BDF v_s + f_int(d_s) - f_body - S_s H^T g``
``R_m``              ``K_h d_m`` on free mesh rows, ``d_m|Gamma = d_s|Gamma`` on
                     wet vertices (one-to-one reference matching, ``H_d = I``),
                     ``d_m - d_m,D`` on the other exterior rows
``R_g``              ``g - g_f_gamma(u_f, p_f, d_m, d_s)`` with ``g_f_gamma``
                     from the same candidate interface geometry

The BDF operator is ``D_BDF w = a_0 w + sum_{j>=1} a_j w_j`` with the stencil
coefficients of :mod:`fsi_ice.time.bdf`.  Replaced rows define the residual:
prescribed rows become unit rows (the known value stays in the residual) and
coupling rows keep the derivative of their constraint equation.

Jacobian
--------

Before row replacement the blocks are

======================  =====================================================
``R_u``                 ``[J_uu, J_up, 0, 0, J_um, 0]``
``R_p``                 ``[J_pu, 0, 0, 0, J_pm, 0]``
``R_sd``                ``[0, 0, a_0 I, -I, 0, 0]``
``R_sv``                ``[0, 0, K_s, a_0 M_s, 0, -S_s H^T]``
``R_m``                 ``[0, 0, J_m_ds, 0, K_h, 0]``
``R_g``                 ``[-L_u, -L_p, -L_ds, 0, -L_dm, I]``
======================  =====================================================

with ``J_uu/J_up/J_pu`` the analytic tangent of
:func:`~fsi_ice.fluid.ale.ale_navier_stokes_residual`, ``J_um/J_pm`` columns
built from :func:`~fsi_ice.fluid.ale.ale_geometry_jvp` (its GCL, history-mass,
convection and mesh-transport terms are counted exactly once), ``K_s`` the
consistent StVK tangent and ``L_u/L_p/L_ds/L_dm`` from
:func:`~fsi_ice.interface.traction.interface_load_jvp`.  The only inherited
approximation is the documented central-difference response of a *callable*
fluid body force inside ``ale_geometry_jvp`` (fixed step
``fluid.ale.BODY_FORCE_DIFFERENCE_STEP``); it is listed in
:data:`JACOBIAN_BLOCK_SOURCES` and must be checked by central differences (see
:mod:`fsi_ice.validation.monolithic`).

Versioning
----------

``candidate_id`` travels through the candidate ALE geometry, the interface
geometry and the interface load; :func:`require_candidate` rejects stale or
mixed versions.  A stale stencil (``stencil.step_id`` different from the
accepted-state generation of the base) is rejected when the context is built.
:func:`evaluate_monolithic` rebuilds every candidate object from ``x`` and
never mutates ``x``, the context, the base histories, the reference mesh or the
topology; repeated evaluation of the same inputs is bitwise reproducible.

中文说明
--------

本模块实现整体耦合（monolithic）的全局残差、完整耦合 Jacobian 与矩阵无关 JVP，是 T21 的
冻结求值契约。它只负责"给定候选未知量 ``x``，算出 ``R(x)``、``dR/dx`` 或 ``dR[q]``"，不
实现 Newton 迭代与时间推进循环（这两者分别在 :mod:`fsi_ice.coupling.newton` 与
:mod:`fsi_ice.coupling.monolithic_newton` 中）。

关键约定：

* **统一残差块划分**：全局未知向量固定拆成 6 个块
  ``BLOCK_NAMES = ("u_f", "p_f", "d_s", "v_s", "d_m", "g")``，块的顺序、形状与全局起始行号
  由 :class:`BlockLayout` 唯一确定；各块物理含义见上方表格，二维节点块一律用
  :func:`~fsi_ice.solid.linear.vector_dofs` 的交错排序。
* **完整耦合 Jacobian 的来源**：每一块都调用共享单场内核的解析切线拼装而成，逐块来源记录在
  :data:`JACOBIAN_BLOCK_SOURCES`；唯一的近似是 ``fluid.ale`` 中"可调用流体体积力"的中心差分，
  由 :data:`APPROXIMATIONS` 显式标注，必须用中心差分单独校验；没有任何耦合项被丢弃。
* **伪时间定常推进**：本模块只给出单个 BDF 步的残差/Jacobian；定常解通过上层（
  :func:`~fsi_ice.coupling.monolithic_newton.march_to_steady`）用恒定 ``dt`` 的伪时间步反复
  求解并接受历史得到，本模块不含时间推进循环。

符号与单位约定：界面迹力 ``g`` 量纲为力（N），已含厚度一次，正方向为"流体作用在固体上"
（fluid-on-solid），故固体动量残差中以 ``-interface_load`` 出现。
"""

from dataclasses import dataclass, field
from types import MappingProxyType
import time

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix, kron

from fsi_ice.ale.geometry import build_ale_geometry
from fsi_ice.ale.gcl import build_gcl_transport
from fsi_ice.ale.harmonic import HarmonicExtension
from fsi_ice.fluid.ale import ale_geometry_jvp, ale_navier_stokes_residual, build_ale_flow_context
from fsi_ice.fluid.taylor_hood import assemble_stokes
from fsi_ice.interface.geometry import evaluate_interface_geometry
from fsi_ice.interface.topology import InterfaceTopology
from fsi_ice.interface.traction import assemble_interface_load, interface_load_jvp
from fsi_ice.mesh import TriMesh
from fsi_ice.solid.linear import ElasticMaterial, assemble_elasticity, vector_dofs, vector_values
from fsi_ice.solid.stvk import assemble_mass, stvk_residual
from fsi_ice.time import BDFStencil

# 全局未知向量的块顺序（不可改动）：流体速度、流体压力、固体位移、固体速度、
# 流体参考网格位移、流体界面迹力。每个块的形状与全局行号由 BlockLayout 决定。
BLOCK_NAMES = ("u_f", "p_f", "d_s", "v_s", "d_m", "g")

#: Per-row-block derivative provenance (T21 item 21-04 approximation audit).
#: Every entry names the shared single-field kernel that supplies the block; the
#: only non-analytic path is the callable-body-force central difference inherited
#: from ``fluid.ale`` and it is marked as an approximation.
JACOBIAN_BLOCK_SOURCES = MappingProxyType({
    "R_u/J_uu,J_up": {"source": "fluid.ale.ale_navier_stokes_residual(with_jacobian=True)",
                      "kind": "analytic", "rows": "u_f"},
    "R_p/J_pu": {"source": "fluid.ale.ale_navier_stokes_residual(with_jacobian=True)",
                 "kind": "analytic", "rows": "p_f"},
    "R_u,R_p/J_um,J_pm": {"source": "fluid.ale.ale_geometry_jvp per mesh DOF (rows of one cell depend "
                                    "only on that cell's vertices)",
                          "kind": "analytic; a callable fluid body force uses central differences "
                                  "(fluid.ale.BODY_FORCE_DIFFERENCE_STEP)",
                          "rows": "u_f, p_f"},
    "R_sd": {"source": "BDF stencil coefficients and identity blocks",
             "kind": "analytic", "rows": "d_s"},
    "R_sv": {"source": "solid.stvk.stvk_residual consistent tangent, solid.stvk.assemble_mass, "
                        "H^T through topology.solid_dofs",
             "kind": "analytic", "rows": "v_s"},
    "R_m": {"source": "ale.harmonic.HarmonicExtension.stiffness; exact interface/boundary rows",
            "kind": "analytic", "rows": "d_m"},
    "R_g": {"source": "interface.traction.interface_load_jvp, one call per supporting DOF",
            "kind": "analytic", "rows": "g"},
    "row_replacement": {"source": "prescribed DOFs and coupling constraints; unit rows, known values "
                                  "in the residual",
                        "kind": "analytic", "rows": "u_f, p_f, d_s, v_s, d_m"},
})

# 装配 Jacobian 中所有被显式标注的近似（目前仅一处：可调用流体体积力的中心差分）。
#: Every explicitly marked approximation of the assembled Jacobian.
APPROXIMATIONS = (
    "A callable fluid body force is differentiated by central differences inside "
    "fluid.ale.ale_geometry_jvp (fixed step BODY_FORCE_DIFFERENCE_STEP); all other blocks are "
    "analytic and no coupling term is dropped.",
)


def _freeze_array(array):
    array.setflags(write=False)
    return array


def _freeze_sparse(matrix):
    matrix = csr_matrix(matrix, copy=True)
    matrix.data.setflags(write=False)
    matrix.indices.setflags(write=False)
    matrix.indptr.setflags(write=False)
    return matrix


def _finite_vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def _index_array(value, size, name):
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array.")
    array = array.astype(np.int64, copy=True)
    if np.any(array < 0) or np.any(array >= size):
        raise ValueError(f"{name} contains an index outside [0, {size}).")
    if len(np.unique(array)) != len(array):
        raise ValueError(f"{name} contains duplicate indices.")
    return np.unique(array)


def _zero(rows, columns):
    return csr_matrix((int(rows), int(columns)))


@dataclass(frozen=True)
class PressurePin:
    """``p_node - value = 0`` row replacing one P1 pressure equation.

    中文说明：用一条 ``p_node - value = 0`` 的行替换某个 P1 压力自由度方程，以固定压力的
    参考值、消除纯 Neumann 压力场的常数零空间。``node`` 为 P1 节点下标（非负整数），
    ``value`` 为该点的参考压力（与压力同量纲），默认 0。
    """

    node: int
    value: float = 0.0

    def __post_init__(self):
        if isinstance(self.node, bool) or not isinstance(self.node, (int, np.integer)) or self.node < 0:
            raise ValueError("PressurePin.node must be a nonnegative integer P1 node index.")
        if isinstance(self.value, bool) or not np.isfinite(self.value):
            raise ValueError("PressurePin.value must be finite.")
        object.__setattr__(self, "node", int(self.node))
        object.__setattr__(self, "value", float(self.value))


@dataclass(frozen=True)
class BlockLayout:
    """Slices, shapes, gather/scatter and constraint rows of ``x``.

    ``shapes`` holds the original ``(n, 2)``/``(n,)`` shape of every block in
    ``BLOCK_NAMES`` order.  The Dirichlet/interface index arrays are block-local
    (``0 .. block size``); the global row is ``starts[block] + index``.  Wet
    interface rows take precedence over Dirichlet rows, so wet nodes supplied as
    Dirichlet are removed and reported in ``excluded_fluid_dirichlet_nodes`` /
    ``excluded_mesh_dirichlet_nodes``.

    中文说明：描述全局未知向量 ``x`` 的分块布局。``shapes``/``sizes``/``starts``/``slices``
    按 :data:`BLOCK_NAMES` 顺序给出每块的原始形状、展平长度、全局起始行号与切片。所有
    Dirichlet/界面下标数组都是块内局部下标（``0 .. 块长-1``），全局行号为
    ``starts[block] + 局部下标``。湿界面行优先于 Dirichlet 行，因此被 Dirichlet 重复指定的
    湿节点会被剔除并记录在 ``excluded_*_dirichlet_nodes``。``gather``/``unpack`` 返回只读
    视图（不可写回），``pack``/``scatter`` 返回副本。
    """

    shapes: tuple
    fluid_interface_dofs: np.ndarray
    fluid_dirichlet_dofs: np.ndarray
    fluid_dirichlet_nodes: np.ndarray
    mesh_interface_dofs: np.ndarray
    mesh_dirichlet_dofs: np.ndarray
    mesh_dirichlet_nodes: np.ndarray
    solid_displacement_dirichlet_dofs: np.ndarray
    solid_displacement_dirichlet_nodes: np.ndarray
    solid_velocity_dirichlet_dofs: np.ndarray
    solid_velocity_dirichlet_nodes: np.ndarray
    pressure_pin: object = None
    excluded_fluid_dirichlet_nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    excluded_mesh_dirichlet_nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def __post_init__(self):
        shapes = tuple(tuple(int(size) for size in shape) for shape in self.shapes)
        if len(shapes) != len(BLOCK_NAMES) or any(not shape for shape in shapes):
            raise ValueError(f"shapes must describe the {len(BLOCK_NAMES)} blocks {BLOCK_NAMES}.")
        sizes = []
        for shape in shapes:
            size = 1
            for entry in shape:
                if entry <= 0:
                    raise ValueError(f"Block shape {shape} must be positive.")
                size *= entry
            sizes.append(size)
        # Half-open slices tile [0, n_total) without overlap by construction; the
        # stored arrays are checked so a hand-built layout cannot desynchronize.
        # 中文：半开切片按块顺序无重叠地铺满 [0, n_total)，由此得到各块全局起始行号。
        starts, total = [], 0
        for size in sizes:
            starts.append(total)
            total += size
        if total <= 0:
            raise ValueError("The monolithic system must have at least one unknown.")
        object.__setattr__(self, "shapes", shapes)
        object.__setattr__(self, "sizes", tuple(sizes))
        object.__setattr__(self, "starts", tuple(starts))
        object.__setattr__(self, "slices", tuple(slice(start, start+size)
                                                   for start, size in zip(starts, sizes)))
        object.__setattr__(self, "n_total", total)
        for name, size in zip(("fluid_interface_dofs", "fluid_dirichlet_dofs", "mesh_interface_dofs",
                               "mesh_dirichlet_dofs", "solid_displacement_dirichlet_dofs",
                               "solid_velocity_dirichlet_dofs"),
                              (sizes[0], sizes[0], sizes[4], sizes[4], sizes[2], sizes[3])):
            array = _index_array(getattr(self, name), size, name)
            _freeze_array(array)
            object.__setattr__(self, name, array)
        for name in ("fluid_dirichlet_nodes", "mesh_dirichlet_nodes", "solid_displacement_dirichlet_nodes",
                     "solid_velocity_dirichlet_nodes", "excluded_fluid_dirichlet_nodes",
                     "excluded_mesh_dirichlet_nodes"):
            array = np.asarray(getattr(self, name), dtype=np.int64).ravel()
            _freeze_array(array)
            object.__setattr__(self, name, array)
        if np.intersect1d(self.fluid_interface_dofs, self.fluid_dirichlet_dofs).size:
            raise ValueError("Wet fluid interface rows and Dirichlet rows must be disjoint; "
                             "build the layout so interface rows take precedence.")
        if np.intersect1d(self.mesh_interface_dofs, self.mesh_dirichlet_dofs).size:
            raise ValueError("Wet mesh interface rows and Dirichlet rows must be disjoint; "
                             "build the layout so interface rows take precedence.")
        if self.pressure_pin is not None:
            if not isinstance(self.pressure_pin, PressurePin):
                raise ValueError("pressure_pin must be a PressurePin or None.")
            if self.pressure_pin.node >= sizes[1]:
                raise ValueError(f"PressurePin node {self.pressure_pin.node} is outside the "
                                 f"{sizes[1]} pressure DOFs.")

    def size_of(self, block):
        return self.sizes[BLOCK_NAMES.index(block)]

    def shape_of(self, block):
        return self.shapes[BLOCK_NAMES.index(block)]

    def slice_of(self, block):
        return self.slices[BLOCK_NAMES.index(block)]

    def start_of(self, block):
        return self.starts[BLOCK_NAMES.index(block)]

    def gather(self, block, x):
        """Read-only view of one block of ``x`` in its original shape.

        中文：按块名取出 ``x`` 的一个块，返回其原始形状的只读视图（不可写回原向量）。
        """
        index = BLOCK_NAMES.index(block)
        values = np.asarray(x, dtype=float)
        if values.ndim != 1 or values.shape[0] != self.n_total:
            raise ValueError(f"x must have shape ({self.n_total},), got {values.shape}.")
        view = values[self.slices[index]].reshape(self.shapes[index])
        view.setflags(write=False)
        return view

    def unpack(self, x):
        """Read-only views of every block; a caller cannot write through them.

        中文：按 :data:`BLOCK_NAMES` 顺序返回全部块的只读视图，调用者无法透过视图写入 ``x``。
        """
        values = np.asarray(x, dtype=float)
        if values.ndim != 1 or values.shape[0] != self.n_total:
            raise ValueError(f"x must have shape ({self.n_total},), got {values.shape}.")
        return {name: self.gather(name, values) for name in BLOCK_NAMES}

    def pack(self, values):
        """Copy blocks (mapping by name or sequence in ``BLOCK_NAMES`` order) into ``x``.

        中文：把各块（按名映射，或按 :data:`BLOCK_NAMES` 顺序的序列）拷贝拼装成完整向量并返回；
        输入块本身不被修改。
        """
        if hasattr(values, "keys"):
            missing = [name for name in BLOCK_NAMES if name not in values]
            unknown = [name for name in values if name not in BLOCK_NAMES]
            if missing or unknown:
                raise ValueError(f"pack requires exactly the blocks {BLOCK_NAMES}; "
                                 f"missing {missing}, unknown {unknown}.")
            ordered = [values[name] for name in BLOCK_NAMES]
        else:
            ordered = list(values)
            if len(ordered) != len(BLOCK_NAMES):
                raise ValueError(f"pack requires {len(BLOCK_NAMES)} blocks, got {len(ordered)}.")
        result = np.empty(self.n_total)
        for name, shape, target, block in zip(BLOCK_NAMES, self.shapes, self.slices, ordered):
            array = np.asarray(block, dtype=float)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"Block {name!r} must be a finite array of shape {shape}, got {array.shape}.")
            result[target] = array.ravel()
        return result

    def scatter(self, block, values):
        """Place one block into a zero vector of the full length (a copy).

        中文：把单个块放入一个全零的全长向量并返回（副本），其余块保持为 0。
        """
        index = BLOCK_NAMES.index(block)
        array = np.asarray(values, dtype=float)
        if array.shape != self.shapes[index]:
            raise ValueError(f"Block {block!r} must have shape {self.shapes[index]}, got {array.shape}.")
        result = np.zeros(self.n_total)
        result[self.slices[index]] = array.ravel()
        return result


def build_block_layout(topology, *, fluid_dirichlet_nodes=None, fluid_dirichlet_tags=None,
                       mesh_dirichlet_nodes=None, solid_dirichlet_nodes=None, solid_dirichlet_tags=None,
                       solid_velocity_dirichlet_nodes=None, pressure_pin=None):
    """Build the frozen block layout from a matching interface topology.

    ``fluid_dirichlet_nodes`` (P2 nodes) or ``fluid_dirichlet_tags`` select the
    non-wet-wall known-velocity rows; wet interface nodes are removed and
    reported.  ``mesh_dirichlet_nodes`` defaults to every exterior fluid P1
    vertex except the wet interface vertices.  ``solid_dirichlet_nodes`` (or
    ``solid_dirichlet_tags``) select the clamped solid rows; the velocity rows
    default to the same nodes.  ``pressure_pin`` is an optional
    :class:`PressurePin`.

    中文说明：由匹配好的界面拓扑构造冻结的 :class:`BlockLayout`。输入为流体 P2、网格 P1 与
    固体 P1 的 Dirichlet 节点（或标签），以及可选的压力钉；输出即分块布局。约定湿界面节点
    优先，故传入的 Dirichlet 中被界面占用的节点会被剔除并单独报告；``mesh_dirichlet_nodes``
    缺省为除湿界面顶点外的全部流体外边界顶点。每对同名节点/标签参数只能二选一。
    """
    if not isinstance(topology, InterfaceTopology):
        raise TypeError("topology must be an InterfaceTopology.")
    fluid_mesh = topology.fluid_mesh
    solid_mesh = topology.solid_mesh
    fluid_space = topology.fluid_space
    n_solid = len(solid_mesh.points)

    if fluid_dirichlet_nodes is not None and fluid_dirichlet_tags is not None:
        raise ValueError("Pass either fluid_dirichlet_nodes or fluid_dirichlet_tags, not both.")
    if fluid_dirichlet_tags is not None:
        nodes = fluid_space.boundary_nodes(list(fluid_dirichlet_tags))
    elif fluid_dirichlet_nodes is None:
        nodes = np.zeros(0, dtype=np.int64)
    else:
        nodes = np.asarray(fluid_dirichlet_nodes)
    nodes = _index_array(nodes, len(fluid_space.points), "fluid_dirichlet_nodes")
    keep = np.setdiff1d(nodes, topology.fluid_p2_nodes)
    excluded_fluid = np.setdiff1d(nodes, keep)

    if mesh_dirichlet_nodes is None:
        exterior = fluid_mesh.boundary_nodes()
    else:
        exterior = _index_array(np.asarray(mesh_dirichlet_nodes), fluid_mesh.n_nodes, "mesh_dirichlet_nodes")
    mesh_keep = np.setdiff1d(exterior, topology.fluid_vertex_nodes)
    excluded_mesh = np.setdiff1d(exterior, mesh_keep)

    if solid_dirichlet_nodes is not None and solid_dirichlet_tags is not None:
        raise ValueError("Pass either solid_dirichlet_nodes or solid_dirichlet_tags, not both.")
    if solid_dirichlet_tags is not None:
        solid_nodes = solid_mesh.boundary_nodes(list(solid_dirichlet_tags))
    elif solid_dirichlet_nodes is None:
        solid_nodes = np.zeros(0, dtype=np.int64)
    else:
        solid_nodes = np.asarray(solid_dirichlet_nodes)
    solid_nodes = _index_array(solid_nodes, n_solid, "solid_dirichlet_nodes")
    if solid_velocity_dirichlet_nodes is None:
        velocity_nodes = solid_nodes
    else:
        velocity_nodes = _index_array(np.asarray(solid_velocity_dirichlet_nodes), n_solid,
                                      "solid_velocity_dirichlet_nodes")

    return BlockLayout(
        shapes=((len(fluid_space.points), 2), (fluid_mesh.n_nodes,), (n_solid, 2), (n_solid, 2),
                (fluid_mesh.n_nodes, 2), (topology.n_fluid_interface_nodes, 2)),
        fluid_interface_dofs=vector_dofs(topology.fluid_p2_nodes),
        fluid_dirichlet_dofs=vector_dofs(keep),
        fluid_dirichlet_nodes=keep,
        mesh_interface_dofs=vector_dofs(topology.fluid_vertex_nodes),
        mesh_dirichlet_dofs=vector_dofs(mesh_keep),
        mesh_dirichlet_nodes=mesh_keep,
        solid_displacement_dirichlet_dofs=vector_dofs(solid_nodes),
        solid_displacement_dirichlet_nodes=solid_nodes,
        solid_velocity_dirichlet_dofs=vector_dofs(velocity_nodes),
        solid_velocity_dirichlet_nodes=velocity_nodes,
        pressure_pin=pressure_pin,
        excluded_fluid_dirichlet_nodes=excluded_fluid,
        excluded_mesh_dirichlet_nodes=excluded_mesh,
    )


@dataclass(frozen=True)
class MonolithicBase:
    """Accepted (frozen) history and boundary data of one BDF step.

    ``solid_states`` holds ``(d_s, v_s)`` full solid arrays latest-first,
    ``fluid_states`` the accepted P2 velocities and ``accepted_geometries`` the
    accepted ALE configurations; all three have ``stencil.order`` entries and
    are copied and frozen here.  ``generation`` is the BDF generation the base
    belongs to; a context rejects a stencil whose ``step_id`` differs.  Boundary
    data may be a constant pair, an ``(n, 2)`` array or a callable evaluated at
    the current (fluid velocity) / reference (mesh, solid) nodes.

    中文说明：一个 BDF 步的"已接受（冻结）"历史与边界数据。``solid_states`` 是"最近优先"
    排列的固体 ``(d_s, v_s)`` 完整数组，``fluid_states`` 是已接受的流体 P2 速度，
    ``accepted_geometries`` 是已接受的 ALE 构型；三者都含 ``stencil.order`` 个条目并在此复制
    冻结。``generation`` 是基准所属的 BDF 代数，``step_id`` 与之一致的模板才会被接受。边界
    数据可为常数二元组、``(n, 2)`` 数组或可调用对象（在"当前流体速度节点 / 参考网格与固体
    节点"处求值）。本类要求固体材料为平面应变 StVK。
    """

    reference_mesh: TriMesh
    material: ElasticMaterial
    fluid_density: float
    fluid_viscosity: float
    solid_states: tuple
    fluid_states: tuple
    accepted_geometries: tuple
    generation: int
    solid_body_force: object = (0.0, 0.0)
    fluid_body_force: object = (0.0, 0.0)
    fluid_boundary_velocity: object = (0.0, 0.0)
    mesh_boundary_displacement: object = (0.0, 0.0)
    solid_boundary_displacement: object = (0.0, 0.0)
    solid_boundary_velocity: object = (0.0, 0.0)
    quality_limits: object = None

    def __post_init__(self):
        if not isinstance(self.reference_mesh, TriMesh):
            raise ValueError("reference_mesh must be a TriMesh.")
        if not isinstance(self.material, ElasticMaterial):
            raise ValueError("material must be an ElasticMaterial.")
        if self.material.model != "plane_strain":
            raise ValueError("The monolithic solid block requires plane_strain StVK.")
        for name in ("fluid_density", "fluid_viscosity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number.")
        if isinstance(self.generation, bool) or not isinstance(self.generation, (int, np.integer)) \
                or self.generation < 0:
            raise ValueError("generation must be a nonnegative integer.")
        states = []
        for index, state in enumerate(self.solid_states):
            pair = tuple(state)
            if len(pair) != 2:
                raise ValueError(f"solid_states[{index}] must be a (displacement, velocity) pair.")
            frozen = []
            for name, values in zip(("displacement", "velocity"), pair):
                array = np.asarray(values, dtype=float)
                if array.ndim != 2 or array.shape[1] != 2 or not np.all(np.isfinite(array)):
                    raise ValueError(f"solid_states[{index}].{name} must be a finite (n_nodes, 2) array.")
                frozen.append(_freeze_array(np.array(array, copy=True)))
            states.append(tuple(frozen))
        object.__setattr__(self, "solid_states", tuple(states))
        fluid_states = []
        for index, values in enumerate(self.fluid_states):
            array = np.asarray(values, dtype=float)
            if array.ndim != 2 or array.shape[1] != 2 or not np.all(np.isfinite(array)):
                raise ValueError(f"fluid_states[{index}] must be a finite (n_P2, 2) array.")
            fluid_states.append(_freeze_array(np.array(array, copy=True)))
        object.__setattr__(self, "fluid_states", tuple(fluid_states))
        object.__setattr__(self, "accepted_geometries", tuple(self.accepted_geometries))
        object.__setattr__(self, "generation", int(self.generation))


@dataclass(frozen=True)
class MonolithicContext:
    """Frozen inputs of one monolithic BDF step: layout, base, stencil, topology.

    ``harmonic`` supplies the reference-domain P1 Laplace stiffness ``K_h`` of
    the mesh block.  ``candidate_id`` labels the candidate geometry, interface
    geometry and load; ``thickness`` is the unit-depth solid thickness counted
    exactly once by the interface load.  ``pressure_constraint`` is a
    :class:`PressurePin` (or ``None`` to inherit the pin stored in the layout).

    中文说明：一次整体耦合 BDF 步的冻结输入集合：分块布局、历史基准、BDF 模板、界面拓扑、
    参考域 P1 Laplace 刚度 ``K_h``、候选版本号、厚度与压力约束。``thickness`` 是单位深度
    固体厚度，由界面载荷恰好计入一次；``pressure_constraint`` 为 ``None`` 时继承布局中的压力
    钉。``candidate_id`` 标记候选几何/界面几何/载荷的版本。构造时会校验几何尺寸、界面自由度
    与拓扑一致、历史条目数等于 ``stencil.order``，并且 ``base.generation == stencil.step_id``
    （否则判为过期候选并抛出 ``ValueError``）。
    """

    layout: BlockLayout
    base: MonolithicBase
    stencil: BDFStencil
    topology: InterfaceTopology
    harmonic: HarmonicExtension
    candidate_id: object
    thickness: float
    pressure_constraint: object = None

    def __post_init__(self):
        if not isinstance(self.layout, BlockLayout):
            raise TypeError("layout must be a BlockLayout.")
        if not isinstance(self.base, MonolithicBase):
            raise TypeError("base must be a MonolithicBase.")
        if not isinstance(self.stencil, BDFStencil):
            raise TypeError("stencil must be a BDFStencil.")
        if not isinstance(self.topology, InterfaceTopology):
            raise TypeError("topology must be an InterfaceTopology.")
        if not isinstance(self.harmonic, HarmonicExtension):
            raise TypeError("harmonic must be a HarmonicExtension.")
        if isinstance(self.thickness, bool) or not np.isfinite(self.thickness) or self.thickness <= 0:
            raise ValueError("thickness must be a finite positive number.")
        if self.candidate_id is None:
            raise ValueError("candidate_id must not be None; it labels the geometry/load/cache version.")
        try:
            hash(self.candidate_id)
        except TypeError:
            raise ValueError("candidate_id must be hashable to key geometry/load versions.") from None
        topology = self.topology
        if self.base.reference_mesh is not topology.fluid_mesh:
            raise ValueError("base.reference_mesh must be topology.fluid_mesh (the fluid reference mesh).")
        if self.harmonic.reference_mesh is not topology.fluid_mesh:
            raise ValueError("harmonic must be built on topology.fluid_mesh.")
        expected = (2*len(topology.fluid_space.points), topology.fluid_mesh.n_nodes,
                    2*len(topology.solid_mesh.points), 2*len(topology.solid_mesh.points),
                    2*topology.fluid_mesh.n_nodes, 2*topology.n_fluid_interface_nodes)
        if self.layout.sizes != expected:
            raise ValueError(f"layout sizes {self.layout.sizes} do not match the topology {expected}.")
        if not np.array_equal(self.layout.fluid_interface_dofs, vector_dofs(topology.fluid_p2_nodes)):
            raise ValueError("layout.fluid_interface_dofs must be vector_dofs(topology.fluid_p2_nodes).")
        if not np.array_equal(self.layout.mesh_interface_dofs, vector_dofs(topology.fluid_vertex_nodes)):
            raise ValueError("layout.mesh_interface_dofs must be vector_dofs(topology.fluid_vertex_nodes).")
        order = self.stencil.order
        if len(self.base.solid_states) != order or len(self.base.fluid_states) != order \
                or len(self.base.accepted_geometries) != order:
            raise ValueError(f"base histories must have stencil.order={order} entries; got "
                             f"{len(self.base.solid_states)} solid, {len(self.base.fluid_states)} fluid and "
                             f"{len(self.base.accepted_geometries)} geometries.")
        for index, (displacement, velocity) in enumerate(self.base.solid_states):
            if displacement.shape != (len(topology.solid_mesh.points), 2) or velocity.shape != displacement.shape:
                raise ValueError(f"solid_states[{index}] must match the topology solid mesh "
                                 f"({len(topology.solid_mesh.points)}, 2).")
        for index, velocity in enumerate(self.base.fluid_states):
            if velocity.shape != (len(topology.fluid_space.points), 2):
                raise ValueError(f"fluid_states[{index}] must match the topology fluid P2 space "
                                 f"({len(topology.fluid_space.points)}, 2).")
        for index, geometry in enumerate(self.base.accepted_geometries):
            if getattr(geometry, "reference_mesh", None) is not topology.fluid_mesh:
                raise ValueError(f"accepted_geometries[{index}] was built on a different reference mesh.")
        # 过期候选判定：BDF 模板的 step_id 必须等于基准"已接受状态"的代数。
        if self.base.generation != self.stencil.step_id:
            raise ValueError(f"stale candidate: stencil.step_id={self.stencil.step_id} does not match the "
                             f"accepted-state generation {self.base.generation}; rebuild the context from "
                             "the current accepted history.")
        pin = self.pressure_constraint
        if pin is None:
            pin = self.layout.pressure_pin
        elif not isinstance(pin, PressurePin) or pin != self.layout.pressure_pin:
            raise ValueError("pressure_constraint must be a PressurePin equal to layout.pressure_pin, "
                             "or None to inherit the layout pin.")
        object.__setattr__(self, "pressure_constraint", pin)


def require_candidate(context, *, geometry=None, interface_geometry=None, load=None):
    """Reject stale or mixed candidate versions of geometry, interface geometry and load.

    Every candidate object must carry the context ``candidate_id`` and belong to
    the context topology; an interface load must also have been assembled on the
    given interface geometry.  Raises ``ValueError`` naming the offending object
    otherwise.

    中文说明：校验候选几何、候选界面几何与界面载荷是否与上下文属于同一版本，用于拒绝过期或
    混用的候选。所有候选对象必须携带上下文的 ``candidate_id`` 且属于该拓扑；界面载荷还必须
    在给定界面几何上装配。不一致时抛出 ``ValueError`` 并指明出错对象。
    """
    if not isinstance(context, MonolithicContext):
        raise TypeError("context must be a MonolithicContext.")
    if geometry is not None:
        if getattr(geometry, "candidate_id", None) != context.candidate_id:
            raise ValueError(f"stale candidate geometry {getattr(geometry, 'candidate_id', None)!r}; "
                             f"the context is {context.candidate_id!r}.")
        if geometry.reference_mesh is not context.base.reference_mesh:
            raise ValueError("candidate geometry was built on a different reference mesh (mixed version).")
    if interface_geometry is not None:
        if getattr(interface_geometry, "candidate_id", None) != context.candidate_id:
            raise ValueError(f"stale candidate interface geometry "
                             f"{getattr(interface_geometry, 'candidate_id', None)!r}; "
                             f"the context is {context.candidate_id!r}.")
        if interface_geometry.topology_id != context.topology.topology_id:
            raise ValueError("interface geometry belongs to a different topology (mixed version).")
    if load is not None:
        if getattr(load, "candidate_id", None) != context.candidate_id:
            raise ValueError(f"stale interface load {getattr(load, 'candidate_id', None)!r}; "
                             f"the context is {context.candidate_id!r}.")
        if load.topology_id != context.topology.topology_id:
            raise ValueError("interface load belongs to a different topology (mixed version).")
        if interface_geometry is not None and load.geometry_id != interface_geometry.geometry_id:
            raise ValueError("interface load was assembled on a different interface geometry (mixed version).")


@dataclass(frozen=True)
class _FluidState:
    """Minimal fluid state adapter accepted by the shared stress/load kernels.

    中文：供共享应力/载荷内核使用的最小流体状态适配器，只暴露速度与压力。
    """

    velocity: np.ndarray
    pressure: np.ndarray


@dataclass(frozen=True)
class _Candidate:
    """One evaluation's rebuilt candidate objects (geometry, operators, load...).

    中文：``_evaluate_candidate`` 的输出集合，成员（候选几何、流算子与 ALE 上下文、界面
    几何、流体状态、界面载荷及各阶段耗时）全部由当前 ``x`` 与冻结上下文重建，不跨调用缓存。
    """

    geometry: object
    operators: object
    ale_context: object
    flow_values: np.ndarray
    interface_geometry: object
    fluid_state: object
    load: object
    seconds: dict


@dataclass(frozen=True)
class _Physics:
    """Shared single-field kernels of one evaluation (residual/tangent pieces).

    中文：一次求值中用到的共享单场内核结果：流体残差与切线、固体质量阵、内力矢量、
    一致切线刚度、以及网格块刚度 ``kron(K_h, I_2)``。
    """

    flow_residual: np.ndarray
    flow_tangent: object
    mass: object
    internal: np.ndarray
    stiffness: object
    mesh_stiffness: object


@dataclass(frozen=True)
class _Rows:
    """Global row indices of every replaced (constraint/Dirichlet) row.

    中文：被替换行（湿界面约束行与各 Dirichlet 行、压力钉行）的全局行号集合，由
    :func:`_global_rows` 依布局一次性算出。
    """

    u_interface: np.ndarray
    u_dirichlet: np.ndarray
    p_pin: np.ndarray
    d_s_dirichlet: np.ndarray
    v_s_dirichlet: np.ndarray
    m_interface: np.ndarray
    m_dirichlet: np.ndarray


def _global_rows(layout):
    # 全局行号 = 各块起始行号 starts[block] + 块内局部下标（见 BlockLayout 约定）。
    u0, p0, ds0, vs0, dm0, _ = layout.starts
    pin = layout.pressure_pin
    return _Rows(
        u_interface=u0+layout.fluid_interface_dofs,
        u_dirichlet=u0+layout.fluid_dirichlet_dofs,
        p_pin=np.array([p0+pin.node], dtype=np.int64) if pin is not None else np.zeros(0, dtype=np.int64),
        d_s_dirichlet=ds0+layout.solid_displacement_dirichlet_dofs,
        v_s_dirichlet=vs0+layout.solid_velocity_dirichlet_dofs,
        m_interface=dm0+layout.mesh_interface_dofs,
        m_dirichlet=dm0+layout.mesh_dirichlet_dofs,
    )


def _solid_history_terms(context):
    """``hd``/``hv`` = sum of a_j * accepted states for displacement/velocity.

    中文：BDF 历史项 ``hd``/``hv`` = ``sum_{j>=1} a_j * 已接受状态``（位移/速度），与当前项
    ``a_0 w`` 一起构成 ``D_BDF w = a_0 w + sum_{j>=1} a_j w_j``。系数取自
    ``stencil.coefficients``，状态按"最近优先"与 ``base.solid_states`` 逐一对应。
    """
    coefficients = np.asarray(context.stencil.coefficients, dtype=float)[1:]
    size = 2*len(context.topology.solid_mesh.points)
    displacement = np.zeros(size)
    velocity = np.zeros(size)
    for coefficient, (d_state, v_state) in zip(coefficients, context.base.solid_states):
        displacement += coefficient*d_state.ravel()
        velocity += coefficient*v_state.ravel()
    return displacement, velocity


def _solid_body_load(base, mesh, material):
    # 常数零体力时直接返回零载荷，跳过装配；否则调用共享弹性装配并取载荷项 [1]。
    body_force = base.solid_body_force
    if not callable(body_force):
        values = np.asarray(body_force, dtype=float)
        if values.shape == (2,) and not np.any(values):
            return np.zeros(2*mesh.n_nodes)
    return assemble_elasticity(mesh, material, body_force=body_force)[1]


def _evaluate_candidate(x, context):
    """Build every candidate object from ``x``; nothing is cached across calls.

    中文：从候选未知量 ``x`` 重建全部候选对象——由 ``d_m`` 构造候选 ALE 几何与 GCL 输运、
    在候选网格上装配流算子并建 ALE 流动上下文、由 ``d_s`` 求界面几何、组装界面载荷；每次
    调用都重建、不跨调用缓存，最后用 :func:`require_candidate` 校验版本一致。
    """
    base = context.base
    blocks = context.layout.unpack(x)
    started = time.perf_counter()
    geometry = build_ale_geometry(base.reference_mesh, np.array(blocks["d_m"], copy=True),
                                  quality_limits=base.quality_limits, candidate_id=context.candidate_id)
    transport = build_gcl_transport(base.reference_mesh, geometry, base.accepted_geometries, context.stencil)
    geometry_seconds = time.perf_counter()-started

    started = time.perf_counter()
    operators = assemble_stokes(geometry.current_mesh, viscosity=base.fluid_viscosity,
                                density=base.fluid_density, body_force=base.fluid_body_force)
    ale_context = build_ale_flow_context(operators, geometry, transport, base.fluid_states)
    flow_values = np.concatenate((blocks["u_f"].ravel(), blocks["p_f"].ravel()))
    operators_seconds = time.perf_counter()-started

    started = time.perf_counter()
    interface_geometry = evaluate_interface_geometry(
        context.topology, geometry.current_mesh, operators.space, np.array(blocks["d_s"], copy=True),
        candidate_id=context.candidate_id)
    fluid_state = _FluidState(np.array(blocks["u_f"], copy=True), np.array(blocks["p_f"], copy=True))
    load = assemble_interface_load(context.topology, interface_geometry, operators, fluid_state,
                                   thickness=context.thickness, path="current")
    interface_seconds = time.perf_counter()-started
    candidate = _Candidate(geometry=geometry, operators=operators, ale_context=ale_context,
                           flow_values=flow_values, interface_geometry=interface_geometry,
                           fluid_state=fluid_state, load=load,
                           seconds={"geometry_seconds": geometry_seconds,
                                    "operators_seconds": operators_seconds,
                                    "interface_seconds": interface_seconds})
    require_candidate(context, geometry=candidate.geometry, interface_geometry=candidate.interface_geometry,
                      load=candidate.load)
    return candidate


def _physics(context, candidate, d_s):
    """Shared single-field residual/tangent kernels of one evaluation.

    中文：调用共享单场内核求一次求值所需的量——流体 ALE Navier-Stokes 残差与解析切线、
    固体质量阵与 StVK 内力/一致切线、以及网格块刚度。
    """
    flow_residual, flow_tangent = ale_navier_stokes_residual(
        candidate.operators, candidate.flow_values, candidate.ale_context, with_jacobian=True)
    mesh = context.topology.solid_mesh
    material = context.base.material
    mass = assemble_mass(mesh, material)
    body = _solid_body_load(context.base, mesh, material)
    internal, stiffness = stvk_residual(mesh, material, np.asarray(d_s), external=body)
    # Both components of the reference-domain mesh displacement use the same
    # scalar Laplace stiffness of HarmonicExtension, i.e. kron(K_h, I_2) in the
    # interleaved nodal ordering; the harmonic solve applies exactly this matrix
    # component by component.
    mesh_stiffness = kron(context.harmonic.stiffness, csr_matrix(np.eye(2)), format="csr")
    return _Physics(flow_residual, flow_tangent, mass, internal, stiffness, mesh_stiffness)


def _residual_vector(x, context, candidate, physics):
    """Constrained global residual; only this function defines ``R(x)``.

    中文：定义唯一的全局约束残差 ``R(x)``，按 :data:`BLOCK_NAMES` 顺序拼成。前
    ``n_velocity+n_pressure`` 行是流体 ALE 残差（其中湿壁与已知速度行用约束替换），随后依次为
    固体位移行 ``D_BDF d_s - v_s``、固体速度行、网格位移行 ``K_h d_m`` 与界面迹力行
    ``g - g_f_gamma``；最后一段把被替换行写回约束方程。
    """
    layout = context.layout
    base = context.base
    topology = context.topology
    rows = _global_rows(layout)
    blocks = layout.unpack(x)
    n_velocity = layout.sizes[0]
    n_pressure = layout.sizes[1]
    u0, p0, ds0, vs0, dm0, g0 = layout.starts

    residual = np.empty(layout.n_total)
    residual[:n_velocity+n_pressure] = physics.flow_residual

    coefficient = float(context.stencil.coefficients[0])
    # a_0 是 BDF 当前值系数；D_BDF w = a_0 w + 历史项。
    history_displacement, history_velocity = _solid_history_terms(context)
    d_s = blocks["d_s"].ravel()
    v_s = blocks["v_s"].ravel()
    residual[ds0:vs0] = coefficient*d_s+history_displacement-v_s

    solid_mesh = topology.solid_mesh
    g = np.asarray(blocks["g"]).ravel()
    interface_load = np.zeros(2*len(solid_mesh.points))
    # g 为流体作用在固体上的界面迹力（fluid-on-solid，单位 N，已含厚度一次）；
    # 经 H^T 散射到固体自由度后，以 -interface_load 进入固体动量残差。
    interface_load[topology.solid_dofs] = topology.H.T @ g
    residual[vs0:dm0] = physics.mass @ (coefficient*v_s+history_velocity)+physics.internal-interface_load

    residual[dm0:g0] = physics.mesh_stiffness @ blocks["d_m"].ravel()
    residual[g0:] = g-candidate.load.g_f_gamma

    # Replaced rows: wet interface rows and prescribed rows.
    # 中文：以下用约束方程覆盖被替换的行——湿界面行 u_f|Gamma - H v_s|Gamma = 0，
    # 各 Dirichlet 行（已知值留在残差右侧），以及压力钉行 p_node - value = 0。
    residual[rows.u_interface] = (blocks["u_f"].ravel()[layout.fluid_interface_dofs]
                                  - topology.H @ blocks["v_s"].ravel()[topology.solid_dofs])
    u_dirichlet = vector_values(base.fluid_boundary_velocity,
                                candidate.operators.space.points[layout.fluid_dirichlet_nodes],
                                "Fluid boundary velocity").ravel()
    residual[rows.u_dirichlet] = blocks["u_f"].ravel()[layout.fluid_dirichlet_dofs]-u_dirichlet
    pin = layout.pressure_pin
    if pin is not None:
        residual[rows.p_pin] = blocks["p_f"].ravel()[pin.node]-pin.value
    d_s_dirichlet = vector_values(base.solid_boundary_displacement,
                                  solid_mesh.points[layout.solid_displacement_dirichlet_nodes],
                                  "Solid boundary displacement").ravel()
    residual[rows.d_s_dirichlet] = d_s[layout.solid_displacement_dirichlet_dofs]-d_s_dirichlet
    v_s_dirichlet = vector_values(base.solid_boundary_velocity,
                                  solid_mesh.points[layout.solid_velocity_dirichlet_nodes],
                                  "Solid boundary velocity").ravel()
    residual[rows.v_s_dirichlet] = v_s[layout.solid_velocity_dirichlet_dofs]-v_s_dirichlet
    d_m = blocks["d_m"].ravel()
    residual[rows.m_interface] = d_m[layout.mesh_interface_dofs]-d_s[topology.solid_dofs]
    d_m_dirichlet = vector_values(base.mesh_boundary_displacement,
                                  base.reference_mesh.points[layout.mesh_dirichlet_nodes],
                                  "Mesh boundary displacement").ravel()
    residual[rows.m_dirichlet] = d_m[layout.mesh_dirichlet_dofs]-d_m_dirichlet
    return residual


def _geometry_columns(operators, flow_values, ale_context):
    """``J_um``/``J_pm``: ``ale_geometry_jvp`` columns for every mesh DOF.

    Every residual term of one cell depends only on that cell's own vertices.
    Two mesh nodes have disjoint affected residual rows exactly when the vertex
    sets of their incident cells are disjoint, so a proper coloring of that
    conflict graph lets one ``ale_geometry_jvp`` call serve a whole color class
    and the columns be read off the disjoint row supports.  This is the same
    analytic directional derivative as one call per DOF, only batched (a test
    compares both paths over every column of a small mesh).

    中文：给出几何耦合块 ``J_um``/``J_pm``（流体残差对网格位移 ``d_m`` 的解析交叉导数），
    按列组装。由于一个单元的残差只依赖本单元顶点，可用冲突图着色把互不影响的网格节点并为
    一类，每类一次 ``ale_geometry_jvp`` 即可读出各列，结果与逐自由度调用等价，只是批量化。
    """
    mesh = operators.space.mesh
    cells = np.asarray(mesh.cells, dtype=np.int64)
    space_cells = np.asarray(operators.space.cells, dtype=np.int64)
    n_velocity = operators.n_velocity
    n_pressure = mesh.n_nodes
    n_mesh = 2*mesh.n_nodes
    node_cells = [[] for _ in range(mesh.n_nodes)]
    incident_vertices = [set() for _ in range(mesh.n_nodes)]
    for cell_id, (a, b, c) in enumerate(cells.tolist()):
        for node in (a, b, c):
            node_cells[node].append(cell_id)
            incident_vertices[node].update((a, b, c))
    # G(v) = nodes whose incident-cell vertices contain v; co-membership in one
    # G(v) is exactly the conflict relation (some incident cells then share v).
    # 中文：两个网格节点若其关联单元的顶点集有交集，则它们的残差行会重叠，属冲突关系；
    # 按此冲突图正常着色，同色节点可共享一次 JVP。partners[node] 即 node 的冲突邻居。
    groups = {}
    for node, vertices in enumerate(incident_vertices):
        for vertex in vertices:
            groups.setdefault(vertex, []).append(node)
    partners = [set() for _ in range(mesh.n_nodes)]
    for members in groups.values():
        member_set = set(members)
        for node in members:
            partners[node].update(member_set)
    colors = np.full(mesh.n_nodes, -1, dtype=np.int64)
    for node in range(mesh.n_nodes):
        used = {int(colors[other]) for other in partners[node] if colors[other] >= 0}
        color = 0
        while color in used:
            color += 1
        colors[node] = color
    row_u, col_u, val_u, row_p, col_p, val_p = [], [], [], [], [], []
    for color in range(int(colors.max())+1):
        nodes = np.flatnonzero(colors == color)
        for component in (0, 1):
            direction = np.zeros((mesh.n_nodes, 2))
            direction[nodes, component] = 1.0
            delta = ale_geometry_jvp(operators, flow_values, ale_context=ale_context, delta_d_m=direction)
            for node in nodes.tolist():
                incident = node_cells[node]
                if not incident:
                    continue
                local_cells = cells[incident]
                dofs_u = np.unique(vector_dofs(space_cells[incident]).ravel())
                dofs_p = np.unique(local_cells.ravel())
                column = 2*node+component
                nonzero_u = delta[dofs_u]
                row_u.extend(dofs_u.tolist())
                col_u.extend([column]*len(dofs_u))
                val_u.extend(nonzero_u.tolist())
                nonzero_p = delta[n_velocity+dofs_p]
                row_p.extend(dofs_p.tolist())
                col_p.extend([column]*len(dofs_p))
                val_p.extend(nonzero_p.tolist())
    geometry_u = coo_matrix((val_u, (row_u, col_u)), shape=(n_velocity, n_mesh)).tocsr()
    geometry_p = coo_matrix((val_p, (row_p, col_p)), shape=(n_pressure, n_mesh)).tocsr()
    return geometry_u, geometry_p


def _load_blocks(context, candidate):
    """``L_u/L_p/L_dm/L_ds`` from ``interface_load_jvp`` (one call per supporting DOF).

    ``g_f_gamma`` depends on the fluid P2 velocity/pressure and the mesh
    displacement only through the interface owner cells and on the solid
    displacement only through the solid owner cells (which contain the edge
    endpoints); columns outside those supports are structurally zero.  Each
    column is one analytic ``interface_load_jvp`` call, so no chain term is
    added twice.

    中文：装配界面迹力对流体速度/压力、网格位移、固体位移的导数块 ``L_u/L_p/L_dm/L_ds``
    （即残差 ``R_g`` 对应的 ``-L``）。``g_f_gamma`` 只通过界面所属单元依赖这些量，支撑集之外
    的列结构为零；每列一次解析 ``interface_load_jvp``，不做重复链式叠加。返回按
    ``{"velocity","pressure","mesh_displacement","solid_displacement"}`` 索引的 CSR 块。
    """
    topology = context.topology
    operators = candidate.operators
    geometry = candidate.interface_geometry
    state = candidate.fluid_state
    mesh = operators.space.mesh
    cells = np.asarray(mesh.cells, dtype=np.int64)
    space_cells = np.asarray(operators.space.cells, dtype=np.int64)
    solid_cells = np.asarray(topology.solid_mesh.cells, dtype=np.int64)
    owners = geometry.fluid_owner
    solid_owners = geometry.solid_owner
    supports = {
        "velocity": ("u_f", vector_dofs(np.unique(space_cells[owners].ravel()))),
        "pressure": ("p_f", np.unique(cells[owners].ravel())),
        "mesh_displacement": ("d_m", vector_dofs(np.unique(cells[owners].ravel()))),
        "solid_displacement": ("d_s", vector_dofs(np.unique(solid_cells[solid_owners].ravel()))),
    }
    n_flux = 2*topology.n_fluid_interface_nodes
    blocks = {}
    for key, (name, dofs) in supports.items():
        shape = context.layout.shape_of(name)
        size = context.layout.size_of(name)
        value = np.zeros(shape)
        flat = value.ravel()
        rows, columns, values = [], [], []
        for dof in dofs.tolist():
            flat[dof] = 1.0
            derivative = interface_load_jvp(topology, geometry, operators, state, {key: value},
                                            thickness=context.thickness, path="current")
            flat[dof] = 0.0
            column = derivative.delta_g_f_gamma
            nonzero = np.flatnonzero(column)
            rows.extend(nonzero.tolist())
            columns.extend([dof]*len(nonzero))
            values.extend(column[nonzero].tolist())
        blocks[key] = coo_matrix((values, (rows, columns)), shape=(n_flux, size)).tocsr()
    return blocks


def _constraint_entries(layout, topology):
    """COO entries of every replaced row (unit rows and wet interface constraints).

    中文：给出所有被替换行的 COO 三元组——各 Dirichlet 行与压力钉行化为单位行（已知值留在
    残差里），流体湿界面行与网格湿界面行化为约束方程的导数（对角 1 与 ``-H``/``-1`` 交叉项）。
    """
    u0, p0, ds0, vs0, dm0, _ = layout.starts
    coo_rows, coo_columns, coo_values = [], [], []

    def add(row, column, value):
        coo_rows.append(row)
        coo_columns.append(column)
        coo_values.append(value)

    for dof in layout.fluid_dirichlet_dofs.tolist():
        add(u0+dof, u0+dof, 1.0)
    if layout.pressure_pin is not None:
        add(p0+layout.pressure_pin.node, p0+layout.pressure_pin.node, 1.0)
    for dof in layout.solid_displacement_dirichlet_dofs.tolist():
        add(ds0+dof, ds0+dof, 1.0)
    for dof in layout.solid_velocity_dirichlet_dofs.tolist():
        add(vs0+dof, vs0+dof, 1.0)
    for dof in layout.mesh_dirichlet_dofs.tolist():
        add(dm0+dof, dm0+dof, 1.0)
    for dof in layout.fluid_interface_dofs.tolist():
        add(u0+dof, u0+dof, 1.0)
    h = topology.H.tocoo()
    # 流体湿界面行：u_f|Gamma - H v_s|Gamma = 0，故在固体自由度处加 -H 项。
    for row, column, value in zip(h.row.tolist(), h.col.tolist(), h.data.tolist()):
        add(u0+int(layout.fluid_interface_dofs[row]), vs0+int(topology.solid_dofs[column]), -float(value))
    for index, dof in enumerate(layout.mesh_interface_dofs.tolist()):
        # 网格湿界面行：d_m|Gamma - d_s|Gamma = 0（一一参考匹配，H_d = I）。
        add(dm0+dof, dm0+dof, 1.0)
        add(dm0+dof, ds0+int(topology.solid_dofs[index]), -1.0)
    return np.array(coo_rows, dtype=np.int64), np.array(coo_columns, dtype=np.int64), np.array(coo_values)


def _zero_rows(matrix, rows):
    # 把指定全局行的所有非零元清零（不改结构），用于被替换行；rows 为空则原样返回。
    matrix = csr_matrix(matrix, copy=True)
    if not len(rows):
        return matrix
    kill = np.zeros(matrix.shape[0], dtype=bool)
    kill[np.asarray(rows, dtype=np.int64)] = True
    row_of = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    matrix.data[kill[row_of]] = 0.0
    return matrix


def _assembled_jacobian(context, candidate, physics):
    """Assemble the complete coupling Jacobian as a 6x6 block sparse matrix.

    中文：装配完整耦合 Jacobian ``dR/dx``（6×6 块稀疏 CSR 矩阵，块布局见模块 docstring）。
    非零块全部来自共享单场内核的解析切线；被替换行的物理列先清零，再由
    :func:`_constraint_entries` 叠加约束方程导数。
    """
    layout = context.layout
    topology = context.topology
    rows = _global_rows(layout)
    n_velocity, n_pressure = layout.sizes[0], layout.sizes[1]
    n_solid, n_mesh, n_flux = layout.sizes[2], layout.sizes[4], layout.sizes[5]
    u0, p0, ds0, vs0, dm0, g0 = layout.starts
    coefficient = float(context.stencil.coefficients[0])

    geometry_u, geometry_p = _geometry_columns(candidate.operators, candidate.flow_values,
                                               candidate.ale_context)
    load_blocks = _load_blocks(context, candidate)

    replaced_fluid = np.concatenate((rows.u_interface-u0, rows.u_dirichlet-u0))
    replaced_mesh = np.concatenate((rows.m_interface-dm0, rows.m_dirichlet-dm0))
    # 被替换行的解析列先清零，稍后由约束行覆盖（否则物理切线与约束行冲突）。
    j_uu = _zero_rows(physics.flow_tangent[:n_velocity, :n_velocity], replaced_fluid)
    j_up = _zero_rows(physics.flow_tangent[:n_velocity, n_velocity:], replaced_fluid)
    j_um = _zero_rows(geometry_u, replaced_fluid)
    j_pu = _zero_rows(physics.flow_tangent[n_velocity:, :n_velocity], rows.p_pin-p0)
    j_pm = _zero_rows(geometry_p, rows.p_pin-p0)

    identity_solid = _zero(n_solid, n_solid)
    identity_solid.setdiag(1.0)
    j_sd_ds = _zero_rows(coefficient*identity_solid, rows.d_s_dirichlet-ds0)
    j_sd_vs = _zero_rows(-identity_solid, rows.d_s_dirichlet-ds0)
    j_sv_ds = _zero_rows(physics.stiffness, rows.v_s_dirichlet-vs0)
    j_sv_vs = _zero_rows(coefficient*physics.mass, rows.v_s_dirichlet-vs0)
    scatter = coo_matrix((np.ones(2*topology.n_solid_interface_nodes),
                          (topology.solid_dofs, np.arange(2*topology.n_solid_interface_nodes))),
                         shape=(n_solid, 2*topology.n_solid_interface_nodes)).tocsr()
    # 固体速度行对 g 的导数：-S_s H^T（S_s 选固体界面自由度，H^T 把界面迹力映射到固体）。
    j_sv_g = _zero_rows(-(scatter @ topology.H.T), rows.v_s_dirichlet-vs0)

    j_m_m = _zero_rows(physics.mesh_stiffness, replaced_mesh)
    j_m_ds = coo_matrix((-np.ones(len(layout.mesh_interface_dofs)),
                         (layout.mesh_interface_dofs, topology.solid_dofs)),
                        shape=(n_mesh, n_solid)).tocsr()
    j_m_ds = _zero_rows(j_m_ds, replaced_mesh)

    j_g_u = -load_blocks["velocity"]
    j_g_p = -load_blocks["pressure"]
    j_g_ds = -load_blocks["solid_displacement"]
    j_g_dm = -load_blocks["mesh_displacement"]
    j_g_g = _zero(n_flux, n_flux)
    j_g_g.setdiag(1.0)

    # 6×6 块矩阵，行/列块均按 BLOCK_NAMES 顺序 u_f,p_f,d_s,v_s,d_m,g；零块显式填 0。
    matrix = bmat([
        [j_uu, j_up, _zero(n_velocity, n_solid), _zero(n_velocity, n_solid), j_um, _zero(n_velocity, n_flux)],
        [j_pu, _zero(n_pressure, n_pressure), _zero(n_pressure, n_solid), _zero(n_pressure, n_solid), j_pm,
         _zero(n_pressure, n_flux)],
        [_zero(n_solid, n_velocity), _zero(n_solid, n_pressure), j_sd_ds, j_sd_vs, _zero(n_solid, n_mesh),
         _zero(n_solid, n_flux)],
        [_zero(n_solid, n_velocity), _zero(n_solid, n_pressure), j_sv_ds, j_sv_vs, _zero(n_solid, n_mesh),
         j_sv_g],
        [_zero(n_mesh, n_velocity), _zero(n_mesh, n_pressure), j_m_ds, _zero(n_mesh, n_solid), j_m_m,
         _zero(n_mesh, n_flux)],
        [j_g_u, j_g_p, j_g_ds, _zero(n_flux, n_solid), j_g_dm, j_g_g],
    ], format="csr")
    coo_rows, coo_columns, coo_values = _constraint_entries(layout, topology)
    # 叠加被替换行：单位行覆盖 Dirichlet/压力钉行，约束导数覆盖湿界面行。
    constraints = coo_matrix((coo_values, (coo_rows, coo_columns)), shape=(layout.n_total, layout.n_total))
    matrix = (matrix+constraints.tocsr()).tocsr()
    matrix.eliminate_zeros()
    return matrix


def _jvp_single(context, candidate, physics, q):
    """One matrix-free directional derivative ``dR[q]`` at the candidate state.

    中文：对单个方向 ``q`` 计算 ``dR[q]``，不装配矩阵，只用切线作用与
    ``ale_geometry_jvp``/``interface_load_jvp`` 的方向导数，逐块拼出结果；被替换行则用约束
    方程的导数。方向 ``q`` 与输出 ``delta`` 均按 :data:`BLOCK_NAMES` 分块。
    """
    layout = context.layout
    topology = context.topology
    rows = _global_rows(layout)
    blocks = layout.unpack(q)
    coefficient = float(context.stencil.coefficients[0])
    n_velocity, n_pressure = layout.sizes[0], layout.sizes[1]
    u0, p0, ds0, vs0, dm0, g0 = layout.starts
    q_u = blocks["u_f"].ravel()
    q_p = blocks["p_f"].ravel()
    q_ds = blocks["d_s"].ravel()
    q_vs = blocks["v_s"].ravel()
    q_dm = blocks["d_m"].ravel()
    q_g = blocks["g"].ravel()

    delta = np.zeros(layout.n_total)
    delta[:n_velocity+n_pressure] = physics.flow_tangent @ np.concatenate((q_u, q_p))
    # 几何耦合：网格位移方向导数经 ale_geometry_jvp 加到流体残差行（GCL/对流/网格输运各计一次）。
    if np.any(q_dm):
        delta[:n_velocity+n_pressure] += ale_geometry_jvp(candidate.operators, candidate.flow_values,
                                                          ale_context=candidate.ale_context,
                                                          delta_d_m=q_dm.reshape(-1, 2))
    delta[ds0:vs0] = coefficient*q_ds-q_vs
    interface_scatter = np.zeros(2*len(topology.solid_mesh.points))
    interface_scatter[topology.solid_dofs] = topology.H.T @ q_g
    delta[vs0:dm0] = physics.stiffness @ q_ds+coefficient*(physics.mass @ q_vs)-interface_scatter
    delta[dm0:g0] = physics.mesh_stiffness @ q_dm

    load_direction = np.zeros(layout.sizes[5])
    # 界面迹力方向导数：对 u_f/p_f/d_s/d_m 逐个贡献合成。R_g 中为 q_g - load_direction。
    for key, block, name in (("velocity", q_u, "u_f"), ("pressure", q_p, "p_f"),
                             ("solid_displacement", q_ds, "d_s"), ("mesh_displacement", q_dm, "d_m")):
        if not np.any(block):
            continue
        derivative = interface_load_jvp(topology, candidate.interface_geometry, candidate.operators,
                                        candidate.fluid_state,
                                        {key: np.array(block).reshape(layout.shape_of(name))},
                                        thickness=context.thickness, path="current")
        load_direction = load_direction+derivative.delta_g_f_gamma
    delta[g0:] = q_g-load_direction

    # Replaced rows: derivatives of the constraint equations.
    # 中文：被替换行（湿界面与 Dirichlet/压力钉行）填入约束方程的方向导数。
    delta[rows.u_interface] = q_u[layout.fluid_interface_dofs]-topology.H @ q_vs[topology.solid_dofs]
    delta[rows.u_dirichlet] = q_u[layout.fluid_dirichlet_dofs]
    pin = layout.pressure_pin
    if pin is not None:
        delta[rows.p_pin] = q_p[pin.node]
    delta[rows.d_s_dirichlet] = q_ds[layout.solid_displacement_dirichlet_dofs]
    delta[rows.v_s_dirichlet] = q_vs[layout.solid_velocity_dirichlet_dofs]
    delta[rows.m_interface] = q_dm[layout.mesh_interface_dofs]-q_ds[topology.solid_dofs]
    delta[rows.m_dirichlet] = q_dm[layout.mesh_dirichlet_dofs]
    return delta


@dataclass(frozen=True)
class MonolithicEvaluation:
    """Frozen result of one monolithic evaluation.

    ``residual`` is read-only and owned by this object; ``jacobian`` (CSR) owns
    and freezes its ``data``/``indices``/``indptr``; ``jvp_values`` is a
    read-only copy.  ``block_norms`` are ``||R_b||_2 / max(||x_b||_2, 1)`` per
    block (dimensionless residual monitors), ``geometry_diagnostics`` the
    minimum candidate-geometry quantities and ``timings`` the wall-clock seconds
    of each stage.

    中文：一次整体耦合求值的冻结结果。``residual`` 只读且归本对象所有；``jacobian``（CSR）
    会冻结其 ``data``/``indices``/``indptr``；``jvp_values`` 为只读副本。``block_norms`` 是
    逐块无量纲残差监视量 ``||R_b||_2 / max(||x_b||_2, 1)``，``geometry_diagnostics`` 给出
    候选几何的最小质量量，``timings`` 记录各阶段墙钟秒数。
    """

    residual: np.ndarray
    jacobian: object
    jvp_values: object
    candidate_id: object
    block_norms: object
    geometry_diagnostics: object
    timings: object

    def __post_init__(self):
        _freeze_array(self.residual)
        if self.jvp_values is not None:
            _freeze_array(self.jvp_values)
        if self.jacobian is not None:
            object.__setattr__(self, "jacobian", _freeze_sparse(self.jacobian))
        for name in ("block_norms", "geometry_diagnostics", "timings"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


def _block_norms(x, residual, layout):
    # 逐块无量纲残差：||R_b||_2 / max(||x_b||_2, 1)，用于收敛监视。
    norms = {}
    for name in BLOCK_NAMES:
        values = layout.gather(name, x).ravel()
        norms[name] = float(np.linalg.norm(residual[layout.slice_of(name)])
                            / max(float(np.linalg.norm(values)), 1.0))
    return norms


def _geometry_diagnostics(candidate):
    geometry = candidate.geometry
    interface = candidate.interface_geometry
    return {
        "minimum_cell_determinant": float(np.min(geometry.current_mesh.determinants)),
        "minimum_volume_ratio": float(geometry.minimum_volume_ratio),
        "minimum_quality": float(geometry.minimum_quality),
        "interface_gap_error": float(interface.gap_error),
        "interface_gap_tolerance": float(interface.gap_tolerance),
        "minimum_solid_jacobian": float(interface.minimum_jacobian),
        "interface_normal_error": float(interface.normal_error),
        "interface_surface_ratio_error": float(interface.surface_ratio_error),
    }


def evaluate_monolithic(x, *, context, mode="residual", directions=None):
    """Evaluate the monolithic residual (and optionally the Jacobian or the JVP).

    ``mode="residual"`` returns the residual, ``mode="assembled"`` the residual
    and the complete coupling Jacobian and ``mode="jvp"`` the residual and the
    matrix-free directional derivative for ``directions`` (``(n,)`` or
    ``(batch, n)``).  The candidate geometry, operators, interface geometry and
    interface load are rebuilt from ``x`` on every call with
    ``context.candidate_id``; nothing is cached and no input is modified.

    中文：求整体耦合残差，可选同时给出 Jacobian 或 JVP。``mode="residual"`` 只返回残差；
    ``"assembled"`` 额外返回完整耦合 Jacobian；``"jvp"`` 额外返回对 ``directions``（``(n,)``
    或 ``(batch, n)``）的矩阵无关方向导数。每次调用都用 ``context.candidate_id`` 从 ``x``
    重建候选几何、算子、界面几何与界面载荷，不缓存、不修改任何输入。
    """
    if mode not in ("residual", "assembled", "jvp"):
        raise ValueError(f"Unknown mode {mode!r}; allowed modes are 'residual', 'assembled' and 'jvp'.")
    if not isinstance(context, MonolithicContext):
        raise TypeError("context must be a MonolithicContext.")
    values = _finite_vector(x, context.layout.n_total, "x")
    if mode == "jvp" and directions is None:
        raise ValueError("mode='jvp' requires directions.")
    if mode != "jvp" and directions is not None:
        raise ValueError("directions are only used with mode='jvp'.")

    started = time.perf_counter()
    evaluation_started = started
    candidate = _evaluate_candidate(values, context)
    candidate_seconds = time.perf_counter()-started

    started = time.perf_counter()
    physics = _physics(context, candidate, context.layout.unpack(values)["d_s"])
    physics_seconds = time.perf_counter()-started

    started = time.perf_counter()
    residual = _residual_vector(values, context, candidate, physics)
    residual_seconds = time.perf_counter()-started

    jacobian = None
    jvp_values = None
    jacobian_seconds = 0.0
    jvp_seconds = 0.0
    if mode == "assembled":
        started = time.perf_counter()
        jacobian = _assembled_jacobian(context, candidate, physics)
        jacobian_seconds = time.perf_counter()-started
    elif mode == "jvp":
        started = time.perf_counter()
        raw = np.asarray(directions, dtype=float)
        batch = raw.reshape(1, -1) if raw.ndim == 1 else raw
        if batch.ndim != 2 or batch.shape[1] != context.layout.n_total or not np.all(np.isfinite(batch)):
            raise ValueError(f"directions must be ({context.layout.n_total},) or (batch, "
                             f"{context.layout.n_total}) and finite, got {raw.shape}.")
        rows = [_jvp_single(context, candidate, physics, row) for row in batch]
        jvp_values = np.stack(rows)
        if raw.ndim == 1:
            jvp_values = jvp_values[0]
        jvp_seconds = time.perf_counter()-started

    timings = dict(candidate.seconds)
    timings.update({"candidate_seconds": candidate_seconds, "physics_seconds": physics_seconds,
                    "residual_seconds": residual_seconds, "jacobian_seconds": jacobian_seconds,
                    "jvp_seconds": jvp_seconds, "total_seconds": time.perf_counter()-evaluation_started})
    return MonolithicEvaluation(
        residual=_freeze_array(np.array(residual, copy=True)),
        jacobian=jacobian,
        jvp_values=jvp_values,
        candidate_id=context.candidate_id,
        block_norms=_block_norms(values, residual, context.layout),
        geometry_diagnostics=_geometry_diagnostics(candidate),
        timings=timings,
    )


def monolithic_residual(x, *, context):
    """Global constrained residual ``R(x)`` (read-only array).

    中文：返回全局约束残差 ``R(x)``（只读数组），等价于 ``mode="residual"`` 求值。
    """
    return evaluate_monolithic(x, context=context, mode="residual").residual


def monolithic_jacobian(x, *, context):
    """Complete coupling Jacobian ``dR/dx`` as a frozen CSR matrix.

    中文：返回完整耦合 Jacobian ``dR/dx``（冻结的 CSR 矩阵），等价于 ``mode="assembled"``。
    """
    return evaluate_monolithic(x, context=context, mode="assembled").jacobian


def monolithic_jvp(x, directions, *, context):
    """Matrix-free directional derivative of ``R`` at ``x``.

    ``directions`` is ``(n,)`` or ``(batch, n)``; the output has the same shape.

    中文：返回 ``R`` 在 ``x`` 处、沿 ``directions`` 的矩阵无关方向导数；``directions`` 为
    ``(n,)`` 或 ``(batch, n)``，输出与输入同形状。
    """
    values = np.asarray(directions, dtype=float)
    if values.ndim not in (1, 2):
        raise ValueError(f"directions must be (n,) or (batch, n), got {values.shape}.")
    return evaluate_monolithic(x, context=context, mode="jvp", directions=values).jvp_values

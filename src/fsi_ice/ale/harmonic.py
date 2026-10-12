"""Harmonic (Laplace) extension of prescribed P1 boundary displacements.

本模块用调和（Laplace）方程把给定的 P1 边界位移扩展到内部节点（网格位移 d_m）。关键约定：
- 求解 -Div(diffusion grad(d_m)) = 0，两个位移分量各解一次；
- 参考网格上的 Laplace 刚度矩阵只装配一次，候选步复用（extend）；
- 必须给出所有外部边界顶点的位移，否则拒绝；
- boundary_error 用冻结的验证长度尺度 L* = 1 m 归一（见 m3_freeze.md 第 6 节）。
"""

from dataclasses import dataclass

import numpy as np

from fsi_ice.fem.poisson import assemble_poisson, solve_dirichlet
from fsi_ice.mesh import TriMesh

# 冻结的验证长度尺度 L* = 1 m（m3_freeze.md 第 6 节），用于归一化边界误差。
LENGTH_SCALE = 1.0  # Frozen validation length scale L*=1 m (m3_freeze.md section 6).


@dataclass(frozen=True)
class MeshDisplacement:
    """Absolute reference-domain P1 displacement d_m with its solve diagnostics.

    参考域上的绝对 P1 位移 d_m（形状 (n_P1_nodes, 2)）及其求解诊断：两个分量的相对残差、
    边界施加误差（按 LENGTH_SCALE 归一）与被约束的边界节点。所有数组只读。
    """

    values: np.ndarray
    component_residuals: np.ndarray
    boundary_error: float
    boundary_nodes: np.ndarray


class HarmonicExtension:
    """Reusable Laplace stiffness on one reference mesh for both P1 components.

    在一个参考网格上可复用的调和扩展器：只装配一次 P1 Laplace 刚度 K_h（可读只读 property
    stiffness），extend 用同一矩阵对两个位移分量分别求解 Dirichlet 问题。diffusion 为
    扩散系数（有限正标量）。输入：参考网格与 diffusion；输出：MeshDisplacement。
    """

    def __init__(self, reference_mesh, *, diffusion=1.0):
        if not isinstance(reference_mesh, TriMesh):
            raise ValueError("reference_mesh must be a TriMesh.")
        if isinstance(diffusion, bool):
            raise ValueError("diffusion must be a finite positive scalar.")
        try:
            value = float(diffusion)
        except (TypeError, ValueError):
            raise ValueError("diffusion must be a finite positive scalar.") from None
        if not np.isfinite(value) or value <= 0:
            raise ValueError("diffusion must be a finite positive scalar.")
        self.reference_mesh = reference_mesh
        self.diffusion = value
        # 源项为 0 的 Laplace 问题：只装配刚度与零右端，两个分量共用。
        self._matrix, self._rhs = assemble_poisson(reference_mesh, 0.0, value)
        self._matrix.data.setflags(write=False)
        self._matrix.indices.setflags(write=False)
        self._matrix.indptr.setflags(write=False)
        self._rhs.setflags(write=False)

    @property
    def stiffness(self):
        """Read-only P1 Laplace stiffness ``K_h`` assembled on the reference mesh.

        This is the same matrix :meth:`extend` solves with (zero source).  The
        arrays are frozen, so a caller can reuse it directly, e.g. as the mesh
        block of the monolithic residual, without copying or risking an
        accidental modification of the shared extension operator.

        中文补充：只读的参考网格 P1 Laplace 刚度 K_h，与 extend 求解所用矩阵相同（零源项）。
        数组已冻结，调用方可直接复用（例如作为整体残差的网格块）而无需复制或担心被意外修改。
        """
        return self._matrix

    def extend(self, boundary_nodes, displacement_values):
        """Solve -Div(diffusion grad(d_m)) = 0 with the given exterior values.

        在给定外部边界位移下求解 -Div(diffusion grad(d_m)) = 0。boundary_nodes 必须唯一、
        覆盖所有外部边界顶点；displacement_values 形状 (len(nodes), 2)。两个分量分别用
        solve_dirichlet 求解，返回 MeshDisplacement（含残差与按 LENGTH_SCALE 归一的边界误差）。
        """
        mesh = self.reference_mesh
        raw_nodes = np.asarray(boundary_nodes)
        if raw_nodes.ndim != 1 or raw_nodes.size == 0 or not np.issubdtype(raw_nodes.dtype, np.integer):
            raise ValueError("Boundary nodes must be a nonempty one-dimensional integer array of P1 vertices.")
        nodes = np.array(raw_nodes, dtype=np.int64)
        if np.any(nodes < 0) or np.any(nodes >= mesh.n_nodes):
            raise ValueError(f"Unknown boundary node; valid P1 vertices are 0..{mesh.n_nodes - 1}.")
        if len(np.unique(nodes)) != len(nodes):
            raise ValueError("Boundary nodes must be unique; duplicate or conflicting prescriptions are rejected.")
        exterior = mesh.boundary_nodes()
        missing = np.setdiff1d(exterior, nodes)
        if missing.size:
            raise ValueError(
                "Missing exterior boundary node(s) "
                f"{missing[:8].tolist() if missing.size > 8 else missing.tolist()}; "
                "every external boundary vertex must be prescribed."
            )
        values = np.asarray(displacement_values, dtype=float)
        if values.shape != (len(nodes), 2):
            raise ValueError(f"displacement_values must have shape ({len(nodes)}, 2), got {values.shape}.")
        if not np.all(np.isfinite(values)):
            raise ValueError("displacement_values must be finite.")
        solution = np.empty((mesh.n_nodes, 2))
        residuals = np.empty(2)
        # 两个位移分量共用同一刚度矩阵，各求解一次 Dirichlet 问题。
        for component in range(2):
            solved = solve_dirichlet(self._matrix, self._rhs, nodes, values[:, component])
            solution[:, component] = solved.values
            residuals[component] = solved.relative_residual
        # 边界施加误差：|求解值 - 给定值| 的最大值，按 L* 归一。
        boundary_error = float(np.max(np.abs(solution[nodes] - values)) / LENGTH_SCALE)
        solution.setflags(write=False)
        residuals.setflags(write=False)
        prescribed = np.array(nodes, copy=True)
        prescribed.setflags(write=False)
        return MeshDisplacement(solution, residuals, boundary_error, prescribed)


def build_harmonic_extension(reference_mesh, *, diffusion=1.0):
    """Assemble the reference Laplace stiffness once; extend() solves candidates.

    在参考网格上一次性装配 Laplace 刚度；之后对每个候选调用 extend() 求解。
    输入：参考网格与 diffusion；输出：HarmonicExtension。
    """
    return HarmonicExtension(reference_mesh, diffusion=diffusion)

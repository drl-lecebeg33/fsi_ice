"""Full Newton systems, solved by GMRES with an adaptively reused LU preconditioner.

本模块提供全 Newton 线性系统求解器 ReusedLUSolver：用 GMRES 迭代求解，并以可自适应复用
的稀疏 LU 分解作为右预条件子。关键约定：
- 仅在自由自由度（剔除约束）上求解；约束自由度的修正量恒为 0；
- 压力以运动学压力 q = p/rho 求解，并把动量行按 1/rho 缩放，使速度/压力量纲平衡而不改变物理方程；
- LU 分解在自由度集合变化、时间系数/几何签名变化或复用次数达到阈值时刷新；
- 每次求解后都用原始（未缩放）矩阵校验真实残差，失败时退回直接 LU 求解。
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres, splu


class ReusedLUSolver:
    """GMRES Newton solver with an adaptively reused sparse LU preconditioner.

    用 GMRES 求解（约束后的）Newton 系统的线性修正量，LU 分解作为预条件子并按需刷新。
    velocity_dofs 为速度自由度上界，用于把压力列缩放 rho、动量行缩放 1/rho；
    density 为流体密度 rho；refresh_interval 为刷新前最大复用次数。solve 返回完整修正向量。
    """

    def __init__(self, *, refresh_interval=20, velocity_dofs=None, density=1.0):
        if not isinstance(refresh_interval, int) or isinstance(refresh_interval, bool) or refresh_interval < 1:
            raise ValueError("LU refresh interval must be a positive integer.")
        self.refresh_interval = refresh_interval
        if not np.isfinite(density) or density <= 0 or (velocity_dofs is not None and
                (not isinstance(velocity_dofs, int) or velocity_dofs < 1)):
            raise ValueError("Invalid pressure scaling parameters.")
        self.velocity_dofs = velocity_dofs
        self.density = density
        self._factor = None
        self._free = None
        self._uses = 0
        self.factorizations = 0
        self.iterations = 0
        self._time_coefficient = None
        self._signature = None

    def prepare(self, time_coefficient, *, signature=None):
        """Reset the cached factorization when the time coefficient or geometry signature changes.

        当时间系数或几何/约束签名发生变化时丢弃缓存的 LU 分解，下一次 solve 将重新分解。
        """
        if self._time_coefficient != time_coefficient or (signature is not None and signature != self._signature):
            self._factor = None
        self._time_coefficient = time_coefficient
        self._signature = signature

    def solve(self, matrix, rhs, constrained, *, signature=None):
        """Solve one Newton correction on the free DOFs and return the full correction vector.

        在自由自由度上求解 Newton 修正并返回完整长度的修正向量（约束位置为 0）。会校验矩阵、
        右端项与约束；若 GMRES 失败或真实残差不达标则退回直接 LU 求解。signature 须与 prepare 一致。
        """
        if signature is not None and self._signature is not None and signature != self._signature:
            raise ValueError("LU solver signature does not match the prepared geometry/constraints; "
                             "prepare the current candidate before solving.")
        n = matrix.shape[0]
        constrained = np.asarray(constrained)
        if constrained.ndim != 1 or not np.issubdtype(constrained.dtype, np.integer) or np.any(constrained < 0) or np.any(constrained >= n):
            raise ValueError("Invalid homogeneous constraints.")
        rhs = np.asarray(rhs, dtype=float)
        if matrix.shape != (n, n) or rhs.shape != (n,) or not np.all(np.isfinite(rhs)) or not np.all(np.isfinite(matrix.data)):
            raise ValueError("Invalid Newton linear system.")
        free = np.setdiff1d(np.arange(n), constrained)
        # 仅在自由自由度上取子矩阵/子向量；约束自由度的修正量固定为 0。
        physical_matrix = matrix[free][:, free].tocsr()
        physical_load = rhs[free]
        # Solve for kinematic pressure q=p/rho and divide momentum rows by rho.
        # The transformation balances velocity/pressure units without changing
        # the physical Newton equation, which is checked again below.
        row_scale = np.ones(len(free))
        column_scale = np.ones(len(free))
        if self.velocity_dofs is not None:
            row_scale[free < self.velocity_dofs] = 1/self.density
            column_scale[free >= self.velocity_dofs] = self.density
        reduced = physical_matrix.multiply(row_scale[:, None]).multiply(column_scale[None, :]).tocsr()
        load = row_scale*physical_load
        correction = np.zeros(n)
        if not len(free):
            # 全部自由度都被约束：修正量为零，直接返回。
            return correction

        def refactor():
            try:
                self._factor = splu(reduced.tocsc())
            except RuntimeError as error:
                raise RuntimeError("Newton linear factorization failed.") from error
            self._free = free.copy()
            self._uses = 0
            self.factorizations += 1

        if self._factor is None or not np.array_equal(free, self._free) or self._uses >= self.refresh_interval:
            refactor()
        # 用（约化后的）LU 分解作为 GMRES 右预条件子。
        preconditioner = LinearOperator(reduced.shape, matvec=self._factor.solve)
        counts = []
        values, info = gmres(reduced, load, M=preconditioner, rtol=1e-11, atol=1e-12/max(self.density, 1),
                             restart=20, maxiter=3, callback=counts.append, callback_type="pr_norm")
        self.iterations += len(counts)
        defect = np.linalg.norm(physical_matrix @ (column_scale*values)-physical_load)/max(np.linalg.norm(physical_load), 1.0)
        if info or not np.all(np.isfinite(values)) or defect > 1e-10:
            # 用原始（未缩放）矩阵校验真实残差；不达标则退回直接 LU 求解。
            refactor()
            values = self._factor.solve(load)
            defect = np.linalg.norm(physical_matrix @ (column_scale*values)-physical_load)/max(np.linalg.norm(physical_load), 1.0)
            if not np.all(np.isfinite(values)) or defect > 1e-10:
                raise RuntimeError(f"Newton linear solve failed the true residual check: {defect:.3e}.")
        self._uses += 1
        if len(counts) > 15:
            # GMRES 迭代次数过多说明预条件质量下降，强制下次求解刷新分解。
            self._uses = self.refresh_interval
        correction[free] = column_scale*values
        return correction

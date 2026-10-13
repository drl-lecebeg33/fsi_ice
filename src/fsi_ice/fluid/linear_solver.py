"""Full Newton systems, solved by GMRES with an adaptively reused LU preconditioner."""

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres, splu


class ReusedLUSolver:
    def __init__(self, *, refresh_interval=20, velocity_dofs=None, density=1.0,
                 reuse_across_candidates=False):
        if not isinstance(refresh_interval, int) or isinstance(refresh_interval, bool) or refresh_interval < 1:
            raise ValueError("LU refresh interval must be a positive integer.")
        self.refresh_interval = refresh_interval
        if not np.isfinite(density) or density <= 0 or (velocity_dofs is not None and
                (not isinstance(velocity_dofs, int) or velocity_dofs < 1)):
            raise ValueError("Invalid pressure scaling parameters.")
        self.velocity_dofs = velocity_dofs
        self.density = density
        if not isinstance(reuse_across_candidates, bool):
            raise ValueError("reuse_across_candidates must be a bool.")
        self.reuse_across_candidates = reuse_across_candidates
        self._factor = None
        self._free = None
        self._uses = 0
        self.factorizations = 0
        self.iterations = 0
        self._time_coefficient = None
        self._signature = None

    def prepare(self, time_coefficient, *, signature=None):
        # A factor from another candidate may only serve as a preconditioner
        # for GMRES on the *current* matrix. solve() checks the true current
        # residual and refactors on any failure. The default keeps the older
        # strict invalidation behavior used outside the FSI2 step workspace.
        if self._time_coefficient != time_coefficient or (not self.reuse_across_candidates and
                signature is not None and signature != self._signature):
            self._factor = None
        self._time_coefficient = time_coefficient
        self._signature = signature

    def solve(self, matrix, rhs, constrained, *, signature=None):
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
        preconditioner = LinearOperator(reduced.shape, matvec=self._factor.solve)
        counts = []
        values, info = gmres(reduced, load, M=preconditioner, rtol=1e-11, atol=1e-12/max(self.density, 1),
                             restart=20, maxiter=3, callback=counts.append, callback_type="pr_norm")
        self.iterations += len(counts)
        defect = np.linalg.norm(physical_matrix @ (column_scale*values)-physical_load)/max(np.linalg.norm(physical_load), 1.0)
        if info or not np.all(np.isfinite(values)) or defect > 1e-10:
            refactor()
            values = self._factor.solve(load)
            defect = np.linalg.norm(physical_matrix @ (column_scale*values)-physical_load)/max(np.linalg.norm(physical_load), 1.0)
            if not np.all(np.isfinite(values)) or defect > 1e-10:
                raise RuntimeError(f"Newton linear solve failed the true residual check: {defect:.3e}.")
        self._uses += 1
        if len(counts) > 15:
            self._uses = self.refresh_interval
        correction[free] = column_scale*values
        return correction

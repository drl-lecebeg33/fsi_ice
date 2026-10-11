"""Partitioned coupling over shared single-field solvers and interface maps."""

from .partitioned import (CouplingFailure, CouplingIteration, CouplingOptions,
                          CouplingResult, DirichletNeumannMap, PartitionedEvaluation,
                          solve_dirichlet_neumann)

__all__ = ["CouplingFailure", "CouplingIteration", "CouplingOptions", "CouplingResult",
           "DirichletNeumannMap", "PartitionedEvaluation", "solve_dirichlet_neumann"]

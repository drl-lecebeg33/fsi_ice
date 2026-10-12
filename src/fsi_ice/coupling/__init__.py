"""Partitioned coupling over shared single-field solvers and interface maps."""

from .partitioned import (CouplingFailure, CouplingIteration, CouplingOptions,
                          CouplingResult, DirichletNeumannMap, PartitionedEvaluation,
                          solve_dirichlet_neumann)
from .monolithic import (APPROXIMATIONS, BLOCK_NAMES, JACOBIAN_BLOCK_SOURCES, BlockLayout,
                         MonolithicBase, MonolithicContext, MonolithicEvaluation, PressurePin,
                         build_block_layout, evaluate_monolithic, monolithic_jacobian,
                         monolithic_jvp, monolithic_residual, require_candidate)

__all__ = ["CouplingFailure", "CouplingIteration", "CouplingOptions", "CouplingResult",
           "DirichletNeumannMap", "PartitionedEvaluation", "solve_dirichlet_neumann",
           "APPROXIMATIONS", "BLOCK_NAMES", "JACOBIAN_BLOCK_SOURCES", "BlockLayout", "MonolithicBase",
           "MonolithicContext", "MonolithicEvaluation", "PressurePin", "build_block_layout",
           "evaluate_monolithic", "monolithic_jacobian", "monolithic_jvp", "monolithic_residual",
           "require_candidate"]

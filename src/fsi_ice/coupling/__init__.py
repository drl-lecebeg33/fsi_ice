"""Partitioned coupling over shared single-field solvers and interface maps."""

from .partitioned import (CouplingFailure, CouplingIteration, CouplingOptions,
                          CouplingResult, DirichletNeumannMap, PartitionedEvaluation,
                          solve_dirichlet_neumann)
from .monolithic import (APPROXIMATIONS, BLOCK_NAMES, JACOBIAN_BLOCK_SOURCES, BlockLayout,
                         MonolithicBase, MonolithicContext, MonolithicEvaluation, PressurePin,
                         build_block_layout, evaluate_monolithic, monolithic_jacobian,
                         monolithic_jvp, monolithic_residual, require_candidate)
from .newton import (Evaluation, Evaluator, NewtonFailure, NewtonIterationRecord, NewtonOptions,
                     NewtonResult, NewtonTelemetry, solve_newton)
from .monolithic_newton import (JACOBIAN_MODES, MonolithicCounters, MonolithicMarchResult,
                                MonolithicMarchStep, MonolithicNewtonFailure, MonolithicStepResult,
                                interface_compatibility_gap, march_to_steady,
                                project_interface_compatibility, scaled_time_terms,
                                solve_monolithic_step)

__all__ = ["CouplingFailure", "CouplingIteration", "CouplingOptions", "CouplingResult",
           "DirichletNeumannMap", "PartitionedEvaluation", "solve_dirichlet_neumann",
           "APPROXIMATIONS", "BLOCK_NAMES", "JACOBIAN_BLOCK_SOURCES", "BlockLayout", "MonolithicBase",
           "MonolithicContext", "MonolithicEvaluation", "PressurePin", "build_block_layout",
           "evaluate_monolithic", "monolithic_jacobian", "monolithic_jvp", "monolithic_residual",
           "require_candidate",
           "Evaluation", "Evaluator", "NewtonFailure", "NewtonIterationRecord", "NewtonOptions",
           "NewtonResult", "NewtonTelemetry", "solve_newton",
           "JACOBIAN_MODES", "MonolithicCounters", "MonolithicMarchResult", "MonolithicMarchStep",
           "MonolithicNewtonFailure", "MonolithicStepResult", "interface_compatibility_gap",
           "march_to_steady", "project_interface_compatibility", "scaled_time_terms",
           "solve_monolithic_step"]

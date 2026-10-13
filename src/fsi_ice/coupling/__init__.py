"""Partitioned coupling over shared single-field solvers and interface maps.

中文说明：本包是流固耦合（FSI）求解的统一入口，本身不含求解逻辑，只聚合导出各子模块的
公共接口。分区耦合（``partitioned``）在共享的单场求解器与界面映射之上做迭代耦合；整体
（单体）耦合由 ``monolithic`` 提供统一残差块划分与完整耦合 Jacobian（来源见
``JACOBIAN_BLOCK_SOURCES``，唯一近似由 ``APPROXIMATIONS`` 标注）以及矩阵无关 JVP，由
``monolithic_newton`` 在此基础上做阻尼 Newton 迭代与伪时间定常推进。所有未知量按
``BLOCK_NAMES`` 固定分块拼成一个全局向量。此文件改动仅影响导出，不改变任何行为。
"""

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

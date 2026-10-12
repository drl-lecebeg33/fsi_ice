"""BDF fluid step: candidate solves never modify accepted velocity history.

本模块封装单个 BDF 流场时间步的求解入口 solve_flow_step：把时间上下文（context）
转换/校验后交给 solve_navier_stokes。关键约定：候选求解绝不修改已接受的速度历史；
固定网格分支把历史项作为右端时间项 -(M·history) 传入，ALE 分支则转交 ALE 上下文
（并要求速度 BDF 上下文与 ALE 模板描述同一步）。
"""

from fsi_ice.fluid.taylor_hood import solve_navier_stokes


def solve_flow_step(operators, velocity, context, *, ale_context=None, **kwargs):
    """Solve one BDF fluid time step from a frozen time context (fixed-grid or ALE).

    求解一个 BDF 流场时间步。context.history_term 必须为 (n_P2, 2) 的历史速度。
    固定网格分支把历史项作为右端 -(M·history_term) 传入 solve_navier_stokes；
    传入 ale_context 时先校验一致性并走 ALE 分支。其余关键字参数透传。返回 FlowSolution。
    """
    if context.history_term.shape != (len(operators.space.points), 2):
        raise ValueError("Flow history must contain the P2 nodal velocity.")
    if ale_context is None:
        # 固定网格：历史项移到右端，等价于 a_0 M u - M·history。
        history_rhs = -(operators.mass @ context.history_term.ravel())
        return solve_navier_stokes(operators, velocity, time_coefficient=context.coefficient,
                                   time_rhs=history_rhs, **kwargs)
    from fsi_ice.fluid.ale import validate_ale_velocity_context

    validate_ale_velocity_context(ale_context, context)
    return solve_navier_stokes(operators, velocity, ale_context=ale_context, **kwargs)

"""BDF fluid step: candidate solves never modify accepted velocity history."""

from fsi_ice.fluid.taylor_hood import solve_navier_stokes


def solve_flow_step(operators, velocity, context, *, ale_context=None, **kwargs):
    if context.history_term.shape != (len(operators.space.points), 2):
        raise ValueError("Flow history must contain the P2 nodal velocity.")
    if ale_context is None:
        history_rhs = -(operators.mass @ context.history_term.ravel())
        return solve_navier_stokes(operators, velocity, time_coefficient=context.coefficient,
                                   time_rhs=history_rhs, **kwargs)
    from fsi_ice.fluid.ale import validate_ale_velocity_context

    validate_ale_velocity_context(ale_context, context)
    return solve_navier_stokes(operators, velocity, ale_context=ale_context, **kwargs)

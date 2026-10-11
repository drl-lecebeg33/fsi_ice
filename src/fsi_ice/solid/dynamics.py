"""BDF discretization of d'=v, rho*v'=Div(P)+f on a fixed reference mesh."""

import numpy as np

from fsi_ice.solid.stvk import assemble_mass, solve_stvk


def solve_solid_step(mesh, material, external, fixed_nodes, context, *, initial=None, **kwargs):
    if context.history_term.shape != (2, mesh.n_nodes, 2):
        raise ValueError("Solid history shape must be (2,n_nodes,2) for displacement and velocity.")
    alpha = context.coefficient
    hd, hv = context.history_term
    mass = assemble_mass(mesh, material)
    # Eliminating v gives acceleration alpha^2*d + alpha*hd + hv.
    history_rhs = -(mass @ (alpha*hd+hv).ravel())
    solution = solve_stvk(mesh, material, external, fixed_nodes, initial=initial,
                          mass_coefficient=alpha**2, mass_rhs=history_rhs, **kwargs)
    velocity = alpha*solution.displacement+hd
    return np.stack((solution.displacement, velocity)), solution

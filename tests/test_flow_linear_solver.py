import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fsi_ice.fem.poisson import solve_dirichlet
from fsi_ice.fluid.linear_solver import ReusedLUSolver
from fsi_ice.fluid.taylor_hood import assemble_stokes, flow_constraints, navier_stokes_residual
from fsi_ice.fluid.dynamics import solve_flow_step
from fsi_ice.mesh import rectangle_mesh
from fsi_ice.time import BDFHistory
from fsi_ice.validation.fluid import exact_velocity, manufactured_force


def test_reused_factor_solves_current_jacobian_and_refreshes():
    operators = assemble_stokes(rectangle_mesh(4), density=1000)
    fixed, _ = flow_constraints(operators, [0, 0])
    rng = np.random.default_rng(51)
    solver = ReusedLUSolver(refresh_interval=3, velocity_dofs=operators.n_velocity, density=1000)
    for _ in range(4):
        state = 0.01*rng.normal(size=operators.n_total)
        _, matrix = navier_stokes_residual(operators, state, time_coefficient=100,
                                          time_rhs=np.zeros(operators.n_velocity))
        rhs = rng.normal(size=operators.n_total)
        expected = solve_dirichlet(matrix, rhs, fixed, 0).values
        actual = solver.solve(matrix, rhs, fixed)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-8)
    assert solver.factorizations >= 2
    with pytest.raises(RuntimeError, match="factorization"):
        ReusedLUSolver().solve(csr_matrix((3, 3)), np.ones(3), np.array([0]))


def test_transient_flow_direct_and_reused_factor_reach_same_solution():
    mesh = rectangle_mesh(4)
    base = assemble_stokes(mesh, body_force=manufactured_force)
    histories = [BDFHistory(exact_velocity(base.space.points), 0.05) for _ in range(2)]
    guesses = [None, None]
    solver = ReusedLUSolver()
    for step in range(5):
        amplitude = np.exp(-(step+1)*0.05)
        operators = assemble_stokes(mesh, body_force=lambda points: amplitude*manufactured_force(points))
        results = []
        for index, history in enumerate(histories):
            context = history.context()
            solution = solve_flow_step(operators, lambda points: amplitude*exact_velocity(points), context,
                                      initial=guesses[index], tolerance=1e-11,
                                      linear_solver=solver if index else None)
            guesses[index] = np.r_[solution.velocity.ravel(), solution.pressure]
            history.accept(context, solution.velocity)
            results.append(guesses[index])
        np.testing.assert_allclose(results[0], results[1], atol=1e-9, rtol=1e-9)
    assert solver.factorizations < 10


def test_residual_only_matches_full_operator():
    operators = assemble_stokes(rectangle_mesh(2))
    state = np.random.default_rng(75).normal(size=operators.n_total)
    full, matrix = navier_stokes_residual(operators, state)
    only, absent = navier_stokes_residual(operators, state, with_jacobian=False)
    np.testing.assert_array_equal(full, only)
    assert absent is None and matrix.shape == (operators.n_total,)*2

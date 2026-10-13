import numpy as np
import pytest

from fsi_ice.fem.p2 import P2Space, p2_reference_gradients, p2_values
from fsi_ice.fem.quadrature import triangle_quadrature
from fsi_ice.fluid.taylor_hood import (assemble_stokes, convection, flow_error_norms,
                                     navier_stokes_residual, solve_navier_stokes, solve_stokes)
from fsi_ice.mesh import rectangle_mesh


def test_p2_nodal_basis_shared_edges_and_polynomial_reproduction():
    nodes = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]])
    np.testing.assert_allclose(p2_values(nodes), np.eye(6), atol=1e-15)
    points, _ = triangle_quadrature(5)
    np.testing.assert_allclose(p2_values(points).sum(axis=1), 1, atol=2e-15)
    np.testing.assert_allclose(p2_reference_gradients(points).sum(axis=1), 0, atol=2e-15)
    polynomial = lambda x: 2+3*x[:, 0]-x[:, 1]+x[:, 0]**2+2*x[:, 0]*x[:, 1]+4*x[:, 1]**2
    np.testing.assert_allclose(p2_values(points) @ polynomial(nodes), polynomial(points), atol=4e-15)
    space = P2Space.from_mesh(rectangle_mesh(2))
    assert len(space.points) == 25
    assert len(space.boundary_nodes()) == 16
    assert len(space.boundary_nodes("left")) == 5
    for edge, midpoint in space.edge_nodes.items():
        np.testing.assert_allclose(space.points[midpoint], space.mesh.points[list(edge)].mean(axis=0))


def test_poiseuille_exact_velocity_pressure_and_full_continuity():
    mesh = rectangle_mesh(4, 3, width=2, height=1)
    operators = assemble_stokes(mesh, viscosity=2.0)
    velocity = lambda x: np.column_stack((x[:, 1]*(1-x[:, 1])/4, np.zeros(len(x))))
    pressure = lambda x: 2-x[:, 0]
    solution = solve_stokes(operators, velocity, pressure_pin=0, pressure_value=2)
    errors = flow_error_norms(operators, solution, velocity, pressure)
    assert errors["velocity_l2"] < 1e-12 and errors["pressure_l2"] < 1e-11
    assert solution.continuity_residual < 1e-13
    np.testing.assert_allclose(operators.matrix.toarray(), operators.matrix.T.toarray(), atol=1e-14)


def test_convection_and_mixed_jacobian_directional_derivative():
    operators = assemble_stokes(rectangle_mesh(3), density=2.0)
    rng = np.random.default_rng(31)
    state, direction = rng.normal(size=(2, operators.n_total))
    residual, jacobian = navier_stokes_residual(operators, state)
    shifted, _ = navier_stokes_residual(operators, state+1e-7*direction)
    reference = jacobian @ direction
    assert np.linalg.norm((shifted-residual)/1e-7-reference)/np.linalg.norm(reference) < 1e-7
    zero, _ = convection(operators, np.tile([1, 0], (len(operators.space.points), 1)))
    np.testing.assert_allclose(zero, 0, atol=1e-14)


def test_navier_stokes_poiseuille_and_failed_newton_are_explicit():
    operators = assemble_stokes(rectangle_mesh(4), viscosity=0.1, density=10)
    velocity = lambda x: np.column_stack((x[:, 1]*(1-x[:, 1]), np.zeros(len(x))))
    solution = solve_navier_stokes(operators, velocity, pressure_value=0.2)
    np.testing.assert_allclose(solution.velocity, velocity(operators.space.points), atol=2e-12)
    assert solution.relative_residual < 1e-10
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_navier_stokes(operators, velocity, pressure_value=0.2,
                            initial=np.ones(operators.n_total), max_iterations=1, tolerance=1e-14)


def test_navier_stokes_absolute_bound_controls_raw_stop():
    operators = assemble_stokes(rectangle_mesh(4), viscosity=0.1, density=10)
    velocity = lambda x: np.column_stack((x[:, 1]*(1-x[:, 1]), np.zeros(len(x))))
    expected = velocity(operators.space.points)
    initial = np.ones(operators.n_total)
    # A loose relative tolerance alone stops while the raw residual is still large.
    loose = solve_navier_stokes(operators, velocity, pressure_value=0.2, initial=initial, tolerance=1.0)
    assert np.linalg.norm(loose.velocity-expected) > .5
    # The absolute bound keeps iterating until the raw free residual is small.
    exact = solve_navier_stokes(operators, velocity, pressure_value=0.2, initial=initial,
                                tolerance=1.0, absolute_tolerance=1e-12)
    assert len(exact.nonlinear_history) > len(loose.nonlinear_history)
    np.testing.assert_allclose(exact.velocity, expected, atol=2e-12)
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_navier_stokes(operators, velocity, pressure_value=0.2, initial=initial,
                            tolerance=1.0, absolute_tolerance=1e-20, max_iterations=4)
    with pytest.raises(ValueError, match="absolute_tolerance"):
        solve_navier_stokes(operators, velocity, pressure_value=0.2, absolute_tolerance=0.0)


def test_incompatible_boundary_flux_is_rejected():
    operators = assemble_stokes(rectangle_mesh(3))
    with pytest.raises(RuntimeError, match="incompressibility"):
        solve_stokes(operators, lambda x: x)

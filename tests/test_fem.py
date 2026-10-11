from math import factorial

import numpy as np
import pytest

from fsi_ice.fem.p1 import map_points, physical_gradients, shape_values
from fsi_ice.fem.poisson import assemble_poisson, solve_dirichlet
from fsi_ice.fem.quadrature import triangle_quadrature
from fsi_ice.mesh import TriMesh, rectangle_mesh


@pytest.mark.parametrize("degree", [1, 2, 5])
def test_quadrature_against_analytical_monomial_integrals(degree):
    points, weights = triangle_quadrature(degree)
    assert np.all(weights > 0)
    assert np.all(points >= 0) and np.all(points.sum(axis=1) <= 1)
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            exact = factorial(i) * factorial(j) / factorial(i + j + 2)
            measured = weights @ (points[:, 0]**i * points[:, 1]**j)
            np.testing.assert_allclose(measured, exact, rtol=2e-14, atol=1e-15)


def test_shape_and_affine_reproduction_on_a_skew_triangle():
    mesh = TriMesh([[0.2, 0.1], [2.3, 0.4], [0.7, 1.8]], [[0, 1, 2]])
    np.testing.assert_allclose(shape_values([[0, 0], [1, 0], [0, 1]]), np.eye(3))
    points, _ = triangle_quadrature(5)
    shapes = shape_values(points)
    np.testing.assert_allclose(shapes.sum(axis=1), 1.0, atol=1e-15)
    gradients = physical_gradients(mesh)
    np.testing.assert_allclose(gradients.sum(axis=1), 0.0, atol=1e-15)
    nodal = 2 + 3 * mesh.points[:, 0] - 4 * mesh.points[:, 1]
    mapped = map_points(mesh, points)[0]
    np.testing.assert_allclose(shapes @ nodal, 2 + 3 * mapped[:, 0] - 4 * mapped[:, 1], atol=2e-14)
    np.testing.assert_allclose(nodal @ gradients[0], [3, -4], atol=2e-14)


def test_element_stiffness_and_load_have_independent_analytical_values():
    mesh = TriMesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]])
    matrix, rhs = assemble_poisson(mesh, 6.0)
    expected = np.array([[1, -0.5, -0.5], [-0.5, 0.5, 0], [-0.5, 0, 0.5]])
    np.testing.assert_allclose(matrix.toarray(), expected, atol=2e-14)
    np.testing.assert_allclose(rhs, [1, 1, 1], atol=2e-14)


def test_sparse_assembly_preserves_symmetry_constant_nullspace_and_total_load():
    mesh = rectangle_mesh(4, 3, width=2, height=3)
    matrix, rhs = assemble_poisson(mesh, 2.0)
    np.testing.assert_allclose(matrix.toarray(), matrix.T.toarray(), atol=1e-14)
    np.testing.assert_allclose(matrix @ np.ones(mesh.n_nodes), 0, atol=1e-14)
    np.testing.assert_allclose(rhs.sum(), 12.0, atol=1e-13)
    assert np.linalg.eigvalsh(matrix.toarray()).min() > -1e-13
    assert matrix.nnz < mesh.n_nodes**2


def test_nonzero_unsorted_dirichlet_lifting_preserves_original_system():
    matrix = np.array([[4, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=float)
    rhs = np.array([12, 20, 16], dtype=float)
    original = matrix.copy()
    solution = solve_dirichlet(matrix, rhs, np.array([2, 0]), np.array([6, 2]))
    np.testing.assert_allclose(solution.values, [2, 4, 6], atol=1e-14)
    np.testing.assert_array_equal(matrix, original)
    assert solution.relative_residual < 1e-14


def test_all_prescribed_dofs_and_singular_free_system():
    solution = solve_dirichlet(np.eye(2), [0, 0], np.array([0, 1]), [2, 3])
    np.testing.assert_allclose(solution.values, [2, 3])
    assert solution.relative_residual == 0
    with pytest.raises(RuntimeError, match="singular"):
        solve_dirichlet(np.diag([1.0, 0.0]), [0, 0], np.array([0]), 0.0)


@pytest.mark.parametrize("source,diffusion", [(np.nan, 1), (0, 0), (0, -1), (0, np.inf)])
def test_invalid_fields_are_rejected(source, diffusion):
    with pytest.raises(ValueError):
        assemble_poisson(rectangle_mesh(2), source, diffusion)

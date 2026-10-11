import numpy as np
import pytest

from fsi_ice.mesh import rectangle_mesh
from fsi_ice.solid.linear import (ElasticMaterial, assemble_elasticity, linear_residual,
                                  solve_elasticity)


@pytest.mark.parametrize("model", ["plane_strain", "plane_stress"])
def test_linear_patch_and_known_stress(model):
    mesh = rectangle_mesh(4, 3, width=2, height=1)
    material = ElasticMaterial(200, 0.25, model)
    exact = lambda x: np.column_stack((0.01*x[:, 0]+0.02*x[:, 1]+0.1,
                                      -0.03*x[:, 0]+0.04*x[:, 1]-0.2))
    result = solve_elasticity(mesh, material, mesh.boundary_nodes(), exact)
    np.testing.assert_allclose(result.displacement, exact(mesh.points), atol=2e-14)
    np.testing.assert_allclose(result.strain, np.tile([0.01, 0.04, -0.01], (mesh.n_cells, 1)), atol=1e-14)
    np.testing.assert_allclose(result.stress, np.tile(material.constitutive @ [0.01, 0.04, -0.01],
                                                   (mesh.n_cells, 1)), atol=2e-12)


def test_rigid_modes_symmetry_and_energy():
    mesh = rectangle_mesh(3)
    matrix, _ = assemble_elasticity(mesh, ElasticMaterial(10, 0.3))
    np.testing.assert_allclose(matrix.toarray(), matrix.T.toarray(), atol=2e-14)
    for field in (np.tile([1, 0], (mesh.n_nodes, 1)), np.tile([0, 1], (mesh.n_nodes, 1)),
                  np.column_stack((-mesh.points[:, 1], mesh.points[:, 0]))):
        np.testing.assert_allclose(matrix @ field.ravel(), 0, atol=2e-14)
    eigenvalues = np.linalg.eigvalsh(matrix.toarray())
    assert np.count_nonzero(np.abs(eigenvalues) < 1e-12) == 3
    assert eigenvalues.min() > -1e-12


def test_boundary_load_and_clamp_reaction_balance():
    mesh = rectangle_mesh(16, 4, width=2, height=0.5)
    material = ElasticMaterial(1e4, 0.3, "plane_stress", thickness=0.2)
    _, rhs = assemble_elasticity(mesh, material, tractions={"right": [0, -10]})
    np.testing.assert_allclose(rhs.reshape(-1, 2).sum(axis=0), [0, -1], atol=1e-14)
    result = solve_elasticity(mesh, material, mesh.boundary_nodes("left"), tractions={"right": [0, -10]})
    np.testing.assert_allclose(result.reactions.sum(axis=0), [0, 1], atol=1e-11)
    # Out-of-plane moment z: x*Fy - y*Fx. Reaction torque cancels the right-edge load.
    torque = (mesh.points[:, 0]*result.reactions[:, 1] - mesh.points[:, 1]*result.reactions[:, 0]).sum()
    np.testing.assert_allclose(torque, 2.0, atol=2e-11)


def test_linear_jacobian_directional_derivative():
    mesh = rectangle_mesh(3)
    material = ElasticMaterial(100, 0.3)
    rng = np.random.default_rng(12)
    u, direction = rng.normal(size=(2, mesh.n_nodes, 2))
    residual, jacobian = linear_residual(mesh, material, u)
    shifted, _ = linear_residual(mesh, material, u+1e-6*direction)
    measured = (shifted-residual)/1e-6
    expected = jacobian @ direction.ravel()
    assert np.linalg.norm(measured-expected)/np.linalg.norm(expected) < 1e-8


@pytest.mark.parametrize("young,poisson", [(0, 0.3), (1, 0.5), (1, -1), (np.inf, 0.3)])
def test_invalid_material(young, poisson):
    with pytest.raises(ValueError):
        ElasticMaterial(young, poisson)

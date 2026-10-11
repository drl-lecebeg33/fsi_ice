import numpy as np
import pytest

from fsi_ice.mesh import rectangle_mesh
from fsi_ice.solid.linear import ElasticMaterial, assemble_elasticity
from fsi_ice.solid.stvk import assemble_mass, solve_static_continuation, solve_stvk, stvk_residual
from fsi_ice.solid.dynamics import solve_solid_step
from fsi_ice.time import BDFHistory


def test_stvk_objectivity_small_strain_and_jacobian_scan():
    mesh = rectangle_mesh(3)
    material = ElasticMaterial(100, 0.3)
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rigid = mesh.points @ (rotation-np.eye(2)).T + [0.5, -0.2]
    residual, _ = stvk_residual(mesh, material, rigid)
    assert np.linalg.norm(residual) < 1e-12
    stiffness, _ = assemble_elasticity(mesh, material)
    zero_residual, tangent = stvk_residual(mesh, material, np.zeros_like(mesh.points))
    np.testing.assert_allclose(tangent.toarray(), stiffness.toarray(), atol=1e-12)
    assert np.linalg.norm(zero_residual) == 0
    rng = np.random.default_rng(65)
    values = 0.01*rng.normal(size=mesh.points.shape)
    direction = rng.normal(size=values.shape)
    _, matrix = stvk_residual(mesh, material, values)
    expected = matrix @ direction.ravel()
    errors = []
    for eps in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        plus, _ = stvk_residual(mesh, material, values+eps*direction)
        minus, _ = stvk_residual(mesh, material, values-eps*direction)
        errors.append(np.linalg.norm((plus-minus)/(2*eps)-expected)/np.linalg.norm(expected))
    assert errors[2] < errors[0]/1000
    assert min(errors) < 1e-8
    np.testing.assert_allclose(matrix.toarray(), matrix.toarray().T, atol=1e-12)
    with pytest.raises(ValueError, match="Jacobian"):
        stvk_residual(mesh, material, mesh.points*[-2, 0])


def test_nonlinear_load_balance_and_failure():
    mesh = rectangle_mesh(10, 2, width=1, height=0.2)
    material = ElasticMaterial(1e4, 0.3)
    _, force = assemble_elasticity(mesh, material, body_force=[0, -10])
    fixed = mesh.boundary_nodes("left")
    result, histories = solve_static_continuation(mesh, material, force, fixed, load_steps=3)
    assert result.nonlinear_history[-1] < 1e-9
    assert len(histories) == 3 and result.minimum_jacobian > 0
    np.testing.assert_allclose(result.reactions.sum(axis=0), [0, 2], atol=1e-8)
    with pytest.raises(RuntimeError, match="failed after"):
        solve_stvk(mesh, material, force, fixed, max_iterations=1)


@pytest.mark.parametrize("order,minimum", [(1, 0.9), (2, 1.8)])
def test_bdf_ode_convergence(order, minimum):
    errors = []
    for dt in (0.1, 0.05, 0.025):
        history = BDFHistory(np.array([1.0]), dt, order=order)
        for _ in range(round(1/dt)):
            context = history.context()
            candidate = -context.history_term/(context.coefficient+1)
            history.accept(context, candidate)
        errors.append(abs(history.states[0][0]-np.exp(-1)))
    assert min(np.log2(np.array(errors[:-1])/errors[1:])) > minimum


def test_time_history_candidate_repeat_accept_restore():
    original = np.array([1.0, 2.0])
    history = BDFHistory(original, 0.1)
    original[:] = 9
    context = history.context()
    before = history.snapshot()
    assert context.order == 1
    np.testing.assert_array_equal(context.derivative([1, 2]), [0, 0])
    np.testing.assert_array_equal(history.states[0], [1, 2])
    exposed = history.states[0]
    exposed.flags.writeable = True
    exposed[:] = -1
    np.testing.assert_array_equal(history.states[0], [1, 2])
    history.accept(context, [1.1, 2.2])
    assert history.context().order == 2
    with pytest.raises(ValueError, match="stale"):
        history.accept(context, [1.1, 2.2])
    history.restore(before)
    assert history.time == 0
    np.testing.assert_array_equal(history.states[0], [1, 2])


def test_solid_dynamic_step_repeat_and_mass():
    mesh = rectangle_mesh(5, 2, width=1, height=0.2)
    material = ElasticMaterial(1e4, 0.3, density=10)
    mass = assemble_mass(mesh, material)
    np.testing.assert_allclose(mass @ np.tile([1, 0], mesh.n_nodes),
                               assemble_elasticity(mesh, material, body_force=[10, 0])[1], atol=1e-14)
    assert np.linalg.eigvalsh(mass.toarray()).min() > 0
    _, force = assemble_elasticity(mesh, material, body_force=[0, -10])
    history = BDFHistory(np.zeros((2, mesh.n_nodes, 2)), 0.01)
    context = history.context()
    first, result = solve_solid_step(mesh, material, force, mesh.boundary_nodes("left"), context)
    second, _ = solve_solid_step(mesh, material, force, mesh.boundary_nodes("left"), context)
    np.testing.assert_allclose(first, second, atol=0)
    np.testing.assert_array_equal(history.states[0], 0)
    assert result.nonlinear_history[-1] < 1e-9
    history.accept(context, first)
    assert history.context().order == 2

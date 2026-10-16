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


def test_stvk_absolute_bound_controls_raw_stop():
    mesh = rectangle_mesh(10, 2, width=1, height=0.2)
    material = ElasticMaterial(1e4, 0.3)
    _, force = assemble_elasticity(mesh, material, body_force=[0, -10])
    fixed = mesh.boundary_nodes("left")
    reference = solve_stvk(mesh, material, force, fixed)
    # A loose relative tolerance alone stops at the initial state.
    loose = solve_stvk(mesh, material, force, fixed, tolerance=1.0)
    assert np.linalg.norm(loose.displacement-reference.displacement) > 1e-3
    # The absolute bound keeps iterating down to the raw residual floor.
    exact = solve_stvk(mesh, material, force, fixed, tolerance=1.0, absolute_tolerance=1e-12)
    assert len(exact.nonlinear_history) > len(loose.nonlinear_history)
    np.testing.assert_allclose(exact.displacement, reference.displacement, atol=1e-14)
    with pytest.raises(RuntimeError, match="failed after"):
        solve_stvk(mesh, material, force, fixed, tolerance=1.0,
                   absolute_tolerance=1e-30, max_iterations=4)
    with pytest.raises(ValueError, match="absolute_tolerance"):
        solve_stvk(mesh, material, force, fixed, absolute_tolerance=-1.0)


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


@pytest.mark.parametrize("extension", [1e-8, 1e-12, 1e-16, -1e-12])
def test_stvk_tiny_affine_extension_retains_analytic_stress_and_energy(extension):
    # E_xx = extension + extension**2/2, evaluated without subtracting unit tensors.
    # This exercises precision required when tight FSI iterations perturb a small strain.
    from fsi_ice.solid.stvk import stvk_energy
    mesh = rectangle_mesh(3, 2, width=1., height=.2)
    material = ElasticMaterial(1.4e6, .4)
    displacement = np.column_stack((extension*mesh.points[:, 0], np.zeros(mesh.n_nodes)))
    residual, _ = stvk_residual(mesh, material, displacement)
    green_xx = extension + .5*extension**2
    lam, mu = material.lame
    exact_force = (1+extension)*(lam+2*mu)*green_xx*.2
    right_force = residual.reshape(-1, 2)[mesh.boundary_nodes("right"), 0].sum()
    assert right_force == pytest.approx(exact_force, rel=5e-13, abs=0.)
    exact_energy = (.5*lam+mu)*green_xx**2*.2
    assert stvk_energy(mesh, material, displacement, np.zeros(2*mesh.n_nodes)) == \
        pytest.approx(exact_energy, rel=5e-13, abs=0.)


def test_solid_correction_accuracy_prevents_loose_residual_early_stop(monkeypatch):
    import fsi_ice.solid.stvk as stvk
    mesh = rectangle_mesh(4, 2, width=1., height=.3)
    material = ElasticMaterial(1., .3)
    _, force = assemble_elasticity(mesh, material, body_force=[0., -1e-8])
    fixed = mesh.boundary_nodes('left')
    reference = solve_stvk(mesh, material, force, fixed, tolerance=1e-16)
    loose = solve_stvk(mesh, material, force, fixed, tolerance=1e-6, absolute_tolerance=1e-6)
    assert np.max(np.abs(loose.displacement-reference.displacement)) > 1e-8
    original, calls = stvk.solve_dirichlet, []
    def counted(*args, **kwargs):
        out = original(*args, **kwargs)
        calls.append(out.values)
        return out
    monkeypatch.setattr(stvk, 'solve_dirichlet', counted)
    accurate = solve_stvk(mesh, material, force, fixed, tolerance=1e-6,
                          absolute_tolerance=1e-6, correction_tolerance=1e-13)
    np.testing.assert_allclose(accurate.displacement, reference.displacement, atol=1e-13, rtol=0.)
    assert np.max(np.abs(calls[-1])) <= 1e-13
    assert accurate.linear_solves == len(calls) == len(accurate.nonlinear_history)


def test_small_correction_does_not_override_residual_gate():
    mesh = rectangle_mesh(4, 2, width=1., height=.3)
    material = ElasticMaterial(1e4, .3)
    _, force = assemble_elasticity(mesh, material, body_force=[0., -10.])
    with pytest.raises(RuntimeError):
        solve_stvk(mesh, material, force, mesh.boundary_nodes('left'), tolerance=1.,
                   absolute_tolerance=1e-30, correction_tolerance=1e3, max_iterations=4)


@pytest.mark.parametrize('bad', [0., -1., np.nan, np.inf, True, 'bad', 1+1j, [1e-12]])
def test_invalid_solid_correction_tolerance(bad):
    mesh = rectangle_mesh(1)
    with pytest.raises(ValueError, match='correction_tolerance'):
        solve_stvk(mesh, ElasticMaterial(1., .3), np.zeros(2*mesh.n_nodes),
                   mesh.boundary_nodes(), correction_tolerance=bad)


@pytest.mark.parametrize('case', ['time_half', 'space_half'])
def test_real_solid_roundoff_snapshots_meet_displacement_accuracy(case):
    from pathlib import Path
    from fsi_ice.mesh import TriMesh
    from fsi_ice.solid.linear import vector_dofs
    from fsi_ice.fem.poisson import solve_dirichlet
    with np.load(Path(__file__).parent/'data'/'solid_precision_floor'/f'{case}.npz',
                 allow_pickle=False) as data:
        mesh = TriMesh(data['points'], data['cells'])
        material = ElasticMaterial(1.4e6, .4, density=10000.)
        fixed = data['fixed_nodes']
        solution = solve_stvk(mesh, material, data['external'], fixed, initial=data['initial'],
            mass_coefficient=float(data['mass_coefficient']), mass_rhs=data['mass_rhs'],
            tolerance=1e-10, absolute_tolerance=1e-8, correction_tolerance=4e-15)
        # Independently reassemble the returned state and certify both gates.
        r, k = stvk_residual(mesh, material, solution.displacement, data['external'])
        mass = assemble_mass(mesh, material)
        alpha2 = float(data['mass_coefficient'])
        r += alpha2*(mass@solution.displacement.ravel())-data['mass_rhs']
        k += alpha2*mass
        dofs = vector_dofs(fixed)
        free = np.setdiff1d(np.arange(len(r)), dofs)
        assert np.linalg.norm(r[free]) <= min(1e-8, 1e-10*max(np.linalg.norm(data['external']), 1.))
        delta = solve_dirichlet(k, -r, dofs, 0.).values
        assert np.max(np.abs(delta)) <= 4e-15
        assert solution.minimum_jacobian > 0
        assert solution.linear_solves >= len(solution.nonlinear_history)

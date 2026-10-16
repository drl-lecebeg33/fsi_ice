"""Weighted ALE correctness and performance-path numerical equivalence."""
from dataclasses import replace
import copy
from pathlib import Path

import numpy as np
import pytest

from fsi_ice.ale import build_interface_weighted_extension, build_gcl_transport
from fsi_ice.coupling.monolithic import monolithic_residual, monolithic_jvp, monolithic_jacobian
from fsi_ice.fluid import ale as ale_module
from fsi_ice.fluid.taylor_hood import assemble_stokes, assemble_velocity_mass, navier_stokes_residual
from fsi_ice.fem.p2 import p2_values
from fsi_ice.mesh import rectangle_mesh
from fsi_ice.validation.fsi2 import load_fsi2_config, validate_fsi2_config
from fsi_ice.validation.fsi2_checkpoint import save_checkpoint, load_checkpoint
from test_fsi2 import small_problem, COUPLING, NEWTON
from test_monolithic_residual import build_problem

ROOT = Path(__file__).resolve().parents[1]
MOTION = {"model": "weighted_harmonic", "length": .005, "power": 1.}


def test_reference_convection_tensor_matches_independent_quadrature():
    problem = build_problem(nx=3, ny_fluid=2)
    ops = problem.operators
    shapes = p2_values(ops.quadrature)
    expected = np.einsum('q,qa,qb,eqcj->eabcj', ops.weights, shapes, shapes, ops.gradients)
    expected *= (ops.density*ops.space.mesh.determinants)[:, None, None, None, None]
    np.testing.assert_allclose(ops.convection_tensor, expected, rtol=2e-13, atol=2e-15)


def test_ale_cache_matches_fresh_assembly_and_isolates_density_history_geometry(monkeypatch):
    problem = build_problem(nx=3, ny_fluid=2, order=2)
    geometry = problem.candidate_geometry
    transport = build_gcl_transport(problem.fluid, geometry, problem.base.accepted_geometries, problem.stencil)
    original_mass = ale_module.assemble_velocity_mass
    original_transport = ale_module.mesh_transport_matrix
    counts = {'mass': 0, 'transport': 0}
    def mass(*args, **kwargs):
        counts['mass'] += 1
        return original_mass(*args, **kwargs)
    def matrix(*args, **kwargs):
        counts['transport'] += 1
        return original_transport(*args, **kwargs)
    monkeypatch.setattr(ale_module, 'assemble_velocity_mass', mass)
    monkeypatch.setattr(ale_module, 'mesh_transport_matrix', matrix)
    ops = problem.operators
    context = ale_module.build_ale_flow_context(ops, geometry, transport, problem.base.fluid_states)
    for density in (1., 3.):
        ops = assemble_stokes(geometry.current_mesh, density=density, viscosity=.1)
        expected = sum(c*(original_mass(g.current_mesh, density=density)@u.ravel())
                       for c,g,u in zip(problem.stencil.coefficients[1:],
                                       problem.base.accepted_geometries, problem.base.fluid_states))
        values = np.linspace(-.02,.03,ops.n_total)
        for _ in range(3):
            residual, jac = navier_stokes_residual(ops, values, ale_context=context)
            # Evaluate independent uncached terms rather than a second cached residual.
            standard, _ = navier_stokes_residual(ops, values)
            mesh_matrix = original_transport(ops, transport)
            coefficient = problem.stencil.coefficients[0]
            standard[:ops.n_velocity] += expected + (coefficient*ops.mass+mesh_matrix)@values[:ops.n_velocity]
            np.testing.assert_allclose(residual, standard, atol=2e-13, rtol=1e-12)
            np.testing.assert_allclose(ale_module.history_mass_load(ops,context), expected, atol=1e-14)
        assert not ale_module.history_mass_load(ops,context).flags.writeable
    assert counts == {'mass': 4, 'transport': 2}
    changed = ale_module.build_ale_flow_context(ops, geometry, transport,
                                              tuple(2*u for u in problem.base.fluid_states))
    np.testing.assert_allclose(ale_module.history_mass_load(ops, changed), 2*expected, atol=1e-14)
    alien = assemble_stokes(problem.fluid, density=3.)
    with pytest.raises(ValueError, match='own candidate'):
        ale_module.history_mass_load(alien,context)


@pytest.mark.parametrize("power", [1., 1.5])
def test_weighted_mesh_translation_and_tangential_affine_patch(power):
    mesh=rectangle_mesh(8)
    extension=build_interface_weighted_extension(mesh, interface_tag='bottom', length=.1, power=power)
    nodes=mesh.boundary_nodes()
    exact=np.column_stack((.02+.03*mesh.points[:,0], -.01*np.ones(mesh.n_nodes)))
    result=extension.extend(nodes,exact[nodes])
    np.testing.assert_allclose(result.values,exact,rtol=0,atol=1e-12)
    assert max(result.component_residuals)<1e-10
    assert result.boundary_error==0


@pytest.mark.parametrize("power", [1., 1.5])
def test_weighted_mesh_normal_manufactured_solution_converges(power):
    length=.1
    from scipy.integrate import quad
    def primitive(y):
        return np.array([quad(lambda z: (np.sqrt(z*z+length*length)/length)**power,
                              0., float(value), epsabs=1e-12)[0]
                         for value in np.asarray(y).reshape(-1)]).reshape(np.asarray(y).shape)
    errors=[]
    for n in (8,16,32):
        mesh=rectangle_mesh(n)
        exact=.02*primitive(mesh.points[:,1])/primitive(1.)
        nodes=mesh.boundary_nodes()
        extension=build_interface_weighted_extension(mesh,interface_tag='bottom',length=length,power=power)
        result=extension.extend(nodes,np.column_stack((exact[nodes],np.zeros(len(nodes)))))
        errors.append(np.max(np.abs(result.values[:,0]-exact)))
    assert errors[1]<errors[0]/3 and errors[2]<errors[1]/3, errors


@pytest.mark.parametrize("power", [1., 1.5])
def test_weighted_mesh_full_coupled_jacobian(power):
    problem=build_problem(nx=3,ny_fluid=2,order=2)
    extension=build_interface_weighted_extension(problem.fluid,interface_tag='interface',length=.05,power=power)
    context=replace(problem.context,harmonic=extension)
    direction=np.random.default_rng(923).normal(size=len(problem.x))*problem.variable_scales
    # Perturb one matching interface: independent random traces would crack it.
    dm=direction[problem.layout.slice_of('d_m')].reshape(-1,2)
    ds=direction[problem.layout.slice_of('d_s')].reshape(-1,2)
    dm[problem.topology.fluid_vertex_nodes]=ds[problem.topology.solid_nodes]
    exact=monolithic_jvp(problem.x,direction,context=context)
    assembled=monolithic_jacobian(problem.x,context=context)@direction
    np.testing.assert_allclose(exact,assembled,rtol=1e-10,atol=1e-11)
    errors=[]
    for eps in (1e-3,1e-4,1e-5,1e-6):
        fd=(monolithic_residual(problem.x+eps*direction,context=context)-
            monolithic_residual(problem.x-eps*direction,context=context))/(2*eps)
        errors.append(np.linalg.norm((fd-exact)/problem.residual_scales)/
                      np.linalg.norm(exact/problem.residual_scales))
    assert min(errors)<1e-5, errors


@pytest.mark.parametrize('order',[1,2])
@pytest.mark.parametrize('power',[1.,1.5])
def test_weighted_four_methods_and_checkpoint_retry(tmp_path,order,power):
    drivers={method:small_problem(order) for method in ('fixed','aitken','iqn','monolithic')}
    def weighted(driver):
        driver.harmonic=build_interface_weighted_extension(driver.topology.fluid_mesh,
                           interface_tag='fluid-interface',length=.05,power=power)
        return driver
    drivers={m:weighted(d) for m,d in drivers.items()}
    config=load_fsi2_config(ROOT/'cases/fsi2.json')
    config.update(dt=.05,end_time=.15,order=order,method='iqn',mesh_motion={**MOTION,'length':.05})
    identity={'config':config,'mesh_sha256':'mesh','code_sha256':'code'}
    for step in range(3):
        states={m:d.step(m,coupling=COUPLING,newton=NEWTON) for m,d in drivers.items()}
        expected=drivers['monolithic']._initial_vector(states['monolithic'])
        for m,d in drivers.items():
            actual=d._initial_vector(states[m])
            threshold=1e-8*np.maximum(np.abs(actual),np.abs(expected))+1e-10*d.scales
            assert np.all(np.abs(actual-expected)<=threshold)
        if step==0:
            path=tmp_path/'checkpoint.npz'
            save_checkpoint(path,states['iqn'],**identity)
            resumed=weighted(small_problem(order))
            resumed.coordinator.load(load_checkpoint(path,**identity))
            before=resumed.coordinator.state
            def reject(view): raise RuntimeError('injected after private solves')
            with pytest.raises(RuntimeError):
                resumed.step('iqn',coupling=COUPLING,newton=NEWTON,checks={'reject':reject})
            assert resumed.coordinator.state is before
        else:
            resumed.step('iqn',coupling=COUPLING,newton=NEWTON)
            np.testing.assert_array_equal(resumed._initial_vector(resumed.coordinator.state),
                                          drivers['iqn']._initial_vector(states['iqn']))
    altered=copy.deepcopy(config);altered['mesh_motion']['length']=.1
    with pytest.raises(ValueError,match='config differs'):
        load_checkpoint(path,**{**identity,'config':altered})


@pytest.mark.parametrize('motion',[{}, {'model':'wrong'}, {'model':'harmonic','power':1},
    {**MOTION,'length':0}, {**MOTION,'power':True}, {**MOTION,'power':float('nan')},
    {**MOTION,'unknown':1}])
def test_bad_mesh_motion_config_rejected(motion):
    config=load_fsi2_config(ROOT/'cases/fsi2.json');config['mesh_motion']=motion
    with pytest.raises(ValueError,match='mesh_motion'):
        validate_fsi2_config(config)


@pytest.mark.parametrize('kwargs',[{'length':0},{'length':True},{'length':.1,'power':-1},
                                   {'length':.1,'interface_tag':'missing'}])
def test_weighted_extension_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        build_interface_weighted_extension(rectangle_mesh(2),**{'interface_tag':'bottom',**kwargs})


def test_power15_preserves_real_fine_mesh_large_deflection():
    from fsi_ice.mesh import TriMesh
    from fsi_ice.ale import build_ale_geometry
    with np.load(ROOT/'tests/data/mesh_large_deflection/space927.npz', allow_pickle=False) as data:
        mesh = TriMesh(data['points'],data['cells'],
                       {k[4:]:data[k] for k in data.files if k.startswith('tag_')})
        nodes, boundary = data['nodes'],data['boundary']
    extension = build_interface_weighted_extension(mesh,interface_tag='fluid-interface',length=.005,power=1.5)
    free = np.setdiff1d(np.arange(mesh.n_nodes),nodes)
    for scale in (1., 1.15, 1.3):
        result = extension.extend(nodes,scale*boundary)
        geo = build_ale_geometry(mesh,result.values,candidate_id='scaled geometry stress fixture')
        assert geo.minimum_volume_ratio >= .2 and geo.minimum_quality >= .1
        np.testing.assert_array_equal(result.values[nodes], scale*boundary)
        # Independently verify the same stiffness used by the monolithic block.
        residual = extension.stiffness @ result.values
        assert np.max(np.abs(residual[free])) < 1e-12


def test_mesh15_research_configs_change_only_mesh_weight():
    import json
    names = ['baseline','time_half','space_half']
    configs = {n:load_fsi2_config(ROOT/f'cases/fsi2_mesh15_{n}.json') for n in names}
    baseline = configs['baseline']
    for name,key in [('time_half','dt'),('space_half','mesh_size')]:
        cfg = copy.deepcopy(configs[name]);cfg[key]=baseline[key]
        assert cfg == baseline
        original=load_fsi2_config(ROOT/f'cases/fsi2_frequency_{name}.json')
        restored=copy.deepcopy(configs[name]);restored['mesh_motion']['power']=1.
        assert restored==original

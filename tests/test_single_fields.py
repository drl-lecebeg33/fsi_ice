import json
from pathlib import Path

import numpy as np
import pytest

from fsi_ice.fluid.dynamics import solve_flow_step
from fsi_ice.fluid.forces import BoundaryMonitor, boundary_force, boundary_flux
from fsi_ice.fluid.taylor_hood import assemble_stokes, solve_stokes, navier_stokes_residual
from fsi_ice.mesh import rectangle_mesh
from fsi_ice.mesh.turek import csm_mesh, read_gmsh2
from fsi_ice.time import BDFHistory
from fsi_ice.validation.common import load_single_config
from fsi_ice.validation.fluid import temporal_fluid
from fsi_ice.validation.solid import temporal_solid


def test_force_normal_sign_pressure_and_flux_integrals():
    mesh = rectangle_mesh(3, width=2)
    operators = assemble_stokes(mesh, viscosity=2)
    def velocity(points):
        return np.column_stack((points[:, 1]*(1-points[:, 1])/4, np.zeros(len(points))))
    solution = solve_stokes(operators, velocity, pressure_value=2)
    expected = {"left": [-2, 0], "right": [0, 0], "bottom": [1, -2], "top": [1, 2]}
    total = np.zeros(2)
    for tag, force in expected.items():
        monitor = BoundaryMonitor(operators, tag)
        np.testing.assert_allclose(monitor.force(solution), force, atol=1e-11)
        np.testing.assert_allclose(boundary_force(operators, solution, tag), force, atol=1e-11)
        assert abs(monitor.flux(solution)-boundary_flux(operators, solution, tag)) < 1e-14
        total += monitor.force(solution)
    np.testing.assert_allclose(total, 0, atol=1e-11)
    assert abs(BoundaryMonitor(operators, "left").flux(solution)+1/24) < 1e-13


def test_convective_jacobian_perturbation_scan_with_time_mass():
    operators = assemble_stokes(rectangle_mesh(3), density=2)
    rng = np.random.default_rng(81)
    values = rng.normal(size=operators.n_total)
    direction = rng.normal(size=values.shape)
    context = BDFHistory(np.zeros_like(operators.space.points), 0.05).context()
    history_rhs = -(operators.mass @ context.history_term.ravel())
    def evaluate(value):
        return navier_stokes_residual(operators, value, time_coefficient=context.coefficient, time_rhs=history_rhs)
    _, jacobian = evaluate(values)
    exact = jacobian @ direction
    errors = []
    for eps in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        plus, _ = evaluate(values+eps*direction)
        minus, _ = evaluate(values-eps*direction)
        errors.append(np.linalg.norm((plus-minus)/(2*eps)-exact)/np.linalg.norm(exact))
    # The convective residual is quadratic, so a centered difference is exact
    # apart from rounding; no artificial second-order slope is expected here.
    assert max(errors) < 1e-8


def test_failed_fluid_candidate_preserves_history_and_repeated_step():
    operators = assemble_stokes(rectangle_mesh(3), viscosity=0.1)
    history = BDFHistory(np.zeros_like(operators.space.points), 0.1)
    context = history.context()
    other = BDFHistory(np.zeros_like(operators.space.points), 0.1)
    with pytest.raises(ValueError, match="unrelated"):
        history.accept(other.context(), np.zeros_like(operators.space.points))
    bad_initial = np.ones(operators.n_total)
    with pytest.raises(RuntimeError):
        solve_flow_step(operators, [0, 0], context, initial=bad_initial, max_iterations=1, tolerance=1e-15)
    assert history.time == 0
    np.testing.assert_array_equal(history.states[0], 0)
    result = solve_flow_step(operators, [0, 0], context)
    np.testing.assert_array_equal(result.velocity, 0)


@pytest.mark.parametrize("solver", [temporal_solid, temporal_fluid])
@pytest.mark.parametrize("order,minimum", [(1, 0.9), (2, 1.8)])
def test_actual_single_field_temporal_orders(solver, order, minimum):
    rows = solver([0.1, 0.05, 0.025], order=order)
    key = "state_error_order" if solver is temporal_solid else "velocity_l2_order"
    assert min(row[key] for row in rows[1:]) > minimum


def test_curved_attachment_and_monitoring_node():
    mesh = csm_mesh(14, 4)
    clamp = mesh.boundary_nodes("left")
    np.testing.assert_allclose(np.linalg.norm(mesh.points[clamp]-[0.2, 0.2], axis=1), 0.05, atol=1e-14)
    assert np.min(np.linalg.norm(mesh.points-[0.6, 0.2], axis=1)) < 1e-14


def test_msh2_sparse_node_ids_clockwise_import_and_tags(tmp_path):
    path = tmp_path / "mesh.msh"
    path.write_text('''$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
1 7 "wall"
2 9 "domain"
$EndPhysicalNames
$Nodes
3
10 0 0 0
20 1 0 0
30 0 1 0
$EndNodes
$Elements
4
1 1 2 7 1 10 20
2 1 2 7 1 20 30
3 1 2 7 1 30 10
4 2 2 9 1 10 30 20
$EndElements
''')
    mesh = read_gmsh2(path)
    np.testing.assert_allclose(mesh.areas, [0.5])
    assert set(mesh.boundaries) == {"wall"}
    assert len(mesh.boundary_nodes("wall")) == 3


def test_single_field_config_validation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = load_single_config(root / "cases/single_fields.json")
    path = tmp_path / "bad.json"
    config["time_steps"] = [0.1, 0.03, 0.01]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="divide"):
        load_single_config(path)

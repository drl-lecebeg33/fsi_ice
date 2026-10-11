import numpy as np
import pytest

from fsi_ice.fem.poisson import error_norms, solve_poisson
from fsi_ice.mesh import rectangle_mesh
from fsi_ice.validation.poisson import PoissonConfig, run_convergence


def test_affine_nonzero_boundary_is_reproduced_to_roundoff():
    mesh = rectangle_mesh(5, 4, width=2, height=1)
    exact = lambda x: 2 + 3 * x[:, 0] - 4 * x[:, 1]
    gradient = lambda x: np.tile([3, -4], (len(x), 1))
    solution = solve_poisson(mesh, 0, exact)
    np.testing.assert_allclose(solution.values, exact(mesh.points), rtol=1e-13, atol=2e-13)
    errors = error_norms(mesh, solution.values, exact, gradient)
    assert errors["l2"] < 1e-13 and errors["h1_seminorm"] < 2e-13


def test_variable_diffusion_linear_manufactured_solution():
    mesh = rectangle_mesh(6)
    # k=1+x and u=x+2: -div(k grad(u))=-1, represented exactly by P1.
    exact = lambda x: x[:, 0] + 2
    coefficient = lambda x: x[:, 0] + 1
    solution = solve_poisson(mesh, -1, exact, coefficient)
    np.testing.assert_allclose(solution.values, exact(mesh.points), atol=5e-14)


def test_quadratic_nonzero_boundary_converges():
    exact = lambda x: x[:, 0]**2 + x[:, 1]**2 + 2*x[:, 0] - 3*x[:, 1] + 1
    gradient = lambda x: np.column_stack((2*x[:, 0] + 2, 2*x[:, 1] - 3))
    errors = []
    for level in [4, 8, 16]:
        mesh = rectangle_mesh(level)
        solution = solve_poisson(mesh, -4, exact)
        np.testing.assert_allclose(solution.values[mesh.boundary_nodes()], exact(mesh.points[mesh.boundary_nodes()]))
        errors.append(error_norms(mesh, solution.values, exact, gradient))
    for a, b in zip(errors, errors[1:]):
        assert np.log2(a["l2"] / b["l2"]) > 1.9
        assert np.log2(a["h1_seminorm"] / b["h1_seminorm"]) > 0.95


def test_sine_convergence_and_machine_readable_evidence(tmp_path):
    report = run_convergence(PoissonConfig(), tmp_path, plot=False)
    assert report["status"] == "PASS"
    assert all(report["checks"].values())
    assert len(report["results"]) == 3
    assert report["results"][-1]["l2_order"] > 1.9
    assert report["results"][-1]["h1_order"] > 0.95
    assert (tmp_path / "convergence.csv").is_file()
    assert (tmp_path / "report.json").is_file()
    with np.load(tmp_path / "solution.npz") as data:
        assert data["points"].shape == (33**2, 2)
        assert np.all(np.isfinite(data["values"]))


@pytest.mark.parametrize("levels", [(4, 8), (4, 4, 8), (8, 4, 16), (True, 4, 8), (2.5, 4, 8)])
def test_invalid_convergence_grids(levels):
    with pytest.raises(ValueError):
        PoissonConfig(mesh_levels=levels)

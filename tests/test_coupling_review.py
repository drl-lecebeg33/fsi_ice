"""A7 independent T18 checks and output-failure regressions.

Physical balances are reconstructed without using the reported combined metric.
The D1/D2 output-failure defects were fixed, so those checks now run as normal
assertions instead of strict xfails; see docs/validation/partitioned_review.md
and the fix record docs/validation/t18_fixes.md.
"""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fsi_ice.coupling import CouplingFailure, CouplingOptions, PartitionedEvaluation, solve_dirichlet_neumann
from fsi_ice.fluid.taylor_hood import unpack_flow
from fsi_ice.interface import scatter_solid_load
from fsi_ice.validation.partitioned import LinearCoupledProblem, load_partitioned_config, run_partitioned_check

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_partitioned_config(ROOT / "cases/partitioned_check.json")


@pytest.fixture
def problem(config):
    return LinearCoupledProblem(config)


def weak_balances(problem, solid, fluid, load):
    """Original physical rows, before replacing interface rows by constraints."""
    solid_free = np.setdiff1d(np.arange(problem.solid_matrix.shape[0]), problem.solid_fixed)
    fluid_free = np.setdiff1d(np.arange(problem.operators.n_total),
                            np.union1d(problem.outer_dofs, problem.gamma_dofs))
    ds = (problem.solid_matrix @ solid.displacement.ravel() - problem.solid_rhs
          - scatter_solid_load(problem.topology, load.f_s_gamma))
    df = problem.fluid_matrix @ np.r_[fluid.velocity.ravel(), fluid.pressure] - problem.fluid_rhs
    motion = fluid.velocity.ravel()[problem.gamma_dofs] - problem.topology.H @ (
        solid.velocity.ravel()[problem.topology.solid_dofs])
    return ds[solid_free], df[fluid_free], motion


def test_small_scaled_component_cannot_hide_behind_large_other_units():
    # The second component is physically 10 scales wrong despite its tiny units.
    with pytest.raises(CouplingFailure) as caught:
        solve_dirichlet_neumann(lambda x: PartitionedEvaluation(np.array([1e6, 1e-8]), None, None, None),
                               [1e6, 0], scales=[1e6, 1e-9],
                               options=CouplingOptions(absolute_tolerance=.1, relative_tolerance=0,
                                                       max_iterations=1))
    assert caught.value.history[0].residual_norm == 10


@pytest.mark.parametrize("omega", [.25, .5, 1.])
def test_actual_fields_satisfy_original_weak_equations(problem, config, omega):
    result = problem.solve(options=CouplingOptions(**{**config["coupling"], "relaxation": omega}))
    ds, df, motion = weak_balances(problem, result.evaluation.solid,
                                  result.evaluation.fluid, result.evaluation.load)
    assert np.max(np.abs(ds)) < 1e-12
    assert np.max(np.abs(df)) < 1e-12
    assert np.max(np.abs(motion)) <= config["acceptance"]["motion_error_max"]
    # Fresh solves reconstruct the returned field from the actual final input.
    fresh = problem.adapter(result.interface_values)
    for first, second in ((fresh.fluid.velocity, result.evaluation.fluid.velocity),
                          (fresh.fluid.pressure, result.evaluation.fluid.pressure),
                          (fresh.solid.displacement, result.evaluation.solid.displacement),
                          (fresh.load.f_s_gamma, result.evaluation.load.f_s_gamma)):
        np.testing.assert_array_equal(first, second)


def test_nonzero_fixed_point_gap_predicts_only_kinematic_physical_defect(problem):
    result = problem.solve(options=CouplingOptions(absolute_tolerance=2e-4, relative_tolerance=0))
    ds, df, motion = weak_balances(problem, result.evaluation.solid,
                                  result.evaluation.fluid, result.evaluation.load)
    predicted = -problem.alpha * (problem.topology.H @ result.history[-1].residual)
    assert np.max(np.abs(predicted)) > 1e-8
    np.testing.assert_allclose(motion, predicted, atol=2e-17, rtol=1e-10)
    assert np.max(np.abs(ds)) < 1e-12 and np.max(np.abs(df)) < 1e-12
    # Small interface gap alone is not a generic all-field residual guarantee.
    assert not np.array_equal(result.interface_values, result.evaluation.target)


def test_block_load_action_matches_actual_stress_integration_on_unconverged_random_fields(problem):
    rng = np.random.default_rng(174)
    matrix, rhs, _, _ = problem.block_system
    ns = problem.solid_matrix.shape[0]
    for _ in range(3):
        d = rng.normal(size=ns)
        z = rng.normal(size=problem.operators.n_total)
        flow = unpack_flow(problem.operators, z, 0.)
        force = scatter_solid_load(problem.topology, problem.transfer_load(flow).f_s_gamma)
        expected = problem.solid_matrix @ d - force - problem.solid_rhs
        np.testing.assert_allclose((matrix @ np.r_[d, z] - rhs)[:ns], expected,
                                   atol=2e-12, rtol=1e-12)


def test_direct_reference_satisfies_original_equations_and_all_prescribed_values(problem):
    solid, fluid, load = problem.direct()
    ds, df, motion = weak_balances(problem, solid, fluid, load)
    assert max(np.max(np.abs(ds)), np.max(np.abs(df)), np.max(np.abs(motion))) < 1e-12
    np.testing.assert_array_equal(solid.displacement.ravel()[problem.solid_fixed], 0.)
    np.testing.assert_allclose(fluid.velocity.ravel()[problem.outer_dofs], problem.outer_values, atol=1e-15)


def test_corner_overlap_is_clamped_and_unpinned_pressure_is_fixed_by_natural_outlet(problem):
    overlap = np.intersect1d(problem.outer_dofs, problem.gamma_dofs)
    assert len(overlap) == 2  # x,y at the bottom-left endpoint, no extra constraints.
    solid_overlap = np.intersect1d(problem.topology.solid_dofs, problem.solid_fixed)
    assert len(solid_overlap) == 2
    assert np.all(problem.outer_dofs < problem.operators.n_velocity)  # no pressure pin.
    solid, fluid, _ = problem.direct()
    np.testing.assert_allclose(fluid.velocity.ravel()[overlap], 0, atol=1e-15)
    shifted = np.r_[fluid.velocity.ravel(), fluid.pressure + 1.]
    free = np.setdiff1d(np.arange(problem.operators.n_total),
                       np.union1d(problem.outer_dofs, problem.gamma_dofs))
    defect = problem.fluid_matrix @ shifted - problem.fluid_rhs
    assert np.max(np.abs(defect[free])) > .01  # pressure constant is not a null mode.
    initial = problem.initial_interface()
    local = np.flatnonzero(np.isin(problem.topology.solid_dofs, solid_overlap))[0]
    initial[local] = 1e-4
    with pytest.raises(CouplingFailure, match="conflicts") as caught:
        problem.solve(initial=initial)
    assert caught.value.stage == "fluid"


@pytest.mark.parametrize("failure", ["after_fluid", "exhaustion"])
def test_failure_retry_preserves_every_history_and_reference_geometry(problem, failure):
    snapshot = problem.history_snapshot()
    mesh_points = (problem.solid_mesh.points.copy(), problem.fluid_mesh.points.copy())
    reference = problem.solve()
    if failure == "after_fluid":
        from fsi_ice.coupling import DirichletNeumannMap
        def fail_load(_):
            raise RuntimeError("A7 injected failure after actual Stokes solve")
        adapter = DirichletNeumannMap(problem.solve_fluid, fail_load, problem.solve_solid,
                                     lambda s: s.displacement.ravel()[problem.topology.solid_dofs])
        with pytest.raises(CouplingFailure):
            solve_dirichlet_neumann(adapter, problem.initial_interface(),
                                   scales=np.full(len(problem.topology.solid_dofs), .01))
    else:
        with pytest.raises(CouplingFailure):
            problem.solve(options=CouplingOptions(max_iterations=1))
    assert problem.histories_equal(snapshot)
    retry = problem.solve()
    for a, b in ((retry.interface_values, reference.interface_values),
                 (retry.evaluation.solid.displacement, reference.evaluation.solid.displacement),
                 (retry.evaluation.fluid.velocity, reference.evaluation.fluid.velocity),
                 (retry.evaluation.fluid.pressure, reference.evaluation.fluid.pressure),
                 (retry.evaluation.load.f_s_gamma, reference.evaluation.load.f_s_gamma)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(problem.solid_mesh.points, mesh_points[0])
    np.testing.assert_array_equal(problem.fluid_mesh.points, mesh_points[1])
    assert problem.histories_equal(snapshot)


def test_convergence_failure_replaces_success_with_durable_fail(config, tmp_path):
    assert run_partitioned_check(config, tmp_path)["status"] == "PASS"
    fail = deepcopy(config)
    fail["coupling"]["max_iterations"] = 1
    with pytest.raises(CouplingFailure):
        run_partitioned_check(fail, tmp_path)
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "solution.npz").exists()
    evidence = json.loads((tmp_path / "failure.json").read_text())
    assert evidence["status"] == "FAIL" and evidence["histories_unchanged"]


@pytest.mark.parametrize("bad_input", ["dt", "material", "malformed_json", "missing_file"])
def test_invalid_config_cli_rerun_cannot_retain_old_success(config, tmp_path, bad_input):
    assert run_partitioned_check(config, tmp_path / "output")["status"] == "PASS"
    path = tmp_path / "invalid.json"
    if bad_input == "dt":
        config["dt"] = 0
    elif bad_input == "material":
        config["solid"]["poisson"] = .5
    if bad_input == "malformed_json":
        path.write_text("{")
    elif bad_input != "missing_file":
        path.write_text(json.dumps(config))
    run = subprocess.run([sys.executable, "-m", "fsi_ice", "partitioned-check", "--config", str(path),
                          "--output", str(tmp_path / "output")], capture_output=True, text=True)
    assert run.returncode == 2
    assert not (tmp_path / "output/report.json").exists()
    assert not (tmp_path / "output/solution.npz").exists()
    assert json.loads((tmp_path / "output/failure.json").read_text())["status"] == "FAIL"


@pytest.mark.parametrize("stage", ["setup", "metrics"])
def test_setup_and_postsolve_failures_have_durable_diagnostics(config, tmp_path, monkeypatch, stage):
    from fsi_ice.validation import partitioned
    assert run_partitioned_check(config, tmp_path)["status"] == "PASS"
    def fail(*args, **kwargs):
        raise RuntimeError("A7 injected " + stage + " failure")
    if stage == "setup":
        monkeypatch.setattr(partitioned, "assemble_stokes", fail)
    else:
        monkeypatch.setattr(LinearCoupledProblem, "metrics", fail)
    with pytest.raises(RuntimeError, match="A7 injected"):
        run_partitioned_check(config, tmp_path)
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "solution.npz").exists()
    assert json.loads((tmp_path / "failure.json").read_text())["status"] == "FAIL"

"""Known periodic signals, truthful partial reports and study isolation."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scripts.fsi2_analyze import COUNTS, analyze_directory, main, plot_history, read_snapshot, summarize
from scripts.m6_prepare_study import prepare, study_cases
from fsi_ice.validation.fsi2 import load_fsi2_config
from fsi_ice.validation.fsi2_checkpoint import save_checkpoint, source_digest
from test_fsi2 import small_problem, COUPLING, NEWTON


ROOT = Path(__file__).resolve().parents[1]


def harmonic(end=8., amplitude=.0816, frequency=2., growth=False):
    config = load_fsi2_config(ROOT/"cases/fsi2.json")
    config.update(dt=.005, end_time=end, method="iqn")
    rows = []
    for step in range(1, round(end/config["dt"])+1):
        time = step*config["dt"]
        factor = 1+.1*time if growth else 1.
        rows.append({"step": step, "time": time, "order": 1 if step == 1 else 2,
            "method": "iqn", "tip_x": -.001, "tip_y": factor*amplitude*np.sin(2*np.pi*frequency*time),
            "drag": 215.06+12*np.cos(4*np.pi*frequency*time),
            "lift": 150*np.sin(2*np.pi*frequency*time), "solve_seconds": .5,
            "residual": 1e-9, "continuity": 1e-15, "minimum_quality": .7,
            "minimum_cell_determinant": 1e-4, "minimum_solid_jacobian": .99,
            **dict.fromkeys(COUNTS, 1)})
    return config, rows


def test_harmonic_correct_period_amplitude_time_weighted_force_and_reference_errors():
    config, rows = harmonic()
    result = summarize(rows, config)
    assert result["status"] == result["periodic_status"] == "PASS"
    stats = result["periodic"]["statistics"]["tip_y"]
    assert stats["frequency"] == pytest.approx(2., abs=1e-10)
    assert stats["amplitude"] == pytest.approx(.0816, abs=1e-12)
    assert result["reference_comparison"]["drag_mean"]["value"] == pytest.approx(215.06, abs=1e-10)
    assert result["reference_comparison"]["frequency"]["relative_error"] == pytest.approx(2/1.93-1)
    assert all(not row["provisional"] for row in result["reference_comparison"].values())
    assert result["work"]["fluid_solves"] == len(rows)
    assert result["work"]["accepted_solve_seconds"] == .5*len(rows)
    assert not result["work"]["fair_timing_evidence"]


def test_partial_even_with_stable_good_cycles_is_incomplete_and_provisional():
    config, rows = harmonic()
    config["end_time"] = 20.
    result = summarize(rows, config)
    assert result["periodic_status"] == "PASS"
    assert result["status"] == "INCOMPLETE"
    assert not result["target_time_complete"]
    assert all(row["provisional"] for row in result["reference_comparison"].values())


def test_completed_unstable_growth_cannot_claim_reference_pass():
    config, rows = harmonic(growth=True)
    result = summarize(rows, config)
    assert result["status"] == "INCOMPLETE"
    assert result["periodic"]["checks"]["stable"] is False
    assert result["reference_comparison"]["tip_y_amplitude"]["provisional"]


@pytest.mark.parametrize("amplitude,frequency", [(.12, 2.), (.0816, 3.)])
def test_stable_completed_reference_failure(amplitude, frequency):
    config, rows = harmonic(amplitude=amplitude, frequency=frequency)
    result = summarize(rows, config)
    assert result["status"] == "FAIL"
    assert result["periodic"]["checks"]["stable"] is True


def test_solver_failure_prevents_pass_even_with_accepted_good_cycles():
    config, rows = harmonic()
    result = summarize(rows, config, failure={"reason": "Output failure"})
    assert result["status"] == "FAIL"
    assert result["periodic_status"] == "PASS"
    assert all(row["provisional"] for row in result["reference_comparison"].values())


@pytest.mark.parametrize("key,value", [("residual", 2e-7), ("continuity", 2e-10),
    ("minimum_quality", .09), ("minimum_cell_determinant", 0), ("minimum_solid_jacobian", -.1)])
def test_any_accepted_physical_gate_violation_blocks_pass(key, value):
    config, rows = harmonic()
    rows[50][key] = value
    result = summarize(rows, config)
    assert result["status"] == "FAIL"
    assert not result["accepted_physics_passed"]


@pytest.mark.parametrize("key,value", [("tip_y", float("nan")), ("residual", -1.),
    ("minimum_quality", 2.), ("linear_solves", -.5), ("time", .4), ("step", 0), ("method", "fixed")])
def test_invalid_ledger_rejected(key, value):
    config, rows = harmonic()
    rows[0][key] = value
    with pytest.raises(ValueError):
        summarize(rows, config)


def test_zero_and_short_prefix_have_no_cycles():
    config, rows = harmonic(end=.1)
    for prefix in ([], rows):
        result = summarize(prefix, config)
        assert result["status"] == "INCOMPLETE"
        assert result["reference_comparison"] == {}


def test_resolved_cycle_plot_is_generated_from_known_signal(tmp_path):
    config, rows = harmonic()
    plot_history(rows, summarize(rows, config), tmp_path)
    assert (tmp_path/"history.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path/"cycles.png").read_bytes().startswith(b"\x89PNG")


@pytest.fixture
def snapshot(tmp_path):
    driver = small_problem()
    driver.monitor = lambda view, blocks: {"tip_x": float(blocks["d_s"][-1, 0]),
        "tip_y": float(blocks["d_s"][-1, 1]), "drag": float(blocks["g"][:, 0].sum()),
        "lift": float(blocks["g"][:, 1].sum())}
    driver.step("iqn", coupling=COUPLING, newton=NEWTON)
    config = load_fsi2_config(ROOT/"cases/fsi2.json")
    config.update(dt=.05, end_time=.2, method="iqn")
    folder = tmp_path/"solver"
    save_checkpoint(folder/"checkpoint.npz", driver.coordinator.state, config=config,
                    mesh_sha256="a"*64, code_sha256=source_digest())
    return folder


def test_real_fe_snapshot_analysis_preserves_bytes_and_outputs_actual_figure(snapshot, tmp_path):
    path = snapshot/"checkpoint.npz"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = analyze_directory(snapshot, tmp_path/"analysis")
    assert result["status"] == "INCOMPLETE"
    assert result["accepted_steps"] == 1
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert {p.name for p in (tmp_path/"analysis").iterdir()} == {
        "summary.json", "summary.md", "cycles.csv", "history.png"}
    assert (tmp_path/"analysis/history.png").read_bytes().startswith(b"\x89PNG")


def test_same_nested_or_parent_output_cannot_overwrite_solver(snapshot):
    for output in (snapshot, snapshot/"analysis", snapshot.parent):
        with pytest.raises(ValueError, match="separate"):
            analyze_directory(snapshot, output)
    assert list(snapshot.iterdir()) == [snapshot/"checkpoint.npz"]


def test_broken_manifest_rejected(snapshot, tmp_path):
    path = snapshot/"checkpoint.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    arrays["pressure"].flat[0] += 1.
    np.savez_compressed(path, **arrays)
    assert main([str(snapshot), "--output", str(tmp_path/"analysis")]) == 2
    assert not (tmp_path/"analysis").exists()


def test_stale_failure_record_requires_consistent_snapshot(snapshot):
    (snapshot/"failure.json").write_text(json.dumps({"status": "FAIL", "accepted_steps": 999}))
    with pytest.raises(ValueError, match="Failure record"):
        read_snapshot(snapshot)


def test_refinement_groups_vary_exactly_one_parameter_and_share_finest():
    base = load_fsi2_config(ROOT/"cases/fsi2.json")
    previous = copy.deepcopy(base)
    cases, groups = study_cases(base)
    assert base == previous
    configs = {case["id"]: case["config"] for case in cases}
    for group, varying in (("space", "mesh_size"), ("time", "dt")):
        selected = [copy.deepcopy(configs[name]) for name in groups[group]]
        assert len({config.pop(varying) for config in selected}) == 3
        assert selected[0] == selected[1] == selected[2]
    assert set(groups["space"]) & set(groups["time"]) == {"intersection_fine"}
    costs = [case for case in cases if case["phase"] == "algorithms"]
    assert len(costs) == 12
    for method in ("fixed", "aitken", "iqn", "monolithic"):
        assert {case["repeat"] for case in costs if case["config"]["method"] == method} == {1, 2, 3}
    normalized = []
    for case in costs:
        config = copy.deepcopy(case["config"])
        config.pop("method")
        normalized.append(config)
    assert all(config == normalized[0] for config in normalized)


def test_preparation_only_writes_configs_commands_and_never_launches(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Study preparation must not execute subprocesses")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    output = tmp_path/"study"
    manifest = prepare(ROOT/"cases/fsi2.json", output)
    assert manifest["status"] == "PREPARED_NOT_RUN"
    assert manifest["solver_started"] is False
    assert len(list((output/"configs").glob("*.json"))) == 17
    assert not (output/"runs").exists()
    assert not any(case["permitted_to_start"] for case in manifest["cases"] if case["phase"] == "algorithms")
    before = (output/"manifest.json").read_bytes()
    with pytest.raises(ValueError, match="existing evidence"):
        prepare(ROOT/"cases/fsi2.json", output)
    assert (output/"manifest.json").read_bytes() == before

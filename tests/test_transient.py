import json
from pathlib import Path

import numpy as np
import pytest

from fsi_ice.validation.transient import load_transient_config, periodic_statistics, read_complete_trajectory
from fsi_ice.validation.common import write_csv


def test_period_statistics_and_startup_exclusion():
    times = np.arange(0, 5, 0.007)
    signal = -0.6+2*np.sin(2*np.pi*1.3*times)
    stats = periodic_statistics(times, signal, minimum_prominence=1)
    assert stats["resolved"] and stats["cycles"] == 3
    assert abs(stats["frequency"]/1.3-1) < 1e-5
    assert abs(stats["amplitude"]/2-1) < 5e-4
    assert abs(stats["mean"]+0.6) < 1e-3
    assert stats["window"][0] > 1
    assert stats["amplitude_variation"] < 1e-3


def test_no_invented_frequency_for_unresolved_flow():
    times = np.linspace(0, 10, 501)
    for signal in (np.ones_like(times), np.exp(-times), np.sin(times)):
        stats = periodic_statistics(times, signal, minimum_prominence=0.1)
        assert not stats["resolved"] and "frequency" not in stats


def test_period_diagnostics_detect_growing_oscillations():
    times = np.arange(0, 5, 0.005)
    signal = np.exp(0.2*times)*np.sin(2*np.pi*times)
    stats = periodic_statistics(times, signal, minimum_prominence=0.5)
    assert stats["resolved"] and stats["amplitude_variation"] > 0.3


def test_transient_config_and_invalid_case(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for case in ("csm3", "cfd3"):
        assert load_transient_config(root / f"cases/{case}.json")["case"] == case
    path = tmp_path / "config.json"
    data = json.loads((root / "cases/csm3.json").read_text())
    data["time_steps"] = [0.02, 0.017, 0.01]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="divide"):
        load_transient_config(path)


def saved_csm_history(folder):
    times = np.arange(0.01, 5.001, 0.01)
    rows = [{"time": float(t), "tip_x": float(-0.014+0.014*np.sin(2*np.pi*1.1*t)), "tip_y": float(-0.06+0.06*np.sin(2*np.pi*1.1*t)),
             "residual": 1e-9, "newton_iterations": 2, "minimum_jacobian": 0.8} for t in times]
    write_csv(folder / "history.csv", rows)
    (folder / "summary.json").write_text(json.dumps({"mesh": [10, 2], "dt": 0.01, "cells": 40,
                                                    "statistics": {"frequency": 999}, "maximum_residual": 0}))
    return rows


def test_saved_diagnostics_are_recomputed_and_original_data_preserved(tmp_path):
    saved_csm_history(tmp_path)
    originals = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    summary = read_complete_trajectory("csm3", [10, 2], 0.01, 5, tmp_path)
    assert abs(summary["statistics"]["frequency"]/1.1-1) < 1e-4
    x_stats = summary["statistics"]["tip_x"]
    assert abs(x_stats["frequency"]/1.1-1) < 1e-4
    assert abs(x_stats["mean"]+0.014) < 1e-5 and abs(x_stats["amplitude"]-0.014) < 1e-5
    assert summary["maximum_residual"] == 1e-9 and summary["minimum_jacobian"] == 0.8
    assert len(summary["saved_evidence"]["files_sha256"]["history.csv"]) == 64
    assert originals == {path.name: path.read_bytes() for path in tmp_path.iterdir()}


@pytest.mark.parametrize("damage", ["truncated", "time_gap", "nonfinite", "negative_residual", "wrong_case", "wrong_dt", "wrong_mesh", "wrong_cells", "failed_marker"])
def test_saved_incomplete_or_incompatible_evidence_is_rejected(tmp_path, damage):
    rows = saved_csm_history(tmp_path)
    if damage == "truncated":
        rows = rows[:-1]
    elif damage == "time_gap":
        rows[30]["time"] += 0.002
    elif damage == "nonfinite":
        rows[30]["tip_y"] = float("nan")
    elif damage == "negative_residual":
        rows[30]["residual"] = -1e-9
    elif damage == "wrong_cells":
        path = tmp_path / "summary.json"
        summary = json.loads(path.read_text())
        summary["cells"] += 1
        path.write_text(json.dumps(summary))
    elif damage == "failed_marker":
        (tmp_path / "failure.json").write_text('{"status":"FAIL"}')
    write_csv(tmp_path / "history.csv", rows)
    with pytest.raises(ValueError):
        read_complete_trajectory("cfd3" if damage == "wrong_case" else "csm3",
                                 [11, 2] if damage == "wrong_mesh" else [10, 2],
                                 0.02 if damage == "wrong_dt" else 0.01, 5, tmp_path)


def test_trajectory_failure_keeps_accepted_history_and_identifies_failed_step(tmp_path, monkeypatch):
    import fsi_ice.validation.transient as transient
    real_solve, real_history = transient.solve_solid_step, transient.BDFHistory
    (tmp_path / "summary.json").write_text('{"statistics":{"resolved":true}}')
    histories, snapshots = [], []
    def recording_history(*args, **kwargs):
        history = real_history(*args, **kwargs)
        histories.append(history)
        return history
    def fail_second_step(*args, **kwargs):
        context = args[4]
        if context.time > 0.015:
            snapshots.append([state.copy() for state in histories[0].states])
            raise RuntimeError("Injected Newton failure")
        return real_solve(*args, **kwargs)
    monkeypatch.setattr(transient, "BDFHistory", recording_history)
    monkeypatch.setattr(transient, "solve_solid_step", fail_second_step)
    with pytest.raises(RuntimeError, match=r"CSM3.*step=2, t=0.02.*Injected Newton failure"):
        transient.run_trajectory("csm3", [4, 2], 0.01, 0.03, tmp_path)
    for before, after in zip(snapshots[0], histories[0].states):
        np.testing.assert_array_equal(before, after)
    assert histories[0].context().time == 0.02
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["status"] == "FAIL" and failure["failed_time"] == 0.02 and failure["accepted_steps"] == 1
    data = np.atleast_1d(np.genfromtxt(tmp_path / "history.csv", delimiter=",", names=True))
    assert len(data) == 1 and data["time"][0] == 0.01
    assert not (tmp_path / "summary.json").exists()

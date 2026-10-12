"""Light unit tests of the M5/T23 comparison, scale, cost and report helpers.

These tests never run the h=0.02 FSI1 solves: the full end-to-end evidence is
produced manually by ``scripts/m5_compare.py`` and recorded in
``docs/validation/t23.md``.  A gated consistency check of the produced results
is available through ``FSI_ICE_T23_RESULTS=<results/m5/t23>``.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fsi_ice.coupling.monolithic_newton import MonolithicCounters, MonolithicNewtonFailure
from fsi_ice.validation import monolithic_case
from fsi_ice.validation.monolithic_case import (
    CONSISTENCY_SPECS, COST_ITEMS, FSI1_GATES, T23_ATOL_SCALE, T23_BLOCK_SCALES, T23_RTOL,
    T23_SCALES, compare_values, consistency_rows, cost_rows, environment_snapshot,
    pressure_gauge_diagnostic, summarize_values)


# ---------------------------------------------------------------------------
# criterion and scales
# ---------------------------------------------------------------------------

def test_frozen_scale_and_tolerance_values():
    assert T23_SCALES == {"L_star": 0.41, "U_star": 0.2, "P_star": 40.0, "F_star": 14.29426}
    assert T23_RTOL == 1e-8
    assert T23_ATOL_SCALE == 1e-10
    assert T23_BLOCK_SCALES["d_s"] == 0.41 and T23_BLOCK_SCALES["p_f"] == 40.0
    names = [name for name, *_ in CONSISTENCY_SPECS]
    assert names == ["d_s", "tip_A", "u_f", "p_f", "d_m", "v_s", "interface_x", "interface_H",
                     "g", "lambda"]
    assert dict((name, unit) for name, unit, *_ in CONSISTENCY_SPECS)["g"] == "N/m"
    assert FSI1_GATES["tip_x_absolute_error_max"] == 1e-5


def test_compare_values_boundary_and_failure_detection():
    equal = compare_values("d_s", [1.0, -2.0], [1.0, -2.0], unit="m", scale=0.41)
    assert equal.passed and equal.max_abs_delta == 0.0 and equal.atol == pytest.approx(4.1e-11)
    magnitude = 3.0
    limit = T23_RTOL*magnitude+equal.atol
    exactly = compare_values("d_s", [3.0], [3.0-limit], unit="m", scale=0.41)
    assert exactly.max_abs_delta == pytest.approx(limit) and exactly.passed  # <= limit
    beyond = compare_values("d_s", [3.0], [3.0-1.01*limit], unit="m", scale=0.41)
    assert not beyond.passed and beyond.limit_ratio > 1.0
    with pytest.raises(ValueError):
        compare_values("u", [0.0, 1.0], [0.0], unit="m/s", scale=0.2)
    with pytest.raises(ValueError):
        compare_values("u", [np.nan], [0.0], unit="m/s", scale=0.2)
    with pytest.raises(ValueError):
        compare_values("u", [0.0], [0.0], unit="m/s", scale=0.0)
    with pytest.raises(ValueError):
        compare_values("u", [0.0], [0.0], unit="m/s", scale=0.2, rtol=-1.0)


def test_compare_values_zero_quantity_uses_absolute_floor():
    # A near-zero vector compares against atol only, never an unreliable relative error.
    row = compare_values("v_s", np.zeros(4), np.full(4, 1e-12), unit="m/s", scale=T23_SCALES["U_star"])
    assert row.passed and row.atol == pytest.approx(2e-11)
    row = compare_values("v_s", np.zeros(4), np.full(4, 1e-9), unit="m/s", scale=T23_SCALES["U_star"])
    assert not row.passed


def _bundle(d_s_value, u_value, p_value, dm_value, g_value, size=3):
    zero = np.zeros(size)
    return {
        "d_s": np.full(size, d_s_value), "tip_A": np.array([d_s_value, d_s_value]),
        "u_f": np.full(size, u_value), "p_f": np.full(size, p_value),
        "d_m": np.full(size, dm_value), "v_s": zero.copy(),
        "interface_x": np.full(size, d_s_value), "interface_H": np.full(size, d_s_value),
        "g": np.full(size, g_value), "lambda": np.full(size, g_value),
    }


def test_consistency_rows_cover_every_variable_and_label_pair():
    rows = consistency_rows(_bundle(1e-4, 0.2, 40.0, 1e-5, 14.0), _bundle(1e-4, 0.2, 40.0, 1e-5, 14.0),
                            label_a="monolithic", label_b="partitioned-aitken")
    assert [row.name for row in rows] == [name for name, *_ in CONSISTENCY_SPECS]
    assert all(row.passed for row in rows)
    assert all("[monolithic vs partitioned-aitken]" in row.description for row in rows)
    shifted = consistency_rows(_bundle(0.0, 0.0, 0.0, 0.0, 0.0), _bundle(0.0, 0.0, 0.0, 0.0, 0.0))
    assert all(row.passed for row in shifted)  # identical zeros pass through atol


def test_pressure_gauge_diagnostic_reports_shift_and_residual():
    a = _bundle(0.0, 0.0, 10.0, 0.0, 0.0)["p_f"]
    b = a+1.0
    bundle_a = _bundle(0.0, 0.0, 0.0, 0.0, 0.0)
    bundle_a["p_f"] = a
    bundle_b = _bundle(0.0, 0.0, 0.0, 0.0, 0.0)
    bundle_b["p_f"] = b
    gauge = pressure_gauge_diagnostic(bundle_a, bundle_b)
    assert gauge["mean_difference"] == pytest.approx(-1.0)
    assert gauge["max_after_mean_shift"] == pytest.approx(0.0)
    assert gauge["max_raw"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# physical error gates and gaps
# ---------------------------------------------------------------------------

def _metrics(tip_x_err, tip_y_rel, drag_rel, lift_err):
    return {"tip_x_absolute_error": tip_x_err, "tip_y_relative_error": tip_y_rel,
            "drag_relative_error": drag_rel, "lift_absolute_error": lift_err}


def test_physical_error_gates_and_gaps():
    fixed, gaps = monolithic_case.physical_error_gaps({
        "monolithic": _metrics(1e-6, 0.02, 0.01, 0.05),
        "partitioned-aitken": _metrics(2e-6, 0.022, 0.012, 0.053)})
    assert all(all(values.values()) for values in fixed.values())
    assert len(gaps) == 1 and gaps[0]["passed"]
    _, gaps_fail = monolithic_case.physical_error_gaps({
        "a": _metrics(1e-6, 0.001, 0.001, 0.01), "b": _metrics(1e-6, 0.02, 0.001, 0.01)})
    assert not gaps_fail[0]["passed"]
    fixed_fail, _ = monolithic_case.physical_error_gaps({
        "a": _metrics(1e-4, 0.02, 0.01, 0.05), "b": _metrics(2e-6, 0.022, 0.012, 0.06)})
    assert not fixed_fail["a"]["tip_x_absolute_error_max"]


# ---------------------------------------------------------------------------
# cost aggregation
# ---------------------------------------------------------------------------

def test_cost_rows_median_min_max_and_missing_items():
    entries = [
        {"method": "partitioned-aitken", "level": "1e-08", "repeat": 0,
         "cost": {"total_seconds": 6.0, "coupling_iterations": 8, "pseudo_time_steps": None}},
        {"method": "partitioned-aitken", "level": "1e-08", "repeat": 1,
         "cost": {"total_seconds": 8.0, "coupling_iterations": 9, "pseudo_time_steps": None}},
        {"method": "partitioned-aitken", "level": "1e-08", "repeat": 2,
         "cost": {"total_seconds": 10.0, "coupling_iterations": 8, "pseudo_time_steps": None}},
    ]
    rows = cost_rows(entries)
    total = next(row for row in rows if row["item"] == "total_seconds")
    assert (total["repeats"], total["median"], total["min"], total["max"]) == (3, 8.0, 6.0, 10.0)
    assert all(row["item"] != "pseudo_time_steps" for row in rows)
    assert set(COST_ITEMS).issuperset({row["item"] for row in rows})
    assert summarize_values([None, float("nan")]) == {"count": 0, "median": None, "min": None,
                                                      "max": None}


def test_environment_snapshot_has_required_fields():
    environment = environment_snapshot()
    for key in ("python", "executable", "platform", "cpu", "memory", "packages", "threads",
                "git_head", "git_status", "timestamp"):
        assert key in environment
    assert set(environment["threads"]) >= {"OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"}


def test_previous_level_state_uses_the_closest_less_tight_level(tmp_path):
    base = tmp_path/"monolithic"/"h_0.02"
    for name, present in (("level_1e-06", True), ("level_1e-08", True), ("level_1e-10", False)):
        directory = base/name
        directory.mkdir(parents=True)
        if present:
            np.savez(directory/"final_state.npz", x=np.zeros(2))
    assert (monolithic_case.previous_level_state(tmp_path, 0.02, 1e-10)
            == base/"level_1e-08"/"final_state.npz")
    assert (monolithic_case.previous_level_state(tmp_path, 0.02, 1e-8)
            == base/"level_1e-06"/"final_state.npz")
    assert monolithic_case.previous_level_state(tmp_path, 0.02, 1e-6) is None
    assert monolithic_case.previous_level_state(tmp_path, 0.05, 1e-8) is None


# ---------------------------------------------------------------------------
# report structure and failure handling
# ---------------------------------------------------------------------------


def test_write_json_rejects_non_finite(tmp_path):
    path = tmp_path/"bad.json"
    with pytest.raises(ValueError):
        monolithic_case.write_json(path, {"value": float("nan")})
    monolithic_case.write_json(path, {"value": 1.0, "nested": {"ok": [1, 2]}})
    assert json.loads(path.read_text())["nested"]["ok"] == [1, 2]


def test_failure_writer_clears_success_artifacts(tmp_path):
    output = tmp_path/"level"
    output.mkdir()
    for name in ("summary.json", "steps.csv", "solution.npz"):
        (output/name).write_text("stale", encoding="utf-8")
    payload = monolithic_case.write_monolithic_failure(output, level=1e-6, stage="march",
                                                       reason="line_search failed")
    assert payload["status"] == "FAIL"
    assert (output/"failure.json").exists()
    for name in ("summary.json", "steps.csv", "solution.npz"):
        assert not (output/name).exists()
    stored = json.loads((output/"failure.json").read_text())
    assert stored["level"] == pytest.approx(1e-6) and "line_search" in stored["reason"]


def _fake_step(index=1):
    return {
        "index": index, "chunk": 0, "generation": index, "order": 2, "dt": 0.1,
        "newton_iterations": 3, "residual_norm": 1e-9, "threshold": 1e-8, "time_term": 1e-7,
        "time_term_u_f": 1e-8, "time_term_d_s": 1e-7, "time_term_v_s": 1e-9,
        "interface_gap": 0.0,
        "counters": MonolithicCounters(steps=1, newton_iterations=3, linear_solves=3).as_dict(),
        "timings": {"evaluate_seconds": 10.0, "linear_seconds": 1.0, "newton_seconds": 12.0,
                    "total_seconds": 13.0},
    }


def test_march_level_aggregates_chunks_and_handles_failure(monkeypatch):
    calls = {"count": 0}
    received_contexts = []
    continuation = {"contexts": []}

    def fake_state_context(case, x, *, dt, candidate_id):
        continuation["contexts"].append({"x": np.array(x, copy=True), "candidate_id": candidate_id})
        return "continuation-context"

    def fake_march(context, x0, **kwargs):
        calls["count"] += 1
        received_contexts.append(context)
        if calls["count"] == 1:
            steps = (SimpleNamespace(generation=1, order=2, dt=0.1, newton_iterations=3,
                                     residual_norm=1e-9, threshold=1e-8, time_term=1e-7,
                                     time_terms={"u_f": 1e-8, "d_s": 1e-7, "v_s": 1e-9},
                                     interface_gap=0.0,
                                     counters=MonolithicCounters(steps=1, newton_iterations=3,
                                                                 linear_solves=3),
                                     timings={"evaluate_seconds": 10.0, "linear_seconds": 1.0,
                                              "newton_seconds": 12.0, "total_seconds": 13.0}),)
            return SimpleNamespace(steps=steps, x=np.zeros(2),
                                   counters=MonolithicCounters(steps=1, newton_iterations=3,
                                                               linear_solves=3),
                                   timings={"evaluate_seconds": 10.0, "linear_seconds": 1.0,
                                            "newton_seconds": 12.0, "total_seconds": 13.0},
                                   final_time_term=1e-7, final_time_terms={"u_f": 1e-8, "d_s": 1e-7,
                                                                          "v_s": 1e-9},
                                   steady_reached=False)
        raise MonolithicNewtonFailure(phase="line_search", iteration=1, reason="no descent",
                                      counters=MonolithicCounters(failed_steps=1),
                                      timings={"evaluate_seconds": 2.0, "linear_seconds": 0.5,
                                               "newton_seconds": 2.5, "total_seconds": 3.0},
                                      march_steps=(SimpleNamespace(
                                          generation=2, order=2, dt=0.1, newton_iterations=2,
                                          residual_norm=1e-8, threshold=1e-8, time_term=1e-6,
                                          time_terms={"u_f": 1e-8, "d_s": 1e-6, "v_s": 1e-9},
                                          interface_gap=0.0,
                                          counters=MonolithicCounters(steps=1, newton_iterations=2),
                                          timings={"evaluate_seconds": 2.0, "linear_seconds": 0.5,
                                                   "newton_seconds": 2.5, "total_seconds": 3.0}),))

    monkeypatch.setattr(monolithic_case, "march_to_steady", fake_march)
    monkeypatch.setattr(monolithic_case, "state_context", fake_state_context)
    case = SimpleNamespace(x0=np.zeros(2), dt=0.1, template="template-context", scales=np.ones(2),
                           candidate_id=("test", "case"))
    record = monolithic_case.march_level(case, level=1e-8, steady_tolerance=1e-8,
                                         newton_rtol=1e-8, newton_atol=1e-8,
                                         chunk_steps=6, max_steps=12, wall_seconds=100.0)
    assert received_contexts == ["template-context", "continuation-context"]
    assert len(continuation["contexts"]) == 1
    assert np.array_equal(continuation["contexts"][0]["x"], np.zeros(2))
    assert record.reason == "newton_failure" and record.failure is not None
    assert record.failure["phase"] == "line_search"
    assert len(record.steps) == 2  # one completed chunk step plus the failing chunk's step
    assert record.counters["steps"] == 1 and record.counters["failed_steps"] == 1
    assert record.timings["evaluate_seconds"] == pytest.approx(12.0)


def test_march_level_reaches_steady_and_writes_zero_iteration_chunk(monkeypatch):
    def fake_march(context, x0, **kwargs):
        return SimpleNamespace(steps=(), x=np.zeros(2), counters=MonolithicCounters(),
                               timings={"evaluate_seconds": 0.0, "linear_seconds": 0.0,
                                        "newton_seconds": 0.0, "total_seconds": 0.0},
                               final_time_term=1e-12, final_time_terms={"u_f": 1e-13, "d_s": 1e-12,
                                                                        "v_s": 1e-14},
                               steady_reached=True)

    monkeypatch.setattr(monolithic_case, "march_to_steady", fake_march)
    case = SimpleNamespace(x0=np.zeros(2), dt=0.1, template=object(), scales=np.ones(2))
    record = monolithic_case.march_level(case, level=1e-10, steady_tolerance=1e-10,
                                         newton_rtol=1e-10, newton_atol=1e-10, max_steps=6)
    assert record.steady_reached and record.reason == "steady"
    assert record.final_time_term == pytest.approx(1e-12) and record.chunks == 1


def test_write_level_outputs_structure(tmp_path):
    record = monolithic_case.MarchRecord(
        level=1e-8, steady_tolerance=1e-8, newton_rtol=1e-8, newton_atol=1e-8, dt=0.1, order=2,
        reason="steady", steady_reached=True, final_time_term=5e-9,
        final_time_terms={"u_f": 1e-9, "d_s": 5e-9, "v_s": 1e-10}, steps=(_fake_step(),),
        x=np.array([0.0, 1.0]), counters=MonolithicCounters(steps=1, newton_iterations=3,
                                                            linear_solves=3).as_dict(),
        timings={"evaluate_seconds": 10.0, "linear_seconds": 1.0, "newton_seconds": 12.0,
                 "total_seconds": 13.0},
        chunks=1, wall_seconds=13.0, failure=None, start={"max_steps": 6})
    payload = monolithic_case.write_monolithic_level(tmp_path, record, metrics={"tip_x": 1.0})
    assert payload["status"] == "PASS"
    summary = json.loads((tmp_path/"summary.json").read_text())
    assert summary["steps_total"] == 1 and summary["steady_reached"] is True
    assert summary["metrics"]["tip_x"] == 1.0
    rows = (tmp_path/"steps.csv").read_text().splitlines()
    assert "counter_newton_iterations" in rows[0] and len(rows) == 2
    assert (tmp_path/"final_state.npz").exists()
    monolithic_case.clear_monolithic_outputs(tmp_path)
    assert not (tmp_path/"summary.json").exists() and not (tmp_path/"final_state.npz").exists()


def test_write_monolithic_outputs_after_failure_stage(tmp_path):
    """The failure convention removes success artifacts and keeps failure.json."""
    output = tmp_path/"level"
    monolithic_case.write_monolithic_failure(output, level=1e-10, stage="run",
                                             reason="RuntimeError: boom")
    stored = json.loads((output/"failure.json").read_text())
    assert stored["status"] == "FAIL" and stored["stage"] == "run"


# ---------------------------------------------------------------------------
# gated end-to-end artifact check (no solve)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("FSI_ICE_T23_RESULTS") is None,
                    reason="set FSI_ICE_T23_RESULTS=results/m5/t23 to check produced artifacts")
def test_produced_t23_results_are_consistent():
    root = Path(os.environ["FSI_ICE_T23_RESULTS"])
    summary_path = root/"comparison"/"summary.json"
    if not summary_path.exists():
        pytest.skip(f"no comparison summary under {root}")
    summary = json.loads(summary_path.read_text())
    assert summary["case"] == "m5_t23"
    for key in ("availability", "platform", "consistency", "physical", "cost", "verdicts"):
        assert key in summary
    assert summary["comparison_level"] is not None
    assert (root/"comparison"/"consistency_per_variable.csv").exists()
    assert (root/"comparison"/"cost_per_item.csv").exists()
    for item in [f"23-{index:02d}" for index in range(1, 11)]:
        assert item in summary["verdicts"]

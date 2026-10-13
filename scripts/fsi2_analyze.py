"""Read-only FSI2 accepted-history analysis, including a running checkpoint."""

import argparse
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from fsi_ice.validation.fsi2 import analyze_fsi2, validate_fsi2_config
from fsi_ice.validation.fsi2_checkpoint import load_checkpoint, source_digest


COUNTS = ("fluid_solves", "solid_solves", "newton_iterations", "linear_solves",
          "coupling_iterations", "iqn_updates", "iqn_restarts")
SIGNALS = ("tip_x", "tip_y", "drag", "lift")
GATES = ("residual", "continuity", "minimum_quality", "minimum_cell_determinant",
         "minimum_solid_jacobian")


def summarize(rows, config, *, identity=None, failure=None):
    """Reuse the frozen cycle estimator; partial or rejected runs cannot PASS."""
    validate_fsi2_config(config)
    target_steps = round(config["end_time"]/config["dt"])
    if len(rows) > target_steps:
        raise ValueError("Accepted ledger exceeds the configured end time.")
    for index, row in enumerate(rows, 1):
        numeric = ("time", "solve_seconds", *SIGNALS, *GATES)
        if any(isinstance(row[key], bool) or not isinstance(row[key], (int, float)) or
               not np.isfinite(row[key]) for key in numeric):
            raise ValueError("Nonfinite or invalid monitoring value.")
        if row["step"] != index or row["method"] != config["method"] or \
                row["order"] != (1 if index == 1 else config["order"]) or \
                not np.isclose(row["time"], index*config["dt"], rtol=0, atol=1e-10):
            raise ValueError("Monitoring ledger has inconsistent steps/time/method/order.")
        if row["solve_seconds"] < 0 or any(isinstance(row[key], bool) or
                not isinstance(row[key], int) or row[key] < 0 for key in COUNTS):
            raise ValueError("Invalid work counts or solve time.")
        if row["residual"] < 0 or row["continuity"] < 0 or not 0 <= row["minimum_quality"] <= 1:
            raise ValueError("Invalid residual norm or mesh quality.")
    limits = config["acceptance"]
    physical = bool(rows) and all(row["residual"] <= limits["residual_max"] and
        row["continuity"] <= limits["continuity_max"] and
        row["minimum_quality"] >= limits["minimum_mesh_quality"] and
        row["minimum_cell_determinant"] > 0 and row["minimum_solid_jacobian"] > 0
        for row in rows)
    periodic = analyze_fsi2(rows, config)
    complete = len(rows) == target_steps
    if failure is not None or (rows and not physical):
        status, reason = "FAIL", "Solver failure or accepted-history physical gate failure"
    elif not complete:
        status, reason = "INCOMPLETE", "Accepted prefix has not reached configured end_time"
    else:
        status, reason = periodic["status"], periodic.get("reason", "Stable-cycle reference comparison")
    extrema = {key: float(max(row[key] for row in rows)) for key in ("residual", "continuity")} if rows else {}
    extrema.update({key: float(min(row[key] for row in rows)) for key in
                   ("minimum_quality", "minimum_cell_determinant", "minimum_solid_jacobian")} if rows else {})
    candidate = None
    tip = periodic.get("statistics", {}).get("tip_y", {})
    if tip.get("resolved"):
        first, last = tip["window"]
        times = np.array([row["time"] for row in rows])
        sample_times = np.r_[first, times[(times > first) & (times < last)], last]
        drag = np.interp(sample_times, times, [row["drag"] for row in rows])
        candidate = {"tip_y_amplitude": tip["amplitude"], "frequency": tip["frequency"],
                     "drag_mean": float(np.trapezoid(drag, sample_times)/(last-first))}
    comparison = {} if candidate is None else {key: {
        "value": value, "reference": config["reference"][key],
        "relative_error": abs(value/config["reference"][key]-1),
        "limit": limits[{"frequency": "frequency_relative_max"}.get(key, key+"_relative_max")],
        "provisional": not (complete and physical and failure is None and periodic["checks"].get("stable", False))}
        for key, value in candidate.items()}
    return {"schema": "m6-analysis-v1", "status": status, "reason": reason,
        "config": config, "identity": identity or {}, "accepted_steps": len(rows),
        "accepted_time": rows[-1]["time"] if rows else 0., "target_steps": target_steps,
        "target_time_complete": complete, "accepted_physics_passed": physical,
        "periodic_status": periodic["status"], "periodic": periodic,
        "reference_comparison": comparison, "solver_failure": failure,
        "physical_extrema": extrema,
        "work": {**{key: sum(row[key] for row in rows) for key in COUNTS},
            "accepted_solve_seconds": sum(row["solve_seconds"] for row in rows),
            "fair_timing_evidence": False,
            "scope": "Accepted step solves only; excludes setup, output and failed attempts. "
                     "Concurrent activity and repeatability are not certified."}}


def read_snapshot(folder):
    """Read one immutable NPZ byte snapshot, then verify its complete manifest."""
    folder = Path(folder)
    blob = (folder/"checkpoint.npz").read_bytes()
    with np.load(BytesIO(blob), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
    config = validate_fsi2_config(metadata["config"])
    state = load_checkpoint(BytesIO(blob), config=config,
        mesh_sha256=metadata["mesh_sha256"], code_sha256=metadata["code_sha256"])
    failure = None
    try:
        failure = json.loads((folder/"failure.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        pass
    if failure is not None and (failure.get("config") != config or
            failure.get("accepted_steps") != state.step or failure.get("status") != "FAIL"):
        raise ValueError("Failure record does not match the accepted snapshot; retry after the writer settles.")
    identity = {"solver_source_sha256": metadata["code_sha256"],
        "analysis_package_sha256": source_digest(), "mesh_sha256": metadata["mesh_sha256"],
        "checkpoint_sha256": hashlib.sha256(blob).hexdigest(),
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_directory": str(folder.resolve()),
        "identity_scope": "Stored solver identity and array integrity; analysis does not resume a solver."}
    rows = list(state.monitoring)
    return rows, summarize(rows, config, identity=identity, failure=failure)


def markdown(summary):
    lines = ["# FSI2 accepted-history analysis", "",
        f"Status: **{summary['status']}** — {summary['reason']}", "",
        f"Accepted time: {summary['accepted_time']:.8g} / {summary['config']['end_time']:g} s; "
        f"steps: {summary['accepted_steps']} / {summary['target_steps']}.", "",
        f"Accepted physics gates: {summary['accepted_physics_passed']}; "
        f"periodic status: {summary['periodic_status']}.", "",
        "Estimates marked provisional are not a stable, completed benchmark verdict.", "",
        "| Quantity | Value | Reference | Relative error | Gate | Provisional |",
        "| --- | ---: | ---: | ---: | ---: | --- |"]
    for key, result in summary["reference_comparison"].items():
        lines.append(f"| {key} | {result['value']:.8g} | {result['reference']:g} | "
            f"{result['relative_error']:.3%} | {result['limit']:.1%} | {result['provisional']} |")
    if not summary["reference_comparison"]:
        lines.append("| No resolved tip-motion window | — | — | — | — | True |")
    lines += ["", "| Signal | Window (s) | Frequency (Hz) | Amplitude | Mean | "
              "Period / amplitude / mean variation |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for name, stats in summary["periodic"].get("statistics", {}).items():
        if stats.get("resolved"):
            lines.append(f"| {name} | {stats['window'][0]:.5f}–{stats['window'][1]:.5f} | "
                f"{stats['frequency']:.7g} | {stats['amplitude']:.7g} | {stats['mean']:.7g} | "
                f"{stats['period_variation']:.3%} / {stats['amplitude_variation']:.3%} / {stats['mean_variation']:.3%} |")
        else:
            lines.append(f"| {name} | Insufficient complete cycles | — | — | — | — |")
    if summary["solver_failure"]:
        lines += ["", "Solver failure: "+summary["solver_failure"]["reason"]]
    lines += ["", "Units: tip displacement m, drag/lift N/m, frequency Hz. Amplitude is half peak-to-peak.",
        "", "Work totals cover accepted solves only; these timings cannot establish an algorithm ranking.",
        "", "```json", json.dumps(summary["work"], indent=2), "```", ""]
    return "\n".join(lines)


def plot_history(rows, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.array([row["time"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for name in ("tip_x", "tip_y"):
        axes[0, 0].plot(times, [row[name] for row in rows], label=name)
    for name in ("drag", "lift"):
        axes[0, 1].plot(times, [row[name] for row in rows], label=name)
    axes[1, 0].plot(times, [row["minimum_quality"] for row in rows], label="minimum quality")
    axes[1, 0].axhline(summary["config"]["acceptance"]["minimum_mesh_quality"], color="red", linestyle="--", label="gate")
    for name in ("residual", "continuity"):
        axes[1, 1].semilogy(times, np.maximum([row[name] for row in rows], np.finfo(float).tiny), label=name)
        axes[1, 1].axhline(summary["config"]["acceptance"][name+"_max"], linestyle="--", label=name+" gate")
    for ax, ylabel in zip(axes.flat, ("Tip displacement (m)", "Force per depth (N/m)", "Mesh quality", "Equation residual / continuity")):
        ax.set(xlabel="Time (s)", ylabel=ylabel)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    for name, stats in summary["periodic"].get("statistics", {}).items():
        if stats.get("resolved"):
            ax = axes[0, 0] if name == "tip_y" else axes[0, 1]
            ax.axvspan(*stats["window"], alpha=.08, color="green")
    config = summary["config"]
    fig.suptitle(f"FSI2 {config['method']}: h={config['mesh_size']:g}, dt={config['dt']:g}; "
                 f"accepted {summary['accepted_time']:.2f}/{config['end_time']:g} s — {summary['status']}")
    fig.savefig(output/"history.png", dpi=150)
    plt.close(fig)
    tip = summary["periodic"].get("statistics", {}).get("tip_y", {})
    if tip.get("resolved"):
        first, last = tip["window"]
        sample = np.r_[first, times[(times > first) & (times < last)], last]
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
        for ax, name, unit in zip(axes, ("tip_y", "drag", "lift"), ("m", "N/m", "N/m")):
            ax.plot(sample, np.interp(sample, times, [row[name] for row in rows]))
            ax.set_ylabel(f"{name} ({unit})")
            ax.grid(alpha=.25)
        axes[-1].set_xlabel("Time (s)")
        provisional = summary["reference_comparison"]["tip_y_amplitude"]["provisional"]
        fig.suptitle("Last three tip-motion cycles — "+("provisional window" if provisional else "stable window")+
                     f"; overall {summary['status']}")
        fig.savefig(output/"cycles.png", dpi=150)
        plt.close(fig)


def analyze_directory(folder, output):
    folder, output = Path(folder).resolve(), Path(output).resolve()
    if folder == output or folder in output.parents or output in folder.parents:
        raise ValueError("Analysis output must be separate from the solver directory and its parents.")
    rows, summary = read_snapshot(folder)
    output.mkdir(parents=True, exist_ok=True)
    # Remove only the optional analysis artifact from an earlier snapshot.
    (output/"cycles.png").unlink(missing_ok=True)
    plot_history(rows, summary, output)
    columns = ("signal", "resolved", "cycles", "frequency", "amplitude", "mean",
               "period_variation", "amplitude_variation", "mean_variation", "window_start", "window_end")
    with (output/"cycles.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, stats in summary["periodic"].get("statistics", {}).items():
            window = stats.get("window", [None, None])
            writer.writerow({"signal": name, **{key: stats.get(key) for key in columns[1:-2]},
                             "window_start": window[0], "window_end": window[1]})
    (output/"summary.md").write_text(markdown(summary), encoding="utf-8")
    (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Solver output directory with checkpoint.npz")
    parser.add_argument("--output", required=True, type=Path, help="Separate analysis artifact directory")
    args = parser.parse_args(argv)
    try:
        summary = analyze_directory(args.input, args.output)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"FSI2 analysis error: {error}")
        return 2
    print(f"{summary['status']}: {summary['accepted_steps']}/{summary['target_steps']} accepted steps; "
          f"periodic={summary['periodic_status']}; {args.output/'summary.json'}")
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[summary["status"]]


if __name__ == "__main__":
    raise SystemExit(main())

"""FSI2 trajectory driver; short runs are INCOMPLETE, never periodic PASS."""

import json
import hashlib
import os
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from fsi_ice.coupling.newton import NewtonOptions
from fsi_ice.coupling.partitioned import CouplingOptions
from fsi_ice.coupling.transient import TransientFSI
from fsi_ice.fluid.forces import BoundaryMonitor
from fsi_ice.fluid.taylor_hood import assemble_stokes
from fsi_ice.validation.common import write_csv, write_report
from fsi_ice.validation.fluid import inlet_profile
from fsi_ice.validation.fsi1 import SteadyFSI1Problem


def inflow_factor(time):
    if not np.isfinite(time) or time < 0:
        raise ValueError("Inlet time must be finite and nonnegative.")
    return float((1-np.cos(np.pi*time/2))/2) if time < 2 else 1.


def validate_fsi2_config(config):
    required = {"case", "mesh_size", "dt", "end_time", "order", "method", "fluid", "solid",
                "coupling", "solver", "acceptance", "reference", "provenance"}
    if not isinstance(config, dict) or set(config) != required or config["case"] != "fsi2":
        raise ValueError("Invalid FSI2 config schema.")

    def positive(value):
        return not isinstance(value, bool) and isinstance(value, (int, float)) and np.isfinite(value) and value > 0

    for name in ("mesh_size", "dt", "end_time"):
        if not positive(config[name]):
            raise ValueError(f"{name} must be finite and positive.")
    if isinstance(config["order"], bool) or not isinstance(config["order"], int) or config["order"] not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    if config["method"] not in ("fixed", "aitken", "monolithic"):
        raise ValueError("method must be fixed, aitken or monolithic.")
    steps = config["end_time"]/config["dt"]
    if not np.isfinite(steps) or steps < 1 or not np.isclose(steps, round(steps), rtol=0, atol=1e-8):
        raise ValueError("end_time must be a positive integer multiple of dt.")
    # Modified materials belong to explicitly named derived cases, not FSI2.
    if config["fluid"] != {"density": 1000., "dynamic_viscosity": 1., "mean_velocity": 1.} or \
       config["solid"] != {"density": 10000., "poisson": .4, "young": 1400000., "thickness": 1.}:
        raise ValueError("Standard FSI2 requires the verified original fluid/solid parameters.")
    if any(not positive(value) for group in ("fluid", "solid") for value in config[group].values()):
        raise ValueError("Material values must be finite positive numbers, not booleans.")
    if not isinstance(config["coupling"], dict) or set(config["coupling"]) != {
            "relaxation", "absolute_tolerance", "relative_tolerance", "max_iterations"}:
        raise ValueError("Invalid coupling controls.")
    CouplingOptions(**config["coupling"], method="fixed" if config["method"] == "monolithic" else config["method"])
    solver = config["solver"]
    if not isinstance(solver, dict) or set(solver) != {
            "single_field_tolerance", "newton_atol", "newton_rtol", "max_iterations"} or \
            any(not positive(solver[key]) for key in ("single_field_tolerance", "newton_atol", "newton_rtol")):
        raise ValueError("Invalid solver controls.")
    NewtonOptions(atol=solver["newton_atol"], rtol=solver["newton_rtol"], max_iterations=solver["max_iterations"])
    acceptance = config["acceptance"]
    limits = {"residual_max": 1e-7, "continuity_max": 1e-10,
        "period_variation_max": .02, "amplitude_variation_max": .02, "mean_variation_max": .02,
        "tip_y_amplitude_relative_max": .1, "frequency_relative_max": .05, "drag_mean_relative_max": .1}
    if not isinstance(acceptance, dict) or set(acceptance) != {*limits, "minimum_mesh_quality"} or \
            any(not positive(value) for value in acceptance.values()) or \
            any(acceptance[name] > value for name, value in limits.items()) or \
            not .1 <= acceptance["minimum_mesh_quality"] <= 1:
        raise ValueError("Invalid acceptance controls or weakened frozen gate.")
    if config["reference"] != {"tip_y_amplitude": .0816, "frequency": 1.93, "drag_mean": 215.06}:
        raise ValueError("FSI2 reference must match the verified frozen targets.")
    if not isinstance(config["provenance"], dict) or not {"url", "verified_on", "table"} <= config["provenance"].keys():
        raise ValueError("Missing FSI2 reference provenance.")
    return config


def load_fsi2_config(path):
    return validate_fsi2_config(json.loads(Path(path).read_text(encoding="utf-8")))


def cycle_statistics(times, signal, *, prominence, cycles=3):
    """Last complete peak-to-peak cycles; means are time-weighted, not midranges."""
    times, signal = np.asarray(times, dtype=float), np.asarray(signal, dtype=float)
    if times.ndim != 1 or signal.shape != times.shape or len(times) < 3 or \
            not np.all(np.isfinite(times)) or not np.all(np.isfinite(signal)) or np.any(np.diff(times) <= 0):
        raise ValueError("Cycle samples must be finite, equally sized and strictly ordered.")
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 3 or not np.isfinite(prominence) or prominence <= 0:
        raise ValueError("Require at least three cycles and positive prominence.")
    peaks = find_peaks(signal, prominence=prominence)[0]
    if len(peaks) < cycles+1:
        return {"resolved": False, "reason": "Insufficient complete cycles"}
    peaks = peaks[-cycles-1:]
    peak_times = []
    for peak in peaks:
        local_times = times[peak-1:peak+2]-times[peak]
        a, b, _ = np.polyfit(local_times, signal[peak-1:peak+2], 2)
        shift = -b/(2*a) if a < 0 else 0.
        peak_times.append(float(times[peak]+np.clip(shift, local_times[0], local_times[-1])))
    periods = np.diff(peak_times)
    amplitudes, means = [], []
    for first, last in zip(peaks[:-1], peaks[1:]):
        values = signal[first:last+1]
        amplitudes.append(float(np.ptp(values)/2))
        means.append(float(np.trapezoid(values, times[first:last+1])/(times[last]-times[first])))
    scale = max(abs(float(np.mean(means))), float(np.mean(amplitudes)), np.finfo(float).tiny)
    return {"resolved": True, "cycles": cycles, "frequency": float(1/np.mean(periods)),
        "amplitude": float(np.mean(amplitudes)), "mean": float(np.mean(means)),
        "period_variation": float(np.ptp(periods)/np.mean(periods)),
        "amplitude_variation": float(np.ptp(amplitudes)/max(np.mean(amplitudes), np.finfo(float).tiny)),
        "mean_variation": float(np.ptp(means)/scale),
        "window": [peak_times[0], peak_times[-1]]}


def analyze_fsi2(rows, config):
    late = [row for row in rows if row["time"] >= 2.]
    if len(late) < 3:
        return {"status": "INCOMPLETE", "reason": "No post-startup periodic window", "checks": {}}
    times = [row["time"] for row in late]
    stats = {name: cycle_statistics(times, [row[name] for row in late], prominence=prominence)
             for name, prominence in (("tip_y", .005), ("drag", 5.), ("lift", 5.))}
    if not all(stat["resolved"] for stat in stats.values()):
        return {"status": "INCOMPLETE", "reason": "Insufficient resolved complete cycles", "statistics": stats, "checks": {}}
    acceptance = config["acceptance"]
    stable = all(stat[key] <= acceptance[key+"_max"] for stat in stats.values()
                 for key in ("period_variation", "amplitude_variation", "mean_variation"))
    if not stable:
        return {"status": "INCOMPLETE", "reason": "Periodic response has not stabilized", "statistics": stats, "checks": {"stable": False}}
    # Drag is evaluated over the y-motion window, which covers full FSI cycles.
    first, last = stats["tip_y"]["window"]
    window_times = np.r_[first, [time for time in times if first < time < last], last]
    window_drag = np.interp(window_times, times, [row["drag"] for row in late])
    drag_mean = float(np.trapezoid(window_drag, window_times)/(last-first))
    reference = config["reference"]
    errors = {"tip_y_amplitude_relative": abs(stats["tip_y"]["amplitude"]/reference["tip_y_amplitude"]-1),
              "frequency_relative": abs(stats["tip_y"]["frequency"]/reference["frequency"]-1),
              "drag_mean_relative": abs(drag_mean/reference["drag_mean"]-1)}
    checks = {name: error <= acceptance[name+"_max"] for name, error in errors.items()}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "statistics": stats,
            "drag_mean": drag_mean, "errors": errors, "checks": {"stable": True, **checks}}


def build_fsi2_problem(config, output):
    problem = SteadyFSI1Problem(config, config["mesh_size"], Path(output)/"mesh")

    def monitor(view, blocks):
        operators = assemble_stokes(view.geometry.current_mesh, density=1000., viscosity=1.)
        # Both BoundaryMonitor and interface.g report force on the solid.
        from types import SimpleNamespace
        solution = SimpleNamespace(velocity=blocks["u_f"], pressure=blocks["p_f"])
        cylinder = BoundaryMonitor(operators, "cylinder-rigid-wall").force(solution)
        force = cylinder + blocks["g"].sum(axis=0)
        return {"tip_x": float(blocks["d_s"][problem.tip_node, 0]),
                "tip_y": float(blocks["d_s"][problem.tip_node, 1]),
                "drag": float(force[0]), "lift": float(force[1])}

    return TransientFSI(problem.topology, problem.extension, problem.material,
        density=1000., viscosity=1.,
        boundary_velocity=lambda time, points: inflow_factor(time)*inlet_profile(points, 1.),
        fluid_tags=["inlet", "walls", "cylinder-rigid-wall"], solid_nodes=problem.fixed_solid_nodes,
        dt=config["dt"], order=config["order"], monitor=monitor)


def run_fsi2_cli(config_path, output, *, method=None, end_time=None, mesh_size=None, progress=print):
    """Fresh trajectory. Does not resume/reuse old solution files implicitly."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("report.json", "failure.json", "monitor.csv", "solution.npz"):
        (output/name).unlink(missing_ok=True)
    stage, driver, config = "input", None, None
    try:
        config = load_fsi2_config(config_path)
        for key, value in (("method", method), ("end_time", end_time), ("mesh_size", mesh_size)):
            if value is not None:
                config[key] = value
        validate_fsi2_config(config)
        stage = "setup"
        driver = build_fsi2_problem(config, output)
        solver, acceptance = config["solver"], config["acceptance"]
        options = NewtonOptions(atol=solver["newton_atol"], rtol=solver["newton_rtol"],
                                max_iterations=solver["max_iterations"])
        stage = "solve"
        for _ in range(round(config["end_time"]/config["dt"])):
            stage = "solve"
            state = driver.step(config["method"], coupling=config["coupling"], newton=options,
                single_tolerance=solver["single_field_tolerance"], residual_max=acceptance["residual_max"],
                continuity_max=acceptance["continuity_max"], minimum_quality=acceptance["minimum_mesh_quality"])
            row = state.monitoring[-1]
            if progress is not None:
                progress(f"FSI2 {config['method']} step {state.step}, t={state.time:.4f}, R={row['residual']:.2e}", flush=True)
            # Only committed rows reach disk; keep partial evidence on failure.
            stage = "monitor_output"
            write_csv(output/"monitor.csv", state.monitoring)
        stage = "analysis"
        analysis = analyze_fsi2(state.monitoring, config)
        stage = "output"
        np.savez_compressed(output/"solution.npz", solid=state.solid_state, velocity=state.velocity,
                            pressure=state.pressure, mesh_displacement=state.mesh_displacement,
                            interface_g=state.interface["g"], mesh_velocity=state.mesh_velocity,
                            time=state.time, step=state.step)
        report = write_report(output, "fsi2", config, analysis,
            {"periodic_validation_complete": analysis["status"] != "INCOMPLETE", **analysis["checks"]}, {
            "scope": "physical trajectory; PASS requires stable cycles and reference gates",
            "residual": "max_i |T21 constrained R_i/S_i|; blocks S=(1,1000,.41,1,.41,215.06)",
            "force": "cylinder + beam, N per unit depth", "restart": "not implemented",
            "reference": config["provenance"]})
        report["status"] = analysis["status"]
        report["accepted_steps"] = state.step
        report["mesh_sha256"] = hashlib.sha256((output/"mesh"/"fsi1.msh").read_bytes()).hexdigest()
        report["threads"] = {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
        (output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        return report
    except Exception as error:
        (output/"report.json").unlink(missing_ok=True)
        (output/"solution.npz").unlink(missing_ok=True)
        failure = {"case": "fsi2", "status": "FAIL", "stage": stage, "reason": str(error),
                   "config_path": str(config_path), "config": None if stage == "input" else config,
                   "accepted_steps": 0 if driver is None else driver.coordinator.state.step}
        (output/"failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        raise

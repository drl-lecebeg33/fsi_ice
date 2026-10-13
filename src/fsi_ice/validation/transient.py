"""Physical CSM3/CFD3 runs, separate mesh/time studies and period diagnostics.

中文说明：本模块负责瞬态单场基准 CSM3（固体）与 CFD3（流体）的实际物理时间推进，
以及相互独立的网格研究与时间步研究，并做周期/谐波诊断。关键约定：固定步长 BDF2
（状态量一阶形式、用 BDF1 启动），每个时间步只有在子问题求解收敛后才接受；网格研究与
时间步研究分别只变一个因素（固定另一因素）；周期统计取“最后若干完整周期”，用抛物线
插值定峰值时刻，均值不是时间平均。后处理模式只重算已保存完整历史的诊断，不重新求解
物理方程；粗网格/大时间步的欠解析情形如实记录，不作为与参考值相符的通过依据。
"""

from pathlib import Path
import json
import hashlib
import re
import time

import numpy as np
from scipy.signal import find_peaks

from fsi_ice.mesh.turek import csm_mesh, cfd_mesh, read_gmsh2
from fsi_ice.solid.linear import ElasticMaterial, assemble_elasticity
from fsi_ice.solid.dynamics import solve_solid_step
from fsi_ice.fluid.taylor_hood import assemble_stokes
from fsi_ice.fluid.dynamics import solve_flow_step
from fsi_ice.fluid.forces import BoundaryMonitor
from fsi_ice.fluid.linear_solver import ReusedLUSolver
from fsi_ice.time import BDFHistory
from fsi_ice.validation.common import write_csv, write_report
from fsi_ice.validation.fluid import inlet_profile, CFD_SOURCE
from fsi_ice.validation.solid import CSM_SOURCE


def load_transient_config(path):
    """中文：读取并校验瞬态基准配置（CSM3 或 CFD3）。要求键恰为
    {case, mesh_levels, time_steps, mesh_study_dt, end_time, acceptance}；网格层数
    与时间步数均≥3 且严格加密/减小；``mesh_study_dt`` 必须属于 ``time_steps``、
    ``end_time``≥4 且可被每个 dt 整除；验收阈值全为有限正数。非法时抛 ``ValueError``。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"case", "mesh_levels", "time_steps", "mesh_study_dt", "end_time", "acceptance"}
    if not isinstance(data, dict) or set(data) != required or data["case"] not in ("csm3", "cfd3"):
        raise ValueError("Invalid transient benchmark schema.")
    levels = data["mesh_levels"]
    if not isinstance(levels, list) or len(levels) < 3:
        raise ValueError("Require at least three mesh levels.")
    if data["case"] == "csm3":
        if any(not isinstance(pair, list) or len(pair) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in pair) for pair in levels) or any(any(b <= a for a, b in zip(first, second)) for first, second in zip(levels, levels[1:])):
            raise ValueError("CSM meshes must strictly refine nx and ny.")
    else:
        if any(isinstance(h, bool) or not isinstance(h, (int, float)) or not np.isfinite(h) or h <= 0 for h in levels) or any(b >= a for a, b in zip(levels, levels[1:])):
            raise ValueError("CFD mesh sizes must be positive and decreasing.")
    steps = data["time_steps"]
    if not isinstance(steps, list) or len(steps) < 3 or any(isinstance(dt, bool) or not isinstance(dt, (int, float)) or not np.isfinite(dt) or dt <= 0 for dt in steps) or any(b >= a for a, b in zip(steps, steps[1:])):
        raise ValueError("Require at least three decreasing time steps.")
    if data["mesh_study_dt"] not in steps or not np.isfinite(data["end_time"]) or data["end_time"] < 4:
        raise ValueError("Invalid mesh-study step or final time.")
    if any(abs(round(data["end_time"]/dt)*dt-data["end_time"]) > 1e-10 for dt in steps):
        raise ValueError("Every dt must divide end_time.")
    acceptance = data["acceptance"]
    if not isinstance(acceptance, dict) or set(acceptance) != {"frequency_relative_error_max", "amplitude_relative_error_max", "mean_relative_error_max", "small_mean_absolute_error_max", "period_variation_max", "residual_max", "mass_error_max"} or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0 for v in acceptance.values()):
        raise ValueError("Invalid transient acceptance thresholds.")
    return data


def periodic_statistics(times, signal, *, minimum_prominence, cycles=3):
    """中文：从一条时间序列提取周期统计（频率/周期波动/幅值/均值）。以 ``find_peaks``
    按 ``minimum_prominence`` 找峰，取最后 ``cycles`` 个完整周期；峰值时刻用三点抛物线
    插值细化（比采样点更准），周期为相邻峰时刻之差。返回字典含 frequency[Hz]、
    mean、amplitude、period_variation（周期极差/均值）、amplitude_variation 与统计窗口。
    完整周期不足时返回 ``resolved=False``。注意：mean 取最后一段的 (max+min)/2，不是
    时间平均。
    """
    times, signal = np.asarray(times), np.asarray(signal)
    peaks = find_peaks(signal, prominence=minimum_prominence)[0]
    peaks = peaks[(peaks > 0) & (peaks < len(signal)-1)]  # 去掉端点峰，保证可做抛物线插值
    if len(peaks) < cycles+1:
        return {"resolved": False, "reason": "Insufficient resolved complete cycles", "cycles": max(len(peaks)-1, 0)}
    peaks = peaks[-cycles-1:]  # 保留最后 cycles 个完整周期的边界峰
    peak_times, amplitudes, means = [], [], []
    for peak in peaks:
        # 三点抛物线顶点偏移量，把离散峰时刻细化到亚采样点精度。
        left, center, right = signal[peak-1:peak+2]
        denominator = left-2*center+right
        shift = 0.5*(left-right)/denominator if denominator != 0 else 0
        peak_times.append(float(times[peak]+shift*(times[peak+1]-times[peak])))
    for first, last in zip(peaks, peaks[1:]):
        # 每个周期段取半极差为幅值、中点值为“均值”。
        segment = signal[first:last+1]
        amplitudes.append(float((segment.max()-segment.min())/2))
        means.append(float((segment.max()+segment.min())/2))
    periods = np.diff(peak_times)
    amplitude = amplitudes[-1]
    period_variation = float(np.ptp(periods)/np.mean(periods))  # 周期极差相对均值
    amplitude_variation = float(np.ptp(amplitudes)/max(np.mean(amplitudes), 1e-30))
    return {"resolved": True, "cycles": cycles, "frequency": float(1/np.mean(periods)),
            "mean": means[-1], "amplitude": amplitude, "period_variation": period_variation,
            "amplitude_variation": amplitude_variation,
            "window": [float(times[peaks[0]]), float(times[peaks[-1]])]}


def run_trajectory(case, mesh_level, dt, end_time, output, *, progress=print):
    """中文：推进一条完整物理轨迹并返回汇总。``case`` 取 csm3/cfd3，``mesh_level`` 为
    CSM 网格 (nx,ny) 或 CFD 网格尺寸 h，``dt`` 固定步长、``end_time`` 终止时间。
    逐步求解子问题，仅在收敛后接受状态；按秒写 history.csv，结束写 summary.json。
    CSM3：平面应变 P1 StVK + 重力，监测翼尖位移；CFD3：P2/P1 固定圆/方柱绕流，入口
    余弦爬升、监测阻力/升力与质量守恒。失败时经 :func:`trajectory_failure` 记录后抛出。
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # A new attempt must not inherit a previous completion/failure marker.
    # Existing CSV is retained until new accepted monitors replace it.
    (output / "summary.json").unlink(missing_ok=True)
    (output / "failure.json").unlink(missing_ok=True)
    started = time.perf_counter()
    rows = []
    if case == "csm3":
        mesh = csm_mesh(*mesh_level)
        material = ElasticMaterial(1.4e6, 0.4, density=1000)  # 与 CSM3 基准一致的材料参数
        _, force = assemble_elasticity(mesh, material, body_force=[0, -2000])  # 重力体力
        fixed = mesh.boundary_nodes("left")
        tip = int(np.argmin(np.linalg.norm(mesh.points-[0.6, 0.2], axis=1)))  # 最接近 (0.6,0.2) 的节点
        history = BDFHistory(np.zeros((2, mesh.n_nodes, 2)), dt)
    else:
        mesh = cfd_mesh(mesh_level, output / "mesh")
        operators = assemble_stokes(mesh, viscosity=1, density=1000)
        history = BDFHistory(np.zeros_like(operators.space.points), dt)
        initial = np.zeros(operators.n_total)
        obstacle = BoundaryMonitor(operators, "obstacle")
        inlet = BoundaryMonitor(operators, "inlet")
        outlet = BoundaryMonitor(operators, "outlet")
        linear_solver = ReusedLUSolver(refresh_interval=20, velocity_dofs=operators.n_velocity, density=operators.density)
    maximum_residual, maximum_mass, minimum_jacobian = 0.0, 0.0, 1.0  # 全程最坏监控量
    for step in range(round(end_time/dt)):
        context = history.context()
        if case == "csm3":
            try:
                state, solution = solve_solid_step(mesh, material, force, fixed, context,
                                                   initial=history.states[0][0], tolerance=1e-8)
            except (RuntimeError, ValueError, FloatingPointError) as error:
                raise trajectory_failure(case, mesh_level, dt, context.time, output, rows, error) from error
            history.accept(context, state)
            values = state[0, tip]
            residual = solution.nonlinear_history[-1]
            minimum_jacobian = min(minimum_jacobian, solution.minimum_jacobian)
            row = {"time": context.time, "tip_x": float(values[0]), "tip_y": float(values[1]),
                   "residual": residual, "newton_iterations": len(solution.nonlinear_history)-1,
                   "minimum_jacobian": solution.minimum_jacobian}
        else:
            # 入口速度余弦爬升：t<2 时 0→1，t≥2 后保持 1（避免启动冲击）。
            amplitude = (1-np.cos(np.pi*context.time/2))/2 if context.time < 2 else 1.0
            try:
                solution = solve_flow_step(operators, lambda points: inlet_profile(points, 2, amplitude), context,
                                          tags=["inlet", "walls", "obstacle"], pressure_pin=None,
                                          initial=initial, tolerance=1e-8, linear_solver=linear_solver)
            except (RuntimeError, ValueError, FloatingPointError) as error:
                raise trajectory_failure(case, mesh_level, dt, context.time, output, rows, error) from error
            initial = np.r_[solution.velocity.ravel(), solution.pressure]  # 上一步解作为下一步初值
            history.accept(context, solution.velocity)
            values = obstacle.force(solution)
            residual = solution.relative_residual
            inflow, outflow = inlet.flux(solution), outlet.flux(solution)
            flux_error = abs(inflow+outflow)/max(abs(inflow), 1e-12)  # 相对通量不平衡
            maximum_mass = max(maximum_mass, flux_error, solution.continuity_residual)
            row = {"time": context.time, "drag": float(values[0]), "lift": float(values[1]),
                   "residual": residual, "newton_iterations": len(solution.nonlinear_history)-1,
                   "continuity": solution.continuity_residual, "relative_flux_error": flux_error}
        rows.append(row)
        maximum_residual = max(maximum_residual, residual)
        if (step+1) % max(1, round(1/dt)) == 0:  # 约每秒落盘一次，便于中断后查看
            write_csv(output / "history.csv", rows)
            progress(f"{case.upper()} mesh={mesh_level}, dt={dt:g}, t={context.time:.2f}: {values}", flush=True)
    write_csv(output / "history.csv", rows)
    statistics = trajectory_statistics(case, rows)
    # 汇总全程最坏残差/质量误差/最小雅可比，以及周期统计与耗时；失败标记已在开头清除。
    summary = {"mesh": mesh_level, "dt": dt, "cells": mesh.n_cells, "statistics": statistics,
               "maximum_residual": maximum_residual, "maximum_mass_error": maximum_mass,
               "minimum_jacobian": minimum_jacobian, "seconds": time.perf_counter()-started}
    if case == "cfd3":
        # CFD3 额外记录线性求解器配置与代价（重用的 LU 预条件 + 真残差校验）。
        summary["linear_solver"] = {"method": "full Newton + GMRES, reused LU preconditioner with true-residual checks and refactor fallback",
                                    "factorizations": linear_solver.factorizations, "gmres_iterations": linear_solver.iterations}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return summary


def trajectory_failure(case, mesh_level, dt, failed_time, output, rows, error):
    """Keep successful monitors and identify a failed candidate before acceptance.

    中文：在候选步被接受前失败时，保留已成功的监控行并写 ``failure.json``（记录失败
    时刻、已接受步数与原因），返回一个带上下文的 ``RuntimeError`` 供上层抛出。
    """
    if rows:
        write_csv(output / "history.csv", rows)
    failure = {"status": "FAIL", "case": case, "mesh": mesh_level, "dt": dt,
               "failed_time": failed_time, "accepted_steps": len(rows), "reason": str(error)}
    (output / "failure.json").write_text(json.dumps(failure, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return RuntimeError(f"{case.upper()} mesh={mesh_level}, dt={dt:g}, step={len(rows)+1}, t={failed_time:g}: {error}")


def trajectory_statistics(case, rows):
    """中文：从监控行提取周期统计。CSM3 以翼尖 y 为主信号、CFD3 以升力为主信号；
    CSM3 的正向翼尖偏移很小，故对其取负（找波谷）以稳定定峰，且 t=0 的启动峰值不在
    已接受采样内。CSM3 另单独统计翼尖 x；CFD3 用最后一个完整升力周期内的阻力取
    中点/半极差得到 drag_mean/amplitude。
    """
    signal_name = "tip_y" if case == "csm3" else "lift"
    # Use minima for CSM3: its positive tip excursions are tiny and the startup
    # maximum at t=0 is absent from the sampled accepted states.
    signal = np.array([row[signal_name] for row in rows])
    sign = -1 if case == "csm3" else 1  # CSM3 取负后找峰等价于原信号找谷
    statistics = periodic_statistics([row["time"] for row in rows], sign*signal,
                                     minimum_prominence=0.02 if case == "csm3" else 100)
    if statistics["resolved"]:
        statistics["mean"] *= sign  # 还原取负带来的均值符号
    if case == "csm3":
        x_statistics = periodic_statistics([row["time"] for row in rows],
                                           -np.array([row["tip_x"] for row in rows]), minimum_prominence=0.002)
        if x_statistics["resolved"]:
            x_statistics["mean"] *= -1
        statistics["tip_x"] = x_statistics
    if case == "cfd3" and statistics["resolved"]:
        # 以最后完整升力周期为窗口，统计该窗口内的阻力中点与半极差。
        last_period_start = statistics["window"][1]-1/statistics["frequency"]
        selected = np.array([last_period_start <= row["time"] <= statistics["window"][1] for row in rows])
        drag = np.array([row["drag"] for row in rows])[selected]
        statistics["drag_mean"] = float((drag.max()+drag.min())/2)
        statistics["drag_amplitude"] = float((drag.max()-drag.min())/2)
    return statistics


def read_complete_trajectory(case, mesh_level, dt, end_time, folder):
    """Recompute diagnostics from complete accepted-step monitors, not a restart.

    The saved summary identifies the mesh and dt. CSV and summary digests are
    included in the new report; original numerical data are never overwritten.
    This checks saved evidence, and does not re-solve the discrete equations.

    中文：从“已保存的完整接受步监控”重算诊断，而非重启求解；不重新求解离散方程。保存
    的 summary 用于核对网格与 dt；本次报告会附带 CSV/summary（及网格文件）的 SHA-256
    摘要，且绝不覆盖原始数值数据。校验：无 failure 标记、字段齐全、步数等于
    ``round(end_time/dt)``、时间采样为 dt 整数倍且监控有限、计数类诊断非负、
    CFD 几何中的网格尺寸与配置一致、单元数一致。
    """
    folder = Path(folder)
    if (folder / "failure.json").exists():
        raise ValueError(f"Saved trajectory is marked failed: {folder}")
    summary_path, history_path = folder / "summary.json", folder / "history.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("mesh") != mesh_level or summary.get("dt") != dt:
        raise ValueError(f"Saved mesh/dt do not match configuration: {folder}")
    data = np.atleast_1d(np.genfromtxt(history_path, delimiter=",", names=True))
    fields = {"time", "residual", "newton_iterations"} | ({"tip_x", "tip_y", "minimum_jacobian"}
              if case == "csm3" else {"drag", "lift", "continuity", "relative_flux_error"})
    if set(data.dtype.names or ()) != fields or len(data) != round(end_time/dt):
        raise ValueError(f"Incomplete or incompatible accepted-step history: {folder}")
    if any(not np.all(np.isfinite(data[name])) for name in fields) or not np.allclose(
            data["time"], dt*np.arange(1, len(data)+1), rtol=0, atol=1e-9):
        raise ValueError(f"Invalid time samples or nonfinite monitors: {folder}")
    for name in fields & {"residual", "newton_iterations", "continuity", "relative_flux_error"}:
        if np.any(data[name] < 0):
            raise ValueError(f"Negative diagnostic {name}: {folder}")
    artifacts = [history_path, summary_path]
    if case == "csm3":
        cells = csm_mesh(*mesh_level).n_cells
    else:
        mesh_path, geometry_path = folder / "mesh" / "mesh.msh", folder / "mesh" / "mesh.geo"
        cells = read_gmsh2(mesh_path).n_cells
        # 从 .geo 中解析 lc= 网格尺寸，核对与配置一致。
        size = re.search(r"^lc\s*=\s*([^;]+);", geometry_path.read_text(), re.MULTILINE)
        if size is None or float(size[1]) != mesh_level:
            raise ValueError(f"Saved geometry mesh size does not match configuration: {folder}")
        artifacts += [mesh_path, geometry_path]
    if summary.get("cells") != cells:
        raise ValueError(f"Saved cell count does not match mesh: {folder}")
    rows = [{name: float(row[name]) for name in fields} for row in data]
    summary["statistics"] = trajectory_statistics(case, rows)
    summary["maximum_residual"] = float(np.max(data["residual"]))
    summary["maximum_mass_error"] = 0.0 if case == "csm3" else float(max(
        np.max(data["continuity"]), np.max(data["relative_flux_error"])))
    summary["minimum_jacobian"] = float(min(1.0, np.min(data["minimum_jacobian"]))) if case == "csm3" else 1.0
    summary["saved_evidence"] = {"folder": str(folder.resolve()), "files_sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}}
    provenance = folder / "provenance.json"
    if provenance.exists():
        summary["saved_evidence"]["origin"] = json.loads(provenance.read_text(encoding="utf-8"))
    return summary


def run_transient(config, output, *, plot=True, progress=print, postprocess_only=False):
    """中文：瞬态基准总驱动。先做网格研究（固定 dt=mesh_study_dt，遍历 mesh_levels），
    再做时间步研究（固定最细网格，遍历 time_steps），其中与 mesh_study_dt 相同的时间步
    复用网格研究最后一行的结果。随后按参考值与验收阈值生成检查项，可选画图，最后写
    report。``postprocess_only=True`` 时只对已保存的完整历史重算诊断，不重新求解。
    """
    output = Path(output)
    case = config["case"]
    mesh_rows, time_rows = [], []
    runner = read_complete_trajectory if postprocess_only else run_trajectory
    options = {} if postprocess_only else {"progress": progress}
    for index, level in enumerate(config["mesh_levels"]):
        mesh_rows.append(runner(case, level, config["mesh_study_dt"], config["end_time"],
                                output / f"mesh_{index}", **options))
    finest_mesh = config["mesh_levels"][-1]
    for dt in config["time_steps"]:
        if dt == config["mesh_study_dt"]:
            row = mesh_rows[-1]
        else:
            row = runner(case, finest_mesh, dt, config["end_time"], output / f"dt_{dt:g}", **options)
        time_rows.append(row)
    final = time_rows[-1]  # 以最细网格、最小时间步的解作为验收依据
    stats = final["statistics"]
    acceptance = config["acceptance"]
    # 与基准来源一致的冻结参考值（CSM3 频率/均值/幅值、翼尖 x；CFD3 同升/阻力）。
    reference = {"frequency": 1.0995, "mean": -0.063607, "amplitude": 0.065160,
                 "tip_x": {"frequency": 1.0995, "mean": -0.014305, "amplitude": 0.014305}} if case == "csm3" else {
        "frequency": 4.3956, "mean": -11.893, "amplitude": 437.81, "drag_mean": 439.45, "drag_amplitude": 5.6183}
    checks = {"resolved_cycles": stats["resolved"],
              "residual": all(r["maximum_residual"] <= acceptance["residual_max"] for r in mesh_rows+time_rows),
              "positive_jacobian": all(r["minimum_jacobian"] > 0 for r in mesh_rows+time_rows),
              "mass_balance": all(r["maximum_mass_error"] <= acceptance["mass_error_max"] for r in mesh_rows+time_rows)}
    for name in ("frequency", "amplitude"):
        threshold = acceptance["frequency_relative_error_max" if name == "frequency" else "amplitude_relative_error_max"]
        checks[name+"_reference"] = bool(stats["resolved"] and abs(stats[name]/reference[name]-1) <= threshold)
    checks["stable_periods"] = bool(stats["resolved"] and max(stats["period_variation"], stats["amplitude_variation"]) <= acceptance["period_variation_max"])
    if case == "csm3":
        checks["mean_reference"] = bool(stats["resolved"] and abs(stats["mean"]/reference["mean"]-1) <= acceptance["mean_relative_error_max"])
        x_stats = stats["tip_x"]
        checks["tip_x_resolved"] = x_stats["resolved"]
        for name, tolerance in (("frequency", "frequency_relative_error_max"),
                                ("amplitude", "amplitude_relative_error_max"), ("mean", "mean_relative_error_max")):
            checks["tip_x_"+name+"_reference"] = bool(x_stats["resolved"] and
                abs(x_stats[name]/reference["tip_x"][name]-1) <= acceptance[tolerance])
        checks["tip_x_stable_periods"] = bool(x_stats["resolved"] and
            max(x_stats["period_variation"], x_stats["amplitude_variation"]) <= acceptance["period_variation_max"])
    else:
        checks["lift_mean_reference"] = bool(stats["resolved"] and abs(stats["mean"]-reference["mean"]) <= acceptance["small_mean_absolute_error_max"])
        checks["drag_mean_reference"] = bool(stats["resolved"] and abs(stats["drag_mean"]/reference["drag_mean"]-1) <= acceptance["mean_relative_error_max"])
        checks["drag_amplitude_reference"] = bool(stats["resolved"] and abs(stats["drag_amplitude"]/reference["drag_amplitude"]-1) <= acceptance["amplitude_relative_error_max"])
    if plot:
        save_transient_plot(output, config, mesh_rows, time_rows)
    return write_report(output, case, config, {"mesh_study": mesh_rows, "time_study": time_rows}, checks,
                        {"source": CSM_SOURCE if case == "csm3" else CFD_SOURCE, "checked": "2026-10-10", "reference": reference,
                         "statistics": "Last three complete cycles; parabolic extrema timing; last-cycle midpoint/half-range; mean is not time average.",
                         "frequency_signal": "tip_y (tip_x is checked separately)" if case == "csm3" else "lift; drag midpoint and half-range use the last complete lift period",
                         "model": "plane-strain P1 StVK, gravity step from rest" if case == "csm3" else "P2/P1 fixed circle and flag; rho=1000, mu=1, mean inlet=2; cosine ramp until t=2; zero traction outlet",
                         "time": "Fixed-step BDF2 on first-order state, BDF1 startup; accept only after converged solve",
                         "execution": "Postprocess complete saved histories; no new physical solves or state restart" if postprocess_only else "New physical trajectories",
                         "studies": "Three spatial meshes at fixed dt; three dt at fixed finest mesh; coarse underresolution is recorded, not passed as a reference match."})


def save_transient_plot(output, config, meshes, times):
    """中文：生成网格研究与时间步研究的对比图并保存为 ``<case>.png``。上排为不同 dt
    的时程（固定最细网格），下排为不同网格的末尾 2 秒时程；CSM3 画翼尖 x/y[m]，
    CFD3 画阻力/升力[N/m]。仅用于展示，不参与判定。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fields = ("tip_x", "tip_y") if config["case"] == "csm3" else ("drag", "lift")
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for dt in config["time_steps"]:
        folder = output / (f"mesh_{len(meshes)-1}" if dt == config["mesh_study_dt"] else f"dt_{dt:g}")
        data = np.genfromtxt(folder / "history.csv", delimiter=",", names=True)
        for axis, field in zip(axes[0], fields):
            axis.plot(data["time"], data[field], label=f"dt={dt:g}", linewidth=0.8)
    for index, level in enumerate(config["mesh_levels"]):
        data = np.genfromtxt(output / f"mesh_{index}" / "history.csv", delimiter=",", names=True)
        data = data[data["time"] >= max(0, config["end_time"]-2)]
        label = f"mesh={level}, cells={meshes[index]['cells']}"
        for axis, field in zip(axes[1], fields):
            axis.plot(data["time"], data[field], label=label, linewidth=0.8)
            axis.set_xlim(max(0, config["end_time"]-2), config["end_time"])
    for index, row in enumerate(axes):
        for axis, field in zip(row, fields):
            axis.set(xlabel="Time [s]", ylabel=field+(" [m]" if config["case"] == "csm3" else " [N/m]"),
                     title="Time-step study: fixed finest mesh" if index == 0 else f"Mesh study: fixed dt={config['mesh_study_dt']:g}")
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=8)
    figure.suptitle(config["case"].upper()+": measured mesh and time-step studies")
    figure.savefig(output / (config["case"]+".png"), dpi=160)
    plt.close(figure)

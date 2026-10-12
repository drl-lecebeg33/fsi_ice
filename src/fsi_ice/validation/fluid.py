"""Taylor-Hood spatial/time verification and fixed-body CFD1/2 benchmarks.

中文说明：本模块做流体单场验证：制造解（Stokes 与 Navier--Stokes）空间收敛、Poiseuille
精确解，以及固定体 Turek--Hron CFD1/CFD2 基准（阻力/升力对照参考值）。速度/压力分别
用 P2/P1 Taylor--Hood 元；力为障碍物表面 -integral(sigma*n) 的合力，单位 N/m（单位厚度）。
"""

from pathlib import Path
import time

import numpy as np

from fsi_ice.mesh import rectangle_mesh
from fsi_ice.mesh.turek import cfd_mesh
from fsi_ice.fluid.taylor_hood import assemble_stokes, solve_stokes, solve_navier_stokes, flow_error_norms
from fsi_ice.fluid.dynamics import solve_flow_step
from fsi_ice.fluid.forces import BoundaryMonitor
from fsi_ice.time import BDFHistory
from fsi_ice.validation.common import add_orders, write_csv, write_report


CFD_SOURCE = "https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_tests/fsi_cfd_tests.html"


def exact_velocity(points):
    """制造解速度场 u=(sin(pi x)cos(pi y), -cos(pi x)sin(pi y))。

    输入形状 (n,2) 的坐标点，返回同形状速度；无物理单位约定，用于收敛验证。
    """
    x, y = np.pi*points[:, 0], np.pi*points[:, 1]
    return np.column_stack((np.sin(x)*np.cos(y), -np.cos(x)*np.sin(y)))


def exact_pressure(points):
    """制造解压力场 p=sin(pi x)sin(pi y)（非线性，便于独立检验 P1 压力收敛）。"""
    # Nonlinear pressure gives a meaningful independent P1 pressure convergence test.
    return np.sin(np.pi*points[:, 0])*np.sin(np.pi*points[:, 1])


def manufactured_force(points, *, convection=True):
    """制造解对应的体积力，使精确场满足动量方程。

    由 -laplacian(u) + grad(p) 得 2*pi^2*u + grad(p)；convection=True 时再加对流项
    u·grad(u)，得到完整 Navier--Stokes 的体积力（rho=1, mu=1 的标准化形式）。
    """
    x, y = np.pi*points[:, 0], np.pi*points[:, 1]
    gradient_p = np.pi*np.column_stack((np.cos(x)*np.sin(y), np.sin(x)*np.cos(y)))
    force = 2*np.pi**2*exact_velocity(points)+gradient_p
    if convection:
        force += np.pi*np.column_stack((np.sin(x)*np.cos(x), np.sin(y)*np.cos(y)))
    return force


def temporal_fluid(time_steps, *, order, mesh_level=4):
    """时间收敛验证：u=exp(-t)*(y*(1-y),0)、p=0，精确空间表示，终止时刻 1。

    对每个 dt 在固定网格上推进 BDF(order) 时间步，最后与解析解比较速度 L2 误差，
    并用 add_orders 得到时间收敛阶。返回按 dt 排列的结果行。
    """
    mesh = rectangle_mesh(mesh_level)
    def profile(points):
        return np.column_stack((points[:, 1]*(1-points[:, 1]), np.zeros(len(points))))
    initial_operators = assemble_stokes(mesh, viscosity=0.01)
    rows = []
    for dt in time_steps:
        history = BDFHistory(profile(initial_operators.space.points), dt, order=order)
        initial = None
        residual, continuity = [], []
        for _ in range(round(1/dt)):
            context = history.context()
            amplitude = np.exp(-context.time)
            operators = assemble_stokes(mesh, viscosity=0.01,
                                        body_force=lambda points: amplitude*(-profile(points)+[0.02, 0]))
            solution = solve_flow_step(operators, lambda points: amplitude*profile(points), context,
                                      initial=initial, tolerance=1e-11)
            initial = np.r_[solution.velocity.ravel(), solution.pressure]
            history.accept(context, solution.velocity)
            residual.append(solution.relative_residual)
            continuity.append(solution.continuity_residual)
        errors = flow_error_norms(operators, solution, lambda points: np.exp(-1)*profile(points),
                                 lambda points: np.zeros(len(points)))
        rows.append({"dt": dt, **errors, "maximum_residual": max(residual), "maximum_continuity": max(continuity)})
    return add_orders(rows, ("velocity_l2",), "dt")


def inlet_profile(points, mean_velocity, amplitude=1.0):
    """通道入口抛物线速度剖面（CFD 几何高度 0.41 m）。

    仅对 x≈0 的入口节点赋 x 方向速度 6*U*amp*y*(0.41-y)/0.41^2（峰值 1.5*U），使
    截面平均速度等于 mean_velocity*amplitude；其余分量为零。points 形状 (n,2)。
    """
    values = np.zeros_like(points)
    selected = np.isclose(points[:, 0], 0, rtol=0, atol=1e-12)
    values[selected, 0] = 6*mean_velocity*amplitude*points[selected, 1]*(0.41-points[selected, 1])/0.41**2
    return values


def run_fluid(config, output, *, plot=True, progress=print):
    """运行流体单场全套验证并写出报告，返回报告字典。

    依次做制造解空间收敛（Stokes 与 Navier--Stokes）、Poiseuille 精确解、BDF1/2 时间
    收敛，以及 CFD1/CFD2 基准；每个流体网格复用同一离散方程，仅改变来流 Reynolds 数
    （对应平均速度 0.2/1.0）。config 需含 manufactured_meshes/fluid_meshes/time_steps
    与 acceptance。
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    acceptance = config["acceptance"]
    manufactured, poiseuille, cfd = [], [], []
    # 分别在 Stokes 与 Navier--Stokes（含对流项）下验证空间收敛。
    for nonlinear in (False, True):
        rows = []
        for level in config["manufactured_meshes"]:
            mesh = rectangle_mesh(level)
            operators = assemble_stokes(mesh, body_force=lambda points: manufactured_force(points, convection=nonlinear))
            solution = solve_navier_stokes(operators, exact_velocity, tolerance=1e-10) if nonlinear else solve_stokes(operators, exact_velocity)
            errors = flow_error_norms(operators, solution, exact_velocity, exact_pressure)
            rows.append({"case": "Navier-Stokes" if nonlinear else "Stokes", "n": level, "h": mesh.h, **errors,
                         "residual": solution.relative_residual, "continuity": solution.continuity_residual})
        manufactured.extend(add_orders(rows, ("velocity_l2", "pressure_l2"), "h"))
    write_csv(output / "manufactured.csv", manufactured)
    for level in config["manufactured_meshes"]:
        mesh = rectangle_mesh(level, width=2)
        operators = assemble_stokes(mesh, viscosity=2)
        def velocity(points):
            return np.column_stack((points[:, 1]*(1-points[:, 1])/4, np.zeros(len(points))))
        solution = solve_stokes(operators, velocity, pressure_value=2)
        errors = flow_error_norms(operators, solution, velocity, lambda points: 2-points[:, 0])
        poiseuille.append({"n": level, **errors, "continuity": solution.continuity_residual})
    write_csv(output / "poiseuille.csv", poiseuille)
    temporal = {f"bdf{order}": temporal_fluid(config["time_steps"], order=order) for order in (1, 2)}
    for name, rows in temporal.items():
        write_csv(output / f"{name}.csv", rows)
    # Reuse each mesh and the same discrete physical equations for both Reynolds numbers.
    for size in config["fluid_meshes"]:
        mesh = cfd_mesh(size, output / "meshes" / f"h_{size:g}")
        operators = assemble_stokes(mesh, viscosity=1, density=1000)
        monitor = BoundaryMonitor(operators, "obstacle")
        inlet_monitor, outlet_monitor = BoundaryMonitor(operators, "inlet"), BoundaryMonitor(operators, "outlet")
        # 参考 (drag, lift) 取自 FeatFlow CFD 基准，单位 N/m（单位厚度）。
        for case, mean, reference in (("CFD1", 0.2, [14.29, 1.119]), ("CFD2", 1.0, [136.7, 10.53])):
            started = time.perf_counter()
            solution = solve_navier_stokes(operators, lambda points: inlet_profile(points, mean),
                                          ["inlet", "walls", "obstacle"], pressure_pin=None, tolerance=1e-8, max_iterations=40)
            force = monitor.force(solution)
            in_flux, out_flux = inlet_monitor.flux(solution), outlet_monitor.flux(solution)
            row = {"case": case, "mesh_size": size, "cells": mesh.n_cells, "dofs": operators.n_total,
                   "drag": float(force[0]), "lift": float(force[1]), "relative_error_drag": float(abs(force[0]/reference[0]-1)),
                   "relative_error_lift": float(abs(force[1]/reference[1]-1)), "inlet_flux": in_flux, "outlet_flux": out_flux,
                   "relative_flux_error": abs(in_flux+out_flux)/abs(in_flux), "continuity": solution.continuity_residual,
                   "residual": solution.relative_residual, "newton_iterations": len(solution.nonlinear_history)-1,
                   "seconds": time.perf_counter()-started}
            cfd.append(row)
            progress(f"{case} h={size:g}: drag={force[0]:.6g}, lift={force[1]:.6g} N/m", flush=True)
            np.savez_compressed(output / f"{case.lower()}_solution.npz", points=mesh.points, cells=mesh.cells,
                                velocity_points=operators.space.points, velocity=solution.velocity, pressure=solution.pressure)
    write_csv(output / "cfd.csv", cfd)
    finest = [r for r in cfd if r["mesh_size"] == config["fluid_meshes"][-1]]
    # 制造解阶次与参考误差取最细网格；守恒/连续/物理残差覆盖所有算例。
    checks = {"velocity_mesh_order": all(r["velocity_l2_order"] >= acceptance["velocity_order_min"] for r in manufactured if r["velocity_l2_order"] is not None),
              "pressure_mesh_order": all(r["pressure_l2_order"] >= acceptance["pressure_order_min"] for r in manufactured if r["pressure_l2_order"] is not None),
              "poiseuille_exact": all(r["velocity_l2"] < 1e-10 and r["pressure_l2"] < 1e-9 for r in poiseuille),
              "cfd_finest_reference": all(max(r["relative_error_drag"], r["relative_error_lift"]) <= acceptance["cfd_relative_error_max"] for r in finest),
              "cfd_mass_balance": all(r["relative_flux_error"] <= acceptance["mass_error_max"] for r in cfd),
              "continuity": all(r["continuity"] <= acceptance["mass_error_max"] for r in cfd+manufactured+poiseuille),
              "physical_residual": all(r["residual"] <= acceptance["residual_max"] for r in cfd+manufactured),
              "bdf1_order": all(r["velocity_l2_order"] >= acceptance["bdf1_order_min"] for r in temporal["bdf1"][1:]),
              "bdf2_order": all(r["velocity_l2_order"] >= acceptance["bdf2_order_min"] for r in temporal["bdf2"][1:])}
    if plot:
        save_fluid_plot(output, manufactured, cfd, temporal)
    return write_report(output, "fluid_single_field", config, {"manufactured": manufactured, "poiseuille": poiseuille, "cfd": cfd, "temporal": temporal}, checks,
                        {"equation": "rho*(du/dt+(u.grad)u)-div(2*mu*epsilon(u))+grad(p)=f; div(u)=0",
                         "space": "P2/P1 Taylor-Hood on affine triangles, degree-5 quadrature",
                         "manufactured": "u=(sin(pi*x)*cos(pi*y),-cos(pi*x)*sin(pi*y)); p=sin(pi*x)*sin(pi*y)",
                         "pressure": "Manufactured/closed domains: pin p(0,0)=0. CFD outlet: natural zero traction, no pressure pin.",
                         "cfd": {"source": CFD_SOURCE, "checked": "2026-10-10", "density": 1000, "dynamic_viscosity": 1,
                                 "mean_velocity_CFD1": 0.2, "mean_velocity_CFD2": 1, "reference_CFD1": [14.29, 1.119],
                                 "reference_CFD2": [136.7, 10.53], "geometry": "2.5x0.41 channel; circle plus rigid flag",
                                 "force": "-integral(sigma*n_fluid) on entire obstacle; positive drag downstream; N/m unit depth",
                                 "mesher": "Gmsh CLI MSH2; own FE assembly and solver"},
                         "temporal": "u=exp(-t)*(y*(1-y),0), p=0, rho=1, mu=0.01; exact spatial representation; final time 1; BDF1 startup."})


def save_fluid_plot(output, manufactured, cfd, temporal):
    """保存三联诊断图 fluid.png：空间收敛、参考误差随网格加密、时间收敛。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for name in ("Stokes", "Navier-Stokes"):
        rows = [r for r in manufactured if r["case"] == name]
        axes[0].loglog([r["h"] for r in rows], [r["velocity_l2"] for r in rows], "o-", label=name+" velocity")
        axes[0].loglog([r["h"] for r in rows], [r["pressure_l2"] for r in rows], "s--", label=name+" pressure")
    axes[0].set(title="Independent spatial convergence", xlabel="h", ylabel="L2 error")
    for name in ("CFD1", "CFD2"):
        rows = [r for r in cfd if r["case"] == name]
        axes[1].loglog([r["mesh_size"] for r in rows], [r["relative_error_drag"] for r in rows], "o-", label=name+" drag")
    axes[1].set(title="Reference error under refinement", xlabel="Obstacle mesh size [m]", ylabel="Relative drag error")
    for name, rows in temporal.items():
        axes[2].loglog([r["dt"] for r in rows], [r["velocity_l2"] for r in rows], "o-", label=name.upper())
    axes[2].set(title="Independent time convergence", xlabel="dt [s]", ylabel="Velocity L2 error")
    for axis in axes:
        axis.legend(fontsize=7)
        axis.grid(True, which="both", alpha=0.3)
    figure.savefig(output / "fluid.png", dpi=160)
    plt.close(figure)

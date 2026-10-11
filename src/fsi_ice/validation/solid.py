"""Linear cantilever, CSM1/2 and independently manufactured time validation."""

from pathlib import Path
import time

import numpy as np

from fsi_ice.mesh import rectangle_mesh
from fsi_ice.mesh.turek import csm_mesh
from fsi_ice.solid.linear import ElasticMaterial, assemble_elasticity, solve_elasticity
from fsi_ice.solid.stvk import assemble_mass, solve_static_continuation, stvk_residual
from fsi_ice.solid.dynamics import solve_solid_step
from fsi_ice.time import BDFHistory
from fsi_ice.validation.common import add_orders, write_csv, write_report


CSM_SOURCE = "https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_tests/fsi_csm_tests.html"


def temporal_solid(time_steps, *, order):
    mesh = rectangle_mesh(4)
    material = ElasticMaterial(0.1, 0.3, density=1)
    mass = assemble_mass(mesh, material)
    mode = np.column_stack((np.sin(np.pi*mesh.points[:, 0])*np.sin(np.pi*mesh.points[:, 1]), np.zeros(mesh.n_nodes)))
    rows = []
    for dt in time_steps:
        amplitude = 0.001
        history = BDFHistory(np.stack((amplitude*mode, -amplitude*mode)), dt, order=order)
        residuals = []
        for _ in range(round(1/dt)):
            context = history.context()
            exact = amplitude*np.exp(-context.time)*mode
            # Exact semidiscrete manufactured load: f = internal(d_exact)+M*d_exact''.
            force = stvk_residual(mesh, material, exact)[0]+mass @ exact.ravel()
            state, solution = solve_solid_step(mesh, material, force, mesh.boundary_nodes(), context,
                                               initial=history.states[0][0], tolerance=1e-12)
            residuals.append(solution.nonlinear_history[-1])
            history.accept(context, state)
        expected = np.stack((amplitude*np.exp(-1)*mode, -amplitude*np.exp(-1)*mode))
        error = float(np.linalg.norm(state-expected)/amplitude)
        rows.append({"dt": dt, "state_error": error, "maximum_residual": max(residuals)})
    return add_orders(rows, ("state_error",), "dt")


def run_solid(config, output, *, plot=True, progress=print):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    acceptance = config["acceptance"]
    beam_rows, csm_rows = [], []
    # E-B tip value is a slender-beam approximation, not an exact 2D solution.
    beam_reference = -0.004
    for nx, ny in config["beam_meshes"]:
        mesh = rectangle_mesh(nx, ny, width=1, height=0.1)
        material = ElasticMaterial(1e6, 0.3, model="plane_stress")
        solution = solve_elasticity(mesh, material, mesh.boundary_nodes("left"), tractions={"right": [0, -10]})
        tip = int(np.argmin(np.linalg.norm(mesh.points-[1, 0.05], axis=1)))
        beam_rows.append({"nx": nx, "ny": ny, "cells": mesh.n_cells, "tip_y": float(solution.displacement[tip, 1]),
                          "beam_relative_error": float(abs(solution.displacement[tip, 1]/beam_reference-1)),
                          "reaction_x": float(solution.reactions.sum(axis=0)[0]), "reaction_y": float(solution.reactions.sum(axis=0)[1]),
                          "residual": solution.relative_residual})
    write_csv(output / "beam.csv", beam_rows)
    for case, young, reference in (("CSM1", 1.4e6, [-0.007187, -0.06610]), ("CSM2", 5.6e6, [-0.0004690, -0.01697])):
        for nx, ny in config["csm_meshes"]:
            started = time.perf_counter()
            mesh = csm_mesh(nx, ny)
            material = ElasticMaterial(young, 0.4, density=1000)
            _, force = assemble_elasticity(mesh, material, body_force=[0, -2000])
            solution, histories = solve_static_continuation(mesh, material, force, mesh.boundary_nodes("left"),
                                                          load_steps=10, tolerance=1e-8, max_iterations=60)
            tip = int(np.argmin(np.linalg.norm(mesh.points-[0.6, 0.2], axis=1)))
            displacement = solution.displacement[tip]
            row = {"case": case, "nx": nx, "ny": ny, "cells": mesh.n_cells, "tip_x": float(displacement[0]), "tip_y": float(displacement[1]),
                   "relative_error_x": float(abs(displacement[0]/reference[0]-1)), "relative_error_y": float(abs(displacement[1]/reference[1]-1)),
                   "minimum_jacobian": solution.minimum_jacobian, "residual": solution.nonlinear_history[-1],
                   "total_newton_iterations": sum(len(history)-1 for history in histories),
                   "force_balance_error": float(np.linalg.norm(solution.reactions.sum(axis=0)+force.reshape(-1, 2).sum(axis=0))),
                   "seconds": time.perf_counter()-started}
            csm_rows.append(row)
            progress(f"{case} {nx}x{ny}: tip=({displacement[0]:.6g},{displacement[1]:.6g}) m", flush=True)
        np.savez_compressed(output / f"{case.lower()}_solution.npz", points=mesh.points, cells=mesh.cells,
                            displacement=solution.displacement)
    write_csv(output / "csm.csv", csm_rows)
    temporal = {f"bdf{order}": temporal_solid(config["time_steps"], order=order) for order in (1, 2)}
    for name, rows in temporal.items():
        write_csv(output / f"{name}.csv", rows)
    finest = [rows[-1] for rows in ([r for r in csm_rows if r["case"] == name] for name in ("CSM1", "CSM2"))]
    checks = {"beam_approximation": beam_rows[-1]["beam_relative_error"] <= acceptance["beam_relative_error_max"],
              "beam_errors_decrease": all(b["beam_relative_error"] < a["beam_relative_error"] for a, b in zip(beam_rows, beam_rows[1:])),
              "beam_reaction": all(abs(r["reaction_y"]-1) < 1e-8 and abs(r["reaction_x"]) < 1e-8 for r in beam_rows),
              "csm_finest_reference": all(max(r["relative_error_x"], r["relative_error_y"]) <= acceptance["csm_relative_error_max"] for r in finest),
              "csm_positive_jacobian": all(r["minimum_jacobian"] > 0 for r in csm_rows),
              "csm_force_balance": all(r["force_balance_error"] < 1e-6 for r in csm_rows),
              "physical_residual": all(r["residual"] <= acceptance["residual_max"] for r in csm_rows+beam_rows),
              "bdf1_order": all(r["state_error_order"] >= acceptance["bdf1_order_min"] for r in temporal["bdf1"][1:]),
              "bdf2_order": all(r["state_error_order"] >= acceptance["bdf2_order_min"] for r in temporal["bdf2"][1:])}
    if plot:
        save_solid_plot(output, beam_rows, csm_rows, temporal)
    return write_report(output, "solid_single_field", config, {"beam": beam_rows, "csm": csm_rows, "temporal": temporal}, checks,
                        {"beam": {"E": 1e6, "nu": 0.3, "model": "plane_stress", "L": 1, "H": 0.1, "traction_y": -10,
                                  "reference": "-F*L^3/(3*E*I), Euler-Bernoulli approximation", "tip_y": beam_reference},
                         "csm": {"source": CSM_SOURCE, "checked": "2026-10-10", "model": "plane_strain", "gravity_y": -2,
                                 "density": 1000, "nu": 0.4, "unit_depth": 1, "geometry": "curved attachment to radius-0.05 cylinder",
                                 "reference_CSM1": [-0.007187, -0.06610], "reference_CSM2": [-0.0004690, -0.01697]},
                         "time": "Semidiscrete manufactured StVK state: d=0.001*exp(-t)*mode, v=-d; fixed space, exact nodal loads; BDF1 startup for BDF2."})


def save_solid_plot(output, beam, csm, temporal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = np.load(output / "csm1_solution.npz")
    points, cells, displacement = data["points"], data["cells"], data["displacement"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    axes[0].triplot(points[:, 0], points[:, 1], cells, linewidth=0.12, color="gray", label="Reference")
    axes[0].triplot((points+displacement)[:, 0], (points+displacement)[:, 1], cells, linewidth=0.12, color="tab:blue", label="Deformed")
    axes[0].set(title="CSM1: computed deformation (scale 1)", xlabel="x [m]", ylabel="y [m]", aspect="equal")
    for name in ("CSM1", "CSM2"):
        rows = [r for r in csm if r["case"] == name]
        axes[1].loglog([r["cells"] for r in rows], [r["relative_error_y"] for r in rows], "o-", label=name+" tip y")
    axes[1].set(title="Reference error under refinement", xlabel="Triangles", ylabel="Relative error")
    from matplotlib.ticker import NullFormatter
    axes[1].set_xticks([r["cells"] for r in csm if r["case"] == "CSM1"],
                      labels=[f"{r['cells']:,}" for r in csm if r["case"] == "CSM1"])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].legend()
    for name, rows in temporal.items():
        axes[2].loglog([r["dt"] for r in rows], [r["state_error"] for r in rows], "o-", label=name.upper())
    axes[2].set(title="Independent time convergence", xlabel="dt [s]", ylabel="Normalized state error")
    axes[2].legend()
    for axis in axes[1:]:
        axis.grid(True, which="both", alpha=0.3)
    figure.savefig(output / "solid.png", dpi=160)
    plt.close(figure)

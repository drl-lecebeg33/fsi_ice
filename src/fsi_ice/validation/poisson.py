"""Manufactured solution u=sin(pi*x) sin(pi*y) on the unit square.

中文说明：本模块用制造解 u=sin(pi x)sin(pi y)（单位方域，-laplacian(u)=2*pi^2*u）
验证 P1 有限元 Poisson 求解器：多档网格上计算 L2 与 H1 半范数误差、估计收敛阶，并写出
convergence.csv、solution.npz、poisson.png 与 report.json（含 Asia/Shanghai 时间戳与代码版本）。
"""

from dataclasses import dataclass
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from fsi_ice.fem.poisson import error_norms, solve_poisson
from fsi_ice.mesh import rectangle_mesh


@dataclass(frozen=True)
class PoissonConfig:
    """Poisson 收敛验证配置：网格档、验收收敛阶与残差阈值（均为有限正数）。"""

    mesh_levels: tuple[int, ...] = (8, 16, 32)
    l2_order_min: float = 1.8
    h1_order_min: float = 0.9
    residual_max: float = 1e-10

    def __post_init__(self):
        if len(self.mesh_levels) < 3:
            raise ValueError("Convergence validation requires at least three mesh levels.")
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in self.mesh_levels):
            raise ValueError("Mesh levels must be integers >= 2.")
        if any(b <= a for a, b in zip(self.mesh_levels, self.mesh_levels[1:])):
            raise ValueError("Mesh levels must be strictly increasing.")
        for value in (self.l2_order_min, self.h1_order_min, self.residual_max):
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError("Acceptance thresholds must be finite and positive.")

    @classmethod
    def from_file(cls, path):
        """从 JSON 读取配置；要求恰含 case/mesh_levels/acceptance 且 case=="poisson_sine"。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != {"case", "mesh_levels", "acceptance"}:
            raise ValueError("Config must contain exactly case, mesh_levels and acceptance.")
        if data["case"] != "poisson_sine":
            raise ValueError("Only the poisson_sine validation case is implemented.")
        acceptance = data["acceptance"]
        if not isinstance(acceptance, dict) or set(acceptance) != {
            "l2_order_min", "h1_order_min", "residual_max",
        }:
            raise ValueError("Acceptance must define l2_order_min, h1_order_min and residual_max.")
        if not isinstance(data["mesh_levels"], list):
            raise ValueError("mesh_levels must be a list.")
        return cls(tuple(data["mesh_levels"]), **acceptance)

    def as_dict(self):
        """导出为可写回 JSON 的字典（含 case 与验收阈值）。"""
        return {"case": "poisson_sine", "mesh_levels": list(self.mesh_levels), "acceptance": {
            "l2_order_min": self.l2_order_min, "h1_order_min": self.h1_order_min,
            "residual_max": self.residual_max,
        }}


def exact_solution(points):
    """制造解 u=sin(pi x)sin(pi y)，在单位方域四边为零（齐次 Dirichlet）。"""
    return np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])


def exact_gradient(points):
    """制造解梯度 grad(u)=(pi cos(pi x)sin(pi y), pi sin(pi x)cos(pi y))。"""
    x, y = np.pi * points[:, 0], np.pi * points[:, 1]
    return np.pi * np.column_stack((np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)))


def source(points):
    """右端项 f=-laplacian(u)=2*pi^2*u，使制造解精确满足 -laplacian(u)=f。"""
    return 2 * np.pi**2 * exact_solution(points)


def code_metadata():
    """收集代码版本信息：当前 commit 与工作树是否干净（无 git 时返回不可用标记）。"""
    root = Path(__file__).resolve().parents[3]
    if not (root / ".git").exists():
        return {"commit": None, "working_tree": "unavailable"}
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                            check=True, capture_output=True, text=True).stdout.splitlines()
    return {"commit": commit, "working_tree": "modified" if status else "clean", "changes": status}


def run_convergence(config: PoissonConfig, output: Path, *, plot=True):
    """在 config.mesh_levels 上跑收敛验证并写出报告，返回报告字典。

    逐档解 P1 Poisson 并测 L2/H1 半范数误差与相对残差（h 为最大单元边长），由相邻两档
    估计收敛阶；验收检查阶次、残差与误差单调下降，并写出 convergence.csv、solution.npz、
    poisson.png 与 report.json。
    """
    rows = []
    for level in config.mesh_levels:
        mesh = rectangle_mesh(level)
        solution = solve_poisson(mesh, source, exact_solution)
        errors = error_norms(mesh, solution.values, exact_solution, exact_gradient)
        row = {"subdivisions": level, "nodes": mesh.n_nodes, "cells": mesh.n_cells,
               "h": mesh.h, **errors, "relative_residual": solution.relative_residual,
               "minimum_quality": float(mesh.quality.min()), "l2_order": None, "h1_order": None}
        if rows:
            # 收敛阶 = log(e_prev/e_now) / log(h_prev/h_now)；h 减半时即 log2 比值。
            previous = rows[-1]
            ratio = np.log(previous["h"] / row["h"])
            row["l2_order"] = float(np.log(previous["l2"] / row["l2"]) / ratio)
            row["h1_order"] = float(np.log(previous["h1_seminorm"] / row["h1_seminorm"]) / ratio)
        rows.append(row)
    checks = {
        # 首行无前一行、阶次为 None，故只从第二行起检查阶次门槛。
        "l2_order": all(row["l2_order"] >= config.l2_order_min for row in rows[1:]),
        "h1_order": all(row["h1_order"] >= config.h1_order_min for row in rows[1:]),
        "linear_residual": all(row["relative_residual"] <= config.residual_max for row in rows),
        "errors_decrease": all(b["l2"] < a["l2"] and b["h1_seminorm"] < a["h1_seminorm"]
                               for a, b in zip(rows, rows[1:])),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "config": config.as_dict(), "equation": "-laplacian(u)=2*pi^2*sin(pi*x)*sin(pi*y)",
        "domain": "[0,1] x [0,1]", "boundary": "exact solution on every exterior edge",
        "discretization": "P1 triangles, degree-5 quadrature, sparse direct solve",
        "error_definitions": {"l2": "quadrature integral of (u_h-u)^2, square root",
                              "h1_seminorm": "quadrature integral of |grad(u_h-u)|^2, square root",
                              "relative_residual": "free DOF residual / max(||b_free||, ||A_free*u||, 1)"},
        "environment": {"python": platform.python_version(), "executable": sys.executable,
                        "platform": platform.platform(), "packages": {
                            name: importlib.metadata.version(name)
                            for name in ("fsi-ice", "numpy", "scipy", "matplotlib")}},
        "code": code_metadata(), "results": rows,
    }
    with (output / "convergence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "solution.npz", points=mesh.points, cells=mesh.cells,
                        values=solution.values, exact=exact_solution(mesh.points))
    if plot:
        save_plot(mesh, solution.values, rows, output / "poisson.png")
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                 allow_nan=False) + "\n", encoding="utf-8")
    return report


def save_plot(mesh, values, rows, path):
    """保存三联图：P1 数值解、节点误差、以及实测收敛曲线（含 h^2 / h 参考斜率）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    triangulation = Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.cells)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    exact = exact_solution(mesh.points)
    for axis, data, title in ((axes[0], values, "P1 Poisson solution"),
                              (axes[1], np.abs(values - exact), "Nodal error |u_h - u|")):
        contours = axis.tricontourf(triangulation, data, levels=18)
        figure.colorbar(contours, ax=axis)
        axis.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    h = np.array([row["h"] for row in rows])
    l2 = np.array([row["l2"] for row in rows])
    h1 = np.array([row["h1_seminorm"] for row in rows])
    axes[2].loglog(h, l2, "o-", label="L2 error")
    axes[2].loglog(h, h1, "s-", label="H1 seminorm error")
    axes[2].loglog(h, l2[0] * (h / h[0])**2, "--", label="h^2 reference")
    axes[2].loglog(h, h1[0] * h / h[0], ":", label="h reference")
    axes[2].set(title="Measured mesh convergence", xlabel="Maximum element edge h", ylabel="Error")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, which="both", alpha=0.3)
    figure.savefig(path, dpi=160)
    plt.close(figure)

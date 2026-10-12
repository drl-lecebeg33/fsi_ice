"""Steady Turek--Hron FSI1 partitioned validation on matching ALE meshes.

中文说明：本模块实现稳态 Turek--Hron FSI1 分区验证：在匹配的 ALE 网格上按
mesh_sizes 由粗到细跑至少三档网格（P2/P1 流体、P1 StVK 固体、调和网格延伸、
Dirichlet--Neumann 位移/面力耦合），逐档记录监测点位移与圆柱力并写出报告。最细
网格须满足 tip_y/drag 相对误差与 tip_x/lift 小量绝对误差门槛，且参考误差随加密单调下降。
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from fsi_ice.ale import build_ale_geometry, build_harmonic_extension
from fsi_ice.coupling import CouplingOptions, DirichletNeumannMap, solve_dirichlet_neumann
from fsi_ice.fluid.forces import BoundaryMonitor
from fsi_ice.fluid.taylor_hood import assemble_stokes, solve_navier_stokes
from fsi_ice.interface import (assemble_interface_load, build_matching_interface,
                               evaluate_interface_geometry, scatter_solid_load)
from fsi_ice.mesh import fsi1_mesh
from fsi_ice.solid.linear import ElasticMaterial
from fsi_ice.solid.stvk import solve_static_continuation
from fsi_ice.validation.common import write_csv, write_report
from fsi_ice.validation.fluid import inlet_profile


def _clear_fsi1_outputs(output):
    """清空该算例输出目录中的旧产物，避免残留上一次 PASS 报告造成误判。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("report.json", "failure.json", "levels.csv"):
        (output/name).unlink(missing_ok=True)
    for pattern in ("iterations_h_*.csv", "solution_h_*.npz"):
        for path in output.glob(pattern):
            path.unlink()


def _write_fsi1_failure(output, *, stage, reason, config_path=None, completed_levels=()):
    """写出 failure.json（status 恒为 "FAIL"），记录失败阶段、原因与已完成网格档，便于诊断。"""
    failure = {"status": "FAIL", "case": "fsi1", "stage": stage, "reason": str(reason),
               "config_path": None if config_path is None else str(config_path),
               "completed_levels": list(completed_levels),
               "timestamp": datetime.now().astimezone().isoformat()}
    Path(output, "failure.json").write_text(json.dumps(failure, indent=2, allow_nan=False)+"\n",
                                            encoding="utf-8")


def load_fsi1_config(path):
    """读取并校验 FSI1 配置，返回原始字典。

    要求字段恰为 case/mesh_sizes/fluid/solid/coupling/solver/reference/acceptance，
    且 case == "fsi1"。mesh_sizes 须为至少三档、严格递减的有限正数；材料参数经
    ElasticMaterial/CouplingOptions 构造以提前暴露非法值；接受阈值须全为正。
    """
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"case", "mesh_sizes", "fluid", "solid", "coupling", "solver", "reference", "acceptance"}
    if not isinstance(config, dict) or set(config) != required or config["case"] != "fsi1":
        raise ValueError("Invalid FSI1 config schema.")
    sizes = config["mesh_sizes"]
    if not isinstance(sizes, list) or len(sizes) < 3 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0 for v in sizes):
        raise ValueError("mesh_sizes must contain at least three finite positive values.")
    if any(sizes[i+1] >= sizes[i] for i in range(len(sizes)-1)):
        raise ValueError("mesh_sizes must be strictly decreasing.")
    fluid = config["fluid"]
    if set(fluid) != {"density", "dynamic_viscosity", "mean_velocity"} or min(fluid.values()) <= 0:
        raise ValueError("Invalid FSI1 fluid parameters.")
    solid = config["solid"]
    if set(solid) != {"density", "poisson", "young", "thickness"}:
        raise ValueError("Invalid FSI1 solid parameters.")
    ElasticMaterial(solid["young"], solid["poisson"], density=solid["density"], thickness=solid["thickness"])
    CouplingOptions(**config["coupling"])
    solver = config["solver"]
    if set(solver) != {"fluid_tolerance", "fluid_max_iterations", "solid_tolerance",
                       "solid_max_iterations", "solid_load_steps"}:
        raise ValueError("Invalid FSI1 solver controls.")
    if min(solver["fluid_tolerance"], solver["solid_tolerance"]) <= 0 or any(
            isinstance(solver[name], bool) or not isinstance(solver[name], int) or solver[name] < 1
            for name in ("fluid_max_iterations", "solid_max_iterations", "solid_load_steps")):
        raise ValueError("Invalid FSI1 solver controls.")
    if set(config["reference"]) != {"tip_x", "tip_y", "drag", "lift"}:
        raise ValueError("Invalid FSI1 reference values.")
    acceptance = config["acceptance"]
    if set(acceptance) != {"tip_y_relative_error_max", "drag_relative_error_max", "tip_x_absolute_error_max",
                           "lift_absolute_error_max", "continuity_max", "mass_balance_max",
                           "single_field_residual_max", "interface_residual_max", "minimum_mesh_quality"} \
            or min(acceptance.values()) <= 0:
        raise ValueError("Invalid FSI1 acceptance thresholds.")
    return config


@dataclass(frozen=True)
class FSI1FluidCandidate:
    """单次流体求解的冻结结果：解、算子、ALE/界面几何以及圆柱力与进出口流量。"""

    solution: object
    operators: object
    ale_geometry: object
    interface_geometry: object
    cylinder_force: np.ndarray
    inlet_flux: float
    outlet_flux: float


class SteadyFSI1Problem:
    """One mesh level of the steady ALE Dirichlet--Neumann problem.

    中文：某一档网格上的稳态 ALE 分区问题，持有网格、材料、界面拓扑与耦合适配器，
    并提供流体/固体子问题求解与指标汇总。
    """

    def __init__(self, config, mesh_size, mesh_output):
        self.config = config
        self.mesh_size = float(mesh_size)
        pair = fsi1_mesh(self.mesh_size, mesh_output)
        self.fluid_mesh, self.solid_mesh = pair.fluid, pair.solid
        reference_operators = assemble_stokes(
            self.fluid_mesh, viscosity=config["fluid"]["dynamic_viscosity"],
            density=config["fluid"]["density"])
        self.topology = build_matching_interface(
            self.solid_mesh, self.fluid_mesh, reference_operators.space,
            solid_tag="solid-interface", fluid_tag="fluid-interface", atol=1e-12, rtol=0)
        solid = config["solid"]
        self.material = ElasticMaterial(solid["young"], solid["poisson"], density=solid["density"],
                                        thickness=solid["thickness"])
        self.extension = build_harmonic_extension(self.fluid_mesh)
        self.exterior_nodes = self.fluid_mesh.boundary_nodes()
        self.exterior_position = {int(node): index for index, node in enumerate(self.exterior_nodes)}
        self.fixed_solid_nodes = self.solid_mesh.boundary_nodes("solid-clamp")
        self.shared_fixed = np.intersect1d(
            self.topology.fluid_vertex_nodes,
            self.fluid_mesh.boundary_nodes(["walls", "outlet", "inlet", "cylinder-rigid-wall"]))
        # 监测点 A=(0.6,0.2) 的最近固体节点，用作挠度与位移输出位置。
        self.tip_node = int(np.argmin(np.linalg.norm(self.solid_mesh.points-[.6, .2], axis=1)))
        if np.linalg.norm(self.solid_mesh.points[self.tip_node]-[.6, .2]) > 1e-12:
            raise ValueError("FSI1 solid mesh is missing monitoring point A=(0.6,0.2).")
        self.adapter = DirichletNeumannMap(self.solve_fluid, self.transfer_load, self.solve_solid,
                                           lambda state: state.displacement.ravel()[self.topology.solid_dofs])

    def full_displacement(self, interface_values):
        """把界面自由度上的位移铺回整块固体网格（单位 m），其余节点置零。

        输入为一维界面位移向量，长度须等于界面自由度数；输出形状与固体网格点一致。
        """
        values = np.asarray(interface_values, dtype=float)
        # 界面位移维度须与拓扑登记的界面自由度严格一致。
        if values.shape != (len(self.topology.solid_dofs),):
            raise ValueError("FSI1 interface displacement has the wrong shape.")
        full = np.zeros_like(self.solid_mesh.points)
        full.ravel()[self.topology.solid_dofs] = values
        return full

    def solve_fluid(self, interface_values):
        """在界面位移驱动的变形网格上求解流体子问题，返回冻结的流体候选项。

        interface_values 为固体界面位移（m）；先把界面位移值搬到流体界面顶点，
        经调和延伸得到全流场网格位移，再装配并求解 Navier--Stokes，最后评估界面几何、
        圆柱力与进出口流量。湿面/夹持交界处位移须为零，否则报错。
        """
        solid_displacement = self.full_displacement(interface_values)
        # 仅界面顶点带位移，其余外边界保持零，供调和延伸插值。
        boundary_values = np.zeros((len(self.exterior_nodes), 2))
        for solid_node, fluid_node in zip(self.topology.solid_nodes, self.topology.fluid_vertex_nodes):
            boundary_values[self.exterior_position[int(fluid_node)]] = solid_displacement[solid_node]
        # 湿面与固支交界必须无位移，否则界面拓扑或映射有误。
        if np.max(np.abs(boundary_values[[self.exterior_position[int(n)] for n in self.shared_fixed]])) > 1e-13:
            raise ValueError("The wet/clamped junction must have zero displacement.")
        mesh_displacement = self.extension.extend(self.exterior_nodes, boundary_values)
        ale_geometry = build_ale_geometry(self.fluid_mesh, mesh_displacement.values,
                                          candidate_id=("fsi1", self.mesh_size))
        fluid = self.config["fluid"]
        operators = assemble_stokes(ale_geometry.current_mesh, viscosity=fluid["dynamic_viscosity"],
                                    density=fluid["density"])
        solution = solve_navier_stokes(
            operators, lambda points: inlet_profile(points, fluid["mean_velocity"]),
            ["inlet", "walls", "cylinder-rigid-wall", "fluid-interface"], pressure_pin=None,
            tolerance=self.config["solver"]["fluid_tolerance"],
            max_iterations=self.config["solver"]["fluid_max_iterations"])
        geometry = evaluate_interface_geometry(
            self.topology, ale_geometry.current_mesh, operators.space, solid_displacement,
            candidate_id=("fsi1", self.mesh_size))
        cylinder = BoundaryMonitor(operators, "cylinder-rigid-wall").force(solution)
        inlet = BoundaryMonitor(operators, "inlet").flux(solution)
        outlet = BoundaryMonitor(operators, "outlet").flux(solution)
        return FSI1FluidCandidate(solution, operators, ale_geometry, geometry, cylinder, inlet, outlet)

    def transfer_load(self, candidate):
        """把流体界面面力（含厚度）映射为固体侧等效节点载荷。"""
        return assemble_interface_load(self.topology, candidate.interface_geometry, candidate.operators,
                                       candidate.solution, thickness=self.material.thickness)

    def solve_solid(self, load):
        """以界面载荷为外载求解 StVK 固体子问题（带回退的载荷延续），返回固体状态。"""
        external = scatter_solid_load(self.topology, load.f_s_gamma)
        result, _ = solve_static_continuation(
            self.solid_mesh, self.material, external, self.fixed_solid_nodes,
            load_steps=self.config["solver"]["solid_load_steps"],
            tolerance=self.config["solver"]["solid_tolerance"],
            max_iterations=self.config["solver"]["solid_max_iterations"])
        return result

    def solve(self):
        """在零位移初值上进行 Dirichlet--Neumann 分区耦合迭代，返回收敛结果。"""
        options = CouplingOptions(**self.config["coupling"])
        return solve_dirichlet_neumann(
            self.adapter, np.zeros(len(self.topology.solid_dofs)),
            # 界面位移特征量级约 1e-3 m，作为 Aitken 缩放的初值基准。
            scales=np.full(len(self.topology.solid_dofs), .001), options=options)

    def metrics(self, result, seconds):
        """汇总该档网格的全部验证指标（位移 m、力 N/m、误差、残差与网格质量等）。"""
        fluid = result.evaluation.fluid
        solid = result.evaluation.solid
        load = result.evaluation.load
        tip = solid.displacement[self.tip_node]
        total_force = fluid.cylinder_force+load.total_force
        reference = self.config["reference"]
        return {
            "mesh_size": self.mesh_size, "fluid_cells": self.fluid_mesh.n_cells,
            "solid_cells": self.solid_mesh.n_cells, "fluid_dofs": fluid.operators.n_total,
            "solid_dofs": 2*self.solid_mesh.n_nodes, "interface_edges": self.topology.n_edges,
            "tip_x": float(tip[0]), "tip_y": float(tip[1]),
            "drag": float(total_force[0]), "lift": float(total_force[1]),
            "tip_x_absolute_error": float(abs(tip[0]-reference["tip_x"])),
            "tip_y_relative_error": float(abs(tip[1]/reference["tip_y"]-1)),
            "drag_relative_error": float(abs(total_force[0]/reference["drag"]-1)),
            "lift_absolute_error": float(abs(total_force[1]-reference["lift"])),
            "inlet_flux": fluid.inlet_flux, "outlet_flux": fluid.outlet_flux,
            "mass_balance": float(abs(fluid.inlet_flux+fluid.outlet_flux)/abs(fluid.inlet_flux)),
            "continuity": fluid.solution.continuity_residual,
            "fluid_residual": fluid.solution.relative_residual,
            "solid_residual": float(solid.nonlinear_history[-1]),
            "interface_residual": result.history[-1].residual_norm,
            "coupling_iterations": len(result.history),
            "fluid_newton_iterations": len(fluid.solution.nonlinear_history)-1,
            "minimum_mesh_volume_ratio": fluid.ale_geometry.minimum_volume_ratio,
            "minimum_mesh_quality": fluid.ale_geometry.minimum_quality,
            "minimum_solid_jacobian": solid.minimum_jacobian, "seconds": float(seconds),
        }


def run_fsi1(config, output, *, progress=print):
    """按 config["mesh_sizes"] 逐档求解并写出 FSI1 报告。

    每档记录指标行（先写 levels.csv 再落盘迭代历史与解 npz），最后在最细网格上对照
    reference 误差做验收；任一环节异常会删除 report.json 并写出 failure.json 后重抛。
    """
    output = Path(output)
    _clear_fsi1_outputs(output)
    rows = []
    try:
        for size in config["mesh_sizes"]:
            started = time.perf_counter()
            problem = SteadyFSI1Problem(config, size, output/"meshes"/f"h_{size:g}")
            result = problem.solve()
            row = problem.metrics(result, time.perf_counter()-started)
            rows.append(row)
            write_csv(output/"levels.csv", rows)
            write_csv(output/f"iterations_h_{size:g}.csv", [
                # residual 为数组，单独存入 npz；CSV 只保留标量迭代历史字段。
                {key: value for key, value in asdict(item).items() if key != "residual"}
                for item in result.history])
            fluid = result.evaluation.fluid
            np.savez_compressed(
                output/f"solution_h_{size:g}.npz",
                solid_points=problem.solid_mesh.points,
                solid_cells=problem.solid_mesh.cells,
                solid_displacement=result.evaluation.solid.displacement,
                fluid_reference_points=problem.fluid_mesh.points,
                fluid_current_points=fluid.ale_geometry.current_mesh.points,
                fluid_cells=problem.fluid_mesh.cells,
                velocity_points=fluid.operators.space.points,
                fluid_velocity=fluid.solution.velocity,
                fluid_pressure=fluid.solution.pressure,
                interface_values=result.interface_values,
                interface_residuals=np.stack([item.residual for item in result.history]))
            progress(f"FSI1 h={size:g}: uy={row['tip_y']:.6e}, drag={row['drag']:.6g}, "
                     f"coupling={row['coupling_iterations']}", flush=True)
        finest = rows[-1]
        a = config["acceptance"]
        # 参考量验收只在最细网格上做；残差/守恒/网格质量则在全部网格档上检查。
        checks = {
            "tip_y_reference": finest["tip_y_relative_error"] <= a["tip_y_relative_error_max"],
            "drag_reference": finest["drag_relative_error"] <= a["drag_relative_error_max"],
            "tip_x_small_quantity": finest["tip_x_absolute_error"] <= a["tip_x_absolute_error_max"],
            "lift_small_quantity": finest["lift_absolute_error"] <= a["lift_absolute_error_max"],
            "continuity": all(r["continuity"] <= a["continuity_max"] for r in rows),
            "mass_balance": all(r["mass_balance"] <= a["mass_balance_max"] for r in rows),
            "single_field_residuals": all(max(r["fluid_residual"], r["solid_residual"])
                                          <= a["single_field_residual_max"] for r in rows),
            "interface_residual": all(r["interface_residual"] <= a["interface_residual_max"] for r in rows),
            "mesh_quality": all(r["minimum_mesh_quality"] >= a["minimum_mesh_quality"] and
                                r["minimum_mesh_volume_ratio"] > 0 and r["minimum_solid_jacobian"] > 0 for r in rows),
            "tip_y_reference_error_decreases": all(rows[i+1]["tip_y_relative_error"]
                                                     < rows[i]["tip_y_relative_error"]
                                                     for i in range(len(rows)-1)),
            "drag_reference_error_decreases": all(rows[i+1]["drag_relative_error"]
                                                   < rows[i]["drag_relative_error"]
                                                   for i in range(len(rows)-1)),
        }
        return write_report(output, "fsi1", config, {"levels": rows}, checks, {
            "scope": "steady partitioned ALE FSI1; P2/P1 Navier-Stokes, P1 StVK, harmonic mesh extension",
            "coupling": "Dirichlet-Neumann displacement/traction with scaled Aitken",
            "force": "cylinder boundary plus elastic beam interface, unit depth",
        })
    except Exception as error:
        # 失败时移除可能残留的 report.json，防止上一次 PASS 被误读为本次结果。
        for name in ("report.json",):
            (output/name).unlink(missing_ok=True)
        _write_fsi1_failure(output, stage="run", reason=error, completed_levels=rows)
        raise


def run_fsi1_cli(config_path, output, *, progress=print):
    """Own output cleanup so invalid input cannot leave an earlier PASS behind.

    中文：CLI 入口先清空输出再解析配置；配置非法（ValueError/TypeError/OSError）时
    写 failure.json（stage="config"）并重抛，确保不会遗留旧的通过报告。
    """
    _clear_fsi1_outputs(output)
    try:
        config = load_fsi1_config(config_path)
    except (ValueError, TypeError, OSError) as error:
        _write_fsi1_failure(output, stage="config", reason=error, config_path=config_path)
        raise
    return run_fsi1(config, output, progress=progress)

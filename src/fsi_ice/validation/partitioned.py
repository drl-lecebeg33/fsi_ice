"""T18 small linear FE coupling: partitioned iteration versus a block direct solve.

This is a BDF1 step linearized on fixed reference geometry, using unsteady
Stokes and linear elasticity. It is not the Turek--Hron FSI1 benchmark and is
not a complete nonlinear monolithic FSI solver.

中文说明：本模块职责是 T18“小型线性有限元耦合”验证：在固定参考几何上取一个
BDF1 时间步，流体用非稳态 Stokes、固体用线弹性，比较**分区耦合迭代**（Dirichlet-
Neumann + 固定点/Aitken/IQN 松弛）与**整体块直接解**的结果。关键约定：这是固定参考
几何的线性算例，不是 Turek–Hron FSI1 基准，也不是完整的非线性整体 FSI 求解器；
分区解与直接解必须在同一套 FE 矩阵、同一界面应力传递（H 与 H^T）下比较，验收看位移/
速度/压力/载荷差异、耦合残差、运动学误差与连续性等门槛。产物遵循 T20 风格：先清空旧
产物，失败时写 ``failure.json``（不伪造未求值字段）。
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from functools import cached_property
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.sparse import bmat, csr_matrix

from fsi_ice.coupling import CouplingFailure, CouplingOptions, DirichletNeumannMap, solve_dirichlet_neumann
from fsi_ice.fem.poisson import solve_dirichlet
from fsi_ice.fluid.taylor_hood import assemble_stokes, flow_constraints, unpack_flow
from fsi_ice.interface import (assemble_interface_load, build_matching_interface,
                               evaluate_interface_geometry, interface_load_jvp, scatter_solid_load)
from fsi_ice.mesh import TriMesh, rectangle_mesh
from fsi_ice.solid.linear import ElasticMaterial, assemble_elasticity, vector_dofs
from fsi_ice.solid.stvk import assemble_mass
from fsi_ice.time.bdf import BDFHistory
from fsi_ice.validation.common import write_csv, write_report
from fsi_ice.validation.poisson import code_metadata


def validate_config(config):
    """中文：校验并返回 T18 分区检查配置。要求顶层键恰为
    {case, mesh, dt, solid, fluid, coupling, scales, acceptance} 且 case 为
    ``partitioned_check``；mesh 为 [nx,ny]（≥2 的整数）；各数值组为有限实数，
    scales/acceptance 为正、流体密度与粘度为正；并试构造 ElasticMaterial 与
    CouplingOptions 以借用其校验。非法时抛 ``ValueError``。
    """
    schemas = {"solid": {"young", "poisson", "density", "body_force_y"},
               "fluid": {"viscosity", "density", "lid_velocity"},
               "coupling": {"relaxation", "absolute_tolerance", "relative_tolerance", "max_iterations"},
               "scales": {"displacement", "velocity", "pressure", "load"},
               "acceptance": {"direct_difference_max", "residual_max", "motion_error_max", "continuity_max"}}
    if not isinstance(config, dict) or set(config) != {"case", "mesh", "dt", *schemas} or \
            config["case"] != "partitioned_check":
        raise ValueError("Invalid partitioned-check config schema.")
    mesh = config["mesh"]
    if not isinstance(mesh, list) or len(mesh) != 2 or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in mesh):
        raise ValueError("mesh must be [nx,ny] with integers >=2.")
    for group, keys in schemas.items():
        values = config[group]
        optional = {"method", "minimum_relaxation", "maximum_relaxation", "aitken_denominator_tolerance",
                    "iqn_initial_relaxation", "iqn_alpha_max", "iqn_history_size",
                    "iqn_restart_tolerance"} if group == "coupling" else set()
        if not isinstance(values, dict) or not keys <= set(values) or not set(values) <= keys | optional:
            raise ValueError(f"Invalid {group} config schema.")
        for name, value in values.items():
            if group == "coupling" and name == "method":
                continue  # CouplingOptions validates the algorithm selector.
            if group == "coupling" and name == "iqn_history_size":
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                    raise ValueError("iqn_history_size must be a nonnegative integer.")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{group} values must be finite real numbers.")
    dt = config["dt"]
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    for group in ("scales", "acceptance"):
        if min(config[group].values()) <= 0:
            raise ValueError(f"{group} values must be positive.")
    if min(config["fluid"]["viscosity"], config["fluid"]["density"]) <= 0:
        raise ValueError("Fluid density and viscosity must be positive.")
    solid = config["solid"]
    ElasticMaterial(solid["young"], solid["poisson"], density=solid["density"])
    CouplingOptions(**config["coupling"])
    return config


def load_partitioned_config(path):
    """中文：从 JSON 文件读取配置（UTF-8）并交给 :func:`validate_config` 校验。"""
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class LinearSolidCandidate:
    """中文：固体子问题的一次候选解（不可变）：位移(``(n_nodes,2)``，m)、
    由 BDF 模板导出的速度(m/s)、以及线性求解的相对残差。"""

    displacement: np.ndarray
    velocity: np.ndarray
    relative_residual: float


class LinearCoupledProblem:
    """Actual P1/P2 FE fields and stress transfer for one fixed-history BDF1 step.

    中文：T18 的固定历史 BDF1 单步问题，持有真实 P1 固体/P2-P1 流体场与界面应力传递。
    单位约定为 SI（m、s、m/s、Pa、N/m，单位厚度）。构造时建网格、装配 Stokes 与线弹性
    算子、构建匹配界面拓扑与固定参考几何，并组装直接解所需的块系统所用矩阵/右端项。
    """

    def __init__(self, config):
        self.config = validate_config(config)
        nx, ny = config["mesh"]
        base_s = rectangle_mesh(nx, ny, height=0.5)
        base_f = rectangle_mesh(nx, ny, height=0.8)
        s_tags, f_tags = dict(base_s.boundaries), dict(base_f.boundaries)
        s_tags["interface"] = s_tags.pop("top")  # 固体顶边与流体底边共享界面
        f_tags["interface"] = f_tags.pop("bottom")
        # 固体整体下移 0.5，使物理界面落在 y=0 与流体底边重合。
        self.solid_mesh = TriMesh(base_s.points - [0, 0.5], base_s.cells, s_tags)
        self.fluid_mesh = TriMesh(base_f.points, base_f.cells, f_tags)
        fluid, solid = config["fluid"], config["solid"]
        self.operators = assemble_stokes(self.fluid_mesh, viscosity=fluid["viscosity"], density=fluid["density"])
        self.material = ElasticMaterial(solid["young"], solid["poisson"], density=solid["density"])
        self.topology = build_matching_interface(self.solid_mesh, self.fluid_mesh, self.operators.space,
                                                solid_tag="interface", fluid_tag="interface", atol=1e-12, rtol=1e-12)
        self.geometry = evaluate_interface_geometry(self.topology, self.fluid_mesh, self.operators.space,
                                                   np.zeros_like(self.solid_mesh.points), candidate_id="t18-fixed-reference")
        self.solid_history = BDFHistory(np.zeros((2, self.solid_mesh.n_nodes, 2)), config["dt"], order=1)
        self.fluid_history = BDFHistory(np.zeros_like(self.operators.space.points), config["dt"], order=1)
        self.solid_context = self.solid_history.context()
        self.fluid_context = self.fluid_history.context()
        self.alpha = self.solid_context.coefficient  # BDF1 系数 a0/dt
        stiffness, external = assemble_elasticity(self.solid_mesh, self.material,
                                                  body_force=(0, solid["body_force_y"]))
        mass = assemble_mass(self.solid_mesh, self.material)
        hd, hv = self.solid_context.history_term
        self.solid_matrix = stiffness + self.alpha**2*mass  # 线弹性 + a0^2 M（BDF1 惯性项）
        self.solid_rhs = external - mass @ (self.alpha*hd + hv).ravel()  # 历史项移到右端
        self.solid_fixed = vector_dofs(self.solid_mesh.boundary_nodes(["bottom", "left"]))
        self.fluid_matrix = self.operators.matrix + bmat(
            [[self.alpha*self.operators.mass, None], [None, csr_matrix((self.fluid_mesh.n_nodes,)*2)]], format="csr")
        self.fluid_rhs = self.operators.rhs.copy()
        self.fluid_rhs[:self.operators.n_velocity] -= self.operators.mass @ self.fluid_context.history_term.ravel()
        # 外部边界约束：lid/top+left；压力不设 pin（与整体 None 约定一致）。
        self.outer_dofs, self.outer_values = flow_constraints(self.operators, self.lid_profile,
                                                             ["top", "left"], pressure_pin=None)
        self.gamma_dofs = vector_dofs(self.topology.fluid_p2_nodes)  # 界面流体 P2 迹自由度
        self.adapter = DirichletNeumannMap(self.solve_fluid, self.transfer_load, self.solve_solid,
                                          lambda solution: solution.displacement.ravel()[self.topology.solid_dofs])

    def lid_profile(self, points):
        """中文：顶盖速度 Dirichlet：仅在 y=0.8 顶边上取 x 方向
        ``lid_velocity*sin(pi*x)``，其余为零（m/s）。"""
        result = np.zeros_like(points)
        top = np.isclose(points[:, 1], 0.8, rtol=0, atol=1e-12)
        result[top, 0] = self.config["fluid"]["lid_velocity"]*np.sin(np.pi*points[top, 0])
        return result

    def initial_interface(self):
        """中文：界面位移初值：取 BDF 历史第 0 层固体位移在固体界面自由度上的分量
        （m），作为分区迭代的起始界面量。"""
        return self.solid_history.states[0][0].ravel()[self.topology.solid_dofs].copy()

    def solve(self, *, options=None, initial=None):
        """中文：跑一次分区 Dirichlet-Neumann 耦合求解。``options`` 为空时由配置
        coupling 段构造 ``CouplingOptions``；``initial`` 为空时用
        :meth:`initial_interface`。返回耦合结果（含迭代历史与解）。"""
        options = CouplingOptions(**self.config["coupling"], case_name="T18-linear-FE") if options is None else options
        return solve_dirichlet_neumann(self.adapter, self.initial_interface() if initial is None else initial,
                                       scales=np.full(len(self.topology.solid_dofs), self.config["scales"]["displacement"]),
                                       options=options)

    def solve_fluid(self, interface_values):
        """中文：流体子问题求解。由紧凑界面位移经 BDF 模板得到界面速度，映射到流体
        P2 迹（``H``）后与外部 Dirichlet 合并为边界条件，解一次 Stokes 并校验连续性。
        输入为固体界面位移（m）；返回解包后的流场。界面运动与固定外边冲突时抛错。
        """
        # x contains compact displacement, so the common BDF stencil supplies velocity.
        hd = self.solid_context.history_term[0].ravel()[self.topology.solid_dofs]
        motion = self.topology.H @ (self.alpha*interface_values + hd)  # 界面速度 = a0*x + 历史项
        boundary = dict(zip(self.outer_dofs, self.outer_values))
        for dof, value in zip(self.gamma_dofs, motion):
            if dof in boundary and abs(boundary[dof]-value) > 1e-12:
                raise ValueError("Interface motion conflicts with the fixed outer boundary.")
            boundary[dof] = value
        dofs = np.array(sorted(boundary), dtype=int)
        values = np.array([boundary[dof] for dof in dofs])
        result = solve_dirichlet(self.fluid_matrix, self.fluid_rhs, dofs, values)
        solution = unpack_flow(self.operators, result.values, result.relative_residual)
        if solution.continuity_residual > 1e-10:  # 连续性硬门槛（1e-10）
            raise RuntimeError("Coupled Stokes candidate violates continuity.")
        return solution

    def transfer_load(self, fluid):
        """中文：把流场界面应力传递到固体界面载荷（经 H^T、单位厚度），返回界面载荷对象。"""
        return assemble_interface_load(self.topology, self.geometry, self.operators, fluid,
                                       thickness=self.material.thickness)

    def solve_solid(self, load):
        """中文：固体子问题求解。把界面载荷散射到右端，解线弹性方程得到位移，并由 BDF
        模板导出速度；返回 :class:`LinearSolidCandidate`。"""
        rhs = self.solid_rhs + scatter_solid_load(self.topology, load.f_s_gamma)
        result = solve_dirichlet(self.solid_matrix, rhs, self.solid_fixed, 0.0)
        displacement = result.values.reshape(-1, 2)
        velocity = self.alpha*displacement + self.solid_context.history_term[0]
        return LinearSolidCandidate(displacement, velocity, result.relative_residual)

    @cached_property
    def block_system(self):
        """Direct reference rows [solid balance, fluid balance/kinematic constraints].

        中文：构造整体块直接解的矩阵与右端项。上块为固体平衡，下块为流体平衡与运动学
        约束；界面速度行用 ``H`` 把固体界面位移与流体 P2 迹关联，载荷耦合列由
        ``interface_load_jvp``（固定几何解析导数）逐列装配。返回 (矩阵, 右端, 约束自由度,
        约束值)。
        """
        nf, ns = self.operators.n_total, self.solid_matrix.shape[0]
        zero = unpack_flow(self.operators, np.zeros(nf), 0.0)
        load_matrix = np.zeros((ns, nf))
        for column in range(nf):
            # 对流体每个自由度做单位扰动，用解析 JVP 得到固体界面载荷的耦合列。
            direction = np.zeros(nf)
            direction[column] = 1.0
            derivative = interface_load_jvp(self.topology, self.geometry, self.operators, zero,
                                           {"velocity": direction[:self.operators.n_velocity].reshape(-1, 2),
                                            "pressure": direction[self.operators.n_velocity:]},
                                           thickness=self.material.thickness)
            load_matrix[:, column] = scatter_solid_load(self.topology, derivative.delta_f_s_gamma)
        fluid_rows = self.fluid_matrix.tolil()
        cross_rows = np.zeros((nf, ns))
        rhs = self.fluid_rhs.copy()
        trace_map = self.topology.H.toarray()
        hd = self.solid_context.history_term[0].ravel()[self.topology.solid_dofs]
        for row, dof in enumerate(self.gamma_dofs):
            # 界面速度行替换为运动学约束：u_gamma - a0*H*x = H*历史项。
            fluid_rows.rows[dof] = [int(dof)]
            fluid_rows.data[dof] = [1.0]
            cross_rows[dof, self.topology.solid_dofs] = -self.alpha*trace_map[row]
            rhs[dof] = trace_map[row] @ hd
        matrix = bmat([[self.solid_matrix, -csr_matrix(load_matrix)],
                       [csr_matrix(cross_rows), fluid_rows.tocsr()]], format="csr")
        prescribed = np.concatenate((self.solid_fixed, ns+self.outer_dofs))  # 固定/约束自由度
        values = np.concatenate((np.zeros(len(self.solid_fixed)), self.outer_values))
        return matrix, np.concatenate((self.solid_rhs, rhs)), prescribed, values

    def direct(self):
        """中文：直接解整体块系统一次，得到固体候选、流场与界面载荷的参考解（分区解与
        之对照）。返回 ``(solid, fluid, load)``。"""
        matrix, rhs, dofs, values = self.block_system
        result = solve_dirichlet(matrix, rhs, dofs, values)
        ns = self.solid_matrix.shape[0]
        displacement = result.values[:ns].reshape(-1, 2)
        fluid = unpack_flow(self.operators, result.values[ns:], result.relative_residual)
        solid = LinearSolidCandidate(displacement, self.alpha*displacement+self.solid_context.history_term[0],
                                     result.relative_residual)
        return solid, fluid, self.transfer_load(fluid)

    def history_snapshot(self):
        """中文：返回固体/流体 BDF 历史的快照，用于检验求解过程是否意外改动历史。"""
        return self.solid_history.snapshot(), self.fluid_history.snapshot()

    def histories_equal(self, snapshots):
        """中文：比较给定快照与当前历史是否逐项相等（数组用 ``array_equal``），用于
        验证候选求解阶段未接受/改写历史。"""
        return all(old[:2] == new[:2] and len(old[2]) == len(new[2]) and
                   all(np.array_equal(a, b) for a, b in zip(old[2], new[2]))
                   for old, new in zip(snapshots, self.history_snapshot()))

    def metrics(self, result, direct):
        """中文：计算分区解相对直接参考解的各项差异与质量指标：位移/速度/压力/载荷差异
        （按各自尺度归一的最大绝对差）、组合残差（自由行缺陷按行尺度归一的最大值）、
        运动学误差、连续性残差、子问题残差、迭代次数与规模。行尺度约定：力行用 load
        尺度；积分连续性行用 velocity*1 m；运动学行用 velocity 尺度。
        """
        solid, fluid, load = result.evaluation.solid, result.evaluation.fluid, result.evaluation.load
        ds, df, dl = direct
        scales = self.config["scales"]
        difference = lambda a, b, scale: float(np.max(np.abs(a-b))/scale)
        matrix, rhs, fixed, _ = self.block_system
        values = np.concatenate((solid.displacement.ravel(), fluid.velocity.ravel(), fluid.pressure))
        free = np.setdiff1d(np.arange(len(values)), fixed)
        defect = matrix @ values-rhs  # 全系统缺陷；只对自由行计分
        # Force rows use 1-N load scales (unit depth). Continuity is integrated
        # velocity over a 1-m reference length; kinematic rows have velocity units.
        # 各行尺度：默认载荷尺度(N/m)；积分连续性行用速度尺度×1 m；界面运动学行用速度尺度。
        row_scales = np.full(len(values), scales["load"])
        ns = self.solid_matrix.shape[0]
        row_scales[ns+self.operators.n_velocity:] = scales["velocity"]*1.0
        row_scales[ns+self.gamma_dofs] = scales["velocity"]
        motion = self.topology.H @ solid.velocity.ravel()[self.topology.solid_dofs]
        return {"displacement_difference": difference(solid.displacement, ds.displacement, scales["displacement"]),
                "velocity_difference": difference(fluid.velocity, df.velocity, scales["velocity"]),
                "pressure_difference": difference(fluid.pressure, df.pressure, scales["pressure"]),
                "load_difference": difference(load.f_s_gamma, dl.f_s_gamma, scales["load"]),
                "combined_residual": float(np.max(np.abs(defect[free])/row_scales[free])),
                "motion_error": difference(fluid.velocity.ravel()[self.gamma_dofs], motion, 1.0),
                "continuity": float(fluid.continuity_residual),
                "solid_residual": float(solid.relative_residual), "fluid_residual": float(fluid.relative_residual),
                "iterations": len(result.history), "interface_residual": result.history[-1].residual_norm,
                "solid_cells": self.solid_mesh.n_cells, "fluid_cells": self.fluid_mesh.n_cells,
                "solid_dofs": self.solid_matrix.shape[0], "fluid_dofs": self.operators.n_total}


def iteration_rows(history):
    """中文：把耦合迭代历史转为 CSV 行；去掉体积较大的 ``residual`` 数组字段，只保留
    标量诊断（残差范数等）。"""
    return [{key: value for key, value in asdict(row).items() if key != "residual"} for row in history]


def _clear_outputs(output):
    """中文：清空旧产物（report/solution/iterations/failure），确保失败重跑不残留上次
    成功结果。"""
    for name in ("report.json", "failure.json", "solution.npz", "iterations.csv"):
        (output / name).unlink(missing_ok=True)


def _write_failure(output, *, case, config, config_path, stage, reason, iteration, problem, snapshots):
    """Persist FAIL evidence; null fields mean the value was never evaluable.

    Input or setup failures have no problem/snapshot objects and must not fake
    histories_unchanged. Partially written success artifacts are removed by
    the caller before this record is written.

    中文：持久化失败证据到 ``failure.json``（不可求值的字段记为 null）。输入/建场阶段
    失败没有 problem/snapshot 对象，因此不得伪造 ``histories_unchanged``；部分写入的
    成功产物由调用方在本记录写出前移除。
    """
    histories_unchanged = None
    if problem is not None and snapshots is not None:
        try:
            histories_unchanged = problem.histories_equal(snapshots)
        except Exception:  # recording must not mask the original failure
            histories_unchanged = None
    failure = {"status": "FAIL", "case": case, "config": config, "config_path": config_path,
               "iteration": iteration, "stage": stage, "reason": reason,
               "timestamp": datetime.now().astimezone().isoformat(),
               "code": code_metadata(), "environment": {"python": platform.python_version(),
                   "executable": sys.executable,
                   "packages": {name: importlib.metadata.version(name) for name in
                                ("numpy", "scipy", "matplotlib", "pytest")}},
               "histories_unchanged": histories_unchanged}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "failure.json").write_text(json.dumps(failure, indent=2, allow_nan=False)+"\n")


def run_partitioned_check(config, output):
    """中文：T18 分区检查主流程。依次建场、做一次分区耦合求解、求整体直接参考解、重复
    一次分区解（验证确定性）、计算指标与验收检查，写 iterations.csv/solution.npz 与
    report.json。失败时清掉已写成功产物并写 failure.json 后原样抛出异常（stage 用于定位
    失败阶段）。
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # A failed rerun must not leave a previous successful report/solution in place.
    _clear_outputs(output)
    problem = snapshots = None
    history = ()
    stage = "setup"  # 阶段标记：异常时据此定位失败环节
    started = time.perf_counter()
    try:
        problem = LinearCoupledProblem(config)
        snapshots = problem.history_snapshot()  # 记录初始历史快照，供事后比对
        stage = "coupling"
        result = problem.solve()
        history = result.history
        stage = "direct_reference"
        direct = problem.direct()
        stage = "repeat"
        repeat = problem.solve()
        stage = "metrics"
        metrics = problem.metrics(result, direct)
        metrics["method"] = config["coupling"].get("method", "fixed")
        metrics["seconds"] = time.perf_counter()-started
        thresholds = config["acceptance"]
        checks = {name: metrics[name+"_difference"] <= thresholds["direct_difference_max"]
                  for name in ("displacement", "velocity", "pressure", "load")}
        checks.update(combined_residual=metrics["combined_residual"] <= thresholds["residual_max"],
                      motion=metrics["motion_error"] <= thresholds["motion_error_max"],
                      continuity=metrics["continuity"] <= thresholds["continuity_max"],
                      histories_unchanged=problem.histories_equal(snapshots),
                      repeat_equal=all(np.array_equal(a, b) for a, b in (
                          (result.interface_values, repeat.interface_values),
                          (result.evaluation.solid.displacement, repeat.evaluation.solid.displacement),
                          (result.evaluation.fluid.velocity, repeat.evaluation.fluid.velocity),
                          (result.evaluation.fluid.pressure, repeat.evaluation.fluid.pressure),
                          (result.evaluation.load.f_s_gamma, repeat.evaluation.load.f_s_gamma))))
        stage = "outputs"
        write_csv(output / "iterations.csv", iteration_rows(result.history))
        np.savez_compressed(output / "solution.npz", interface_input=result.interface_values,
                            interface_residuals=np.stack([row.residual for row in result.history]),
                            solid_displacement=result.evaluation.solid.displacement,
                            fluid_velocity=result.evaluation.fluid.velocity, fluid_pressure=result.evaluation.fluid.pressure,
                            interface_load=result.evaluation.load.f_s_gamma,
                            direct_displacement=direct[0].displacement, direct_velocity=direct[1].velocity,
                            direct_pressure=direct[1].pressure, direct_load=direct[2].f_s_gamma)
        return write_report(output, config["case"], config, metrics, checks,
                            {"scope": "T18 fixed-reference linear BDF1 FE coupling; not FSI1 or full nonlinear monolithic FSI",
                             "physics": "P1 plane-strain linear solid / P2-P1 unsteady Stokes; shared stress trace and H/H.T",
                             "norm": "max(abs(G(x)-x)/S); atol+rtol*max(max(abs(x)/S),max(abs(G(x))/S))",
                             "combined_residual": "max(abs(free-row defect)/row scale): force=load scale; integrated continuity=velocity scale*1 m; motion=velocity scale",
                             "termination": result.reason,
                             "history": "Zero initial d,v,u; no accept during candidate solves",
                             "force": "-sigma*n_fluid; already integrated force, unit depth 1 m",
                             "direct": "Same FE matrices; analytic interface_load_jvp at fixed geometry; coupled velocity rows"})
    except (ValueError, TypeError, RuntimeError, OSError) as error:
        if isinstance(error, CouplingFailure):
            history = error.history  # 耦合失败时仍保留已产生的迭代历史
        if history:
            write_csv(output / "iterations.csv", iteration_rows(history))
        # A partially written success artifact must not survive a failed run.
        for name in ("report.json", "solution.npz"):  # 清理可能已写的成功产物
            (output / name).unlink(missing_ok=True)
        _write_failure(output, case=config.get("case", "partitioned_check") if isinstance(config, dict)
                       else "partitioned_check", config=config, config_path=None,
                       stage=error.stage if isinstance(error, CouplingFailure) else stage,
                       reason=str(error),
                       iteration=error.iteration if isinstance(error, CouplingFailure)
                       else (len(history) if history else None),
                       problem=problem, snapshots=snapshots)
        raise


def run_partitioned_check_cli(config_path, output, *, method=None):
    """CLI entry: own output cleanup and record input failures with the path.

    Clearing happens before any failure can leave a stale report or solution,
    and an input failure never claims to have loaded a configuration.

    中文：CLI 入口，自行负责输出清理并记录带路径的输入失败。清理在任何可能残留失败
    产物之前完成；读取/解析/校验配置的输入失败分别以 config_read/config_parse/
    config_validate 阶段写入 failure.json（不声称已加载配置），随后抛出异常。
    ``method`` 非空时覆盖配置中的耦合算法选择。
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _clear_outputs(output)
    case = "partitioned_check"
    try:
        text = Path(config_path).read_text(encoding="utf-8")
    except OSError as error:
        _write_failure(output, case=case, config=None, config_path=str(config_path), stage="config_read",
                       reason=str(error), iteration=None, problem=None, snapshots=None)
        raise
    try:
        data = json.loads(text)
    except ValueError as error:
        _write_failure(output, case=case, config=None, config_path=str(config_path), stage="config_parse",
                       reason=str(error), iteration=None, problem=None, snapshots=None)
        raise
    try:
        config = validate_config(data)
    except (ValueError, TypeError) as error:
        _write_failure(output, case=case, config=None, config_path=str(config_path), stage="config_validate",
                       reason=str(error), iteration=None, problem=None, snapshots=None)
        raise
    if method is not None:
        config["coupling"]["method"] = method
    return run_partitioned_check(config, output)

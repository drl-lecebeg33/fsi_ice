"""Shared physical BDF step with partitioned/monolithic solve and atomic accept.

The partitioned fixed point uses the full solid displacement. Interface motion
is its trace; using the full vector also makes the solid-dependent traction
geometry identical to T21. Neither route accepts any history inside iterations.

中文说明：本模块把同一个物理 BDF 时间步交给两条求解路径——分区
（Dirichlet--Neumann 不动点）或单体（monolithic Newton）——并做原子接受。关键
约定：分区不动点以完整固体位移作为界面未知量，界面运动只是其迹；采用完整向量
还能让依赖固体的牵引几何与 T21 完全一致。两条路径的迭代内部都不接受任何时间
历史；单步结果由 StateCoordinator 事务统一提交（publish 恰好一次），失败时已
接受状态保持不变并保留失败审计。
"""

from time import perf_counter

import numpy as np

from fsi_ice.ale import build_ale_geometry, build_gcl_transport, evaluate_mesh_velocity
from fsi_ice.coupling.monolithic import (MonolithicBase, MonolithicContext, build_block_layout,
                                        evaluate_monolithic)
from fsi_ice.coupling.monolithic_newton import solve_monolithic_step
from fsi_ice.coupling.newton import NewtonOptions
from fsi_ice.coupling.partitioned import (CouplingOptions, PartitionedEvaluation,
                                         solve_dirichlet_neumann)
from fsi_ice.fluid.ale import build_ale_flow_context
from fsi_ice.fluid.dynamics import solve_flow_step
from fsi_ice.fluid.linear_solver import ReusedLUSolver
from fsi_ice.fluid.taylor_hood import assemble_stokes
from fsi_ice.interface.geometry import evaluate_interface_geometry
from fsi_ice.interface.traction import assemble_interface_load
from fsi_ice.solid.dynamics import solve_solid_step
from fsi_ice.time.transaction import AcceptedState, StateCoordinator, StepPlan
from fsi_ice.validation.monolithic import block_scale_vector


# Inner Newton solves must also satisfy an absolute share of the coupled gate:
# their relative criterion scales with the equation magnitudes (ALE RHS/history
# loads, external solid load) and can leave a raw residual above "residual_max"
# once the response amplitude grows. The 2-norm bound below implies the
# max-component bound used by the acceptance gate, with one decade of margin.
_INNER_GATE_SHARE = 0.1


class TransientFSI:
    """One reference mesh and one accepted root, common to both solve routes.

    中文：瞬态 FSI 步进器，持有一张参考网格与一个已接受状态根，供分区与单体两条
    求解路径共用；内部用 StateCoordinator 管理事务式提交。
    """

    def __init__(self, topology, harmonic, material, *, density, viscosity, boundary_velocity,
                 fluid_tags, solid_nodes, dt, order=2, monitor=None, scales=None,
                 fluid_interface_tag="fluid-interface"):
        """中文：构造瞬态 FSI 步进器。``topology`` 为分区拓扑（流体/固体网格与
        界面映射），``harmonic`` 为界面调和扩展，``material`` 为固体材料（提供
        thickness）。``density``/``viscosity`` 为流体物性；``boundary_velocity``
        为函数 (time, points)->速度；``fluid_tags``/``solid_nodes`` 为 Dirichlet
        边界标记与节点；``dt`` 与 ``order`` 为 BDF 步长与阶数。``scales`` 可覆盖
        各物理块的无量纲尺度；``monitor`` 为每步监控回调。初始接受状态全为零。
        """
        self.topology, self.harmonic, self.material = topology, harmonic, material
        self.density, self.viscosity = density, viscosity
        self.boundary_velocity, self.fluid_tags = boundary_velocity, tuple(fluid_tags)
        self.fluid_interface_tag = fluid_interface_tag
        self.solid_nodes, self.monitor = np.array(solid_nodes, copy=True), monitor
        self.layout = build_block_layout(topology, fluid_dirichlet_tags=fluid_tags,
                                         solid_dirichlet_nodes=solid_nodes, pressure_pin=None)
        # 各物理块的无量纲尺度向量，用于残差与收敛归一化；默认取本模型典型量级
        self.scales = block_scale_vector(self.layout, scales or {
            "u_f": 1., "p_f": 1000., "d_s": .41, "v_s": 1., "d_m": .41, "g": 215.06})
        fluid, solid = topology.fluid_mesh, topology.solid_mesh
        self.coordinator = StateCoordinator(AcceptedState.initial(
            solid=np.zeros((2, solid.n_nodes, 2)),
            velocity=np.zeros((len(topology.fluid_space.points), 2)),
            pressure=np.zeros(fluid.n_nodes), mesh_displacement=np.zeros_like(fluid.points),
            mesh_current=fluid.points, mesh_velocity=np.zeros_like(fluid.points), dt=dt, order=order))

    def _context(self, view):
        """中文：由候选视图 view 组装单体求解上下文，取前 order 步网格位移构造
        已接受 ALE 几何，并填入固体/流体历史与当前时刻的边界速度。"""

        order = view.stencil.order  # BDF 阶数，决定参与的历史步数
        geometries = tuple(build_ale_geometry(self.topology.fluid_mesh, displacement,
                          candidate_id=(view.attempt_id, "history", index))
                          for index, displacement in enumerate(view.base.mesh_history[:order]))
        base = MonolithicBase(reference_mesh=self.topology.fluid_mesh, material=self.material,
            fluid_density=self.density, fluid_viscosity=self.viscosity,
            solid_states=tuple(tuple(state) for state in view.base.solid_history[:order]),
            fluid_states=view.base.fluid_history[:order], accepted_geometries=geometries,
            generation=view.stencil.step_id,
            fluid_boundary_velocity=lambda points: self.boundary_velocity(view.target_time, points))
        return MonolithicContext(self.layout, base, view.stencil, self.topology, self.harmonic,
                                 candidate_id=(view.attempt_id, "coupled"), thickness=self.material.thickness)

    def _initial_vector(self, state):
        """中文：把已接受状态打包成单体求解的初始块向量 x0。各物理块（流体速度/
        压力、固体位移/速度、网格位移、界面量 g）按 layout 顺序拼接；g 缺省为零。"""
        return self.layout.pack({"u_f": state.velocity, "p_f": state.pressure,
            "d_s": state.solid_state[0], "v_s": state.solid_state[1],
            "d_m": state.mesh_displacement,
            "g": state.interface.get("g", np.zeros(self.layout.shape_of("g")))})

    def _partitioned(self, view, context, options, tolerance, max_iterations, residual_max,
                     fluid_absolute=None, solid_absolute=None):
        """中文：用一个时间步的分区 Dirichlet--Neumann 迭代求解。界面未知量取
        完整固体位移（拉直为一维），``evaluate`` 每次给出界面映象 G(x) 及其打包块
        向量；返回被接受候选的流体解与各类计数字典。``tolerance``/``max_iterations``
        透传给内部流体与固体单场求解，``fluid_absolute``/``solid_absolute`` 为耦合
        门控导出的单场自由残差绝对上限。"""
        topology, fluid, solid = self.topology, self.topology.fluid_mesh, self.topology.solid_mesh
        boundary_nodes = fluid.boundary_nodes()
        boundary_index = {int(node): i for i, node in enumerate(boundary_nodes)}
        counts = {"fluid_solves": 0, "solid_solves": 0, "newton_iterations": 0, "linear_solves": 0,
                  "fluid_factorizations": 0, "fluid_krylov_iterations": 0}
        flow_initial = np.r_[view.base.velocity.ravel(), view.base.pressure]
        linear_solver = ReusedLUSolver(velocity_dofs=self.layout.size_of("u_f"),
                                       density=self.density, reuse_across_candidates=True)
        inner_factor, inner_refinements = 1.0, 0

        def evaluate(displacement):
            nonlocal flow_initial, inner_factor, inner_refinements
            ds = displacement.reshape(solid.points.shape)  # 拉直界面量恢复为 (节点数,2) 位移
            # 由 BDF 系数与历史项直接给出固体速度（瞬时运动学关系）
            vs = view.solid_context.coefficient*ds + view.solid_context.history_term[0]
            boundary = np.zeros((len(boundary_nodes), 2))
            # 按 固体节点->流体顶点 映射把固体位移写入流体边界
            for sn, fn in zip(topology.solid_nodes, topology.fluid_vertex_nodes):
                boundary[boundary_index[int(fn)]] = ds[sn]
            dm = self.harmonic.extend(boundary_nodes, boundary).values  # 调和扩展得网格位移
            geometry = build_ale_geometry(fluid, dm, candidate_id=context.candidate_id)
            transport = build_gcl_transport(fluid, geometry, context.base.accepted_geometries, view.stencil)
            operators = assemble_stokes(geometry.current_mesh, density=self.density, viscosity=self.viscosity)
            ale = build_ale_flow_context(operators, geometry, transport, context.base.fluid_states)
            tags = list(self.fluid_tags) + [self.fluid_interface_tag]
            nodes = operators.space.boundary_nodes(tags)
            prescribed = self.boundary_velocity(view.target_time, operators.space.points)
            # 界面 P2 速度由固体速度经映射 H 给出，体现界面运动
            prescribed[topology.fluid_p2_nodes] = (topology.H @ vs.ravel()[topology.solid_dofs]).reshape(-1, 2)
            solution = solve_flow_step(operators, prescribed[nodes], view.fluid_context,
                tags=tags, pressure_pin=None, ale_context=ale,
                initial=flow_initial, linear_solver=linear_solver,
                tolerance=tolerance, absolute_tolerance=(None if fluid_absolute is None else
                    inner_factor*fluid_absolute), max_iterations=max_iterations)
            flow_initial = np.r_[solution.velocity.ravel(), solution.pressure]
            interface = evaluate_interface_geometry(topology, geometry.current_mesh, operators.space, ds,
                                                     candidate_id=context.candidate_id)
            load = assemble_interface_load(topology, interface, operators, solution,
                                            thickness=self.material.thickness)
            external = np.zeros(2*solid.n_nodes)
            external[topology.solid_dofs] = load.f_s_gamma  # 界面牵引装配为固体外力
            ds_scales = self.scales[self.layout.slice_of("d_s")]
            input_threshold = options.absolute_tolerance + options.relative_tolerance*float(
                np.max(np.abs(displacement/ds_scales)))
            correction_tolerance = (max(100*inner_factor*input_threshold*float(np.min(ds_scales)),
                                        np.finfo(float).tiny) if inner_refinements else None)
            state, solid_result = solve_solid_step(solid, self.material, external, self.solid_nodes,
                view.solid_context, initial=ds, tolerance=tolerance,
                absolute_tolerance=solid_absolute, correction_tolerance=correction_tolerance,
                max_iterations=max_iterations)
            counts["fluid_solves"] += 1
            counts["solid_solves"] += 1
            # 合并两场 Newton 迭代数（各自历史首行为初值，故各减 1）
            iterations = len(solution.nonlinear_history)+len(solid_result.nonlinear_history)-2
            counts["newton_iterations"] += iterations
            counts["linear_solves"] += len(solution.nonlinear_history)-1+solid_result.linear_solves
            # 组装本候选的完整块向量 x（与 layout 分块一致）
            x = self.layout.pack({"u_f": solution.velocity, "p_f": solution.pressure,
                "d_s": ds, "v_s": vs, "d_m": dm,
                "g": load.g_f_gamma.reshape(self.layout.shape_of("g"))})
            # An inexact field solve can leave a noisy map near the much tighter
            # displacement gate. Refine at the same input, at most three decades
            # per step. For the solid certify displacement correction accuracy;
            # forcing ever smaller force residuals can hit a roundoff floor.
            gap = float(np.max(np.abs((state[0].ravel()-displacement)/ds_scales)))
            magnitude = max(float(np.max(np.abs(displacement/ds_scales))),
                            float(np.max(np.abs(state[0].ravel()/ds_scales))))
            threshold = options.absolute_tolerance + options.relative_tolerance*magnitude
            if threshold < gap <= 1000*threshold and inner_refinements < 3 and \
                    fluid_absolute is not None and solid_absolute is not None:
                inner_factor *= .1
                inner_refinements += 1
                return evaluate(displacement)
            # 返回 G(x)：target 为固体新位移，fluid 位置暂存打包块向量 x
            return PartitionedEvaluation(state[0].ravel(), x, load, state)

        def physical_convergence(evaluation):
            # A tiny displacement gap can still produce a large solid momentum
            # residual. Check the same six-block equation used at transaction
            # acceptance before ending the interface iteration.
            residual = evaluate_monolithic(evaluation.fluid, context=context, mode="residual").residual
            norm = float(np.max(np.abs(residual/self.scales)))
            if not np.isfinite(norm):
                raise RuntimeError("Nonfinite coupled residual during partitioned convergence check.")
            return norm <= residual_max

        # 以已接受固体位移为界面初值，界面尺度取 layout 中 d_s 分块的尺度
        result = solve_dirichlet_neumann(evaluate, view.base.solid_state[0].ravel(),
                     scales=self.scales[self.layout.slice_of("d_s")], options=options,
                     convergence_check=physical_convergence)
        # 返回流体解与耦合统计（迭代数、LU/GMRES 工作量、IQN 更新/重启、末次界面残差与松弛因子）
        return result.evaluation.fluid, {**counts,
            "fluid_factorizations": linear_solver.factorizations,
            "fluid_krylov_iterations": linear_solver.iterations,
            "coupling_iterations": len(result.history),
            "iqn_updates": sum(row.relaxation_reason == "iqn" for row in result.history),
            "iqn_restarts": sum(row.relaxation_reason == "iqn_restart" for row in result.history),
            "inner_refinements": inner_refinements,
            "interface_residual": result.history[-1].residual_norm,
            "relaxation": result.history[-1].relaxation}

    def step(self, method="aitken", *, coupling=None, newton=None, single_tolerance=1e-12,
             residual_max=1e-7, continuity_max=1e-10, minimum_quality=.1, hooks=None, checks=None):
        """Solve and validate one physical step, publishing exactly once on success.

        No method switches on failure. Optional hooks/checks are real transaction
        seams for callers and fault tests. A raised exception leaves the accepted
        root untouched; the coordinator retains the failure audit.

        中文：求解并校验一个物理时间步，成功时恰好发布一次。失败时不切换方法；
        可选的 ``hooks``/``checks`` 是给调用方与故障测试的真实事务接缝。抛出的
        异常不改动已接受的状态根，协调器保留失败审计。``method`` 取
        ``fixed``/``aitken``/``monolithic``；``single_tolerance`` 传给单场求解，
        ``residual_max``/``continuity_max``/``minimum_quality`` 为共享物理闸门
        阈值。成功返回协调器当前状态。
        """
        if method not in ("fixed", "aitken", "iqn", "monolithic"):
            raise ValueError("method must be fixed, aitken, iqn or monolithic.")
        for name, value in (("single_tolerance", single_tolerance), ("residual_max", residual_max),
                            ("continuity_max", continuity_max), ("minimum_quality", minimum_quality)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or \
                    not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive real number.")
        if minimum_quality > 1 or "coupled_residual" in dict(checks or {}):
            # 质量下限须在 (0,1]；且不允许外部覆盖共享物理闸门 coupled_residual
            raise ValueError("Invalid quality limit or attempted override of the shared physics gate.")
        coupling = dict(coupling or {})
        # 布置 CouplingOptions：单体路径内部仍用 fixed 松弛，并统一用例名
        coupling.update(method=method if method != "monolithic" else "fixed", case_name="transient-fsi")
        coupling_options = CouplingOptions(**coupling)
        newton = NewtonOptions(atol=1e-10, rtol=1e-12) if newton is None else newton
        # Gate-derived absolute targets for the single-field Newton solves.
        inner_absolute = _INNER_GATE_SHARE*residual_max
        fluid_absolute = inner_absolute*float(np.max(self.scales[self.layout.slice_of("u_f")]))
        solid_absolute = inner_absolute*float(np.max(self.scales[self.layout.slice_of("v_s")]))
        cache = {}  # 跨事务回调共享的暂存区（求解结果、分块、计数、指标）

        def displacement(view):
            # 事务回调：求解本步（单体或分区），缓存结果并返回网格位移 d_m 供几何构建
            started = perf_counter()
            context = self._context(view)
            if method == "monolithic":
                result = solve_monolithic_step(context, self._initial_vector(view.base),
                                               options=newton, scales=self.scales)
                x = result.x
                counts = result.counters.as_dict()
                counts["coupling_iterations"] = 0  # 单体路径无外层耦合迭代
            else:
                x, counts = self._partitioned(view, context, coupling_options, single_tolerance,
                                              newton.max_iterations, residual_max, fluid_absolute, solid_absolute)
            cache.update(context=context, x=x, blocks=self.layout.unpack(x), counts=counts,
                         solve_seconds=perf_counter()-started)
            return cache["blocks"]["d_m"]  # 以网格位移驱动后续几何/网格速度构建

        def physics_check(view):
            # 共享物理闸门：在真实被接受候选上重算耦合残差与弱连续性
            candidate = self.layout.pack({"u_f": view.velocity, "p_f": view.pressure,
                "d_s": view.solid[0], "v_s": view.solid[1], "d_m": view.displacement,
                "g": view.interface["g"]})
            evaluation = evaluate_monolithic(candidate, context=cache["context"], mode="residual")
            norm = float(np.max(np.abs(evaluation.residual/self.scales)))  # 归一化耦合残差无穷范数
            diagnostics = dict(evaluation.geometry_diagnostics)
            # Recompute weak continuity on the actual accepted candidate.
            # 中文：在被接受的真实候选上重算弱连续性（归一化散度）。
            operators = assemble_stokes(view.geometry.current_mesh, density=self.density, viscosity=self.viscosity)
            continuity = float(np.linalg.norm(operators.divergence @ view.velocity.ravel()) /
                               max(np.linalg.norm(view.velocity), 1.))
            # 任一闸门越界或非有限即判物理失败，终止整个事务
            if not np.isfinite(norm) or not np.isfinite(continuity) or norm > residual_max or \
                    continuity > continuity_max or diagnostics["minimum_quality"] < minimum_quality:
                raise RuntimeError(f"transient step physics failed: residual={norm:.3e}, "
                                   f"continuity={continuity:.3e}, quality={diagnostics['minimum_quality']:.3e}")
            cache["metrics"] = {"residual": norm, "continuity": continuity, **diagnostics}
            return dict(cache["metrics"])

        def monitor(view):
            # 组装监控行（时间/步/阶/方法/耗时/指标/计数），并叠加调用方 monitor
            row = {"time": view.target_time, "step": view.target_step, "order": view.stencil.order,
                   "method": method, "solve_seconds": cache["solve_seconds"],
                   **cache["metrics"], **cache["counts"]}
            if self.monitor is not None:
                row.update(self.monitor(view, cache["blocks"]))
            return row

        # 组装单步事务计划；coupled_residual 是不可覆盖的共享物理检查
        plan = StepPlan(build_displacement=displacement,
            build_geometry=lambda view: build_ale_geometry(self.topology.fluid_mesh, view.displacement,
                                                           candidate_id=view.revision),
            build_mesh_velocity=lambda view: evaluate_mesh_velocity(view.geometry, view.mesh_context).values_p1,
            build_mesh_current=lambda view: view.geometry.current_mesh.points,
            solve_solid=lambda view: np.stack((cache["blocks"]["d_s"], cache["blocks"]["v_s"])),
            solve_fluid=lambda view: (cache["blocks"]["u_f"], cache["blocks"]["p_f"]),
            build_interface=lambda view: {"g": cache["blocks"]["g"]},
            interface_shapes={"g": self.layout.shape_of("g")},
            checks={"coupled_residual": physics_check, **dict(checks or {})},
            build_monitor=monitor, hooks=hooks or {})
        attempt = self.coordinator.begin(plan=plan)
        try:
            # 事务四阶段：求值→校验→准备→发布
            attempt.evaluate()
            attempt.validate()
            attempt.prepare()
            attempt.publish()
        except Exception as error:
            # 任一步失败则中止并保留审计；已 committed/aborted 时不再重复变更
            if attempt.status not in ("aborted", "committed"):
                attempt.abort(stage="transient", reason=str(error))
            raise
        return self.coordinator.state

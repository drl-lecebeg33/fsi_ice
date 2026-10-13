"""Shared physical BDF step with partitioned/monolithic solve and atomic accept.

The partitioned fixed point uses the full solid displacement. Interface motion
is its trace; using the full vector also makes the solid-dependent traction
geometry identical to T21. Neither route accepts any history inside iterations.
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
from fsi_ice.fluid.taylor_hood import assemble_stokes
from fsi_ice.interface.geometry import evaluate_interface_geometry
from fsi_ice.interface.traction import assemble_interface_load
from fsi_ice.solid.dynamics import solve_solid_step
from fsi_ice.time.transaction import AcceptedState, StateCoordinator, StepPlan
from fsi_ice.validation.monolithic import block_scale_vector


class TransientFSI:
    """One reference mesh and one accepted root, common to both solve routes."""

    def __init__(self, topology, harmonic, material, *, density, viscosity, boundary_velocity,
                 fluid_tags, solid_nodes, dt, order=2, monitor=None, scales=None,
                 fluid_interface_tag="fluid-interface"):
        self.topology, self.harmonic, self.material = topology, harmonic, material
        self.density, self.viscosity = density, viscosity
        self.boundary_velocity, self.fluid_tags = boundary_velocity, tuple(fluid_tags)
        self.fluid_interface_tag = fluid_interface_tag
        self.solid_nodes, self.monitor = np.array(solid_nodes, copy=True), monitor
        self.layout = build_block_layout(topology, fluid_dirichlet_tags=fluid_tags,
                                         solid_dirichlet_nodes=solid_nodes, pressure_pin=None)
        self.scales = block_scale_vector(self.layout, scales or {
            "u_f": 1., "p_f": 1000., "d_s": .41, "v_s": 1., "d_m": .41, "g": 215.06})
        fluid, solid = topology.fluid_mesh, topology.solid_mesh
        self.coordinator = StateCoordinator(AcceptedState.initial(
            solid=np.zeros((2, solid.n_nodes, 2)),
            velocity=np.zeros((len(topology.fluid_space.points), 2)),
            pressure=np.zeros(fluid.n_nodes), mesh_displacement=np.zeros_like(fluid.points),
            mesh_current=fluid.points, mesh_velocity=np.zeros_like(fluid.points), dt=dt, order=order))

    def _context(self, view):
        order = view.stencil.order
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
        return self.layout.pack({"u_f": state.velocity, "p_f": state.pressure,
            "d_s": state.solid_state[0], "v_s": state.solid_state[1],
            "d_m": state.mesh_displacement,
            "g": state.interface.get("g", np.zeros(self.layout.shape_of("g")))})

    def _partitioned(self, view, context, options, tolerance, max_iterations):
        topology, fluid, solid = self.topology, self.topology.fluid_mesh, self.topology.solid_mesh
        boundary_nodes = fluid.boundary_nodes()
        boundary_index = {int(node): i for i, node in enumerate(boundary_nodes)}
        counts = {"fluid_solves": 0, "solid_solves": 0, "newton_iterations": 0, "linear_solves": 0}

        def evaluate(displacement):
            ds = displacement.reshape(solid.points.shape)
            vs = view.solid_context.coefficient*ds + view.solid_context.history_term[0]
            boundary = np.zeros((len(boundary_nodes), 2))
            for sn, fn in zip(topology.solid_nodes, topology.fluid_vertex_nodes):
                boundary[boundary_index[int(fn)]] = ds[sn]
            dm = self.harmonic.extend(boundary_nodes, boundary).values
            geometry = build_ale_geometry(fluid, dm, candidate_id=context.candidate_id)
            transport = build_gcl_transport(fluid, geometry, context.base.accepted_geometries, view.stencil)
            operators = assemble_stokes(geometry.current_mesh, density=self.density, viscosity=self.viscosity)
            ale = build_ale_flow_context(operators, geometry, transport, context.base.fluid_states)
            tags = list(self.fluid_tags) + [self.fluid_interface_tag]
            nodes = operators.space.boundary_nodes(tags)
            prescribed = self.boundary_velocity(view.target_time, operators.space.points)
            prescribed[topology.fluid_p2_nodes] = (topology.H @ vs.ravel()[topology.solid_dofs]).reshape(-1, 2)
            solution = solve_flow_step(operators, prescribed[nodes], view.fluid_context,
                tags=tags, pressure_pin=None, ale_context=ale,
                initial=np.r_[view.base.velocity.ravel(), view.base.pressure],
                tolerance=tolerance, max_iterations=max_iterations)
            interface = evaluate_interface_geometry(topology, geometry.current_mesh, operators.space, ds,
                                                     candidate_id=context.candidate_id)
            load = assemble_interface_load(topology, interface, operators, solution,
                                            thickness=self.material.thickness)
            external = np.zeros(2*solid.n_nodes)
            external[topology.solid_dofs] = load.f_s_gamma
            state, solid_result = solve_solid_step(solid, self.material, external, self.solid_nodes,
                view.solid_context, initial=ds, tolerance=tolerance, max_iterations=max_iterations)
            counts["fluid_solves"] += 1
            counts["solid_solves"] += 1
            iterations = len(solution.nonlinear_history)+len(solid_result.nonlinear_history)-2
            counts["newton_iterations"] += iterations
            counts["linear_solves"] += iterations
            x = self.layout.pack({"u_f": solution.velocity, "p_f": solution.pressure,
                "d_s": ds, "v_s": vs, "d_m": dm,
                "g": load.g_f_gamma.reshape(self.layout.shape_of("g"))})
            return PartitionedEvaluation(state[0].ravel(), x, load, state)

        result = solve_dirichlet_neumann(evaluate, view.base.solid_state[0].ravel(),
                     scales=self.scales[self.layout.slice_of("d_s")], options=options)
        return result.evaluation.fluid, {**counts, "coupling_iterations": len(result.history),
            "iqn_updates": sum(row.relaxation_reason == "iqn" for row in result.history),
            "iqn_restarts": sum(row.relaxation_reason == "iqn_restart" for row in result.history),
            "interface_residual": result.history[-1].residual_norm,
            "relaxation": result.history[-1].relaxation}

    def step(self, method="aitken", *, coupling=None, newton=None, single_tolerance=1e-12,
             residual_max=1e-7, continuity_max=1e-10, minimum_quality=.1, hooks=None, checks=None):
        """Solve and validate one physical step, publishing exactly once on success.

        No method switches on failure. Optional hooks/checks are real transaction
        seams for callers and fault tests. A raised exception leaves the accepted
        root untouched; the coordinator retains the failure audit.
        """
        if method not in ("fixed", "aitken", "iqn", "monolithic"):
            raise ValueError("method must be fixed, aitken, iqn or monolithic.")
        for name, value in (("single_tolerance", single_tolerance), ("residual_max", residual_max),
                            ("continuity_max", continuity_max), ("minimum_quality", minimum_quality)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or \
                    not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive real number.")
        if minimum_quality > 1 or "coupled_residual" in dict(checks or {}):
            raise ValueError("Invalid quality limit or attempted override of the shared physics gate.")
        coupling = dict(coupling or {})
        coupling.update(method=method if method != "monolithic" else "fixed", case_name="transient-fsi")
        coupling_options = CouplingOptions(**coupling)
        newton = NewtonOptions(atol=1e-10, rtol=1e-12) if newton is None else newton
        cache = {}

        def displacement(view):
            started = perf_counter()
            context = self._context(view)
            if method == "monolithic":
                result = solve_monolithic_step(context, self._initial_vector(view.base),
                                               options=newton, scales=self.scales)
                x = result.x
                counts = result.counters.as_dict()
                counts["coupling_iterations"] = 0
            else:
                x, counts = self._partitioned(view, context, coupling_options, single_tolerance,
                                              newton.max_iterations)
            cache.update(context=context, x=x, blocks=self.layout.unpack(x), counts=counts,
                         solve_seconds=perf_counter()-started)
            return cache["blocks"]["d_m"]

        def physics_check(view):
            candidate = self.layout.pack({"u_f": view.velocity, "p_f": view.pressure,
                "d_s": view.solid[0], "v_s": view.solid[1], "d_m": view.displacement,
                "g": view.interface["g"]})
            evaluation = evaluate_monolithic(candidate, context=cache["context"], mode="residual")
            norm = float(np.max(np.abs(evaluation.residual/self.scales)))
            diagnostics = dict(evaluation.geometry_diagnostics)
            # Recompute weak continuity on the actual accepted candidate.
            operators = assemble_stokes(view.geometry.current_mesh, density=self.density, viscosity=self.viscosity)
            continuity = float(np.linalg.norm(operators.divergence @ view.velocity.ravel()) /
                               max(np.linalg.norm(view.velocity), 1.))
            if not np.isfinite(norm) or not np.isfinite(continuity) or norm > residual_max or \
                    continuity > continuity_max or diagnostics["minimum_quality"] < minimum_quality:
                raise RuntimeError(f"transient step physics failed: residual={norm:.3e}, "
                                   f"continuity={continuity:.3e}, quality={diagnostics['minimum_quality']:.3e}")
            cache["metrics"] = {"residual": norm, "continuity": continuity, **diagnostics}
            return dict(cache["metrics"])

        def monitor(view):
            row = {"time": view.target_time, "step": view.target_step, "order": view.stencil.order,
                   "method": method, "solve_seconds": cache["solve_seconds"],
                   **cache["metrics"], **cache["counts"]}
            if self.monitor is not None:
                row.update(self.monitor(view, cache["blocks"]))
            return row

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
            attempt.evaluate()
            attempt.validate()
            attempt.prepare()
            attempt.publish()
        except Exception as error:
            if attempt.status not in ("aborted", "committed"):
                attempt.abort(stage="transient", reason=str(error))
            raise
        return self.coordinator.state

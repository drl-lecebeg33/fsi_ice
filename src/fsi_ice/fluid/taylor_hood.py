"""Physical pressure convention: -div(2 mu eps(u)) + grad(p) = f."""

from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix

from fsi_ice.fem.p1 import map_points, shape_values
from fsi_ice.fem.p2 import P2Space, p2_physical_gradients, p2_reference_gradients, p2_values
from fsi_ice.fem.poisson import solve_dirichlet
from fsi_ice.fem.quadrature import triangle_quadrature
from fsi_ice.solid.linear import vector_dofs, vector_values


@lru_cache(maxsize=1)
def _reference_convection_tensor():
    points, weights = triangle_quadrature(5)
    shape = p2_values(points)
    tensor = np.einsum("q,qa,qb,qcj->abcj", weights, shape, shape,
                       p2_reference_gradients(points), optimize="greedy")
    tensor.setflags(write=False)
    return tensor


def scatter_matrix(local, row_dofs, column_dofs, shape):
    nrow, ncol = local.shape[1:]
    rows = np.repeat(row_dofs, ncol, axis=1).ravel()
    columns = np.tile(column_dofs, (1, nrow)).ravel()
    return coo_matrix((local.ravel(), (rows, columns)), shape=shape).tocsr()


@dataclass(frozen=True)
class FlowOperators:
    space: P2Space
    matrix: object
    rhs: np.ndarray
    mass: object
    divergence: object
    viscosity: float
    density: float
    quadrature: np.ndarray
    weights: np.ndarray
    gradients: np.ndarray
    body_force: object = (0.0, 0.0)

    @property
    def n_velocity(self):
        return 2*len(self.space.points)

    @property
    def n_total(self):
        return self.n_velocity+self.space.mesh.n_nodes

    @cached_property
    def convection_tensor(self):
        # Integral N_a*N_b*grad(N_c): exact for the degree-five convective form.
        # Affine triangles: integrate once on the reference cell, then transform.
        points, weights = triangle_quadrature(5)
        if np.array_equal(self.quadrature, points) and np.array_equal(self.weights, weights):
            tensor = np.einsum("abck,ekj->eabcj", _reference_convection_tensor(),
                               np.linalg.inv(self.space.mesh.jacobians), optimize="greedy")
        else:
            shape = p2_values(self.quadrature)
            tensor = np.einsum("q,qa,qb,eqcj->eabcj", self.weights, shape, shape, self.gradients,
                               optimize="greedy")
        tensor *= (self.density*self.space.mesh.determinants)[:, None, None, None, None]
        tensor.flags.writeable = False
        return tensor


def assemble_stokes(mesh, *, viscosity=1.0, density=1.0, body_force=(0.0, 0.0)):
    if not np.isfinite(viscosity) or viscosity <= 0 or not np.isfinite(density) or density <= 0:
        raise ValueError("Viscosity and density must be finite and positive.")
    space = P2Space.from_mesh(mesh)
    points, weights = triangle_quadrature(5)
    velocity_shape = p2_values(points)
    pressure_shape = shape_values(points)
    gradients = p2_physical_gradients(mesh, points)
    strains = np.zeros((mesh.n_cells, len(points), 3, 12))
    strains[:, :, 0, 0::2] = gradients[:, :, :, 0]
    strains[:, :, 1, 1::2] = gradients[:, :, :, 1]
    strains[:, :, 2, 0::2] = gradients[:, :, :, 1]
    strains[:, :, 2, 1::2] = gradients[:, :, :, 0]
    constitutive = viscosity*np.diag([2.0, 2.0, 1.0])
    local_a = np.einsum("q,eqai,ab,eqbj->eij", weights, strains, constitutive, strains, optimize="greedy")
    local_a *= mesh.determinants[:, None, None]
    divergence_basis = np.empty((mesh.n_cells, len(points), 12))
    divergence_basis[:, :, 0::2] = gradients[:, :, :, 0]
    divergence_basis[:, :, 1::2] = gradients[:, :, :, 1]
    local_b = np.einsum("q,qp,eqv->epv", weights, pressure_shape, divergence_basis)
    local_b *= mesh.determinants[:, None, None]
    local_mass_scalar = np.einsum("q,qa,qb->ab", weights, velocity_shape, velocity_shape)
    local_mass = np.einsum("ab,ij->aibj", local_mass_scalar, np.eye(2)).reshape(12, 12)
    local_mass = (density*mesh.determinants[:, None, None])*local_mass
    dofs = vector_dofs(space.cells)
    nv = 2*len(space.points)
    a = scatter_matrix(local_a, dofs, dofs, (nv, nv))
    divergence = scatter_matrix(local_b, mesh.cells, dofs, (mesh.n_nodes, nv))
    mass = scatter_matrix(local_mass, dofs, dofs, (nv, nv))
    # Multiplying the incompressibility equation by -1 makes the saddle matrix symmetric.
    matrix = bmat([[a, -divergence.T], [-divergence, None]], format="csr")
    physical_points = map_points(mesh, points)
    force = vector_values(body_force, physical_points.reshape(-1, 2), "Body force").reshape(mesh.n_cells, -1, 2)
    load = np.einsum("q,eqi,qa->eai", weights, force, velocity_shape)
    load *= mesh.determinants[:, None, None]
    rhs = np.zeros(nv+mesh.n_nodes)
    np.add.at(rhs, dofs.ravel(), load.ravel())
    return FlowOperators(space, matrix, rhs, mass, divergence, viscosity, density, points, weights, gradients,
                         body_force)


def assemble_velocity_mass(mesh, *, density=1.0):
    """Consistent P2 velocity mass matrix on one mesh (same quadrature as assemble_stokes)."""
    if not np.isfinite(density) or density <= 0:
        raise ValueError("Density must be finite and positive.")
    space = P2Space.from_mesh(mesh)
    points, weights = triangle_quadrature(5)
    velocity_shape = p2_values(points)
    local_mass_scalar = np.einsum("q,qa,qb->ab", weights, velocity_shape, velocity_shape)
    local_mass = np.einsum("ab,ij->aibj", local_mass_scalar, np.eye(2)).reshape(12, 12)
    local_mass = (density*mesh.determinants[:, None, None])*local_mass
    dofs = vector_dofs(space.cells)
    return scatter_matrix(local_mass, dofs, dofs, (2*len(space.points),)*2)


def flow_constraints(operators, velocity, tags=None, *, pressure_pin=0, pressure_value=0.0):
    nodes = operators.space.boundary_nodes(tags)
    values = vector_values(velocity, operators.space.points[nodes], "Boundary velocity")
    dofs = vector_dofs(nodes)
    prescribed = values.ravel()
    if pressure_pin is not None:
        if isinstance(pressure_pin, bool) or not isinstance(pressure_pin, (int, np.integer)):
            raise ValueError("Pressure pin must be an integer node or None.")
        if not 0 <= pressure_pin < operators.space.mesh.n_nodes or not np.isfinite(pressure_value):
            raise ValueError("Pressure pin/value is invalid.")
        dofs = np.append(dofs, operators.n_velocity+pressure_pin)
        prescribed = np.append(prescribed, pressure_value)
    return dofs, prescribed


@dataclass(frozen=True)
class FlowSolution:
    velocity: np.ndarray
    pressure: np.ndarray
    relative_residual: float
    continuity_residual: float
    nonlinear_history: tuple = ()


def unpack_flow(operators, values, residual, history=()):
    velocity = values[:operators.n_velocity].reshape(-1, 2)
    pressure = values[operators.n_velocity:]
    continuity = np.linalg.norm(operators.divergence @ velocity.ravel()) / max(np.linalg.norm(velocity), 1)
    return FlowSolution(velocity, pressure, float(residual), float(continuity), tuple(history))


def solve_stokes(operators, velocity, tags=None, *, pressure_pin=0, pressure_value=0.0):
    dofs, values = flow_constraints(operators, velocity, tags, pressure_pin=pressure_pin, pressure_value=pressure_value)
    result = solve_dirichlet(operators.matrix, operators.rhs, dofs, values)
    solution = unpack_flow(operators, result.values, result.relative_residual)
    if solution.continuity_residual > 1e-10:
        raise RuntimeError("Velocity boundary data are incompatible with incompressibility.")
    return solution


def convection(operators, velocity, *, with_jacobian=True):
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape != (len(operators.space.points), 2) or not np.all(np.isfinite(velocity)):
        raise ValueError("Velocity must be a finite (n_P2_nodes,2) array.")
    mesh = operators.space.mesh
    local_velocity = velocity[operators.space.cells]
    tensor = operators.convection_tensor
    advected = np.einsum("eabcj,ebj->eac", tensor, local_velocity)
    load = advected @ local_velocity
    dofs = vector_dofs(operators.space.cells)
    residual = np.zeros(operators.n_velocity)
    np.add.at(residual, dofs.ravel(), load.ravel())
    if not with_jacobian:
        return residual, None
    local_jacobian = np.einsum("eab,ij->eaibj", advected, np.eye(2))
    advecting_derivative = tensor.transpose(0, 1, 2, 4, 3) @ local_velocity[:, None, None, :, :]
    local_jacobian += advecting_derivative.transpose(0, 1, 4, 2, 3)
    local_jacobian = local_jacobian.reshape(mesh.n_cells, 12, 12)
    jacobian = scatter_matrix(local_jacobian, dofs, dofs, (operators.n_velocity, operators.n_velocity))
    return residual, jacobian


def navier_stokes_residual(operators, values, *, time_coefficient=0.0, time_rhs=None, with_jacobian=True,
                           ale_context=None):
    values = np.asarray(values, dtype=float)
    if values.shape != (operators.n_total,) or not np.all(np.isfinite(values)):
        raise ValueError("Mixed flow state must be a finite vector of the correct size.")
    if ale_context is not None:
        if time_rhs is not None or (time_coefficient is not None and time_coefficient != 0.0):
            raise ValueError("ale_context cannot be combined with time_coefficient/time_rhs.")
        from fsi_ice.fluid.ale import ale_navier_stokes_residual

        return ale_navier_stokes_residual(operators, values, ale_context, with_jacobian=with_jacobian)
    convective, convective_jacobian = convection(operators, values[:operators.n_velocity].reshape(-1, 2), with_jacobian=with_jacobian)
    residual = operators.matrix @ values - operators.rhs
    residual[:operators.n_velocity] += convective
    momentum_jacobian = convective_jacobian
    if time_coefficient:
        if not np.isfinite(time_coefficient) or time_coefficient <= 0:
            raise ValueError("Time coefficient must be finite and positive.")
        history = np.asarray(time_rhs, dtype=float)
        if history.shape != (operators.n_velocity,) or not np.all(np.isfinite(history)):
            raise ValueError("Time history RHS must match velocity DOFs.")
        residual[:operators.n_velocity] += time_coefficient*(operators.mass @ values[:operators.n_velocity])-history
        if with_jacobian:
            momentum_jacobian = momentum_jacobian + time_coefficient*operators.mass
    if not with_jacobian:
        return residual, None
    correction = bmat([[momentum_jacobian, None], [None, csr_matrix((operators.space.mesh.n_nodes,)*2)]], format="csr")
    return residual, operators.matrix+correction


def solve_navier_stokes(operators, velocity, tags=None, *, pressure_pin=0, pressure_value=0.0,
                        initial=None, tolerance=1e-9, max_iterations=30, absolute_tolerance=None,
                        time_coefficient=0.0, time_rhs=None, linear_solver=None, ale_context=None):
    if not np.isfinite(tolerance) or tolerance <= 0 or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("Invalid nonlinear solve controls.")
    if absolute_tolerance is not None and (not np.isfinite(absolute_tolerance) or absolute_tolerance <= 0):
        raise ValueError("absolute_tolerance must be a finite positive real number when provided.")
    dofs, prescribed = flow_constraints(operators, velocity, tags, pressure_pin=pressure_pin, pressure_value=pressure_value)
    signature = None
    if ale_context is not None:
        if time_rhs is not None or (time_coefficient is not None and time_coefficient != 0.0):
            raise ValueError("ale_context cannot be combined with time_coefficient/time_rhs.")
        from fsi_ice.fluid.ale import ale_linear_signature, history_mass_load

        signature = ale_linear_signature(operators, ale_context, dofs, prescribed)
    if linear_solver is not None:
        if ale_context is None:
            linear_solver.prepare(time_coefficient)
        else:
            linear_solver.prepare(ale_context.stencil.coefficients[0], signature=signature)
    if initial is None:
        starting = solve_dirichlet(operators.matrix, operators.rhs, dofs, prescribed)
        values = starting.values.copy()
    else:
        values = np.array(initial, dtype=float, copy=True)
        if values.shape != (operators.n_total,) or not np.all(np.isfinite(values)):
            raise ValueError("Invalid initial flow state.")
    values[dofs] = prescribed
    free = np.setdiff1d(np.arange(operators.n_total), dofs)
    history = []
    if ale_context is None:
        scale = max(np.linalg.norm(operators.rhs), 1.0)
    else:
        scale = max(np.linalg.norm(operators.rhs), np.linalg.norm(history_mass_load(operators, ale_context)), 1.0)
    for iteration in range(max_iterations+1):
        residual, jacobian = navier_stokes_residual(operators, values, time_coefficient=time_coefficient,
                                                    time_rhs=time_rhs, ale_context=ale_context)
        norm = float(np.linalg.norm(residual[free])/scale)
        history.append(norm)
        # The relative criterion scales with the equation magnitudes; the optional
        # absolute bound keeps the raw free residual below a caller-given limit
        # (used by the coupled driver to meet its absolute acceptance gate).
        if norm <= tolerance and (absolute_tolerance is None or
                                  float(np.linalg.norm(residual[free])) <= absolute_tolerance):
            result = unpack_flow(operators, values, norm, history)
            if result.continuity_residual > max(1e-10, tolerance):
                raise RuntimeError("Flow solution violates the full continuity equation.")
            return result
        if iteration == max_iterations:
            break
        if linear_solver is None:
            correction = solve_dirichlet(jacobian, -residual, dofs, 0.0).values
        elif ale_context is None:
            correction = linear_solver.solve(jacobian, -residual, dofs)
        else:
            correction = linear_solver.solve(jacobian, -residual, dofs, signature=signature)
        step = 1.0
        for _ in range(16):
            candidate = values+step*correction
            candidate_residual, _ = navier_stokes_residual(operators, candidate, time_coefficient=time_coefficient,
                                                           time_rhs=time_rhs, with_jacobian=False,
                                                           ale_context=ale_context)
            if np.linalg.norm(candidate_residual[free])/scale < norm:
                values = candidate
                break
            step *= 0.5
        else:
            raise RuntimeError("Navier-Stokes Newton line search failed.")
    raise RuntimeError(f"Navier-Stokes did not converge in {max_iterations} iterations; residual={history[-1]:.3e}.")


def flow_error_norms(operators, solution, exact_velocity, exact_pressure):
    mesh = operators.space.mesh
    points = map_points(mesh, operators.quadrature)
    velocity = np.einsum("qa,eai->eqi", p2_values(operators.quadrature), solution.velocity[operators.space.cells])
    pressure = solution.pressure[mesh.cells] @ shape_values(operators.quadrature).T
    expected_velocity = vector_values(exact_velocity, points.reshape(-1, 2), "Exact velocity").reshape(points.shape)
    expected_pressure = np.asarray(exact_pressure(points.reshape(-1, 2)), dtype=float).reshape(pressure.shape)
    if not np.all(np.isfinite(expected_pressure)):
        raise ValueError("Exact pressure must be finite.")
    measure = mesh.determinants[:, None]*operators.weights
    return {"velocity_l2": float(np.sqrt(np.sum(measure*np.sum((velocity-expected_velocity)**2, axis=2)))),
            "pressure_l2": float(np.sqrt(np.sum(measure*(pressure-expected_pressure)**2)))}

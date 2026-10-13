"""Total Lagrangian P1 StVK solid, plane strain and reference-volume loads."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix

from fsi_ice.fem.p1 import physical_gradients
from fsi_ice.fem.poisson import solve_dirichlet
from fsi_ice.solid.linear import vector_dofs, vector_values


def assemble_mass(mesh, material):
    scalar = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12
    local = np.einsum("ab,ij->aibj", scalar, np.eye(2)).reshape(6, 6)
    local = (material.density*material.thickness*mesh.areas)[:, None, None]*local
    dofs = vector_dofs(mesh.cells)
    return coo_matrix((local.ravel(), (np.repeat(dofs, 6, axis=1).ravel(),
                                     np.tile(dofs, (1, 6)).ravel())),
                      shape=(2*mesh.n_nodes,)*2).tocsr()


def stvk_residual(mesh, material, displacement, external=None):
    if material.model != "plane_strain":
        raise ValueError("Finite-strain StVK currently requires plane_strain.")
    displacement = np.asarray(displacement, dtype=float)
    if displacement.shape != mesh.points.shape or not np.all(np.isfinite(displacement)):
        raise ValueError("Displacement must be a finite (n_nodes,2) array.")
    gradients = physical_gradients(mesh)
    deformation = np.eye(2)+np.einsum("eai,eaJ->eiJ", displacement[mesh.cells], gradients)
    determinants = np.linalg.det(deformation)
    if np.any(determinants <= 0) or not np.all(np.isfinite(determinants)):
        raise ValueError("Solid deformation has a non-positive Jacobian.")
    strain = 0.5*(np.einsum("ekI,ekJ->eIJ", deformation, deformation)-np.eye(2))
    lam, mu = material.lame
    second_piola = lam*np.trace(strain, axis1=1, axis2=2)[:, None, None]*np.eye(2)+2*mu*strain
    first_piola = deformation @ second_piola
    measure = mesh.areas*material.thickness
    local_residual = measure[:, None, None]*np.einsum("eaJ,eiJ->eai", gradients, first_piola)
    # dP_iJ/dF_kL: geometric stress term plus the derivative of F*S(E(F)).
    tangent = np.einsum("ik,eLJ->eiJkL", np.eye(2), second_piola)
    tangent += lam*np.einsum("eiJ,ekL->eiJkL", deformation, deformation)
    tangent += mu*np.einsum("eiL,ekJ->eiJkL", deformation, deformation)
    tangent += mu*np.einsum("JL,eik->eiJkL", np.eye(2), deformation @ deformation.transpose(0, 2, 1))
    local_matrix = np.einsum("eaJ,eiJkL,ebL->eaibk", gradients, tangent, gradients)
    local_matrix = (measure[:, None, None, None, None]*local_matrix).reshape(mesh.n_cells, 6, 6)
    dofs = vector_dofs(mesh.cells)
    residual = np.zeros(2*mesh.n_nodes)
    np.add.at(residual, dofs.ravel(), local_residual.ravel())
    if external is not None:
        external = np.asarray(external, dtype=float)
        if external.shape != residual.shape or not np.all(np.isfinite(external)):
            raise ValueError("External load must match solid DOFs and be finite.")
        residual -= external
    matrix = coo_matrix((local_matrix.ravel(), (np.repeat(dofs, 6, axis=1).ravel(),
                                             np.tile(dofs, (1, 6)).ravel())),
                        shape=(2*mesh.n_nodes,)*2).tocsr()
    return residual, matrix


@dataclass(frozen=True)
class NonlinearSolidSolution:
    displacement: np.ndarray
    reactions: np.ndarray
    nonlinear_history: tuple
    minimum_jacobian: float


def stvk_energy(mesh, material, displacement, external):
    deformation = np.eye(2)+np.einsum("eai,eaJ->eiJ", displacement[mesh.cells], physical_gradients(mesh))
    strain = 0.5*(deformation.transpose(0, 2, 1) @ deformation-np.eye(2))
    lam, mu = material.lame
    density = 0.5*lam*np.trace(strain, axis1=1, axis2=2)**2+mu*np.sum(strain**2, axis=(1, 2))
    return float(np.dot(mesh.areas*material.thickness, density)-np.dot(np.asarray(external), displacement.ravel()))


def solve_stvk(mesh, material, external, fixed_nodes, *, prescribed=(0.0, 0.0), initial=None,
               tolerance=1e-9, max_iterations=40, absolute_tolerance=None,
               mass_coefficient=0.0, mass_rhs=None):
    if not np.isfinite(tolerance) or tolerance <= 0 or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("Invalid nonlinear solid solve controls.")
    if absolute_tolerance is not None and (not np.isfinite(absolute_tolerance) or absolute_tolerance <= 0):
        raise ValueError("absolute_tolerance must be a finite positive real number when provided.")
    fixed_nodes = np.asarray(fixed_nodes)
    if fixed_nodes.ndim != 1 or not np.issubdtype(fixed_nodes.dtype, np.integer) or np.any(fixed_nodes < 0) or np.any(fixed_nodes >= mesh.n_nodes):
        raise ValueError("Invalid fixed solid nodes.")
    dofs = vector_dofs(fixed_nodes)
    values = np.zeros(mesh.points.shape) if initial is None else np.array(initial, dtype=float, copy=True)
    if values.shape != mesh.points.shape or not np.all(np.isfinite(values)):
        raise ValueError("Invalid solid initial guess.")
    values[fixed_nodes] = vector_values(prescribed, mesh.points[fixed_nodes], "Prescribed displacement")
    free = np.setdiff1d(np.arange(values.size), dofs)
    external = np.asarray(external, dtype=float)
    scale = max(np.linalg.norm(external), 1.0)
    mass = None
    if mass_coefficient:
        if not np.isfinite(mass_coefficient) or mass_coefficient < 0:
            raise ValueError("Invalid solid mass coefficient.")
        mass = assemble_mass(mesh, material)
        mass_rhs = np.asarray(mass_rhs, dtype=float)
        if mass_rhs.shape != (values.size,) or not np.all(np.isfinite(mass_rhs)):
            raise ValueError("Invalid solid mass history RHS.")

    def evaluate(state):
        residual, matrix = stvk_residual(mesh, material, state, external)
        if mass is not None:
            residual += mass_coefficient*(mass @ state.ravel())-mass_rhs
            matrix += mass_coefficient*mass
        return residual, matrix

    def potential(state):
        energy = stvk_energy(mesh, material, state, external)
        if mass is not None:
            energy += 0.5*mass_coefficient*np.dot(state.ravel(), mass @ state.ravel())-np.dot(mass_rhs, state.ravel())
        return energy

    history = []
    for iteration in range(max_iterations+1):
        residual, matrix = evaluate(values)
        norm = float(np.linalg.norm(residual[free])/scale)
        history.append(norm)
        # The relative criterion scales with the external load magnitude; the
        # optional absolute bound keeps the raw free residual below a caller-given
        # limit (used by the coupled driver to meet its absolute acceptance gate).
        if norm <= tolerance and (absolute_tolerance is None or
                                  float(np.linalg.norm(residual[free])) <= absolute_tolerance):
            deformation = np.eye(2)+np.einsum("eai,eaJ->eiJ", values[mesh.cells], physical_gradients(mesh))
            return NonlinearSolidSolution(values, residual.reshape(-1, 2), tuple(history),
                                          float(np.linalg.det(deformation).min()))
        if iteration == max_iterations:
            break
        correction = solve_dirichlet(matrix, -residual, dofs, 0.0).values.reshape(-1, 2)
        energy = potential(values)
        slope = float(np.dot(residual, correction.ravel()))
        step = 1.0
        for _ in range(20):
            candidate = values+step*correction
            try:
                candidate_residual, _ = evaluate(candidate)
            except ValueError as error:
                if "Jacobian" not in str(error):
                    raise
                step *= 0.5
                continue
            descent = slope < 0 and potential(candidate) <= energy+1e-4*step*slope
            if descent or np.linalg.norm(candidate_residual[free])/scale < norm:
                values = candidate
                break
            step *= 0.5
        else:
            raise RuntimeError("StVK Newton line search failed.")
    raise RuntimeError(f"StVK failed after {max_iterations} iterations; residual={history[-1]:.3e}.")


def solve_static_continuation(mesh, material, external, fixed_nodes, *, load_steps=10, **kwargs):
    if not isinstance(load_steps, int) or isinstance(load_steps, bool) or load_steps < 1:
        raise ValueError("load_steps must be a positive integer.")
    initial = None
    histories = []
    for step in range(1, load_steps+1):
        solution = solve_stvk(mesh, material, np.asarray(external)*step/load_steps,
                              fixed_nodes, initial=initial, **kwargs)
        initial = solution.displacement
        histories.append(solution.nonlinear_history)
    return solution, histories

"""ALE flow context: geometry-consistent BDF masses, swept transport and Jacobian blocks."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import bmat, csr_matrix

from fsi_ice.ale.geometry import ALEGeometry
from fsi_ice.ale.gcl import GCLTransport
from fsi_ice.ale.kinematics import MeshVelocity, evaluate_mesh_velocity
from fsi_ice.fem.p1 import map_points, shape_values
from fsi_ice.fem.p2 import p2_physical_gradients, p2_reference_gradients, p2_values
from fsi_ice.fem.quadrature import triangle_quadrature
from fsi_ice.fluid.taylor_hood import assemble_velocity_mass, convection, scatter_matrix
from fsi_ice.solid.linear import vector_dofs, vector_values
from fsi_ice.time import BDFStencil


@dataclass(frozen=True)
class ALEFlowContext:
    """Candidate ALE geometry, mesh velocity, swept transport and BDF template."""

    geometry: ALEGeometry
    mesh_velocity: MeshVelocity
    transport: GCLTransport
    stencil: BDFStencil
    history_velocities: tuple


@dataclass(frozen=True)
class _MeshDisplacementDerivative:
    """Fixed-history adapter so evaluate_mesh_velocity can build w_kin from the stencil."""

    coefficient: float
    term: np.ndarray

    def derivative(self, candidate):
        candidate = np.asarray(candidate, dtype=float)
        return self.coefficient*candidate + self.term


def _stencil_from_transport(transport):
    return BDFStencil(transport.order, transport.dt, transport.step_id, transport.coefficients, transport.times)


def build_ale_flow_context(operators, geometry, transport, history_velocities):
    """Freeze the ALE inputs for one candidate step without advancing any history."""
    if not isinstance(geometry, ALEGeometry):
        raise ValueError("geometry must be an ALEGeometry.")
    if not isinstance(transport, GCLTransport) or transport.candidate_geometry is not geometry:
        raise ValueError("transport must be the GCLTransport built for this candidate geometry.")
    if operators.space.mesh is not geometry.current_mesh:
        raise ValueError("operators must be assembled on geometry.current_mesh.")
    stencil = _stencil_from_transport(transport)
    histories = tuple(history_velocities)
    if len(histories) != stencil.order:
        raise ValueError(f"history_velocities must contain stencil.order={stencil.order} accepted P2 velocities.")
    frozen = []
    for velocity in histories:
        values = np.array(velocity, dtype=float, copy=True)
        if values.shape != (len(operators.space.points), 2) or not np.all(np.isfinite(values)):
            raise ValueError("Accepted velocities must be finite (n_P2, 2) arrays.")
        values.setflags(write=False)
        frozen.append(values)
    term = np.zeros(geometry.displacement.shape)
    for coefficient, accepted in zip(stencil.coefficients[1:], transport.accepted_geometries):
        term += coefficient*accepted.displacement
    mesh_velocity = evaluate_mesh_velocity(
        geometry, _MeshDisplacementDerivative(stencil.coefficients[0], term)
    )
    return ALEFlowContext(geometry, mesh_velocity, transport, stencil, tuple(frozen))


def validate_ale_velocity_context(ale_context, mesh_context):
    """Require the velocity BDF context and the ALE stencil/histories to describe one step."""
    stencil = ale_context.stencil
    if mesh_context.order != stencil.order:
        raise ValueError("Velocity BDF context and ALE stencil must share the same order.")
    if not np.isclose(mesh_context.time, stencil.times[0], rtol=0.0, atol=1e-14):
        raise ValueError("Velocity BDF context and ALE stencil must target the same time.")
    if not np.isclose(mesh_context.coefficient, stencil.coefficients[0], rtol=1e-13, atol=0.0):
        raise ValueError("Velocity BDF coefficient and ALE stencil candidate weight must agree.")
    combined = np.zeros(mesh_context.history_term.shape)
    for coefficient, velocity in zip(stencil.coefficients[1:], ale_context.history_velocities):
        combined += coefficient*velocity
    if not np.allclose(mesh_context.history_term, combined, rtol=1e-9, atol=1e-12):
        raise ValueError("Velocity BDF history term does not match the ALE history velocities.")


def history_mass_load(operators, ale_context):
    """sum_j a_j M_j u_j over accepted configurations (candidate handled separately)."""
    load = np.zeros(operators.n_velocity)
    coefficients = ale_context.stencil.coefficients
    for coefficient, geometry, velocity in zip(
        coefficients[1:], ale_context.transport.accepted_geometries, ale_context.history_velocities
    ):
        mass = assemble_velocity_mass(geometry.current_mesh, density=operators.density)
        load += coefficient*(mass @ velocity.ravel())
    return load


def mesh_transport_matrix(operators, transport):
    """K_mesh = -rho int_X N_a [Q_BDF.grad_X N_b + g_BDF N_b] dX on the reference domain."""
    mesh = operators.space.mesh
    if transport.candidate_geometry.current_mesh is not mesh:
        raise ValueError("The swept transport does not belong to the operators' current mesh.")
    reference_mesh = transport.reference_mesh
    points, weights = triangle_quadrature(5)
    shapes = p2_values(points)
    reference_gradients = p2_physical_gradients(reference_mesh, points)
    advected = np.einsum("eqj,eqbj->eqb", transport.flux, reference_gradients)
    kernel = advected + transport.divergence[:, None, None]*shapes
    local_scalar = -operators.density*np.einsum(
        "e,q,qa,eqb->eab", reference_mesh.determinants, weights, shapes, kernel
    )
    local = np.einsum("eab,ij->eaibj", local_scalar, np.eye(2)).reshape(mesh.n_cells, 12, 12)
    dofs = vector_dofs(operators.space.cells)
    return scatter_matrix(local, dofs, dofs, (operators.n_velocity, operators.n_velocity))


def ale_navier_stokes_residual(operators, values, ale_context, *, with_jacobian=True):
    """Full ALE residual and Jacobian; the fixed-grid branch of taylor_hood stays separate."""
    if ale_context.geometry.current_mesh is not operators.space.mesh:
        raise ValueError("ale_context geometry must be the operators' current mesh.")
    n_velocity = operators.n_velocity
    velocity = values[:n_velocity].reshape(-1, 2)
    residual = operators.matrix @ values - operators.rhs
    convective, convective_jacobian = convection(operators, velocity, with_jacobian=with_jacobian)
    residual[:n_velocity] += convective
    residual[:n_velocity] += history_mass_load(operators, ale_context)
    coefficient = float(ale_context.stencil.coefficients[0])
    residual[:n_velocity] += coefficient*(operators.mass @ values[:n_velocity])
    transport_matrix = mesh_transport_matrix(operators, ale_context.transport)
    residual[:n_velocity] += transport_matrix @ values[:n_velocity]
    if not with_jacobian:
        return residual, None
    momentum = convective_jacobian + coefficient*operators.mass + transport_matrix
    correction = bmat([[momentum, None], [None, csr_matrix((operators.space.mesh.n_nodes,)*2)]], format="csr")
    return residual, operators.matrix + correction


def ale_linear_signature(operators, ale_context, dofs, prescribed):
    """Geometry/constraint signature for ReusedLUSolver invalidation."""
    return (
        repr(ale_context.geometry.candidate_id),
        int(ale_context.stencil.step_id),
        float(operators.viscosity),
        float(operators.density),
        np.asarray(dofs, dtype=np.int64).tobytes(),
        np.asarray(prescribed, dtype=float).tobytes(),
    )


BODY_FORCE_DIFFERENCE_STEP = 1e-6  # spatial central-difference step for callable body forces


def _cofactor(matrices):
    """det(A) A^-1 for (n, 2, 2) matrices (linear in its argument)."""
    result = np.empty_like(matrices)
    result[:, 0, 0] = matrices[:, 1, 1]
    result[:, 0, 1] = -matrices[:, 0, 1]
    result[:, 1, 0] = -matrices[:, 1, 0]
    result[:, 1, 1] = matrices[:, 0, 0]
    return result


def _strain_basis(gradients):
    strains = np.zeros((gradients.shape[0], gradients.shape[1], 3, 12))
    strains[:, :, 0, 0::2] = gradients[:, :, :, 0]
    strains[:, :, 1, 1::2] = gradients[:, :, :, 1]
    strains[:, :, 2, 0::2] = gradients[:, :, :, 1]
    strains[:, :, 2, 1::2] = gradients[:, :, :, 0]
    return strains


def _divergence_basis(gradients):
    basis = np.empty((gradients.shape[0], gradients.shape[1], 12))
    basis[:, :, 0::2] = gradients[:, :, :, 0]
    basis[:, :, 1::2] = gradients[:, :, :, 1]
    return basis


def _transport_response_matrix(operators, reference_mesh, flux, divergence):
    """K(Q, g) with the same reference-domain formula as mesh_transport_matrix."""
    mesh = operators.space.mesh
    points, weights = triangle_quadrature(5)
    shapes = p2_values(points)
    reference_gradients = p2_physical_gradients(reference_mesh, points)
    advected = np.einsum("eqj,eqbj->eqb", flux, reference_gradients)
    kernel = advected + divergence[:, None, None]*shapes
    local_scalar = -operators.density*np.einsum(
        "e,q,qa,eqb->eab", reference_mesh.determinants, weights, shapes, kernel
    )
    local = np.einsum("eab,ij->eaibj", local_scalar, np.eye(2)).reshape(mesh.n_cells, 12, 12)
    dofs = vector_dofs(operators.space.cells)
    return scatter_matrix(local, dofs, dofs, (operators.n_velocity, operators.n_velocity))


def _transport_directional(geometry, transport, delta_jacobian, delta, points):
    """Nodal d(Q_BDF) and d(g_BDF) of the candidate; accepted configurations are fixed."""
    reference_mesh = geometry.reference_mesh
    inverse_reference = np.linalg.inv(reference_mesh.jacobians)
    delta_deformation = np.einsum("eij,ejk->eik", delta_jacobian, inverse_reference)
    delta_determinant = np.einsum(
        "eij,eji->e", _cofactor(geometry.deformation_gradients), delta_deformation
    )
    accepted = transport.accepted_geometries[0]
    midpoint = 0.5*(geometry.deformation_gradients + accepted.deformation_gradients)
    cofactor_midpoint = _cofactor(midpoint)
    delta_cofactor_midpoint = _cofactor(0.5*delta_deformation)
    difference = (geometry.displacement - accepted.displacement)[geometry.current_mesh.cells]
    delta_difference = delta[geometry.current_mesh.cells]
    nodal = transport.interval_weights[0]*(
        np.einsum("eij,enj->eni", delta_cofactor_midpoint, difference)
        + np.einsum("eij,enj->eni", cofactor_midpoint, delta_difference)
    )/transport.dt
    flux = np.einsum("qa,eai->eqi", shape_values(points), nodal)
    source = transport.coefficients[0]*delta_determinant
    return flux, source


def ale_geometry_jvp(operators, values, *, ale_context, delta_d_m):
    """Directional derivative of the uneliminated ALE residual w.r.t. the candidate mesh displacement.

    Every geometry block is differentiated analytically: current J_x/det J_x and its
    inverse, P2 gradients, strains/divergence/mass, the physical convection tensor,
    the swept cofactors Q_BDF and source g_BDF of the candidate interval, and the
    K_mesh response assembled on the reference domain.  Accepted geometries and all
    history terms stay fixed.  The spatial evaluation of a callable body force has no
    derivative protocol in this version, so that single block uses central differences
    of f(x +/- h delta_x) with h = BODY_FORCE_DIFFERENCE_STEP and is not an analytic
    tangent for arbitrary callables.  Boundary values do not enter the uneliminated
    residual.  The accompanying mesh-velocity relation is delta_w_kin = a_0 delta_d_m.
    """
    geometry = ale_context.geometry
    transport = ale_context.transport
    mesh = operators.space.mesh
    if transport.candidate_geometry is not geometry or geometry.current_mesh is not mesh:
        raise ValueError("ale_context must describe the operators' current candidate geometry.")
    values = np.asarray(values, dtype=float)
    if values.shape != (operators.n_total,) or not np.all(np.isfinite(values)):
        raise ValueError("Mixed flow state must be a finite vector of the correct size.")
    delta = np.asarray(delta_d_m, dtype=float)
    if delta.shape != geometry.displacement.shape or not np.all(np.isfinite(delta)):
        raise ValueError(
            f"delta_d_m must have shape {geometry.displacement.shape} and be finite, got {delta.shape}."
        )
    reference_mesh = geometry.reference_mesh
    cells = mesh.cells
    n_velocity = operators.n_velocity

    delta_vertices = delta[cells]
    delta_jacobian = np.stack((delta_vertices[:, 1]-delta_vertices[:, 0],
                               delta_vertices[:, 2]-delta_vertices[:, 0]), axis=2)
    inverse_current = np.linalg.inv(mesh.jacobians)
    delta_inverse = -np.einsum("eij,ejk,ekl->eil", inverse_current, delta_jacobian, inverse_current)
    delta_determinant = np.einsum("eij,eji->e", _cofactor(mesh.jacobians), delta_jacobian)

    points, weights = triangle_quadrature(5)
    velocity_shape = p2_values(points)
    pressure_shape = shape_values(points)
    reference_gradients = p2_reference_gradients(points)
    delta_gradients = np.einsum("qai,eij->eqaj", reference_gradients, delta_inverse)
    gradients = operators.gradients
    determinant = mesh.determinants
    density = operators.density

    constitutive = operators.viscosity*np.diag([2.0, 2.0, 1.0])
    strains = _strain_basis(gradients)
    delta_strains = _strain_basis(delta_gradients)
    raw_stiffness = np.einsum("q,eqai,ab,eqbj->eij", weights, strains, constitutive, strains)
    delta_raw_stiffness = (
        np.einsum("q,eqai,ab,eqbj->eij", weights, delta_strains, constitutive, strains)
        + np.einsum("q,eqai,ab,eqbj->eij", weights, strains, constitutive, delta_strains)
    )
    delta_stiffness = (determinant[:, None, None]*delta_raw_stiffness
                       + delta_determinant[:, None, None]*raw_stiffness)

    raw_divergence = np.einsum("q,qp,eqv->epv", weights, pressure_shape, _divergence_basis(gradients))
    delta_raw_divergence = np.einsum(
        "q,qp,eqv->epv", weights, pressure_shape, _divergence_basis(delta_gradients)
    )
    delta_divergence = (determinant[:, None, None]*delta_raw_divergence
                        + delta_determinant[:, None, None]*raw_divergence)

    physical_points = map_points(mesh, points)
    delta_points = delta[cells[:, 0]][:, None, :] + np.einsum("qj,eij->eqi", points, delta_jacobian)
    body_force = getattr(operators, "body_force", (0.0, 0.0))
    flat_points = physical_points.reshape(-1, 2)
    if callable(body_force):
        flat_direction = delta_points.reshape(-1, 2)
        force = vector_values(body_force, flat_points, "Body force").reshape(physical_points.shape)
        plus = vector_values(body_force, flat_points+BODY_FORCE_DIFFERENCE_STEP*flat_direction,
                             "Body force")
        minus = vector_values(body_force, flat_points-BODY_FORCE_DIFFERENCE_STEP*flat_direction,
                              "Body force")
        delta_force = ((plus-minus)/(2*BODY_FORCE_DIFFERENCE_STEP)).reshape(physical_points.shape)
    else:
        force = vector_values(body_force, flat_points, "Body force").reshape(physical_points.shape)
        delta_force = np.zeros_like(force)
    raw_load = np.einsum("q,eqi,qa->eai", weights, force, velocity_shape)
    delta_raw_load = np.einsum("q,eqi,qa->eai", weights, delta_force, velocity_shape)
    delta_load = (determinant[:, None, None]*delta_raw_load
                  + delta_determinant[:, None, None]*raw_load)

    dofs = vector_dofs(operators.space.cells)
    delta_stiffness_matrix = scatter_matrix(delta_stiffness, dofs, dofs, (n_velocity, n_velocity))
    delta_divergence_matrix = scatter_matrix(delta_divergence, cells, dofs, (mesh.n_nodes, n_velocity))
    delta_rhs = np.zeros(operators.n_total)
    np.add.at(delta_rhs, dofs.ravel(), delta_load.ravel())

    result = np.zeros(operators.n_total)
    velocity = values[:n_velocity]
    pressure = values[n_velocity:]
    result[:n_velocity] += delta_stiffness_matrix @ velocity
    result[:n_velocity] -= delta_divergence_matrix.T @ pressure
    result[n_velocity:] -= delta_divergence_matrix @ velocity
    result[:n_velocity] -= delta_rhs[:n_velocity]

    raw_tensor = np.einsum("q,qa,qb,eqcj->eabcj", weights, velocity_shape, velocity_shape, gradients)
    delta_raw_tensor = np.einsum(
        "q,qa,qb,eqcj->eabcj", weights, velocity_shape, velocity_shape, delta_gradients
    )
    delta_tensor = density*(determinant[:, None, None, None, None]*delta_raw_tensor
                            + delta_determinant[:, None, None, None, None]*raw_tensor)
    local_velocity = velocity.reshape(-1, 2)[operators.space.cells]
    delta_convective = np.einsum("eabcj,ebj->eac", delta_tensor, local_velocity) @ local_velocity
    np.add.at(result, dofs.ravel(), delta_convective.ravel())

    local_mass_scalar = np.einsum("q,qa,qb->ab", weights, velocity_shape, velocity_shape)
    raw_mass = np.einsum("ab,ij->aibj", local_mass_scalar, np.eye(2)).reshape(12, 12)
    delta_mass = (density*delta_determinant)[:, None, None]*raw_mass
    delta_mass_matrix = scatter_matrix(delta_mass, dofs, dofs, (n_velocity, n_velocity))
    result[:n_velocity] += float(ale_context.stencil.coefficients[0])*(delta_mass_matrix @ velocity)

    delta_flux, delta_source = _transport_directional(geometry, transport, delta_jacobian, delta, points)
    delta_transport = _transport_response_matrix(operators, reference_mesh, delta_flux, delta_source)
    result[:n_velocity] += delta_transport @ velocity
    return result

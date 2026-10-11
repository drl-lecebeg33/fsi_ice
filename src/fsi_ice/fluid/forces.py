"""Force exerted by the fluid on a tagged solid boundary, unit depth.

``boundary_cauchy_stress`` is the single Cauchy-stress kernel shared by the
fixed-mesh boundary monitors and the moving-interface traction assembly in
``fsi_ice.interface.traction``: it evaluates
``sigma = -p I + mu (grad u + grad u^T)`` at owner-cell local quadrature
points without averaging the fluid gradient across cells.
"""

import numpy as np

from fsi_ice.fem.p1 import shape_values
from fsi_ice.fem.p2 import p2_reference_gradients


def boundary_cauchy_stress(operators, solution, owner_cells, reference_points):
    """Cauchy stress of a flow state at boundary quadrature points.

    ``owner_cells`` are the fluid cells that own the boundary edges and
    ``reference_points`` are local coordinates in those cells, either shared
    ``(q, 2)`` or per-owner ``(e, q, 2)``; the result has shape
    ``(e, q, 2, 2)``.  Velocity is evaluated on the P2 nodes of each owner
    cell and pressure on its P1 vertices, so the stress uses the fluid trace
    value from one side only and never an inter-cell average.
    """
    mesh = operators.space.mesh
    cells = np.asarray(owner_cells)
    if cells.ndim != 1 or not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("owner_cells must be a one-dimensional integer array.")
    if not len(cells):
        raise ValueError("owner_cells must not be empty.")
    if np.any(cells < 0) or np.any(cells >= mesh.n_cells):
        raise ValueError("owner_cells contains a cell outside the mesh.")
    velocity = np.asarray(solution.velocity, dtype=float)
    if velocity.shape != (len(operators.space.points), 2) or not np.all(np.isfinite(velocity)):
        raise ValueError("Velocity must be a finite (n_P2_nodes, 2) array matching the operators.")
    pressure = np.asarray(solution.pressure, dtype=float)
    if pressure.shape != (mesh.n_nodes,) or not np.all(np.isfinite(pressure)):
        raise ValueError("Pressure must be a finite (n_P1_nodes,) array matching the operators.")
    points = np.asarray(reference_points, dtype=float)
    if points.ndim == 3:
        if points.shape[0] != len(cells) or points.shape[2] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("reference_points must be a finite (e, q, 2) array matching owner_cells.")
    elif points.ndim == 2:
        if points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("reference_points must be a finite (q, 2) array.")
    else:
        raise ValueError("reference_points must have shape (q, 2) or (e, q, 2).")
    inverse_jacobians = np.linalg.inv(mesh.jacobians)
    reference = points if points.ndim == 2 else points.reshape(-1, 2)
    shape = shape_values(reference)
    n_points = reference.shape[0] if points.ndim == 2 else points.shape[1]
    gradients = np.empty((len(cells), n_points, 6, 2))
    if points.ndim == 2:
        reference_gradients = p2_reference_gradients(reference)
        for k, cell in enumerate(cells):
            gradients[k] = reference_gradients @ inverse_jacobians[cell]
        shape = shape[None, :, :]
    else:
        for k, cell in enumerate(cells):
            gradients[k] = p2_reference_gradients(points[k]) @ inverse_jacobians[cell]
        shape = shape.reshape(len(cells), n_points, 3)
    gradient = np.einsum("eai,eqaj->eqij", velocity[operators.space.cells[cells]], gradients)
    values = np.einsum("eqa,ea->eq", shape, pressure[mesh.cells[cells]])
    return operators.viscosity*(gradient+gradient.transpose(0, 1, 3, 2))-values[:, :, None, None]*np.eye(2)


class BoundaryMonitor:
    """Precompute fixed-mesh boundary quadrature for transient monitoring.

    The monitor caches the owner cells, reference quadrature points, boundary
    measures and outward normals of one tagged mesh configuration; the stress
    itself is evaluated through :func:`boundary_cauchy_stress`.  Reusing a
    monitor for another candidate geometry or viscosity is rejected.
    """

    def __init__(self, operators, tag):
        from fsi_ice.fem.p2 import p2_values

        mesh = operators.space.mesh
        owners = {}
        for e, cell in enumerate(mesh.cells):
            for a, b in ((0, 1), (1, 2), (2, 0)):
                owners.setdefault(tuple(sorted((cell[a], cell[b]))), []).append((e, a, b))
        s, w = np.polynomial.legendre.leggauss(3)
        s, w = (s+1)/2, w/2
        reference_vertices = np.array([[0, 0], [1, 0], [0, 1]])
        owner_cells, reference_points, velocity_nodes, velocity_basis, measures, normals = [], [], [], [], [], []
        for edge in mesh.boundaries[tag]:
            entries = owners[tuple(sorted(edge))]
            if len(entries) != 1:
                raise ValueError("Boundary monitor requires exterior edges.")
            e, a, b = entries[0]
            reference = (1-s[:, None])*reference_vertices[a]+s[:, None]*reference_vertices[b]
            tangent = mesh.points[mesh.cells[e, b]]-mesh.points[mesh.cells[e, a]]
            length = np.linalg.norm(tangent)
            owner_cells.append(e)
            reference_points.append(reference)
            velocity_nodes.append(operators.space.cells[e])
            velocity_basis.append(p2_values(reference))
            measures.append(length*w)
            normals.append([tangent[1]/length, -tangent[0]/length])
        self.operators = operators
        self.viscosity = operators.viscosity
        self.geometry_signature = (mesh, float(operators.viscosity))
        self.owner_cells = np.array(owner_cells, dtype=np.int64)
        self.reference_points = np.array(reference_points)
        self.velocity_nodes = np.array(velocity_nodes)
        self.velocity_basis = np.array(velocity_basis)
        self.measures = np.array(measures)
        self.normals = np.array(normals)

    def require_current_geometry(self, operators):
        """Reject a stale monitor used for a different candidate mesh or viscosity."""
        if operators.space.mesh is not self.geometry_signature[0] or float(operators.viscosity) != self.geometry_signature[1]:
            raise ValueError("BoundaryMonitor was built for a different geometry or viscosity; "
                             "rebuild it for the current candidate.")

    def force(self, solution, *, operators=None):
        if operators is not None:
            self.require_current_geometry(operators)
        else:
            operators = self.operators
        stress = boundary_cauchy_stress(operators, solution, self.owner_cells, self.reference_points)
        return -np.einsum("eq,eqij,ej->i", self.measures, stress, self.normals)

    def flux(self, solution, *, operators=None):
        if operators is not None:
            self.require_current_geometry(operators)
        velocity = np.einsum("eqa,eai->eqi", self.velocity_basis, solution.velocity[self.velocity_nodes])
        return float(np.einsum("eq,eqi,ei", self.measures, velocity, self.normals))


def boundary_force(operators, solution, tag):
    mesh = operators.space.mesh
    owners = {}
    for cell_id, cell in enumerate(mesh.cells):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            owners.setdefault(tuple(sorted((cell[a], cell[b]))), []).append((cell_id, a, b))
    abscissae, weights = np.polynomial.legendre.leggauss(3)
    abscissae, weights = (abscissae+1)/2, weights/2
    vertices = np.array([[0, 0], [1, 0], [0, 1]])
    total = np.zeros(2)
    for edge in mesh.boundaries[tag]:
        entries = owners[tuple(sorted(edge))]
        if len(entries) != 1:
            raise ValueError("Force integration requires exterior edges.")
        cell_id, a, b = entries[0]
        reference = (1-abscissae[:, None])*vertices[a]+abscissae[:, None]*vertices[b]
        stress = boundary_cauchy_stress(operators, solution, np.array([cell_id]), reference[None, :, :])[0]
        tangent = mesh.points[mesh.cells[cell_id, b]]-mesh.points[mesh.cells[cell_id, a]]
        length = np.linalg.norm(tangent)
        outward_fluid = np.array([tangent[1], -tangent[0]])/length
        # sigma*n_fluid is the force on the fluid; its negative acts on the obstacle.
        total -= length*np.einsum("q,qij,j->i", weights, stress, outward_fluid)
    return total


def boundary_flux(operators, solution, tag):
    from fsi_ice.fem.p2 import p2_values

    mesh = operators.space.mesh
    owners = {tuple(sorted((cell[a], cell[b]))): (e, a, b)
              for e, cell in enumerate(mesh.cells) for a, b in ((0, 1), (1, 2), (2, 0))}
    abscissae, weights = np.polynomial.legendre.leggauss(3)
    abscissae, weights = (abscissae+1)/2, weights/2
    vertices = np.array([[0, 0], [1, 0], [0, 1]])
    total = 0.0
    for edge in mesh.boundaries[tag]:
        e, a, b = owners[tuple(sorted(edge))]
        reference = (1-abscissae[:, None])*vertices[a]+abscissae[:, None]*vertices[b]
        field = p2_values(reference) @ solution.velocity[operators.space.cells[e]]
        tangent = mesh.points[mesh.cells[e, b]]-mesh.points[mesh.cells[e, a]]
        total += float(np.einsum("q,qi,i", weights, field, [tangent[1], -tangent[0]]))
    return total

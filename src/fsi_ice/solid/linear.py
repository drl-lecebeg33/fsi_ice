"""P1 small-strain elasticity, with interleaved [ux0,uy0,ux1,uy1,...] DOFs."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix

from fsi_ice.fem.p1 import map_points, physical_gradients, shape_values
from fsi_ice.fem.poisson import solve_dirichlet
from fsi_ice.fem.quadrature import triangle_quadrature


@dataclass(frozen=True)
class ElasticMaterial:
    young: float
    poisson: float
    model: str = "plane_strain"
    thickness: float = 1.0
    density: float = 1.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in (self.young, self.poisson, self.thickness, self.density)):
            raise ValueError("Material parameters must be finite.")
        if self.young <= 0 or not -1 < self.poisson < 0.5 or self.thickness <= 0 or self.density <= 0:
            raise ValueError("Require E>0, -1<nu<0.5, thickness>0 and density>0.")
        if self.model not in {"plane_stress", "plane_strain"}:
            raise ValueError("Material model must be plane_stress or plane_strain.")

    @property
    def lame(self):
        mu = self.young / (2 * (1 + self.poisson))
        lam = self.young * self.poisson / ((1 + self.poisson) * (1 - 2*self.poisson))
        if self.model == "plane_stress":
            lam = 2 * mu * lam / (lam + 2 * mu)
        return lam, mu

    @property
    def constitutive(self):
        lam, mu = self.lame
        # Voigt strain = [epsilon_xx, epsilon_yy, 2 epsilon_xy].
        return np.array([[lam + 2*mu, lam, 0], [lam, lam + 2*mu, 0], [0, 0, mu]])


def vector_values(field, points, name):
    values = np.asarray(field(points) if callable(field) else field, dtype=float)
    if values.shape == (2,):
        values = np.tile(values, (len(points), 1))
    if values.shape != (len(points), 2) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must provide a finite (n,2) array or length-two constant.")
    return values


def vector_dofs(nodes):
    nodes = np.asarray(nodes)
    if not np.issubdtype(nodes.dtype, np.integer):
        raise ValueError("Node indices must be integers.")
    return (2*nodes[..., None] + np.array([0, 1])).reshape(*nodes.shape[:-1], -1)


def strain_matrices(mesh):
    gradients = physical_gradients(mesh)
    matrices = np.zeros((mesh.n_cells, 3, 6))
    matrices[:, 0, 0::2] = gradients[:, :, 0]
    matrices[:, 1, 1::2] = gradients[:, :, 1]
    matrices[:, 2, 0::2] = gradients[:, :, 1]
    matrices[:, 2, 1::2] = gradients[:, :, 0]
    return matrices


def assemble_elasticity(mesh, material: ElasticMaterial, body_force=(0.0, 0.0), tractions=None):
    b_matrix = strain_matrices(mesh)
    local = np.einsum("eai,ab,ebj->eij", b_matrix, material.constitutive, b_matrix)
    local *= (mesh.areas * material.thickness)[:, None, None]
    dofs = vector_dofs(mesh.cells)
    rows = np.repeat(dofs, 6, axis=1).ravel()
    columns = np.tile(dofs, (1, 6)).ravel()
    matrix = coo_matrix((local.ravel(), (rows, columns)),
                        shape=(2*mesh.n_nodes, 2*mesh.n_nodes)).tocsr()
    reference, weights = triangle_quadrature(5)
    points = map_points(mesh, reference)
    force = vector_values(body_force, points.reshape(-1, 2), "Body force").reshape(mesh.n_cells, -1, 2)
    local_rhs = np.einsum("q,eqi,qa->eai", weights, force, shape_values(reference))
    local_rhs *= (mesh.determinants * material.thickness)[:, None, None]
    rhs = np.zeros(2*mesh.n_nodes)
    np.add.at(rhs, dofs.ravel(), local_rhs.ravel())
    # Traction is force per current boundary area; multiply by the unit-depth thickness.
    abscissae = np.array([0.5 - 0.5/np.sqrt(3), 0.5 + 0.5/np.sqrt(3)])
    edge_shapes = np.column_stack((1-abscissae, abscissae))
    for tag, traction in (tractions or {}).items():
        if tag not in mesh.boundaries:
            raise ValueError(f"Unknown traction boundary: {tag}")
        edges = mesh.boundaries[tag]
        vertices = mesh.points[edges]
        points = np.einsum("qa,eai->eqi", edge_shapes, vertices)
        values = vector_values(traction, points.reshape(-1, 2), "Traction").reshape(len(edges), 2, 2)
        lengths = np.linalg.norm(vertices[:, 1]-vertices[:, 0], axis=1)
        load = 0.5 * np.einsum("eqi,qa->eai", values, edge_shapes)
        load *= (lengths * material.thickness)[:, None, None]
        np.add.at(rhs, vector_dofs(edges).ravel(), load.ravel())
    return matrix, rhs


@dataclass(frozen=True)
class ElasticSolution:
    displacement: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    reactions: np.ndarray
    relative_residual: float


def solve_elasticity(mesh, material, prescribed_nodes, displacement=(0.0, 0.0),
                     *, body_force=(0.0, 0.0), tractions=None):
    nodes = np.asarray(prescribed_nodes)
    if nodes.ndim != 1 or not np.issubdtype(nodes.dtype, np.integer) or not len(nodes):
        raise ValueError("Prescribed nodes must be a nonempty integer vector.")
    if np.any(nodes < 0) or np.any(nodes >= mesh.n_nodes):
        raise ValueError("Prescribed node lies outside the mesh.")
    matrix, rhs = assemble_elasticity(mesh, material, body_force, tractions)
    values = vector_values(displacement, mesh.points[nodes], "Displacement")
    solution = solve_dirichlet(matrix, rhs, vector_dofs(nodes), values.ravel())
    strain = np.einsum("eai,ei->ea", strain_matrices(mesh), solution.values[vector_dofs(mesh.cells)])
    stress = strain @ material.constitutive.T
    return ElasticSolution(solution.values.reshape(-1, 2), strain, stress,
                           (matrix @ solution.values - rhs).reshape(-1, 2), solution.relative_residual)


def linear_residual(mesh, material, displacement, *, body_force=(0.0, 0.0), tractions=None):
    values = np.asarray(displacement, dtype=float)
    if values.shape != (mesh.n_nodes, 2) or not np.all(np.isfinite(values)):
        raise ValueError("Displacement must be a finite (n_nodes,2) array.")
    matrix, rhs = assemble_elasticity(mesh, material, body_force, tractions)
    return matrix @ values.ravel() - rhs, matrix

import numpy as np
import pytest

from fsi_ice.mesh import TriMesh, rectangle_mesh


def test_rectangle_geometry_tags_and_ownership():
    mesh = rectangle_mesh(3, 2, width=2, height=3)
    assert mesh.n_nodes == 12 and mesh.n_cells == 12
    np.testing.assert_allclose(mesh.areas.sum(), 6.0, atol=1e-14)
    assert np.all(mesh.determinants > 0)
    assert np.all((mesh.quality > 0) & (mesh.quality <= 1 + 1e-14))
    assert len(mesh.boundary_nodes()) == 10
    np.testing.assert_allclose(mesh.points[mesh.boundary_nodes("left"), 0], 0)
    np.testing.assert_allclose(mesh.points[mesh.boundary_nodes("top"), 1], 3)
    assert sum(len(edges) for edges in mesh.boundaries.values()) == 10
    with pytest.raises(ValueError):
        mesh.points[0, 0] = 9
    with pytest.raises(TypeError):
        mesh.boundaries["new"] = np.array([[0, 1]])
    with pytest.raises(ValueError, match="Unknown boundary"):
        mesh.boundary_nodes("missing")


@pytest.mark.parametrize("points,cells,message", [
    ([[0, 0], [1, 0], [0, 1]], [[0, 2, 1]], "Negative orientation"),
    ([[0, 0], [1, 0], [2, 0]], [[0, 1, 2]], "Degenerate"),
    ([[0, 0], [1, 0], [0, np.nan]], [[0, 1, 2]], "finite"),
    ([[0, 0], [1, 0], [0, 1]], [[0, 1, 3]], "outside"),
    ([[0, 0], [1, 0], [0, 1]], [[0.0, 1.0, 2.0]], "integers"),
    ([[0, 0], [1, 0], [0, 1]], [[0, 1, 1]], "repeated"),
    ([[0, 0], [1, 0], [0, 1]], [[0, 1, 2], [0, 1, 2]], "Duplicate"),
    ([[0, 0], [1, 0], [0, 1], [4, 4]], [[0, 1, 2]], "Unused"),
])
def test_invalid_mesh_is_rejected(points, cells, message):
    with pytest.raises(ValueError, match=message):
        TriMesh(points, cells)


def test_boundary_cannot_label_an_interior_edge():
    mesh = rectangle_mesh(1)
    with pytest.raises(ValueError, match="non-exterior"):
        TriMesh(mesh.points, mesh.cells, {"wrong": np.array([[0, 3]])})


def test_small_valid_cells_and_geometry_overflow():
    mesh = TriMesh(np.array([[0, 0], [1, 0], [0, 1]]) * 1e-12, [[0, 1, 2]])
    np.testing.assert_allclose(mesh.areas, [5e-25], rtol=1e-14, atol=0)
    with pytest.raises(ValueError, match="overflows"):
        TriMesh([[0, 0], [1e200, 0], [0, 1e200]], [[0, 1, 2]])


def test_nonmanifold_edge_is_rejected():
    with pytest.raises(ValueError, match="Non-manifold"):
        TriMesh([[0, 0], [1, 0], [0.5, 1], [0.5, -1], [0.2, 2]],
                [[0, 1, 2], [1, 0, 3], [0, 1, 4]])


@pytest.mark.parametrize("nx,ny,width", [(0, 1, 1), (1.5, 2, 1), (True, 2, 1), (2, 0, 1), (2, 2, -1)])
def test_invalid_rectangle_parameters(nx, ny, width):
    with pytest.raises(ValueError):
        rectangle_mesh(nx, ny, width=width)

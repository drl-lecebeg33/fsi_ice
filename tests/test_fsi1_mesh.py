from pathlib import Path
import shutil

import numpy as np
import pytest

from fsi_ice.fem.p2 import P2Space
from fsi_ice.interface import build_matching_interface
from fsi_ice.mesh import fsi1_geometry, fsi1_mesh, read_fsi_gmsh2


def test_fsi1_geometry_rejects_invalid_size_and_contains_frozen_tags():
    for value in (0, -1, np.nan, np.inf, True, "0.1"):
        with pytest.raises(ValueError):
            fsi1_geometry(value)
    text = fsi1_geometry(.05)
    for name in ("fluid", "solid", "walls", "outlet", "inlet",
                 "fluid-interface", "solid-interface", "cylinder-rigid-wall", "solid-clamp"):
        assert f'("{name}"' in text


def test_read_fsi_gmsh2_rejects_missing_domain_boundary(tmp_path):
    path = tmp_path / "incomplete.msh"
    path.write_text('''$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
1 1 "walls"
2 2 "fluid"
$EndPhysicalNames
$Nodes
3
1 0 0 0
2 1 0 0
3 0 1 0
$EndNodes
$Elements
2
1 1 2 1 1 1 2
2 2 2 2 1 1 2 3
$EndElements
''', encoding="utf-8")
    with pytest.raises(ValueError, match="solid triangles"):
        read_fsi_gmsh2(path)


@pytest.mark.skipif(shutil.which("gmsh") is None, reason="Gmsh executable not installed")
def test_generated_fsi1_pair_has_shared_global_trace_and_matching_p2_map(tmp_path):
    pair = fsi1_mesh(.04, tmp_path)
    assert set(pair.fluid.boundaries) == {
        "walls", "outlet", "inlet", "cylinder-rigid-wall", "fluid-interface"
    }
    assert set(pair.solid.boundaries) == {"solid-clamp", "solid-interface"}
    assert pair.fluid.quality.min() > .1 and pair.solid.quality.min() > .1

    space = P2Space.from_mesh(pair.fluid)
    topology = build_matching_interface(pair.solid, pair.fluid, space,
                                        solid_tag="solid-interface", fluid_tag="fluid-interface",
                                        atol=1e-12, rtol=0)
    solid_ids = pair.solid_global_nodes[topology.solid_nodes]
    fluid_ids = pair.fluid_global_nodes[topology.fluid_vertex_nodes]
    np.testing.assert_array_equal(solid_ids, fluid_ids)
    np.testing.assert_array_equal(topology.X_s_gamma, topology.X_f_gamma[:len(solid_ids)])
    np.testing.assert_allclose(topology.h @ np.ones(len(solid_ids)), 1, atol=1e-14)
    np.testing.assert_allclose(topology.N_s, -topology.N_f, atol=1e-14)
    assert len(np.intersect1d(pair.solid.boundary_nodes("solid-clamp"), topology.solid_nodes)) == 2
    assert np.min(np.linalg.norm(pair.solid.points-[.6, .2], axis=1)) < 1e-14

    reread = read_fsi_gmsh2(Path(tmp_path) / "fsi1.msh")
    np.testing.assert_array_equal(reread.fluid.points, pair.fluid.points)
    np.testing.assert_array_equal(reread.solid.cells, pair.solid.cells)

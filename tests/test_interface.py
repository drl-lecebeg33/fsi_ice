"""T16 interface tests: topology, motion/load mapping, current geometry and real traction.

The T16a/b cases use explicitly *synthetic integrated dual nodal forces* for
algebraic checks.  The T16c cases below integrate the real fluid Cauchy stress
trace through the shared ``fluid.forces.boundary_cauchy_stress`` kernel, check
the Nanson current/reference conversion, the thickness and the dynamic geometry
caches.
"""

import dataclasses
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from fsi_ice.fem.p2 import P2Space
from fsi_ice.fluid.forces import BoundaryMonitor, boundary_force
from fsi_ice.fluid.taylor_hood import FlowSolution, assemble_stokes, solve_stokes
from fsi_ice.interface import (InterfaceGeometry, InterfaceLoad, InterfaceLoadDerivative, InterfaceMotion,
                               InterfaceTopology, assemble_interface_load, build_matching_interface,
                               evaluate_interface_geometry, interface_diagnostics, interface_load_jvp,
                               map_load, map_motion, pullback_traction, scatter_solid_load)
from fsi_ice.mesh import TriMesh, rectangle_mesh
from fsi_ice.time.bdf import BDFHistory


def two_domain(nx=4, ny_fluid=3, ny_solid=2, width=1.0):
    """Fluid [0,width]x[0,1] above solid [0,width]x[-1,0] sharing y=0 in nx edges."""
    fluid_flat = rectangle_mesh(nx, ny_fluid, width=width, height=1.0)
    solid_flat = rectangle_mesh(nx, ny_solid, width=width, height=1.0)
    solid = TriMesh(solid_flat.points - np.array([0.0, 1.0]), solid_flat.cells,
                    _tags(solid_flat, "top"))
    fluid = TriMesh(fluid_flat.points, fluid_flat.cells, _tags(fluid_flat, "bottom"))
    return solid, fluid


def _tags(mesh, interface_tag):
    tags = {name: mesh.boundaries[name] for name in ("bottom", "top", "left", "right")}
    tags["interface"] = tags.pop(interface_tag)
    return tags


def build_interface(solid, fluid, *, atol=1e-12, rtol=1e-12):
    return build_matching_interface(solid, fluid, P2Space.from_mesh(fluid), solid_tag="interface",
                                    fluid_tag="interface", atol=atol, rtol=rtol)


def renumber(mesh, rng):
    """Return a copy with globally permuted node ids and the new->old map."""
    order = rng.permutation(mesh.n_nodes)
    inverse = np.empty(mesh.n_nodes, dtype=np.int64)
    inverse[order] = np.arange(mesh.n_nodes)
    boundaries = {tag: inverse[edges] for tag, edges in mesh.boundaries.items()}
    return TriMesh(mesh.points[order], inverse[mesh.cells], boundaries), order


def p2_node_map(space_new, space_old, new_to_old):
    """Map fluid P2 node ids of a renumbered copy back to the original ids."""
    n_mesh = len(space_new.mesh.points)
    mapping = {node: int(new_to_old[node]) for node in range(n_mesh)}
    for edge, node in space_new.edge_nodes.items():
        old_edge = tuple(sorted(int(new_to_old[index]) for index in edge))
        mapping[int(node)] = int(space_old.edge_nodes[old_edge])
    return mapping


def stub_mesh(points, cells, boundaries):
    """Minimal mesh-like input for defects that a validated TriMesh rejects at construction."""
    return SimpleNamespace(points=np.array(points, dtype=float), cells=np.array(cells, dtype=np.int64),
                           n_nodes=len(points),
                           boundaries=MappingProxyType({tag: np.asarray(edges)
                                                        for tag, edges in boundaries.items()}))


def test_matching_topology_direction_and_renumbering():
    solid, fluid = two_domain(nx=4)
    topology = build_interface(solid, fluid)
    assert isinstance(topology, InterfaceTopology)
    assert topology.n_solid_interface_nodes == 5
    assert topology.n_fluid_interface_nodes == 9
    assert topology.n_edges == 4
    assert topology.n_solid_nodes == solid.n_nodes
    assert topology.n_fluid_nodes == len(topology.fluid_space.points)
    h = topology.h.toarray()
    np.testing.assert_array_equal(h.sum(axis=1), np.ones(9))
    np.testing.assert_array_equal(topology.H.toarray(), np.kron(h, np.eye(2)))
    np.testing.assert_array_equal(topology.h @ topology.X_s_gamma, topology.X_f_gamma)
    np.testing.assert_array_equal(topology.N_s + topology.N_f, 0)
    np.testing.assert_array_equal(topology.N_s, np.tile([0.0, 1.0], (4, 1)))
    np.testing.assert_array_equal(topology.reference_lengths, np.full(4, 0.25))
    np.testing.assert_array_equal(topology.solid_dofs,
                                  (2*topology.solid_nodes[:, None] + [0, 1]).ravel())
    pairs = sorted(tuple(sorted(pair.solid_edge)) for pair in topology.edge_pairs)
    tagged = sorted(tuple(sorted(edge)) for edge in solid.boundaries["interface"])
    assert pairs == tagged
    assert len(set(int(node) for node in topology.fluid_p2_nodes)) == 9
    assert len({int(pair.fluid_midpoint) for pair in topology.edge_pairs}) == 4
    expected_vertices = {int(node) for node in np.unique(fluid.boundaries["interface"])}
    assert {int(node) for node in topology.fluid_vertex_nodes} == expected_vertices
    with pytest.raises(ValueError):
        topology.solid_nodes[0] = -1
    with pytest.raises(ValueError):
        topology.h.data[0] = 2.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        topology.solid_tag = "other"
    again = build_interface(solid, fluid)
    assert again.topology_id == topology.topology_id
    np.testing.assert_array_equal(again.h.toarray(), h)
    reversed_solid = TriMesh(solid.points, solid.cells,
                             {tag: edges[:, ::-1] for tag, edges in solid.boundaries.items()})
    reversed_fluid = TriMesh(fluid.points, fluid.cells,
                             {tag: edges[:, ::-1] for tag, edges in fluid.boundaries.items()})
    flipped = build_interface(reversed_solid, reversed_fluid)
    assert flipped.topology_id == topology.topology_id
    np.testing.assert_array_equal(flipped.h.toarray(), h)

    rng = np.random.default_rng(2026)
    solid_p, inverse_s = renumber(solid, rng)
    fluid_p, inverse_f = renumber(fluid, rng)
    topology_p = build_matching_interface(solid_p, fluid_p, P2Space.from_mesh(fluid_p),
                                          solid_tag="interface", fluid_tag="interface",
                                          atol=1e-12, rtol=1e-12)
    old_solid_index = {int(node): i for i, node in enumerate(topology.solid_nodes)}
    old_fluid_index = {int(node): i for i, node in enumerate(topology.fluid_p2_nodes)}
    p2_old = p2_node_map(P2Space.from_mesh(fluid_p), P2Space.from_mesh(fluid), inverse_f)
    expected = np.zeros_like(topology_p.h.toarray())
    rows = [old_fluid_index[p2_old[int(node)]] for node in topology_p.fluid_p2_nodes]
    columns = [old_solid_index[int(inverse_s[node])] for node in topology_p.solid_nodes]
    for row, old_row in enumerate(rows):
        for column, old_column in enumerate(columns):
            expected[row, column] = h[old_row, old_column]
    np.testing.assert_array_equal(topology_p.h.toarray(), expected)
    np.testing.assert_array_equal(topology_p.H.toarray(), np.kron(expected, np.eye(2)))

    displacement = rng.normal(size=(solid.n_nodes, 2))
    motion = map_motion(topology, displacement)
    motion_p = map_motion(topology_p, displacement[inverse_s])
    for row, node in enumerate(topology_p.solid_nodes):
        np.testing.assert_array_equal(motion_p.vertex_values[row],
                                      motion.vertex_values[old_solid_index[int(inverse_s[node])]])
    for row, node in enumerate(topology_p.fluid_p2_nodes):
        np.testing.assert_array_equal(motion_p.p2_values[row],
                                      motion.p2_values[old_fluid_index[p2_old[int(node)]]])


def test_coincident_slit_interface_normals_and_ambiguity():
    points = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.0], [0.5, 0.5], [1.0, 0.0]])
    cells = np.array([[0, 1, 3], [2, 4, 3]])
    mesh = TriMesh(points, cells, {"lower": np.array([[1, 3]]), "upper": np.array([[2, 3]])})
    topology = build_matching_interface(mesh, mesh, P2Space.from_mesh(mesh), solid_tag="lower",
                                        fluid_tag="upper", atol=1e-12, rtol=1e-12)
    assert topology.n_edges == 1 and topology.n_solid_interface_nodes == 2
    pair = topology.edge_pairs[0]
    np.testing.assert_array_equal(np.array(pair.N_s) + np.array(pair.N_f), 0)
    np.testing.assert_allclose(pair.N_s, [1.0, 0.0], atol=0)
    assert pair.length == 0.5
    both = TriMesh(points, cells, {"both": np.array([[1, 3], [2, 3]])})
    with pytest.raises(ValueError, match="Ambiguous match"):
        build_matching_interface(both, both, P2Space.from_mesh(both), solid_tag="both",
                                 fluid_tag="both", atol=1e-12, rtol=1e-12)


def test_motion_translation_affine_and_rotation():
    solid, fluid = two_domain(nx=3)
    topology = build_interface(solid, fluid)
    n_solid_gamma = topology.n_solid_interface_nodes
    n_fluid_gamma = topology.n_fluid_interface_nodes
    translation = np.array([0.31, -0.72])
    motion = map_motion(topology, np.tile(translation, (solid.n_nodes, 1)))
    np.testing.assert_allclose(motion.vertex_values, np.tile(translation, (n_solid_gamma, 1)),
                               atol=0, rtol=0)
    np.testing.assert_allclose(motion.p2_values, np.tile(translation, (n_fluid_gamma, 1)),
                               atol=0, rtol=0)
    assert isinstance(motion, InterfaceMotion)

    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(2, 2))
    offset = rng.normal(size=2)
    field = solid.points @ matrix.T + offset
    motion = map_motion(topology, field)
    np.testing.assert_allclose(motion.vertex_values, topology.X_s_gamma @ matrix.T + offset,
                               atol=1e-15, rtol=0)
    np.testing.assert_allclose(motion.p2_values, topology.X_f_gamma @ matrix.T + offset,
                               atol=1e-15, rtol=0)

    angle = 0.37
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    motion = map_motion(topology, solid.points @ (rotation - np.eye(2)).T)
    x_s = topology.X_s_gamma + motion.vertex_values
    x_f = topology.X_f_gamma + motion.p2_values
    np.testing.assert_allclose(x_f, topology.X_f_gamma @ rotation.T, atol=1e-13, rtol=0)
    np.testing.assert_allclose(x_f, topology.h @ x_s, atol=1e-13, rtol=0)
    node_row = {int(node): row for row, node in enumerate(topology.solid_nodes)}
    for offset_index, pair in enumerate(topology.edge_pairs):
        midpoint = motion.p2_values[n_solid_gamma + offset_index]
        average = 0.5*(motion.p2_values[node_row[pair.solid_edge[0]]]
                       + motion.p2_values[node_row[pair.solid_edge[1]]])
        np.testing.assert_array_equal(midpoint, average)

    frozen = field.copy()
    frozen.setflags(write=False)
    map_motion(topology, frozen)
    np.testing.assert_array_equal(frozen, field)
    with pytest.raises(ValueError, match="shape"):
        map_motion(topology, np.zeros((solid.n_nodes, 3)))
    with pytest.raises(ValueError, match="finite"):
        map_motion(topology, np.full((solid.n_nodes, 2), np.nan))
    with pytest.raises(ValueError):
        motion.p2_values[0, 0] = 1.0

    rng2 = np.random.default_rng(4)
    solid_p, inverse_s = renumber(solid, rng2)
    fluid_p, _ = renumber(fluid, rng2)
    topology_p = build_matching_interface(solid_p, fluid_p, P2Space.from_mesh(fluid_p),
                                          solid_tag="interface", fluid_tag="interface",
                                          atol=1e-12, rtol=1e-12)
    motion_p = map_motion(topology_p, field[inverse_s])
    np.testing.assert_allclose(motion_p.p2_values, topology_p.X_f_gamma @ matrix.T + offset,
                               atol=1e-15, rtol=0)


def test_bdf_derivative_commutes_with_H():
    solid, fluid = two_domain(nx=3)
    topology = build_interface(solid, fluid)
    H = topology.H
    rng = np.random.default_rng(11)
    n = 2*topology.n_solid_interface_nodes
    dt = 0.1
    d0, d1, candidate = rng.normal(size=(3, n))

    def compare(history, fluid_history):
        context = history.context()
        fluid_context = fluid_history.context()
        assert context.order == fluid_context.order
        derivative = context.derivative(candidate)
        np.testing.assert_allclose(fluid_context.derivative(H @ candidate), H @ derivative, atol=1e-14)
        snapshot = history.snapshot()
        for _ in range(2):
            context.derivative(candidate + 1.0)
        after = history.snapshot()
        assert after[0] == snapshot[0] and after[1] == snapshot[1]
        for state, previous in zip(after[2], snapshot[2]):
            np.testing.assert_array_equal(state, previous)
        return context, fluid_context

    history1 = BDFHistory(d0, dt, order=1)
    fluid1 = BDFHistory(H @ d0, dt, order=1)
    context1, fluid_context1 = compare(history1, fluid1)
    assert context1.order == 1
    history1.accept(context1, d1)
    fluid1.accept(fluid_context1, H @ d1)
    np.testing.assert_array_equal(fluid1.states[0], H @ history1.states[0])
    compare(history1, fluid1)

    history2 = BDFHistory(d0, dt, order=2)
    fluid2 = BDFHistory(H @ d0, dt, order=2)
    context2, fluid_context2 = compare(history2, fluid2)
    assert context2.order == 1
    history2.accept(context2, d1)
    fluid2.accept(fluid_context2, H @ d1)
    np.testing.assert_array_equal(fluid2.states[0], H @ history2.states[0])
    mature, fluid_mature = compare(history2, fluid2)
    assert mature.order == 2 and fluid_mature.order == 2

    derivative = mature.derivative(candidate)
    exact = interface_diagnostics(topology, motion_residual=(derivative.copy(), derivative))
    assert exact["pass"] and exact["motion_residual"]["value"] == 0.0
    perturbed = derivative + 1e-3*rng.normal(size=n)
    mismatch = interface_diagnostics(topology, motion_residual=(perturbed, derivative))
    assert not mismatch["pass"]
    assert mismatch["motion_residual"]["value"] > mismatch["motion_residual"]["tolerance"]


@pytest.mark.parametrize("thickness", [1.0, 0.2, 2.0])
def test_constant_traction_synthetic_nodal_force(thickness):
    length = 0.7
    fluid_flat = rectangle_mesh(1, 2, width=length, height=0.5)
    solid_flat = rectangle_mesh(1, 1, width=length, height=0.5)
    solid = TriMesh(solid_flat.points - np.array([0.0, 0.5]), solid_flat.cells, _tags(solid_flat, "top"))
    fluid = TriMesh(fluid_flat.points, fluid_flat.cells, _tags(fluid_flat, "bottom"))
    topology = build_interface(solid, fluid)
    assert topology.n_edges == 1 and topology.n_solid_interface_nodes == 2
    traction = np.array([0.25, -0.6])
    weights = np.full(topology.n_fluid_interface_nodes, 1.0/6.0)
    midpoints = np.flatnonzero(topology.fluid_p2_nodes >= fluid.n_nodes)
    assert len(midpoints) == 1
    weights[midpoints] = 2.0/3.0
    loads = np.zeros(2*topology.n_fluid_interface_nodes)
    for node in range(topology.n_fluid_interface_nodes):
        loads[2*node:2*node+2] = thickness*length*traction*weights[node]
    # Synthetic integrated dual nodal force, not a real fluid traction integration.
    compact = map_load(topology, loads)
    np.testing.assert_allclose(compact.reshape(-1, 2),
                               np.tile(0.5*thickness*length*traction, (2, 1)), atol=1e-15, rtol=0)
    unit = map_load(topology, loads/thickness)
    np.testing.assert_allclose(compact, thickness*unit, atol=1e-15, rtol=0)
    full = scatter_solid_load(topology, compact)
    assert full.shape == (2*solid.n_nodes,)
    np.testing.assert_array_equal(full[topology.solid_dofs], compact)
    np.testing.assert_allclose(full.reshape(-1, 2).sum(axis=0), thickness*length*traction,
                               atol=1e-15, rtol=0)
    origin = np.array([0.13, -0.21])
    diagnostics = interface_diagnostics(topology, loads={"compact": compact, "fluid": loads},
                                        analytic_force=thickness*length*traction, moment_origin=origin)
    assert diagnostics["pass"], diagnostics["failures"]
    relative = topology.X_s_gamma.mean(axis=0) - origin
    expected_moment = thickness*length*(relative[0]*traction[1] - relative[1]*traction[0])
    np.testing.assert_allclose(diagnostics["total_moment"]["solid"], expected_moment, atol=1e-15, rtol=0)
    np.testing.assert_allclose(diagnostics["total_moment"]["fluid"], expected_moment, atol=1e-15, rtol=0)
    doubled = interface_diagnostics(topology, loads={"compact": map_load(topology, 2*loads),
                                                     "fluid": 2*loads},
                                    analytic_force=thickness*length*traction)
    assert not doubled["pass"]
    assert doubled["total_force"]["analytic_error"] > doubled["total_force"]["tolerance"]


def test_virtual_work_force_and_moment_translation():
    solid, fluid = two_domain(nx=4)
    topology = build_interface(solid, fluid)
    rng = np.random.default_rng(20261011)
    motion = map_motion(topology, rng.normal(scale=0.05, size=(solid.n_nodes, 2)))
    loads_fluid = rng.normal(size=2*topology.n_fluid_interface_nodes)
    compact = map_load(topology, loads_fluid)
    delta = rng.normal(size=2*topology.n_solid_interface_nodes)
    diagnostics = interface_diagnostics(topology, motion=motion, loads={"compact": compact,
                                                                       "fluid": loads_fluid},
                                        virtual_solid=delta, virtual_fluid=topology.H @ delta,
                                        moment_origin=(0.0, 0.0),
                                        scales={"length": 1.0, "force": 1.0, "moment": 1.0, "work": 1.0})
    assert diagnostics["pass"], diagnostics["failures"]
    assert diagnostics["configuration"] == "current"
    work = diagnostics["virtual_work"]
    assert work["error"] <= work["tolerance"]
    assert work["mapping_error"] == 0.0
    np.testing.assert_allclose(work["solid"], work["fluid"], rtol=1e-13)
    delta_fluid = topology.H @ delta
    np.testing.assert_allclose(float(delta_fluid @ loads_fluid), float(delta @ compact), rtol=1e-13)

    shift = np.array([0.4, -0.25])
    shifted = interface_diagnostics(topology, motion=motion, loads={"compact": compact, "fluid": loads_fluid},
                                    moment_origin=shift)
    force = diagnostics["total_force"]["compact"]
    translation = shift[0]*force[1] - shift[1]*force[0]
    for key in ("solid", "full", "fluid"):
        np.testing.assert_allclose(shifted["total_moment"][key],
                                   diagnostics["total_moment"][key] - translation,
                                   atol=1e-12, rtol=1e-13)
    reference = interface_diagnostics(topology, loads={"compact": compact, "fluid": loads_fluid})
    assert reference["configuration"] == "reference" and reference["pass"]

    vertex_load = np.zeros_like(loads_fluid)
    vertex_load[0:2] = [3.0, -4.0]
    compact_vertex = map_load(topology, vertex_load)
    np.testing.assert_array_equal(compact_vertex[:2], [3.0, -4.0])
    np.testing.assert_array_equal(compact_vertex[2:], 0.0)
    clamped = interface_diagnostics(topology, motion=motion,
                                    loads={"compact": compact_vertex, "fluid": vertex_load},
                                    moment_origin=(0.2, 0.3))
    assert clamped["pass"], clamped["failures"]
    np.testing.assert_allclose(clamped["total_force"]["compact"], [3.0, -4.0], atol=0, rtol=0)
    x0 = topology.X_s_gamma[0] + motion.vertex_values[0]
    expected = (x0[0] - 0.2)*(-4.0) - (x0[1] - 0.3)*3.0
    np.testing.assert_allclose(clamped["total_moment"]["solid"], expected, atol=1e-13, rtol=0)

    bad_mapping = interface_diagnostics(topology, loads={"compact": compact, "fluid": loads_fluid},
                                        virtual_solid=delta, virtual_fluid=delta_fluid + 1e-3)
    assert not bad_mapping["pass"]
    assert bad_mapping["virtual_work"]["mapping_error"] > 0.0


def test_zero_tiny_loads_and_diagnostic_failures():
    solid, fluid = two_domain(nx=2)
    topology = build_interface(solid, fluid)
    zero = np.zeros(2*topology.n_fluid_interface_nodes)
    compact_zero = map_load(topology, zero)
    np.testing.assert_array_equal(compact_zero, 0.0)
    diagnostics = interface_diagnostics(topology, loads={"compact": compact_zero, "fluid": zero},
                                        virtual_solid=np.zeros(2*topology.n_solid_interface_nodes))
    assert diagnostics["pass"], diagnostics["failures"]
    assert diagnostics["total_force"]["scale_fallback"] is True
    assert diagnostics["total_force"]["full_error"] == 0.0
    assert diagnostics["virtual_work"]["scale_fallback"] is True
    tiny = 1e-20*np.random.default_rng(5).normal(size=2*topology.n_fluid_interface_nodes)
    compact_tiny = map_load(topology, tiny)
    small = interface_diagnostics(topology, loads={"compact": compact_tiny, "fluid": tiny})
    assert small["pass"], small["failures"]
    broken = interface_diagnostics(topology, loads={"compact": compact_tiny + 1e-6, "fluid": tiny})
    assert not broken["pass"]
    assert broken["total_force"]["fluid_error"] > broken["total_force"]["tolerance"]
    assert "total_force" in broken["failures"][0]
    overflow = interface_diagnostics(topology, loads=np.full(2*topology.n_solid_interface_nodes, 1e308))
    assert not overflow["pass"] and overflow["total_force"]["passed"] is False

    normals_ok = interface_diagnostics(topology, current_normals=(topology.N_s, topology.N_f))
    assert normals_ok["pass"] and normals_ok["normals"]["current_error"] == 0.0
    normals_bad = interface_diagnostics(topology, current_normals=(topology.N_s, topology.N_s))
    assert not normals_bad["pass"]
    assert normals_bad["normals"]["current_error"] > normals_bad["normals"]["tolerance"]


def test_rejects_unknown_tags_empty_and_mismatched_segments():
    solid, fluid = two_domain(nx=4)
    space = P2Space.from_mesh(fluid)
    with pytest.raises(ValueError, match="Unknown solid boundary tag"):
        build_matching_interface(solid, fluid, space, solid_tag="wet", fluid_tag="interface",
                                 atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError, match="Unknown fluid boundary tag"):
        build_matching_interface(solid, fluid, space, solid_tag="interface", fluid_tag="wet",
                                 atol=1e-12, rtol=1e-12)
    for bad in (np.nan, np.inf, -1.0, 0.0):
        with pytest.raises(ValueError, match="atol"):
            build_matching_interface(solid, fluid, space, solid_tag="interface", fluid_tag="interface",
                                     atol=bad, rtol=1e-12)
    with pytest.raises(ValueError, match="rtol"):
        build_matching_interface(solid, fluid, space, solid_tag="interface", fluid_tag="interface",
                                 atol=1e-12, rtol=np.nan)
    empty = TriMesh(solid.points, solid.cells,
                    {"interface": np.empty((0, 2), dtype=np.int64),
                     "bottom": solid.boundaries["bottom"], "left": solid.boundaries["left"],
                     "right": solid.boundaries["right"]})
    with pytest.raises(ValueError, match="Empty interface"):
        build_matching_interface(empty, fluid, space, solid_tag="interface", fluid_tag="interface",
                                 atol=1e-12, rtol=1e-12)
    solid_coarse, fluid_coarse = two_domain(nx=2)
    space_coarse = P2Space.from_mesh(fluid_coarse)
    with pytest.raises(ValueError, match="not matched by any solid interface node"):
        build_matching_interface(solid_coarse, fluid, space, solid_tag="interface",
                                 fluid_tag="interface", atol=1e-12, rtol=1e-12)
    solid_fine, _ = two_domain(nx=8)
    with pytest.raises(ValueError, match="no fluid candidate"):
        build_matching_interface(solid_fine, fluid, space, solid_tag="interface",
                                 fluid_tag="interface", atol=1e-12, rtol=1e-12)
    shifted = TriMesh(solid.points + np.array([1e-3, 0.0]), solid.cells, dict(solid.boundaries))
    with pytest.raises(ValueError, match="no fluid candidate"):
        build_matching_interface(shifted, fluid, space, solid_tag="interface",
                                 fluid_tag="interface", atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError, match="fluid_space"):
        build_matching_interface(solid, fluid, space_coarse, solid_tag="interface",
                                 fluid_tag="interface", atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError, match="tagged more than once"):
        TriMesh(fluid.points, fluid.cells, {"interface": np.array([[0, 1], [0, 1]])})


def test_rejects_mesh_level_defects_via_duck_typed_meshes():
    # TriMesh rejects non-exterior, repeated, zero-length and non-finite tagged
    # edges at construction; the builder keeps its own guards so an equivalent
    # foreign mesh-like input cannot slip through silently.
    _, fluid = two_domain(nx=2)
    space = P2Space.from_mesh(fluid)

    def build(candidate_solid, tag="interface"):
        return build_matching_interface(candidate_solid, fluid, space, solid_tag=tag,
                                        fluid_tag="interface", atol=1e-12, rtol=1e-12)

    interior = stub_mesh([[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 1, 2], [0, 2, 3]],
                         {"interface": [[0, 2]]})
    with pytest.raises(ValueError, match="2 owner cells"):
        build(interior)
    absent = stub_mesh([[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 1, 2], [0, 2, 3]],
                       {"interface": [[1, 3]]})
    with pytest.raises(ValueError, match="exterior"):
        build(absent)
    zero = stub_mesh([[0, 0], [0, 0], [1, 1]], [[0, 1, 2]], {"interface": [[0, 1]]})
    with pytest.raises(ValueError, match="Zero-length"):
        build(zero)
    non_finite = stub_mesh([[0, 0], [np.nan, 0], [1, 1]], [[0, 1, 2]], {"interface": [[0, 1]]})
    with pytest.raises(ValueError, match="finite"):
        build(non_finite)
    repeated = stub_mesh([[0, 0], [1, 1], [0, 1]], [[0, 1, 2]], {"interface": [[0, 1], [1, 0]]})
    with pytest.raises(ValueError, match="Duplicate"):
        build(repeated)
    outside = stub_mesh([[0, 0], [1, 1], [0, 1]], [[0, 1, 2]], {"interface": [[0, 3]]})
    with pytest.raises(ValueError, match="outside the mesh"):
        build(outside)
    with pytest.raises(ValueError, match="Unknown solid boundary tag"):
        build(outside, tag="wet")
    floats = stub_mesh([[0, 0], [1, 1], [0, 1]], [[0, 1, 2]],
                       {"interface": np.array([[0.0, 1.0]])})
    with pytest.raises(ValueError, match="integer"):
        build(floats)


def test_shape_scale_context_and_nan_rejections():
    solid, fluid = two_domain(nx=3)
    topology = build_interface(solid, fluid)
    n_solid = 2*topology.n_solid_interface_nodes
    n_fluid = 2*topology.n_fluid_interface_nodes
    with pytest.raises(ValueError, match="shape"):
        map_motion(topology, np.zeros((solid.n_nodes,)))
    with pytest.raises(ValueError, match="shape"):
        map_load(topology, np.zeros(n_fluid + 1))
    with pytest.raises(ValueError, match="shape"):
        scatter_solid_load(topology, np.zeros(n_solid + 2))
    with pytest.raises(ValueError, match="finite"):
        map_motion(topology, np.full((solid.n_nodes, 2), np.inf))
    with pytest.raises(ValueError, match="finite"):
        map_load(topology, np.full(n_fluid, np.nan))
    with pytest.raises(ValueError, match="finite"):
        scatter_solid_load(topology, np.full(n_solid, np.nan))
    with pytest.raises(TypeError):
        map_motion(solid, np.zeros((solid.n_nodes, 2)))
    with pytest.raises(ValueError, match="Unknown scale keys"):
        interface_diagnostics(topology, scales={"density": 1.0})
    with pytest.raises(ValueError, match="Scale 'force'"):
        interface_diagnostics(topology, scales={"force": np.nan})
    with pytest.raises(ValueError, match="Unknown loads keys"):
        interface_diagnostics(topology, loads={"wet": np.zeros(n_solid)})
    with pytest.raises(ValueError, match="moment_origin"):
        interface_diagnostics(topology, moment_origin=(0.0,))
    with pytest.raises(ValueError, match="finite"):
        interface_diagnostics(topology, loads=np.array([np.nan]*n_solid))
    with pytest.raises(ValueError, match="virtual_solid"):
        interface_diagnostics(topology, virtual_solid=np.zeros(n_solid + 1))
    with pytest.raises(ValueError, match="motion_residual"):
        interface_diagnostics(topology, motion_residual=np.zeros(n_solid))
    with pytest.raises(ValueError, match="current_normals"):
        interface_diagnostics(topology, current_normals=(np.zeros((topology.n_edges, 2)),
                                                         np.zeros((topology.n_edges + 1, 2))))
    other = build_interface(*two_domain(nx=5))
    with pytest.raises(ValueError, match="different topology"):
        interface_diagnostics(topology, motion=map_motion(other, np.zeros((other.n_solid_nodes, 2))))


# ---------------------------------------------------------------------------
# T16c: real fluid traction, Nanson conversion and dynamic geometry caches
# ---------------------------------------------------------------------------

MU_STAR = 0.7
GAMMA_STAR = 0.35
P_STAR = 0.4
ANGLE = 0.4
ROTATION = np.array([[np.cos(ANGLE), -np.sin(ANGLE)], [np.sin(ANGLE), np.cos(ANGLE)]])
DEFORMATIONS = {
    "identity": (np.eye(2), np.zeros(2)),
    "rotation": (ROTATION, np.array([0.3, -0.2])),
    "stretch": (np.diag([2.0, 0.5]), np.zeros(2)),
    "shear": (np.array([[1.2, 0.6], [0.2, 1.1]]), np.zeros(2)),
}


def scaled_error(actual, expected, explicit=1.0):
    """Frozen ``1e-12 * S`` rule with ``S = max(explicit, sum of contributions)``."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    error = float(np.max(np.abs(actual-expected))) if actual.size else 0.0
    scale = max(float(explicit), float(np.sum(np.abs(expected))), float(np.sum(np.abs(actual))))
    return error, 1e-12*scale


def assert_scaled(actual, expected, explicit=1.0, note=""):
    error, tolerance = scaled_error(actual, expected, explicit)
    assert error <= tolerance, f"{note}: error {error:.3e} > tolerance {tolerance:.3e}"


def oriented_domains(orientation, n=4):
    """Solid/fluid pair sharing one straight interface in four orientations.

    Returns ``(solid, fluid, n_s)`` where ``n_s`` is the constant outward solid
    normal of the interface; both meshes tag it ``"interface"`` and every
    interface edge has length ``1/n``.
    """
    rect = rectangle_mesh(n, n, width=1.0, height=1.0)

    def tagged(mesh, dx, dy, tag):
        boundaries = dict(mesh.boundaries)
        boundaries["interface"] = boundaries.pop(tag)
        return TriMesh(mesh.points + np.array([dx, dy]), mesh.cells, boundaries)

    if orientation == "up":
        return tagged(rect, 0.0, -1.0, "top"), tagged(rect, 0.0, 0.0, "bottom"), np.array([0.0, 1.0])
    if orientation == "down":
        return tagged(rect, 0.0, 0.0, "bottom"), tagged(rect, 0.0, -1.0, "top"), np.array([0.0, -1.0])
    if orientation == "right":
        return tagged(rect, -1.0, 0.0, "right"), tagged(rect, 0.0, 0.0, "left"), np.array([1.0, 0.0])
    if orientation == "left":
        return tagged(rect, 0.0, 0.0, "left"), tagged(rect, -1.0, 0.0, "right"), np.array([-1.0, 0.0])
    raise ValueError(orientation)


def analytic_state(operators, *, pressure=0.0, shear=0.0):
    """Analytic linear state u=(shear*y,0), p=pressure on the current P2/P1 nodes."""
    points = operators.space.points
    velocity = np.column_stack((shear*points[:, 1], np.zeros(len(points))))
    return FlowSolution(velocity, np.full(operators.space.mesh.n_nodes, pressure), 0.0, 0.0)


def analytic_stress(pressure, shear, viscosity):
    return np.array([[-pressure, viscosity*shear], [viscosity*shear, -pressure]])


def evaluate_candidate(topology, solid, fluid, matrix, center=(0.0, 0.0), offset=(0.0, 0.0),
                       candidate_id="candidate"):
    """Affine candidate x=(X-center)@matrix.T+center+offset on both sides."""
    center = np.asarray(center, dtype=float)
    offset = np.asarray(offset, dtype=float)
    current = TriMesh((fluid.points-center) @ matrix.T+center+offset, fluid.cells, fluid.boundaries)
    displacement = (solid.points-center) @ matrix.T+center+offset-solid.points
    geometry = evaluate_interface_geometry(topology, current, P2Space.from_mesh(current), displacement,
                                           candidate_id=candidate_id)
    operators = assemble_stokes(current, viscosity=MU_STAR, density=1.0)
    return current, displacement, geometry, operators


@pytest.mark.parametrize("orientation", ["up", "down", "left", "right"])
def test_16_03_pressure_sign_four_orientations(orientation):
    solid, fluid, n_s = oriented_domains(orientation, n=4)
    topology = build_interface(solid, fluid)
    np.testing.assert_allclose(topology.N_s, np.tile(n_s, (topology.n_edges, 1)), atol=1e-15)
    _, _, geometry, operators = evaluate_candidate(topology, solid, fluid, np.eye(2))
    solution = analytic_state(operators, pressure=P_STAR)
    load = assemble_interface_load(topology, geometry, operators, solution, thickness=1.0)
    expected_t = -P_STAR*n_s
    assert_scaled(load.t_s, np.broadcast_to(expected_t, load.t_s.shape), note=f"{orientation} traction")
    total_length = float(np.sum(geometry.lengths))
    for name, value in (("solid", load.total_force), ("fluid", load.total_force_fluid),
                        ("quadrature", load.total_force_quadrature)):
        assert_scaled(value, total_length*expected_t, note=f"{orientation} {name} force")
    # p>0 pushes into the solid interior, and the unit-depth monitor agrees.
    assert float(load.total_force @ n_s) < 0.0
    monitor_force = boundary_force(operators, solution, "interface")
    assert_scaled(load.total_force, monitor_force, note=f"{orientation} boundary_force")
    for thickness in (0.2, 2.0):
        thick = assemble_interface_load(topology, geometry, operators, solution, thickness=thickness)
        assert_scaled(thick.total_force/thickness, monitor_force, note=f"{orientation} thickness {thickness}")
        assert_scaled(thick.g_f_gamma, thickness*load.g_f_gamma, note=f"{orientation} linear thickness")


def test_16_03_constant_traction_node_weights_and_reversed_labels():
    solid, fluid, n_s = oriented_domains("up", n=1)
    topology = build_interface(solid, fluid)
    assert topology.n_edges == 1 and topology.n_solid_interface_nodes == 2
    assert topology.n_fluid_interface_nodes == 3
    _, _, geometry, operators = evaluate_candidate(topology, solid, fluid, np.eye(2))
    solution = analytic_state(operators, pressure=P_STAR)
    traction = -P_STAR*n_s
    length = float(geometry.lengths[0])
    weights = np.array([1.0/6.0, 1.0/6.0, 2.0/3.0])
    for thickness in (0.2, 1.0, 2.0):
        load = assemble_interface_load(topology, geometry, operators, solution, thickness=thickness)
        expected_g = thickness*length*np.outer(weights, traction).reshape(-1)
        assert_scaled(load.g_f_gamma, expected_g, note=f"P2 weights b={thickness}")
        assert_scaled(load.f_s_gamma, np.tile(thickness*length*traction/2.0, 2), note=f"P1 weights b={thickness}")
        assert_scaled(load.total_force, thickness*length*traction, note=f"total b={thickness}")
    unit = assemble_interface_load(topology, geometry, operators, solution, thickness=1.0)

    reversed_solid = TriMesh(solid.points, solid.cells,
                             {tag: edges[:, ::-1] for tag, edges in solid.boundaries.items()})
    reversed_fluid = TriMesh(fluid.points, fluid.cells,
                             {tag: edges[:, ::-1] for tag, edges in fluid.boundaries.items()})
    topology_reversed = build_interface(reversed_solid, reversed_fluid)
    assert topology_reversed.topology_id == topology.topology_id
    _, _, geometry_reversed, operators_reversed = evaluate_candidate(topology_reversed, reversed_solid,
                                                                     reversed_fluid, np.eye(2))
    load_reversed = assemble_interface_load(topology_reversed, geometry_reversed, operators_reversed,
                                            analytic_state(operators_reversed, pressure=P_STAR), thickness=1.0)
    assert_scaled(load_reversed.g_f_gamma, unit.g_f_gamma, note="reversed labels g")
    assert_scaled(load_reversed.total_force, unit.total_force, note="reversed labels force")
    assert float(load_reversed.total_force @ n_s) < 0.0


def test_16_03_linear_shear_stress_and_monitor_sign():
    solid, fluid, n_s = oriented_domains("up", n=3)
    topology = build_interface(solid, fluid)
    _, _, geometry, operators = evaluate_candidate(topology, solid, fluid, np.eye(2))
    solution = analytic_state(operators, pressure=P_STAR, shear=GAMMA_STAR)
    sigma = analytic_stress(P_STAR, GAMMA_STAR, MU_STAR)
    expected_t = np.einsum("ij,ej->ei", sigma, geometry.n_s)
    load = assemble_interface_load(topology, geometry, operators, solution, thickness=1.0)
    assert_scaled(load.t_s, np.broadcast_to(expected_t[:, None, :], load.t_s.shape), note="shear traction")
    expected_total = np.sum(geometry.lengths[:, None]*expected_t, axis=0)
    assert_scaled(load.total_force, expected_total, note="shear total force")
    assert_scaled(load.total_force, boundary_force(operators, solution, "interface"), note="shear monitor")
    # F.n_s is the pressure part and is negative (pushing into the solid)
    normal_component = float(np.sum(load.total_force @ n_s))
    assert abs(normal_component+float(np.sum(geometry.lengths))*P_STAR) <= 1e-12


def test_16_03_real_stokes_solution_traction():
    solid, fluid, n_s = oriented_domains("up", n=3)
    topology = build_interface(solid, fluid)
    operators = assemble_stokes(fluid, viscosity=MU_STAR, density=1.0)

    def exact_velocity(points):
        return np.column_stack((GAMMA_STAR*points[:, 1], np.zeros(len(points))))

    solution = solve_stokes(operators, exact_velocity, None, pressure_pin=0, pressure_value=P_STAR)
    assert solution.relative_residual <= 1e-12
    assert solution.continuity_residual <= 1e-12
    assert_scaled(solution.velocity, exact_velocity(operators.space.points), note="solve velocity")
    assert_scaled(solution.pressure, np.full(fluid.n_nodes, P_STAR), note="solve pressure")
    geometry = evaluate_interface_geometry(topology, fluid, P2Space.from_mesh(fluid),
                                           np.zeros_like(solid.points), candidate_id="real-solve")
    load = assemble_interface_load(topology, geometry, operators, solution, thickness=1.0)
    expected_t = analytic_stress(P_STAR, GAMMA_STAR, MU_STAR) @ n_s
    assert_scaled(load.t_s, np.broadcast_to(expected_t, load.t_s.shape), note="real solve traction")
    expected_total = float(np.sum(geometry.lengths))*expected_t
    assert_scaled(load.total_force, expected_total, note="real solve total force")
    assert_scaled(load.total_force, boundary_force(operators, solution, "interface"), note="real solve monitor")


@pytest.mark.parametrize("name", list(DEFORMATIONS))
@pytest.mark.parametrize("thickness", [0.2, 1.0, 2.0])
def test_16_05_nanson_reference_current_and_thickness(name, thickness):
    matrix, center = DEFORMATIONS[name]
    solid, fluid, n_s = oriented_domains("up", n=3)
    topology = build_interface(solid, fluid)
    _, _, geometry, operators = evaluate_candidate(topology, solid, fluid, matrix, center, candidate_id=name)
    solution = analytic_state(operators, pressure=P_STAR, shear=GAMMA_STAR)
    expected_t = np.einsum("ij,ej->ei", analytic_stress(P_STAR, GAMMA_STAR, MU_STAR), geometry.n_s)
    jacobian = float(np.linalg.det(matrix))
    expected_length_ratio = float(np.linalg.norm(matrix @ np.array([1.0, 0.0])))
    expected_nanson = jacobian*float(np.linalg.norm(np.linalg.solve(matrix.T, np.array([0.0, 1.0]))))
    assert abs(expected_length_ratio-expected_nanson) <= 1e-12*max(1.0, expected_length_ratio)
    assert_scaled(geometry.J_s, np.full(geometry.n_edges, jacobian), note=f"{name} J_s")
    assert_scaled(geometry.j_Gamma_current, np.full(geometry.n_edges, expected_length_ratio),
                  note=f"{name} length ratio")
    assert_scaled(geometry.j_Gamma_nanson, np.full(geometry.n_edges, expected_nanson), note=f"{name} nanson")
    if name == "stretch":  # J = 1 with length ratio 2 excludes using the volume Jacobian
        assert abs(float(geometry.J_s[0])-1.0) <= 1e-14
        assert abs(float(geometry.j_Gamma_current[0])-2.0) <= 1e-13

    load = assemble_interface_load(topology, geometry, operators, solution, thickness=thickness)
    reference = assemble_interface_load(topology, geometry, operators, solution, thickness=thickness,
                                        path="reference")
    assert_scaled(load.t_s, np.broadcast_to(expected_t[:, None, :], load.t_s.shape), note=f"{name} traction")
    assert_scaled(load.T_s0, geometry.j_Gamma_nanson[:, None, None]*load.t_s, note=f"{name} T_s0")
    assert_scaled(reference.g_f_gamma, load.g_f_gamma, note=f"{name} current/reference g")
    assert_scaled(reference.f_s_gamma, load.f_s_gamma, note=f"{name} current/reference f")
    assert_scaled(reference.total_force, load.total_force, note=f"{name} current/reference total force")
    assert_scaled(reference.total_moment, load.total_moment, explicit=1.0, note=f"{name} current/reference moment")
    assert_scaled(reference.total_moment_quadrature, load.total_moment_quadrature, explicit=1.0,
                  note=f"{name} current/reference quadrature moment")
    expected_total = thickness*np.sum(geometry.lengths[:, None]*expected_t, axis=0)
    for value, label in ((load.total_force, "solid"), (load.total_force_fluid, "fluid"),
                         (load.total_force_quadrature, "quadrature")):
        assert_scaled(value, expected_total, note=f"{name} {label} total")
    rng = np.random.default_rng(20261011)
    delta = rng.normal(size=2*topology.n_solid_interface_nodes)
    for path_load in (load, reference):
        work_solid = float(delta @ path_load.f_s_gamma)
        work_fluid = float((topology.H @ delta) @ path_load.g_f_gamma)
        error, tolerance = scaled_error(work_fluid, work_solid)
        assert error <= tolerance, (name, thickness, path_load.path, error, tolerance)
    unit = assemble_interface_load(topology, geometry, operators, solution, thickness=1.0)
    assert_scaled(load.g_f_gamma, thickness*unit.g_f_gamma, note=f"{name} thickness once (g)")
    assert_scaled(load.f_s_gamma, thickness*unit.f_s_gamma, note=f"{name} thickness once (f)")
    assert_scaled(load.t_s, unit.t_s, note=f"{name} thickness independent traction")


@pytest.mark.parametrize("name", list(DEFORMATIONS))
def test_16_geom_normals_and_surface_ratio(name):
    matrix, center = DEFORMATIONS[name]
    solid, fluid, n_s = oriented_domains("up", n=4)
    topology = build_interface(solid, fluid)
    _, _, geometry, _ = evaluate_candidate(topology, solid, fluid, matrix, center, candidate_id=name)
    assert geometry.normal_error <= 1e-12
    assert geometry.normalization_error <= 1e-12
    assert geometry.nanson_normal_error <= 1e-12
    assert geometry.gap_error <= geometry.gap_tolerance
    assert geometry.minimum_jacobian > 0.0 and geometry.minimum_cell_determinant > 0.0
    np.testing.assert_allclose(geometry.n_s+geometry.n_f, 0.0, atol=1e-15)
    error, tolerance = scaled_error(geometry.j_Gamma_current, geometry.j_Gamma_nanson)
    assert error <= tolerance, (name, error, tolerance)
    if name in ("identity", "rotation"):
        assert_scaled(geometry.j_Gamma_current, np.ones(geometry.n_edges), note=f"{name} rigid ratio")
    if name == "shear":
        assert abs(float(geometry.J_s[0])-1.2) <= 1e-13


def test_16_dyn_geometry_cache_new_length_repeatability_and_stale_monitor():
    solid, fluid, n_s = oriented_domains("up", n=4)
    topology = build_interface(solid, fluid)
    reference_length = float(np.sum(topology.reference_lengths))
    results = {}
    for candidate_id, stretch in (("candidate-a", 1.5), ("candidate-b", 2.2)):
        current, displacement, geometry, operators = evaluate_candidate(
            topology, solid, fluid, np.diag([stretch, 1.0]), candidate_id=candidate_id)
        solution = analytic_state(operators, pressure=P_STAR)
        load = assemble_interface_load(topology, geometry, operators, solution, thickness=1.0)
        results[candidate_id] = (geometry, operators, solution, load)
        length = reference_length*stretch
        assert_scaled(geometry.lengths, np.full(topology.n_edges, length/topology.n_edges),
                      note=f"{candidate_id} lengths")
        assert_scaled(load.total_force, -P_STAR*length*n_s, note=f"{candidate_id} force")
        assert_scaled(load.total_moment, -P_STAR*length**2/2.0, explicit=1.0, note=f"{candidate_id} moment")
    geometry_a, operators_a, _, load_a = results["candidate-a"]
    geometry_b, operators_b, solution_b, load_b = results["candidate-b"]
    assert_scaled(load_b.total_force, 2.2/1.5*load_a.total_force, note="new geometry force")
    assert_scaled(load_b.total_moment, (2.2/1.5)**2*load_a.total_moment, explicit=1.0, note="new geometry moment")
    assert not np.allclose(load_b.g_f_gamma, load_a.g_f_gamma)

    _, _, geometry_again, operators_again = evaluate_candidate(
        topology, solid, fluid, np.diag([1.5, 1.0]), candidate_id="candidate-a")
    load_again = assemble_interface_load(topology, geometry_again, operators_again,
                                         analytic_state(operators_again, pressure=P_STAR), thickness=1.0)
    assert geometry_again.geometry_id == geometry_a.geometry_id
    np.testing.assert_array_equal(load_again.g_f_gamma, load_a.g_f_gamma)
    np.testing.assert_array_equal(load_again.f_s_gamma, load_a.f_s_gamma)

    monitor = BoundaryMonitor(operators_a, "interface")
    with pytest.raises(ValueError, match="different geometry"):
        monitor.force(solution_b, operators=operators_b)
    with pytest.raises(ValueError, match="different candidate"):
        assemble_interface_load(topology, geometry_a, operators_b, solution_b, thickness=1.0)
    with pytest.raises(ValueError, match="different candidate"):
        geometry_a.require_current(operators_b)
    with pytest.raises(ValueError, match="connectivity"):
        evaluate_interface_geometry(build_interface(*oriented_domains("up", n=3)[:2]), fluid,
                                    P2Space.from_mesh(fluid), np.zeros_like(solid.points), candidate_id="bad")


def test_interface_geometry_rejections_and_read_only():
    solid, fluid, n_s = oriented_domains("up", n=4)
    topology = build_interface(solid, fluid)
    space = P2Space.from_mesh(fluid)
    zero = np.zeros_like(solid.points)
    geometry = evaluate_interface_geometry(topology, fluid, space, zero, candidate_id="ok")
    assert isinstance(geometry, InterfaceGeometry)
    with pytest.raises(ValueError):
        geometry.lengths[0] = 0.0
    with pytest.raises(TypeError):
        evaluate_interface_geometry(solid, fluid, space, zero, candidate_id="bad")
    with pytest.raises(ValueError, match="shape"):
        evaluate_interface_geometry(topology, fluid, space, np.zeros((solid.n_nodes, 3)), candidate_id="bad")
    with pytest.raises(ValueError, match="finite"):
        evaluate_interface_geometry(topology, fluid, space, np.full((solid.n_nodes, 2), np.nan),
                                    candidate_id="bad")
    with pytest.raises(ValueError, match="current_space"):
        evaluate_interface_geometry(topology, fluid, P2Space.from_mesh(solid), zero, candidate_id="bad")
    coarse = rectangle_mesh(2, 2)
    with pytest.raises(ValueError, match="connectivity"):
        evaluate_interface_geometry(topology, coarse, P2Space.from_mesh(coarse),
                                    np.zeros((coarse.n_nodes, 2)), candidate_id="bad")
    flipped = solid.points @ np.diag([-1.0, 1.0]).T-solid.points
    with pytest.raises(ValueError, match="Jacobian"):
        evaluate_interface_geometry(topology, fluid, space, flipped, candidate_id="bad")
    collapsed = solid.points @ np.diag([0.0, 1.0]).T-solid.points
    with pytest.raises(ValueError, match="degenerate|Jacobian"):
        evaluate_interface_geometry(topology, fluid, space, collapsed, candidate_id="bad")
    moved = TriMesh(fluid.points+np.array([0.0, 0.01]), fluid.cells, fluid.boundaries)
    with pytest.raises(ValueError, match="coincide"):
        evaluate_interface_geometry(topology, moved, P2Space.from_mesh(moved), zero, candidate_id="bad")
    other = evaluate_interface_geometry(topology, fluid, space, zero, candidate_id="other-version")
    assert other.geometry_id != geometry.geometry_id
    same = evaluate_interface_geometry(topology, fluid, space, zero, candidate_id="ok")
    assert same.geometry_id == geometry.geometry_id


def test_interface_load_rejections_and_diagnostics_blocks():
    solid, fluid, n_s = oriented_domains("up", n=3)
    topology = build_interface(solid, fluid)
    _, _, geometry, operators = evaluate_candidate(topology, solid, fluid, np.eye(2), candidate_id="diag")
    solution = analytic_state(operators, pressure=P_STAR, shear=GAMMA_STAR)
    load = assemble_interface_load(topology, geometry, operators, solution, thickness=1.0)
    reference = assemble_interface_load(topology, geometry, operators, solution, thickness=1.0, path="reference")
    assert isinstance(load, InterfaceLoad)
    with pytest.raises(ValueError, match="thickness"):
        assemble_interface_load(topology, geometry, operators, solution, thickness=0.0)
    with pytest.raises(ValueError, match="thickness"):
        assemble_interface_load(topology, geometry, operators, solution, thickness=True)
    with pytest.raises(ValueError, match="path"):
        assemble_interface_load(topology, geometry, operators, solution, thickness=1.0, path="middle")
    with pytest.raises(ValueError, match="moment_origin"):
        assemble_interface_load(topology, geometry, operators, solution, thickness=1.0, moment_origin=(0.0,))
    with pytest.raises(TypeError):
        assemble_interface_load(topology, geometry, operators, None, thickness=1.0)
    bad = FlowSolution(np.full_like(operators.space.points, np.nan), np.zeros(fluid.n_nodes), 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        assemble_interface_load(topology, geometry, operators, bad, thickness=1.0)
    with pytest.raises(ValueError, match="shape"):
        pullback_traction(geometry, np.zeros((geometry.n_edges, geometry.n_quadrature, 3)))
    with pytest.raises(ValueError, match="path"):
        pullback_traction(geometry, load.t_s, path="middle")
    assert_scaled(load.T_s0, geometry.j_Gamma_nanson[:, None, None]*load.t_s, note="pullback nanson")
    assert_scaled(pullback_traction(geometry, load.t_s, path="length"),
                  geometry.j_Gamma_current[:, None, None]*load.t_s, note="pullback length")

    other_current, _, other_geometry, other_operators = evaluate_candidate(
        topology, solid, fluid, np.diag([1.4, 1.0]), candidate_id="other")
    other_solution = analytic_state(other_operators, pressure=P_STAR)
    other_load = assemble_interface_load(topology, other_geometry, other_operators, other_solution, thickness=1.0)
    with pytest.raises(ValueError, match="different candidate"):
        assemble_interface_load(topology, geometry, other_operators, other_solution, thickness=1.0)
    with pytest.raises(ValueError, match="different geometry version"):
        interface_diagnostics(topology, geometry=geometry, load=other_load)
    with pytest.raises(ValueError, match="either loads or load"):
        interface_diagnostics(topology, geometry=geometry, load=load, loads=load.f_s_gamma)

    rng = np.random.default_rng(20261012)
    delta = rng.normal(size=2*topology.n_solid_interface_nodes)
    result = interface_diagnostics(topology, geometry=geometry, load=load, virtual_solid=delta,
                                   moment_origin=(0.1, -0.2),
                                   scales={"force": 1.0, "moment": 1.0, "work": 1.0, "length": 1.0})
    assert result["pass"], result["failures"]
    assert result["configuration"] == "current"
    assert result["geometry"]["passed"] and result["traction"]["passed"]
    assert result["geometry"]["surface_ratio_error"] <= result["geometry"]["surface_ratio_tolerance"]
    assert result["traction"]["force_quadrature_error"] <= result["traction"]["force_tolerance"]
    assert result["traction"]["moment_quadrature_error"] <= result["traction"]["moment_tolerance"]
    reference_result = interface_diagnostics(topology, geometry=geometry, load=reference, virtual_solid=delta)
    assert reference_result["pass"], reference_result["failures"]
    broken = dataclasses.replace(load, total_force=load.total_force+1.0)
    broken_result = interface_diagnostics(topology, geometry=geometry, load=broken)
    assert not broken_result["pass"] and "traction" in broken_result["failures"][0]


# ---------------------------------------------------------------------------
# T16d / 16-DER: analytic directional derivative of the interface load
# ---------------------------------------------------------------------------

JVP_EPSILONS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
JVP_TRACTION_SCALE = MU_STAR + 1.0  # mu* U*/L* + P* with U* = L* = P* = 1
_JVP_BLOCKS = ("t_s", "g_f_gamma", "f_s_gamma", "total_force", "total_moment")
_JVP_ATTRIBUTES = ("delta_t_s", "delta_g_f_gamma", "delta_f_s_gamma", "delta_total_force",
                   "delta_total_moment")


def jvp_configuration(level=4, seed=1604):
    """Base candidate, fluid state and the direction families for 16-DER.

    The candidate is a general affine map applied to both domains (non-unit
    ``F_s`` and ``J_x``).  ``mesh-only`` moves the fluid P1 nodes that are not
    interface vertices: the two current interfaces stay coincident (so the
    perturbed geometry passes the gap check of
    ``evaluate_interface_geometry``) while the owner-cell ``J_x`` and the P2
    physical gradients change.  ``solid-only`` moves interior solid nodes only;
    it is a *structurally* zero-response direction: the owner-cell third-vertex
    basis gradient is perpendicular to the interface edge, so ``j_Gamma`` and
    the current traction are exactly insensitive at first order, which the test
    checks on an absolute scale.  ``interface-motion`` moves each matched
    interface vertex by the same displacement on the solid and fluid side, and
    ``mixed`` adds interior nodes plus velocity/pressure directions on top.
    """
    solid, fluid, _ = oriented_domains("up", n=level)
    topology = build_interface(solid, fluid)
    matrix = np.array([[1.2, 0.35], [0.12, 0.9]])  # det = 1.038 > 0
    center = np.array([0.17, -0.23])
    offset = np.array([0.06, 0.04])
    current, displacement, geometry, operators = evaluate_candidate(
        topology, solid, fluid, matrix, center, offset, candidate_id="16-der-base")
    rng = np.random.default_rng(seed)
    n_p2 = len(operators.space.points)
    velocity = rng.normal(scale=0.25, size=(n_p2, 2))
    pressure = rng.normal(scale=0.3, size=(fluid.n_nodes,))
    solution = FlowSolution(velocity, pressure, 0.0, 0.0)

    interface_solid = {int(node) for node in topology.solid_nodes}
    interface_fluid = {int(node) for node in topology.fluid_vertex_nodes}
    interior_solid = [node for node in range(solid.n_nodes) if node not in interface_solid]
    interior_fluid = [node for node in range(fluid.n_nodes) if node not in interface_fluid]
    assert len(interior_solid) and len(interior_fluid)

    full_u = rng.normal(size=(n_p2, 2))
    full_p = rng.normal(size=(fluid.n_nodes,))
    full_mesh = rng.normal(size=(fluid.n_nodes, 2))
    full_solid = rng.normal(size=(solid.n_nodes, 2))
    zero_u = np.zeros((n_p2, 2))
    zero_p = np.zeros((fluid.n_nodes,))
    zero_mesh = np.zeros((fluid.n_nodes, 2))
    zero_solid = np.zeros((solid.n_nodes, 2))
    mesh_only = zero_mesh.copy()
    mesh_only[interior_fluid] = full_mesh[interior_fluid]
    solid_only = zero_solid.copy()
    solid_only[interior_solid] = full_solid[interior_solid]
    mesh_interface = zero_mesh.copy()
    mesh_interface[topology.fluid_vertex_nodes] = full_solid[topology.solid_nodes]
    solid_interface = zero_solid.copy()
    solid_interface[topology.solid_nodes] = full_solid[topology.solid_nodes]
    mesh_mixed = full_mesh.copy()
    mesh_mixed[topology.fluid_vertex_nodes] = full_solid[topology.solid_nodes]

    directions = {
        "u-only": {"velocity": full_u, "pressure": zero_p,
                   "mesh_displacement": zero_mesh, "solid_displacement": zero_solid},
        "p-only": {"velocity": zero_u, "pressure": full_p,
                   "mesh_displacement": zero_mesh, "solid_displacement": zero_solid},
        "mesh-only": {"velocity": zero_u, "pressure": zero_p,
                      "mesh_displacement": mesh_only, "solid_displacement": zero_solid},
        "solid-only": {"velocity": zero_u, "pressure": zero_p,
                       "mesh_displacement": zero_mesh, "solid_displacement": solid_only},
        "interface-motion": {"velocity": zero_u, "pressure": zero_p,
                             "mesh_displacement": mesh_interface, "solid_displacement": solid_interface},
        "mixed": {"velocity": full_u, "pressure": full_p,
                  "mesh_displacement": mesh_mixed, "solid_displacement": full_solid},
    }
    return solid, fluid, topology, current, displacement, geometry, operators, solution, directions


def jvp_perturbed_load(topology, current, displacement, solution, direction, *, sign, eps, thickness, path):
    """``assemble_interface_load`` at ``(mesh, d_s, u, p) + sign*eps*delta``."""
    delta_mesh = direction.get("mesh_displacement", np.zeros_like(current.points, dtype=float))
    delta_solid = direction.get("solid_displacement", np.zeros_like(displacement, dtype=float))
    delta_velocity = direction.get("velocity", np.zeros_like(solution.velocity, dtype=float))
    delta_pressure = direction.get("pressure", np.zeros_like(solution.pressure, dtype=float))
    mesh = TriMesh(current.points + sign*eps*delta_mesh, current.cells, current.boundaries)
    space = P2Space.from_mesh(mesh)
    perturbed = evaluate_interface_geometry(topology, mesh, space,
                                            displacement + sign*eps*delta_solid,
                                            candidate_id=("16-der-fd", sign, eps, path))
    operators = assemble_stokes(mesh, viscosity=MU_STAR, density=1.0)
    state = FlowSolution(solution.velocity + sign*eps*delta_velocity,
                         solution.pressure + sign*eps*delta_pressure, 0.0, 0.0)
    return assemble_interface_load(topology, perturbed, operators, state, thickness=thickness, path=path)


def jvp_finite_difference(topology, current, displacement, solution, direction, *, eps, thickness, path):
    plus = jvp_perturbed_load(topology, current, displacement, solution, direction, sign=1.0, eps=eps,
                              thickness=thickness, path=path)
    minus = jvp_perturbed_load(topology, current, displacement, solution, direction, sign=-1.0, eps=eps,
                               thickness=thickness, path=path)
    return {name: (getattr(plus, name)-getattr(minus, name))/(2*eps) for name in _JVP_BLOCKS}


def jvp_scales(thickness, length_scale):
    """Explicit dimensional scales for each derivative block."""
    force = thickness*JVP_TRACTION_SCALE*length_scale
    return {"t_s": JVP_TRACTION_SCALE, "g_f_gamma": force, "f_s_gamma": force,
            "total_force": force, "total_moment": force*length_scale}


def jvp_errors(derivative, finite_difference, scales):
    """Worst relative block error with the frozen ``max(|Jv|, scale)`` denominator."""
    worst = 0.0
    detail = {}
    for name, attribute in zip(_JVP_BLOCKS, _JVP_ATTRIBUTES):
        analytic = np.asarray(getattr(derivative, attribute), dtype=float)
        reference = np.asarray(finite_difference[name], dtype=float)
        scale = max(float(np.max(np.abs(analytic))) if analytic.size else 0.0, scales[name])
        detail[name] = float(np.max(np.abs(reference-analytic)))/scale
        worst = max(worst, detail[name])
    return worst, detail


@pytest.mark.parametrize("path", ["current", "reference"])
@pytest.mark.parametrize("case", ["u-only", "p-only", "mesh-only", "solid-only", "interface-motion", "mixed"])
def test_16_der_analytic_jvp_matches_central_difference_scan(case, path):
    _, _, topology, current, displacement, geometry, operators, solution, directions = jvp_configuration()
    direction = directions[case]
    thickness = 0.7
    derivative = interface_load_jvp(topology, geometry, operators, solution, direction,
                                    thickness=thickness, path=path)
    assert isinstance(derivative, InterfaceLoadDerivative)
    scales = jvp_scales(thickness, float(np.sum(topology.reference_lengths)))
    errors = []
    details = []
    for eps in JVP_EPSILONS:
        finite_difference = jvp_finite_difference(topology, current, displacement, solution, direction,
                                                  eps=eps, thickness=thickness, path=path)
        worst, detail = jvp_errors(derivative, finite_difference, scales)
        errors.append(worst)
        details.append(detail)
    if case == "solid-only":
        # Structurally zero: the third-vertex basis gradient of every owner cell
        # is perpendicular to the (straight) interface edge, so moving interior
        # solid nodes changes F_s but not j_Gamma/n_f/length at first order.
        # The response is checked on the absolute scale, not as a ratio.
        assert max(float(np.max(np.abs(np.asarray(getattr(derivative, attribute)))))
                   for attribute in _JVP_ATTRIBUTES) <= 1e-10*scales["t_s"], (case, path)
        assert max(errors) <= 1e-5, (case, path, errors, details)
        return
    assert np.max(np.abs(derivative.delta_g_f_gamma)) > 1e-2*scales["g_f_gamma"], (case, path)
    assert min(errors) <= 1e-5, (case, path, errors, details)
    if case in ("mesh-only", "interface-motion", "mixed"):
        assert errors[0] > errors[1] > errors[2], (case, path, errors)
        assert errors[-1] > min(errors), (case, path, errors)
    else:
        assert min(errors) <= 1e-12, (case, path, errors)
        assert errors[-1] > errors[0], (case, path, errors)


def test_16_der_zero_response_directions_use_absolute_scale():
    _, _, topology, current, displacement, geometry, operators, solution, _ = jvp_configuration()
    thickness = 0.7
    far_fluid = int(np.argmax(current.points[:, 1]))
    interface_cells = {int(node) for cell in operators.space.cells[geometry.fluid_owner] for node in cell}
    assert far_fluid not in interface_cells
    far_solid = int(np.argmin(np.asarray(topology.solid_mesh.points)[:, 1]))
    solid_cells = {int(node) for pair in topology.edge_pairs
                   for node in topology.solid_mesh.cells[pair.solid_owner]}
    assert far_solid not in solid_cells

    mesh_zero = np.zeros_like(current.points)
    mesh_zero[far_fluid] = [0.9, -1.1]
    solid_zero = np.zeros((topology.n_solid_nodes, 2))
    solid_zero[far_solid] = [-0.4, 0.8]
    velocity_zero = np.zeros_like(solution.velocity)
    velocity_zero[far_fluid] = [1.3, -0.7]
    for path in ("current", "reference"):
        for delta in ({"mesh_displacement": mesh_zero}, {"solid_displacement": solid_zero},
                      {"velocity": velocity_zero}):
            derivative = interface_load_jvp(topology, geometry, operators, solution, delta,
                                            thickness=thickness, path=path)
            for attribute in _JVP_ATTRIBUTES:
                np.testing.assert_array_equal(np.asarray(getattr(derivative, attribute)), 0.0)
            finite_difference = jvp_finite_difference(topology, current, displacement, solution, delta,
                                                      eps=1e-4, thickness=thickness, path=path)
            for name in _JVP_BLOCKS:
                np.testing.assert_array_equal(np.asarray(finite_difference[name]), 0.0)


def test_16_der_mapping_identity_input_preservation_and_read_only():
    _, _, topology, current, displacement, geometry, operators, solution, directions = jvp_configuration()
    direction = directions["mixed"]
    thickness = 0.7
    watch = {
        "fluid points": current.points, "solid displacement": geometry.solid_displacement,
        "x_q": geometry.x_q, "lengths": geometry.lengths, "n_s": geometry.n_s, "n_f": geometry.n_f,
        "F_s": geometry.F_s, "J_s": geometry.J_s, "j_Gamma_nanson": geometry.j_Gamma_nanson,
        "velocity": solution.velocity, "pressure": solution.pressure,
        "matrix": operators.matrix.data, "gradients": operators.gradients, "rhs": operators.rhs,
        "H data": topology.H.data, "H indices": topology.H.indices, "H indptr": topology.H.indptr,
        "X_s_gamma": topology.X_s_gamma, "reference_lengths": topology.reference_lengths,
        "delta velocity": direction["velocity"], "delta pressure": direction["pressure"],
        "delta mesh": direction["mesh_displacement"], "delta solid": direction["solid_displacement"],
    }
    before = {name: np.array(value, copy=True) for name, value in watch.items()}
    result = interface_load_jvp(topology, geometry, operators, solution, direction,
                                thickness=thickness, path="current")
    reference = interface_load_jvp(topology, geometry, operators, solution, direction,
                                   thickness=thickness, path="reference")
    for name, value in watch.items():
        np.testing.assert_array_equal(np.asarray(value), before[name], err_msg=name)
    assert result.geometry_id == geometry.geometry_id and result.topology_id == topology.topology_id
    np.testing.assert_array_equal(result.delta_f_s_gamma,
                                  np.asarray(topology.H.T @ result.delta_g_f_gamma).ravel())
    np.testing.assert_array_equal(reference.delta_f_s_gamma,
                                  np.asarray(topology.H.T @ reference.delta_g_f_gamma).ravel())
    np.testing.assert_array_equal(result.delta_total_force,
                                  result.delta_f_s_gamma.reshape(-1, 2).sum(axis=0))
    np.testing.assert_allclose(result.delta_total_force,
                               result.delta_g_f_gamma.reshape(-1, 2).sum(axis=0), rtol=0, atol=1e-13)
    assert not np.array_equal(result.delta_g_f_gamma, reference.delta_g_f_gamma)
    frozen_solution = FlowSolution(np.array(solution.velocity, copy=True),
                                   np.array(solution.pressure, copy=True), 0.0, 0.0)
    frozen_solution.velocity.setflags(write=False)
    frozen_solution.pressure.setflags(write=False)
    frozen_direction = {key: np.array(value, copy=True) for key, value in direction.items()}
    for value in frozen_direction.values():
        value.setflags(write=False)
    frozen = interface_load_jvp(topology, geometry, operators, frozen_solution, frozen_direction,
                                thickness=thickness, path="current")
    np.testing.assert_array_equal(frozen.delta_g_f_gamma, result.delta_g_f_gamma)
    np.testing.assert_array_equal(frozen.delta_f_s_gamma, result.delta_f_s_gamma)
    for attribute in ("delta_t_s", "delta_g_f_gamma", "delta_f_s_gamma", "delta_total_force"):
        with pytest.raises(ValueError):
            getattr(result, attribute)[0] = 0.0
    assert np.isfinite(result.delta_total_moment) and np.isfinite(reference.delta_total_moment)


def test_16_der_argument_and_direction_rejections():
    solid, fluid, topology, current, displacement, geometry, operators, solution, directions = \
        jvp_configuration()
    direction = directions["mixed"]
    thickness = 1.0
    with pytest.raises(TypeError):
        interface_load_jvp(None, geometry, operators, solution, direction, thickness=thickness)
    with pytest.raises(TypeError):
        interface_load_jvp(topology, None, operators, solution, direction, thickness=thickness)
    other_solid, other_fluid, _ = oriented_domains("up", n=3)
    other_topology = build_interface(other_solid, other_fluid)
    with pytest.raises(ValueError, match="different topology"):
        interface_load_jvp(other_topology, geometry, operators, solution, direction, thickness=thickness)
    _, _, _, other_operators = evaluate_candidate(topology, solid, fluid, np.diag([1.4, 1.0]),
                                                  candidate_id="16-der-other")
    with pytest.raises(ValueError, match="different candidate"):
        interface_load_jvp(topology, geometry, other_operators, solution, direction, thickness=thickness)
    for bad in (0.0, -1.0, np.nan, np.inf, True):
        with pytest.raises(ValueError, match="thickness"):
            interface_load_jvp(topology, geometry, operators, solution, direction, thickness=bad)
    with pytest.raises(ValueError, match="path"):
        interface_load_jvp(topology, geometry, operators, solution, direction, thickness=thickness,
                           path="middle")
    with pytest.raises(TypeError, match="mapping"):
        interface_load_jvp(topology, geometry, operators, solution, np.zeros(3), thickness=thickness)
    with pytest.raises(ValueError, match="Unknown delta keys"):
        interface_load_jvp(topology, geometry, operators, solution, {"mesh": np.zeros((2, 2))},
                           thickness=thickness)
    with pytest.raises(ValueError, match="at least one direction key"):
        interface_load_jvp(topology, geometry, operators, solution, {}, thickness=thickness)
    with pytest.raises(ValueError, match="nonzero"):
        interface_load_jvp(topology, geometry, operators, solution,
                           {"pressure": np.zeros(operators.space.mesh.n_nodes)}, thickness=thickness)
    shapes = {"velocity": (len(operators.space.points)+1, 2), "pressure": (operators.space.mesh.n_nodes+1,),
              "mesh_displacement": (operators.space.mesh.n_nodes, 3),
              "solid_displacement": (topology.n_solid_nodes+2, 2)}
    for key, shape in shapes.items():
        with pytest.raises(ValueError, match="shape"):
            interface_load_jvp(topology, geometry, operators, solution,
                               {key: np.zeros(shape)}, thickness=thickness)
    broken = np.array(direction["pressure"], copy=True)
    broken[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        interface_load_jvp(topology, geometry, operators, solution, {"pressure": broken},
                           thickness=thickness)
    with pytest.raises(TypeError, match="velocity and pressure"):
        interface_load_jvp(topology, geometry, operators, SimpleNamespace(pressure=solution.pressure),
                           direction, thickness=thickness)
    bad_state = SimpleNamespace(velocity=np.zeros((len(operators.space.points)+1, 2)),
                                pressure=solution.pressure)
    with pytest.raises(ValueError, match="Velocity"):
        interface_load_jvp(topology, geometry, operators, bad_state, direction, thickness=thickness)

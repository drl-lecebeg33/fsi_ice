"""T14 ALE mesh basics: harmonic extension, candidate geometry, BDF velocity, GCL flux.

Frozen thresholds follow docs/development/m3_freeze.md section 6 (values repeated
here as module constants; they must not be relaxed). All geometry uses the frozen
validation scales L*=1 m, T*=1 s, U*=L*/T*.
"""

import numpy as np
import pytest

from fsi_ice.ale import (
    GCLTransport,
    MeshDisplacement,
    MeshVelocity,
    build_ale_geometry,
    build_gcl_transport,
    build_harmonic_extension,
    evaluate_mesh_velocity,
)
from fsi_ice.fem.p1 import shape_values
from fsi_ice.fem.p2 import P2Space
from fsi_ice.fem.poisson import error_norms
from fsi_ice.mesh import rectangle_mesh
from fsi_ice.time import BDFHistory

EPS = np.finfo(float).eps
L_STAR = 1.0
T_STAR = 1.0
U_STAR = L_STAR/T_STAR

PATCH_MAX_ERROR = 1e-11                 # max|d_h-d|/L*
PATCH_FREE_RESIDUAL = 1e-10             # scaled free residual per component
PATCH_CONSTRAINT_ERROR = 100*EPS*L_STAR
L2_ORDER_MIN = 1.8
H1_ORDER_MIN = 0.9
VELOCITY_EXACT = 1e-12*U_STAR
BDF1_ORDER_MIN = 0.9
BDF2_ORDER_MIN = 1.8
GCL_CELL_LIMIT = 1e-11                  # |g_BDF-Div_X Q| / S_g
GCL_EDGE_LIMIT = 1e-12*U_STAR*L_STAR    # interior edge cancellation
GCL_BUDGET_LIMIT = 1e-10*U_STAR*L_STAR  # global swept budget
DEFAULT_LIMITS = {"min_volume_ratio": 0.2, "min_quality": 0.1}
TRAJECTORY_TOLERANCE = 100*EPS*L_STAR


def _affine_field(matrix, offset):
    return lambda points: points @ np.asarray(matrix).T + np.asarray(offset)


def _boundary_data(field, mesh):
    nodes = mesh.boundary_nodes()
    return nodes, field(mesh.points[nodes])


def _midpoint_edges(space):
    return np.asarray(tuple(space.edge_nodes.keys()), dtype=np.int64)


# --------------------------------------------------------------------------
# 14-01 affine patch reproduction
# --------------------------------------------------------------------------

@pytest.mark.parametrize("level", [4, 8, 16])
@pytest.mark.parametrize(
    "matrix,offset",
    [
        (np.zeros((2, 2)), np.array([0.3, -0.17])),
        (np.array([[0.15, 0.25], [-0.2, 0.05]]), np.array([0.02, 0.04])),
    ],
    ids=["translation", "shear"],
)
def test_14_01_affine_patch_is_reproduced_exactly(level, matrix, offset):
    mesh = rectangle_mesh(level)
    reference_points = mesh.points.copy()
    reference_cells = mesh.cells.copy()
    field = _affine_field(matrix, offset)
    nodes, prescribed = _boundary_data(field, mesh)
    extension = build_harmonic_extension(mesh, diffusion=1.0)
    displacement = extension.extend(nodes, prescribed)
    np.testing.assert_array_equal(mesh.points, reference_points)
    np.testing.assert_array_equal(mesh.cells, reference_cells)
    assert isinstance(displacement, MeshDisplacement)
    assert displacement.values.shape == mesh.points.shape
    assert not displacement.values.flags.writeable
    assert not displacement.component_residuals.flags.writeable
    assert not displacement.boundary_nodes.flags.writeable
    np.testing.assert_array_equal(displacement.boundary_nodes, nodes)
    error = float(np.max(np.abs(displacement.values - field(mesh.points))))/L_STAR
    assert error <= PATCH_MAX_ERROR
    assert np.all(displacement.component_residuals <= PATCH_FREE_RESIDUAL)
    assert displacement.boundary_error <= PATCH_CONSTRAINT_ERROR


# --------------------------------------------------------------------------
# 14-02 general harmonic solution convergence
# --------------------------------------------------------------------------

def test_14_02_harmonic_solution_converges_on_refined_meshes():
    amplitude = 0.02

    def exact_x(points):
        return amplitude*np.sin(np.pi*points[:, 0])*np.sinh(np.pi*points[:, 1])/np.sinh(np.pi)

    def gradient_x(points):
        return amplitude*np.pi*np.column_stack((
            np.cos(np.pi*points[:, 0])*np.sinh(np.pi*points[:, 1])/np.sinh(np.pi),
            np.sin(np.pi*points[:, 0])*np.cosh(np.pi*points[:, 1])/np.sinh(np.pi),
        ))

    errors = []
    for level in (8, 16, 32):
        mesh = rectangle_mesh(level)
        nodes = mesh.boundary_nodes()
        prescribed = np.column_stack((exact_x(mesh.points[nodes]), np.zeros(len(nodes))))
        displacement = build_harmonic_extension(mesh).extend(nodes, prescribed)
        norms = error_norms(mesh, displacement.values[:, 0], exact_x, gradient_x)
        errors.append((norms["l2"], norms["h1_seminorm"]))
        assert np.max(np.abs(displacement.values[:, 1])) <= PATCH_MAX_ERROR
    for (l2_coarse, h1_coarse), (l2_fine, h1_fine) in zip(errors, errors[1:]):
        assert l2_fine < l2_coarse and h1_fine < h1_coarse
        assert np.log2(l2_coarse/l2_fine) >= L2_ORDER_MIN
        assert np.log2(h1_coarse/h1_fine) >= H1_ORDER_MIN


# --------------------------------------------------------------------------
# 14-03 BDF mesh velocity
# --------------------------------------------------------------------------

def _run_velocity_history(mesh, field, exact, dt, order, steps):
    history = BDFHistory(field(mesh.points, 0.0), dt, order=order)
    errors = []
    for step in range(steps):
        context = history.context()
        time = context.time
        candidate = field(mesh.points, time)
        geometry = build_ale_geometry(mesh, candidate, candidate_id=("velocity", order, dt, step))
        velocity = evaluate_mesh_velocity(geometry, context)
        assert isinstance(velocity, MeshVelocity)
        assert not velocity.values_p1.flags.writeable
        assert not velocity.values_p2.flags.writeable
        errors.append(float(np.max(np.abs(velocity.values_p1 - exact(mesh.points, time)))))
        edges = _midpoint_edges(geometry.current_space)
        expected_midpoints = 0.5*(velocity.values_p1[edges[:, 0]] + velocity.values_p1[edges[:, 1]])
        np.testing.assert_array_equal(velocity.values_p2[mesh.n_nodes:], expected_midpoints)
        history.accept(context, candidate)
    return errors


@pytest.mark.parametrize("dt", [0.05, 0.025, 0.0125])
@pytest.mark.parametrize("order", [1, 2])
def test_14_03_linear_trajectory_is_exact_including_startup(dt, order):
    mesh = rectangle_mesh(4)
    field = _affine_field(np.array([[0.02, -0.01], [0.015, 0.025]]), np.array([0.03, -0.02]))
    trajectory = lambda points, time: field(points)*time
    exact_velocity = lambda points, time: field(points)
    errors = _run_velocity_history(mesh, trajectory, exact_velocity, dt, order, int(round(0.5/dt)))
    assert max(errors) <= VELOCITY_EXACT


def _exponential_velocity_errors(dt, order):
    mesh = rectangle_mesh(4)
    base = _affine_field(np.array([[0.02, 0.0], [0.01, 0.03]]), np.array([0.02, -0.01]))
    trajectory = lambda points, time: base(points)*(1 - np.exp(-time/T_STAR))
    exact = lambda points, time: base(points)*np.exp(-time/T_STAR)/T_STAR
    return _run_velocity_history(mesh, trajectory, exact, dt, order, int(round(0.5/dt)))


@pytest.mark.parametrize("order", [1, 2])
def test_14_03_exponential_field_converges_with_dt(order):
    errors = {dt: _exponential_velocity_errors(dt, order) for dt in (0.05, 0.025, 0.0125)}
    dts = sorted(errors)
    if order == 1:
        observed = [max(errors[dt]) for dt in dts]
        startup = observed
        main_threshold = BDF1_ORDER_MIN
    else:
        observed = [max(errors[dt][1:]) for dt in dts]
        startup = [errors[dt][0] for dt in dts]
        main_threshold = BDF2_ORDER_MIN
    orders = [np.log2(coarse/fine) for fine, coarse in zip(observed, observed[1:])]
    assert min(orders) >= main_threshold
    startup_orders = [np.log2(coarse/fine) for fine, coarse in zip(startup, startup[1:])]
    assert min(startup_orders) >= BDF1_ORDER_MIN


def test_14_03_repeated_candidates_do_not_advance_history():
    mesh = rectangle_mesh(3)
    basis = _affine_field(np.array([[0.02, 0.0], [0.01, 0.03]]), np.array([0.02, -0.01]))
    dt = 0.05
    trajectory = lambda points, time: basis(points)*(1 - np.exp(-time/T_STAR))
    history = BDFHistory(trajectory(mesh.points, 0.0), dt, order=2)
    context = history.context()
    history.accept(context, trajectory(mesh.points, context.time))
    snapshot = history.snapshot()
    context = history.context()
    assert context.stencil.order == 2
    candidate = trajectory(mesh.points, context.time)
    geometry = build_ale_geometry(mesh, candidate, candidate_id="repeat-candidates")
    first = evaluate_mesh_velocity(geometry, context)
    second = evaluate_mesh_velocity(geometry, context)
    np.testing.assert_array_equal(first.values_p1, second.values_p1)
    np.testing.assert_array_equal(first.values_p2, second.values_p2)
    after = history.snapshot()
    assert after[0] == snapshot[0] and after[1] == snapshot[1]
    assert len(after[2]) == len(snapshot[2]) == 2
    for before_state, after_state in zip(snapshot[2], after[2]):
        np.testing.assert_array_equal(before_state, after_state)


# --------------------------------------------------------------------------
# 14-04 repeated candidates over a long trajectory
# --------------------------------------------------------------------------

def _assert_topology_and_health(geometry, reference_space):
    reference_mesh = geometry.reference_mesh
    assert np.array_equal(geometry.current_space.cells, reference_space.cells)
    assert dict(geometry.current_space.edge_nodes) == dict(reference_space.edge_nodes)
    edges = _midpoint_edges(reference_space)
    np.testing.assert_array_equal(
        geometry.current_space.points[reference_mesh.n_nodes:],
        geometry.current_mesh.points[edges].mean(axis=1),
    )
    assert geometry.minimum_volume_ratio >= DEFAULT_LIMITS["min_volume_ratio"]
    assert geometry.minimum_quality >= DEFAULT_LIMITS["min_quality"]


def test_14_04_repeated_candidates_preserve_topology_and_trajectory():
    level = 6
    mesh = rectangle_mesh(level)
    reference_points = mesh.points.copy()
    reference_cells = mesh.cells.copy()
    reference_space = P2Space.from_mesh(mesh)

    def trajectory(points, time):
        phase = 2*np.pi*time
        shear = 0.06*np.sin(phase)
        stretch = 0.03*np.cos(phase)
        x = points[:, 0] - 0.5
        y = points[:, 1] - 0.5
        return np.column_stack((
            0.02*np.sin(phase) + shear*y + stretch*x,
            0.015*np.cos(phase) - 0.5*shear*x + stretch*y,
        ))

    def perturbation(points, step, index):
        scale = 1e-4*(1 + index)
        x = points[:, 0] - 0.5
        y = points[:, 1] - 0.5
        return scale*np.column_stack((
            np.sin((2 + index)*np.pi*x)*np.cos((1 + step % 3)*np.pi*y),
            np.cos(np.pi*x)*np.sin((2 + index)*np.pi*y),
        ))

    dt = 0.01
    history = BDFHistory(trajectory(mesh.points, 0.0), dt, order=2)
    minimum_ratios = []
    minimum_qualities = []
    candidate_count = 0
    for step in range(100):
        context = history.context()
        time = context.time
        base = trajectory(mesh.points, time)
        for index in range(1 + step % 2):
            perturbed = base + perturbation(mesh.points, step, index)
            geometry = build_ale_geometry(mesh, perturbed, candidate_id=("trial", step, index))
            _assert_topology_and_health(geometry, reference_space)
            minimum_ratios.append(geometry.minimum_volume_ratio)
            minimum_qualities.append(geometry.minimum_quality)
            candidate_count += 1
        accepted_geometry = build_ale_geometry(mesh, base, candidate_id=("accept", step))
        _assert_topology_and_health(accepted_geometry, reference_space)
        candidate_count += 1
        np.testing.assert_array_equal(accepted_geometry.displacement, base)
        drift = float(np.max(np.abs(accepted_geometry.current_mesh.points - (mesh.points + base))))
        assert drift <= TRAJECTORY_TOLERANCE
        history.accept(context, base)
    assert candidate_count == 100 + 100 + 50
    assert abs(history.time - 1.0) <= 100*EPS
    assert min(minimum_ratios) >= DEFAULT_LIMITS["min_volume_ratio"]
    assert min(minimum_qualities) >= DEFAULT_LIMITS["min_quality"]
    np.testing.assert_array_equal(mesh.points, reference_points)
    np.testing.assert_array_equal(mesh.cells, reference_cells)


def test_geometry_fields_follow_frozen_definitions():
    mesh = rectangle_mesh(3)
    field = _affine_field(np.array([[0.1, -0.05], [0.04, 0.08]]), np.array([0.01, -0.02]))
    displacement = field(mesh.points)
    reference_points = mesh.points.copy()
    geometry = build_ale_geometry(mesh, displacement, candidate_id="fields")
    np.testing.assert_array_equal(mesh.points, reference_points)
    np.testing.assert_array_equal(geometry.displacement, displacement)
    assert not geometry.displacement.flags.writeable
    assert not np.shares_memory(geometry.displacement, displacement)
    assert not np.shares_memory(geometry.current_mesh.points, mesh.points)
    expected_gradients = np.einsum(
        "eij,ejk->eik", geometry.current_mesh.jacobians, np.linalg.inv(mesh.jacobians)
    )
    np.testing.assert_allclose(geometry.deformation_gradients, expected_gradients, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(
        geometry.volume_ratios, np.linalg.det(geometry.deformation_gradients), rtol=1e-14, atol=1e-15
    )
    np.testing.assert_array_equal(geometry.quality, geometry.current_mesh.quality)
    assert geometry.reference_space.mesh is mesh and geometry.current_space.mesh is geometry.current_mesh
    assert np.array_equal(geometry.reference_space.cells, geometry.current_space.cells)
    assert dict(geometry.reference_space.edge_nodes) == dict(geometry.current_space.edge_nodes)
    assert geometry.minimum_volume_ratio == float(np.min(geometry.volume_ratios))
    assert geometry.minimum_quality == float(np.min(geometry.quality))


# --------------------------------------------------------------------------
# 14-05 failures and rejections
# --------------------------------------------------------------------------

def test_14_05_geometry_rejections_leave_reference_untouched():
    mesh = rectangle_mesh(2)
    reference_points = mesh.points.copy()
    reference_cells = mesh.cells.copy()
    accepted = build_ale_geometry(mesh, np.zeros_like(mesh.points), candidate_id="accepted")
    accepted_values = accepted.displacement.copy()
    accepted_points = accepted.current_mesh.points.copy()
    accepted_snapshots = {
        name: np.array(getattr(accepted, name))
        for name in ("deformation_gradients", "volume_ratios", "quality")
    }

    flipped = np.zeros_like(mesh.points)
    flipped[0] = (0.6, 0.6)
    compressed = -0.7*mesh.points
    sheared = np.column_stack((5.0*mesh.points[:, 1], np.zeros(mesh.n_nodes)))
    not_a_number = np.zeros_like(mesh.points)
    not_a_number[3, 1] = np.nan
    infinite = np.zeros_like(mesh.points)
    infinite[3, 1] = np.inf

    cases = [
        ("flip", flipped, "orientation", True),
        ("volume-ratio", compressed, "volume ratio", True),
        ("quality", sheared, "quality", True),
        ("nan", not_a_number, "non-finite", False),
        ("inf", infinite, "non-finite", False),
    ]
    for name, candidate, pattern, locates_cell in cases:
        with pytest.raises(ValueError, match=pattern) as error:
            build_ale_geometry(mesh, candidate, candidate_id=f"reject-{name}")
        assert f"reject-{name}" in str(error.value)
        assert ("cell" in str(error.value)) is locates_cell
        np.testing.assert_array_equal(mesh.points, reference_points)
        np.testing.assert_array_equal(mesh.cells, reference_cells)
        np.testing.assert_array_equal(accepted.displacement, accepted_values)
        np.testing.assert_array_equal(accepted.current_mesh.points, accepted_points)
        for name_field, snapshot in accepted_snapshots.items():
            np.testing.assert_array_equal(getattr(accepted, name_field), snapshot)

    with pytest.raises(ValueError, match="shape"):
        build_ale_geometry(mesh, np.zeros((mesh.n_nodes, 3)), candidate_id="wrong-2d")
    with pytest.raises(ValueError, match="shape"):
        build_ale_geometry(mesh, np.zeros(mesh.n_nodes), candidate_id="wrong-1d")
    with pytest.raises(ValueError, match="shape"):
        build_ale_geometry(mesh, np.zeros((mesh.n_nodes - 1, 2)), candidate_id="wrong-nodes")
    with pytest.raises(ValueError, match="finite positive"):
        build_ale_geometry(mesh, np.zeros_like(mesh.points), candidate_id="limits",
                           quality_limits={"min_volume_ratio": 0.0})
    with pytest.raises(ValueError, match="unknown key"):
        build_ale_geometry(mesh, np.zeros_like(mesh.points), candidate_id="limits",
                           quality_limits={"min_volume_ratio": 0.2, "extra": 1.0})

    rebuilt = build_ale_geometry(mesh, np.zeros_like(mesh.points), candidate_id="rebuilt")
    np.testing.assert_array_equal(rebuilt.displacement, accepted_values)
    np.testing.assert_array_equal(rebuilt.current_mesh.points, accepted_points)


def test_14_05_quality_limits_are_configurable_and_stored():
    mesh = rectangle_mesh(4)
    contraction = -0.5*mesh.points  # uniform volume ratio 0.25, similarity quality
    tolerated = build_ale_geometry(mesh, contraction, candidate_id="default")
    assert tolerated.minimum_volume_ratio >= DEFAULT_LIMITS["min_volume_ratio"]
    assert dict(tolerated.quality_limits) == DEFAULT_LIMITS
    with pytest.raises(ValueError, match="volume ratio"):
        build_ale_geometry(mesh, contraction, candidate_id="strict",
                           quality_limits={"min_volume_ratio": 0.6})
    custom = build_ale_geometry(mesh, contraction, candidate_id="custom",
                                quality_limits={"min_volume_ratio": 0.2, "min_quality": 0.05})
    assert custom.quality_limits["min_volume_ratio"] == 0.2
    assert custom.quality_limits["min_quality"] == 0.05


def test_14_05_harmonic_rejections_leave_state_untouched():
    mesh = rectangle_mesh(3)
    reference_points = mesh.points.copy()
    reference_cells = mesh.cells.copy()
    extension = build_harmonic_extension(mesh)
    nodes = mesh.boundary_nodes()
    good = extension.extend(nodes, np.zeros((len(nodes), 2)))
    good_values = good.values.copy()

    missing = nodes[:-1]
    unknown = np.append(nodes, mesh.n_nodes)
    duplicated = np.append(nodes, nodes[0])
    conflicted = np.append(nodes, nodes[0])
    conflicted_values = np.zeros((len(conflicted), 2))
    conflicted_values[-1] = (1.0, 1.0)
    not_a_number = np.zeros((len(nodes), 2))
    not_a_number[2, 0] = np.nan

    cases = [
        (missing, np.zeros((len(missing), 2)), "exterior"),
        (unknown, np.zeros((len(unknown), 2)), "Unknown"),
        (duplicated, np.zeros((len(duplicated), 2)), "unique|conflicting"),
        (conflicted, conflicted_values, "unique|conflicting"),
        (nodes, np.zeros((len(nodes), 1)), "shape"),
        (nodes, np.zeros(len(nodes)), "shape"),
        (nodes, np.zeros((len(nodes) - 1, 2)), "shape"),
        (nodes, not_a_number, "finite"),
        (np.array([], dtype=np.int64), np.zeros((0, 2)), "nonempty"),
        (nodes.astype(float), np.zeros((len(nodes), 2)), "integer"),
    ]
    for bad_nodes, bad_values, pattern in cases:
        with pytest.raises(ValueError, match=pattern):
            extension.extend(bad_nodes, bad_values)
        np.testing.assert_array_equal(mesh.points, reference_points)
        np.testing.assert_array_equal(mesh.cells, reference_cells)
        np.testing.assert_array_equal(good.values, good_values)

    recovered = extension.extend(nodes, np.zeros((len(nodes), 2)))
    np.testing.assert_array_equal(recovered.values, good_values)


@pytest.mark.parametrize("diffusion", [0.0, -1.0, np.nan, np.inf, True])
def test_14_05_invalid_diffusion_is_rejected(diffusion):
    with pytest.raises(ValueError, match="diffusion"):
        build_harmonic_extension(rectangle_mesh(2), diffusion=diffusion)


def test_14_05_velocity_rejects_shape_mismatch():
    mesh = rectangle_mesh(2)
    geometry = build_ale_geometry(mesh, np.zeros_like(mesh.points), candidate_id="velocity-shape")
    wrong_context = BDFHistory(np.zeros(mesh.n_nodes), 0.05, order=1).context()
    with pytest.raises(ValueError):
        evaluate_mesh_velocity(geometry, wrong_context)


# --------------------------------------------------------------------------
# Swept transport (T14c primitives, item 4) numerical verification
# --------------------------------------------------------------------------

def test_gcl_interval_divergence_identity_for_random_configurations():
    mesh = rectangle_mesh(4)
    generator = np.random.default_rng(20261011)
    newer = 0.01*generator.standard_normal(mesh.points.shape)
    older = -0.008*generator.standard_normal(mesh.points.shape)
    geometry_newer = build_ale_geometry(mesh, newer, candidate_id="pair-newer")
    geometry_older = build_ale_geometry(mesh, older, candidate_id="pair-older")
    dt = 0.05
    context = BDFHistory(older, dt, order=1).context()
    transport = build_gcl_transport(mesh, geometry_newer, [geometry_older], context.stencil)
    assert isinstance(transport, GCLTransport)
    expected = (geometry_newer.volume_ratios - geometry_older.volume_ratios)/dt
    np.testing.assert_allclose(transport.divergence, expected, rtol=1e-13, atol=1e-15)
    assert np.max(transport.cell_defects) <= GCL_CELL_LIMIT
    assert transport.step_id == context.stencil.step_id
    assert transport.dt == context.stencil.dt
    np.testing.assert_array_equal(transport.edges, np.sort(transport.edges, axis=1))


def _scaling_field(points, time):
    return 0.05*np.sin(time/T_STAR)*points


def _internal_field(points, time):
    envelope = 0.03*np.sin(np.pi*points[:, 0])*np.sin(np.pi*points[:, 1])
    return envelope[:, None]*np.array([np.sin(time/T_STAR), np.cos(time/T_STAR)])


def _check_transport_defects(mesh, transport, stencil):
    coefficients = np.asarray(stencil.coefficients)
    ratios = [transport.candidate_geometry.volume_ratios]
    ratios.extend(geometry.volume_ratios for geometry in transport.accepted_geometries)
    weighted = np.zeros(mesh.n_cells)
    divergence = np.zeros(mesh.n_cells)
    for coefficient, ratio in zip(coefficients, ratios):
        weighted += np.abs(coefficient)*ratio
        divergence += coefficient*ratio

    cell_scale = np.maximum(weighted, 1.0/T_STAR)
    assert np.max(transport.cell_defects/cell_scale) <= GCL_CELL_LIMIT
    np.testing.assert_allclose(transport.divergence, divergence, rtol=1e-14, atol=1e-15)

    areas = mesh.determinants/2
    sweep_scale = np.maximum(weighted*areas, areas/T_STAR)
    swept = np.abs(transport.cell_boundary_fluxes - divergence*areas)/sweep_scale
    assert np.max(swept) <= GCL_CELL_LIMIT

    assert transport.edge_defect <= GCL_EDGE_LIMIT
    assert transport.budget_defect <= GCL_BUDGET_LIMIT
    interior = ~transport.exterior_mask
    if np.any(interior):
        assert np.max(np.abs(transport.edge_fluxes[interior])) <= GCL_EDGE_LIMIT

    assert transport.flux.shape == (mesh.n_cells, len(transport.quadrature_points), 2)
    assert transport.w_gcl.shape == transport.flux.shape
    assert np.all(np.isfinite(transport.flux)) and np.all(np.isfinite(transport.w_gcl))
    for array in (
        transport.flux, transport.divergence, transport.edge_fluxes, transport.cell_boundary_fluxes,
        transport.w_gcl, transport.cell_defects, transport.coefficients, transport.times,
        transport.edges, transport.exterior_mask, transport.quadrature_points, transport.quadrature_weights,
    ):
        assert not array.flags.writeable


@pytest.mark.parametrize("field", [_scaling_field, _internal_field], ids=["global-scaling", "internal-motion"])
@pytest.mark.parametrize("order", [1, 2])
def test_gcl_swept_transport_meets_frozen_defects(order, field):
    mesh = rectangle_mesh(8)
    dt = 0.05
    history = BDFHistory(field(mesh.points, 0.0), dt, order=order)
    accepted = [build_ale_geometry(mesh, field(mesh.points, 0.0), candidate_id=("gcl-init", order))]
    observed_orders = []
    for step in range(4):
        context = history.context()
        stencil = context.stencil
        observed_orders.append(stencil.order)
        candidate = build_ale_geometry(mesh, field(mesh.points, context.time), candidate_id=("gcl", order, step))
        transport = build_gcl_transport(mesh, candidate, accepted[:stencil.order], stencil)
        _check_transport_defects(mesh, transport, stencil)
        history.accept(context, field(mesh.points, context.time))
        accepted.insert(0, candidate)
    expected = [1, 2, 2, 2] if order == 2 else [1, 1, 1, 1]
    assert observed_orders == expected


def test_gcl_effective_velocity_and_flux_follow_the_frozen_formula():
    mesh = rectangle_mesh(5)
    generator = np.random.default_rng(4)
    newer = 0.01*generator.standard_normal(mesh.points.shape)
    older = np.zeros_like(mesh.points)
    geometry_newer = build_ale_geometry(mesh, newer, candidate_id="w-newer")
    geometry_older = build_ale_geometry(mesh, older, candidate_id="w-older")
    dt = 0.1
    context = BDFHistory(older, dt, order=1).context()
    transport = build_gcl_transport(mesh, geometry_newer, [geometry_older], context.stencil)

    midpoint = 0.5*(geometry_newer.deformation_gradients + geometry_older.deformation_gradients)
    cofactors = np.empty_like(midpoint)
    cofactors[:, 0, 0] = midpoint[:, 1, 1]
    cofactors[:, 0, 1] = -midpoint[:, 0, 1]
    cofactors[:, 1, 0] = -midpoint[:, 1, 0]
    cofactors[:, 1, 1] = midpoint[:, 0, 0]
    nodal = np.einsum(
        "eij,enj->eni", cofactors, (geometry_newer.displacement - geometry_older.displacement)[mesh.cells]
    )/dt
    shapes = shape_values(transport.quadrature_points)
    expected_flux = np.einsum("qa,eai->eqi", shapes, nodal)
    np.testing.assert_allclose(transport.flux, expected_flux, rtol=1e-13, atol=1e-16)
    expected_w = np.einsum(
        "eij,eqj->eqi", geometry_newer.deformation_gradients, expected_flux
    )/geometry_newer.volume_ratios[:, None, None]
    np.testing.assert_allclose(transport.w_gcl, expected_w, rtol=1e-13, atol=1e-16)


def test_gcl_rejects_inconsistent_geometries_and_stencils():
    mesh = rectangle_mesh(3)
    other_mesh = rectangle_mesh(3)
    field = _scaling_field
    dt = 0.05
    history = BDFHistory(field(mesh.points, 0.0), dt, order=2)
    startup = history.context().stencil
    assert startup.order == 1
    candidate = build_ale_geometry(mesh, field(mesh.points, 0.0), candidate_id="candidate")
    accepted = build_ale_geometry(mesh, field(mesh.points, 0.0), candidate_id="accepted")
    foreign = build_ale_geometry(other_mesh, field(other_mesh.points, 0.0), candidate_id="foreign")

    with pytest.raises(ValueError, match="accepted_geometries"):
        build_gcl_transport(mesh, candidate, [], startup)
    with pytest.raises(ValueError, match="reference_mesh"):
        build_gcl_transport(mesh, foreign, [accepted], startup)
    with pytest.raises(ValueError, match="reference_mesh"):
        build_gcl_transport(mesh, candidate, [foreign], startup)
    with pytest.raises(ValueError, match="ALEGeometry"):
        build_gcl_transport(mesh, candidate, [object()], startup)

    history.accept(history.context(), field(mesh.points, 0.0))
    mature = history.context().stencil
    assert mature.order == 2
    with pytest.raises(ValueError, match="order=2"):
        build_gcl_transport(mesh, candidate, [accepted], mature)
    transport = build_gcl_transport(mesh, candidate, [accepted, accepted], mature)
    assert transport.order == 2

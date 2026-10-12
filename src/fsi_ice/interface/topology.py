"""One-time reference matching of solid and fluid interface tags.

The solid P1 trace and the fluid P2 trace of a wet interface are matched once
in the reference configuration.  The matching fixes the endpoint
correspondence, the scalar map ``h`` (fluid P2 trace <- solid P1 trace) and
its interleaved block form ``H``; deformed configurations never re-match
nodes and never move reference coordinates.

本模块在参照构形下对湿界面的固体 P1 迹与流体 P2 迹做一次性匹配，确定端点
对应、标量映射 ``h``（流体 P2 迹 <- 固体 P1 迹）及其交错块形式 ``H``。关键
约定：参照构形只匹配一次，形变构形从不重新匹配节点、也从不移动参照坐标；
``atol`` 与坐标同量纲、``rtol`` 无量纲；法向为各子域外法向、两域法向相反。
"""

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, kron

from fsi_ice.solid.linear import vector_dofs


def _digest(*items):
    """Stable short digest of labelled arrays and strings.

    对带标签的数组/字符串做稳定短摘要（SHA-256 前 16 位十六进制）。用于生成可
    复现的 ``topology_id`` / ``mesh_id``：相同参照数据得到相同摘要，重编号或
    重划网格会改变摘要。标签与值之间用分隔符隔开以避免歧义。
    """
    hasher = hashlib.sha256()
    for label, value in items:
        hasher.update(label.encode("utf-8"))
        hasher.update(b"\x00")
        if isinstance(value, str):
            hasher.update(value.encode("utf-8"))
        elif isinstance(value, (int, np.integer)):
            hasher.update(str(int(value)).encode("utf-8"))
        else:
            hasher.update(np.ascontiguousarray(value).tobytes())
        hasher.update(b"\xff")
    return hasher.hexdigest()[:16]


def _read_only(array):
    array.setflags(write=False)
    return array


def _freeze_sparse(matrix):
    matrix = csr_matrix(matrix)
    matrix.data.setflags(write=False)
    matrix.indices.setflags(write=False)
    matrix.indptr.setflags(write=False)
    return matrix


def _format_point(point):
    return "({:.17g}, {:.17g})".format(float(point[0]), float(point[1]))


@dataclass(frozen=True)
class InterfaceEdge:
    """One physical wet edge, listed exactly once.

    ``solid_edge`` and ``fluid_edge`` store global node pairs in the
    counterclockwise traversal order of their exterior owner cells, so
    ``(t_y, -t_x)`` of the stored tangent is the outward normal of that
    domain.  ``orientation`` is ``+1`` when ``solid_edge[k]`` matches
    ``fluid_edge[k]`` and ``-1`` when the stored orders are reversed.

    一条物理湿界面棱，只登记一次。``solid_edge``/``fluid_edge`` 以各自所有者外
    单元逆时针遍历顺序存储全局节点对，因此存储切线的 ``(t_y, -t_x)`` 即该子域
    外法向。``orientation`` 为 ``+1`` 表示 ``solid_edge`` 与 ``fluid_edge`` 顺序
    一致、``-1`` 表示两者存储顺序相反。``length`` 为参照棱长（长度量纲），
    ``N_s``/``N_f`` 为参照构形下的一对相反外法向（无量纲单位向量）。
    """

    solid_edge: tuple
    fluid_edge: tuple
    solid_owner: int
    fluid_owner: int
    fluid_midpoint: int
    orientation: int
    length: float
    N_s: tuple
    N_f: tuple


@dataclass(frozen=True)
class InterfaceTopology:
    """Immutable reference interface topology and mapping data.

    ``solid_nodes`` is sorted ascending.  ``fluid_vertex_nodes[k]`` is the
    fluid node matched to ``solid_nodes[k]``; ``fluid_p2_nodes`` lists those
    vertices first and the fluid edge midpoints afterwards, in ``edge_pairs``
    order.  Compact vectors follow the interleaved convention
    ``[x0, y0, x1, y1, ...]`` of ``vector_dofs``; ``h`` maps the compact
    solid P1 trace to the compact fluid P2 trace and ``H`` is its block form
    ``kron(h, I_2)``.  ``topology_id`` is a stable digest of the matched
    reference interface (tags, node ids, pair table, reference coordinates):
    identical reference data rebuilds the same id, renumbering or re-meshing
    changes it.  ``solid_mesh_id``/``fluid_mesh_id`` identify the full
    reference meshes.

    不可变的参照界面拓扑与映射数据。``solid_nodes`` 按升序排列；
    ``fluid_vertex_nodes[k]`` 是与 ``solid_nodes[k]`` 匹配的流体节点；
    ``fluid_p2_nodes`` 先列这些顶点、再按 ``edge_pairs`` 顺序列出流体棱中点。
    压缩向量遵循 ``vector_dofs`` 的交错约定 ``[x0, y0, x1, y1, ...]``；``h`` 将
    压缩固体 P1 迹映射到压缩流体 P2 迹（顶点单位映射、棱中点取两端点 0.5 平均），
    ``H`` 为其块形式 ``kron(h, I_2)``。``topology_id`` 是匹配结果的稳定摘要，
    相同参照数据重建同一 id，重编号或重划网格会改变它；``solid_mesh_id``/
    ``fluid_mesh_id`` 标识完整参照网格。
    """

    solid_mesh: object
    fluid_mesh: object
    fluid_space: object
    solid_tag: str
    fluid_tag: str
    atol: float
    rtol: float
    solid_nodes: np.ndarray
    fluid_vertex_nodes: np.ndarray
    fluid_p2_nodes: np.ndarray
    edge_pairs: tuple
    reference_lengths: np.ndarray
    N_s: np.ndarray
    N_f: np.ndarray
    X_s_gamma: np.ndarray
    X_f_gamma: np.ndarray
    h: object
    H: object
    solid_dofs: np.ndarray
    fluid_p2_dofs: np.ndarray
    topology_id: str
    solid_mesh_id: str
    fluid_mesh_id: str

    def __post_init__(self):
        for name in ("solid_nodes", "fluid_vertex_nodes", "fluid_p2_nodes", "reference_lengths",
                     "N_s", "N_f", "X_s_gamma", "X_f_gamma", "solid_dofs", "fluid_p2_dofs"):
            _read_only(getattr(self, name))
        object.__setattr__(self, "h", _freeze_sparse(self.h))
        object.__setattr__(self, "H", _freeze_sparse(self.H))
        object.__setattr__(self, "edge_pairs", tuple(self.edge_pairs))

    @property
    def n_solid_nodes(self):
        """Full solid mesh node count (not only interface nodes).

        完整固体网格节点数（含非界面节点）。
        """
        return len(self.solid_mesh.points)

    @property
    def n_fluid_nodes(self):
        """Full fluid P2 node count (not only interface nodes).

        完整流体 P2 节点数（含非界面节点）。
        """
        return len(self.fluid_space.points)

    @property
    def n_solid_interface_nodes(self):
        """固体界面 P1 迹节点数 ``n_sG``（= ``len(solid_nodes)``）。"""
        return len(self.solid_nodes)

    @property
    def n_fluid_interface_nodes(self):
        """流体界面 P2 迹节点数 ``n_fG``（顶点数 + 棱中点数）。"""
        return len(self.fluid_p2_nodes)

    @property
    def n_edges(self):
        """界面棱数（``edge_pairs`` 长度）。"""
        return len(self.edge_pairs)


@dataclass(frozen=True)
class _TaggedEdges:
    """一侧带标签界面棱的校验结果（内部使用）。

    ``nodes`` 为 ``(k, 2)`` 全局节点对，``owners`` 为每棱唯一所有者单元编号，
    ``local`` 为棱在所有者单元内的逆时针局部端点下标，``lengths`` 为参照棱长
    （长度量纲），``normals`` 为按逆时针遍历切线 ``(t_y, -t_x)`` 得到的外法向。
    """

    nodes: np.ndarray
    owners: np.ndarray
    local: np.ndarray
    lengths: np.ndarray
    normals: np.ndarray


def _tagged_edges(mesh, tag, side):
    """Validate one tagged side and recover owner cells and outward normals.

    校验一侧（``side`` 取 "solid"/"fluid"）带标签界面并恢复所有者单元与外法向。
    检查坐标/连通性形状与有限性、标签存在、棱无重复、每条棱恰有一个单元所有者、
    棱长非零，返回 :class:`_TaggedEdges`。``side`` 仅用于错误信息，不影响计算。
    """
    points = np.asarray(mesh.points, dtype=float)
    cells = np.asarray(mesh.cells)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError(f"The {side} mesh must provide finite (n, 2) reference coordinates.")
    if cells.ndim != 2 or cells.shape[1] != 3 or not np.issubdtype(cells.dtype, np.integer):
        raise ValueError(f"The {side} mesh must provide an integer (m, 3) connectivity.")
    if tag not in mesh.boundaries:
        raise ValueError(f"Unknown {side} boundary tag: {tag!r}")
    edges = np.asarray(mesh.boundaries[tag])
    if edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError(f"{side.capitalize()} tag {tag!r} must provide an integer (k, 2) edge array.")
    if len(edges) == 0:
        raise ValueError(f"Empty interface: {side} tag {tag!r} has no edges and cannot build a topology.")
    edges = edges.astype(np.int64, copy=True)
    if np.any(edges < 0) or np.any(edges >= len(points)):
        raise ValueError(f"{side.capitalize()} tag {tag!r} contains an edge node outside the mesh.")
    sorted_edges = np.sort(edges, axis=1)
    # 无向棱去重：同一对节点重复出现即为重复登记。
    unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    repeated = np.flatnonzero(counts > 1)
    if len(repeated):
        key = unique_edges[repeated[0]]
        raise ValueError(f"Duplicate {side} edge ({int(key[0])}, {int(key[1])}) in tag {tag!r}.")
    # 建无向棱 -> [(单元, 局部端点 a, b), ...] 索引，用于校验每条界面棱唯一所有者。
    owners = {}
    for cell_id, cell in enumerate(cells):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            owners.setdefault(tuple(sorted((int(cell[a]), int(cell[b])))), []).append((cell_id, a, b))
    point_rows = points[edges]
    bad = np.flatnonzero(~np.all(np.isfinite(point_rows), axis=(1, 2)))
    if len(bad):
        row = edges[bad[0]]
        raise ValueError(f"Non-finite coordinates on {side} tag {tag!r} edge ({int(row[0])}, {int(row[1])}).")
    owner_cells = np.empty(len(edges), dtype=np.int64)
    local = np.empty((len(edges), 2), dtype=np.int64)
    for index, row in enumerate(edges):
        key = (int(min(row)), int(max(row)))
        entries = owners.get(key, [])
        if not entries:
            raise ValueError(f"{side.capitalize()} tag {tag!r} edge {key} belongs to no cell; "
                             "matching interface edges must be exterior.")
        if len(entries) > 1:
            raise ValueError(f"{side.capitalize()} tag {tag!r} edge {key} has {len(entries)} owner cells; "
                             "matching interface edges must have exactly one owner.")
        owner_cells[index], a, b = entries[0]
        local[index] = (a, b)
    tangent = points[cells[owner_cells, local[:, 1]]] - points[cells[owner_cells, local[:, 0]]]
    lengths = np.linalg.norm(tangent, axis=1)
    bad = np.flatnonzero(lengths <= 0)
    if len(bad):
        row = edges[bad[0]]
        raise ValueError(f"Zero-length {side} edge ({int(row[0])}, {int(row[1])}) in tag {tag!r}.")
    # 外法向 = 逆时针切线 (dx,dy) 旋转 -90° 后归一化：n = (dy,-dx)/|t|，
    # 保证法向指向所有者单元外部。
    normals = np.column_stack((tangent[:, 1], -tangent[:, 0])) / lengths[:, None]
    return _TaggedEdges(edges, owner_cells, local, lengths, normals)


def build_matching_interface(solid_mesh, fluid_mesh, fluid_space, *, solid_tag, fluid_tag, atol, rtol):
    """Match the wet interface once in the reference configuration.

    Endpoints are matched by reference coordinates within
    ``atol + rtol * max(|X_solid|, |X_fluid|)``.  Repeated or missing edges,
    ambiguous or unmatched endpoints, non-exterior or multi-owner edges,
    zero-length edges, non-finite coordinates and segmentation mismatches are
    rejected with the offending tag, edge and node reported.  The meshes are
    never modified and the matching is never repeated for deformed states.

    在参照构形下对湿界面做一次性匹配。输入 ``solid_mesh``/``fluid_mesh`` 为参照
    网格，``fluid_space`` 必须是由 ``fluid_mesh`` 构建的 P2 空间，``solid_tag``/
    ``fluid_tag`` 为界面标签。端点按参照坐标在 ``atol + rtol * max(|X_solid|,
    |X_fluid|)`` 容差内匹配（``atol`` 与坐标同量纲，``rtol`` 无量纲）。重复/缺失
    棱、二义或未匹配端点、非外棱或多所有者棱、零长棱、非有限坐标、分段不一致均
    报错并指出标签、棱、节点。返回固定的 :class:`InterfaceTopology`；参照网格
    不被修改，形变态不再重匹配。
    """
    for name, value in (("atol", atol), ("rtol", rtol)):
        number = float(value)
        if not np.isfinite(number) or (number <= 0 if name == "atol" else number < 0):
            requirement = "positive" if name == "atol" else "nonnegative"
            raise ValueError(f"{name} must be finite and {requirement}, got {value!r}.")
    for name, tag in (("solid", solid_tag), ("fluid", fluid_tag)):
        if not isinstance(tag, str) or not tag:
            raise ValueError(f"{name}_tag must be a nonempty string.")
    if getattr(fluid_space, "mesh", None) is not fluid_mesh:
        raise ValueError("fluid_space must be the P2 space built from fluid_mesh.")
    solid = _tagged_edges(solid_mesh, solid_tag, "solid")
    fluid = _tagged_edges(fluid_mesh, fluid_tag, "fluid")
    points_s = np.asarray(solid_mesh.points, dtype=float)
    points_f = np.asarray(fluid_mesh.points, dtype=float)

    solid_side_nodes = np.unique(solid.nodes)
    fluid_side_nodes = np.unique(fluid.nodes)
    coordinates_s = points_s[solid_side_nodes]
    coordinates_f = points_f[fluid_side_nodes]
    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.linalg.norm(coordinates_s[:, None, :] - coordinates_f[None, :, :], axis=2)
        # 逐对配对容差：atol + rtol * max(|X_s|,|X_f|)，两两比较得到 (n_s_side, n_f_side) 掩码。
        allowed = atol + rtol * np.maximum(np.linalg.norm(coordinates_s, axis=1)[:, None],
                                           np.linalg.norm(coordinates_f, axis=1)[None, :])
    if not np.all(np.isfinite(distances)):
        raise ValueError("Non-finite coordinate differences while matching the interface.")
    mask = distances <= allowed
    solid_to_fluid = np.full(len(solid_side_nodes), -1, dtype=np.int64)
    for index, node in enumerate(solid_side_nodes):
        candidates = np.flatnonzero(mask[index])
        if not len(candidates):
            raise ValueError(
                f"Solid interface node {int(node)} at {_format_point(coordinates_s[index])} has no fluid "
                f"candidate within atol={float(atol)!r}, rtol={float(rtol)!r}; missing or mismatched fluid edge.")
        if len(candidates) > 1:
            listed = ", ".join(f"{int(fluid_side_nodes[c])} at {_format_point(coordinates_f[c])}"
                               for c in candidates)
            raise ValueError(f"Ambiguous match for solid interface node {int(node)} at "
                             f"{_format_point(coordinates_s[index])}: fluid candidates {listed}.")
        solid_to_fluid[index] = candidates[0]
    matched_by = mask.sum(axis=0)
    unmatched = np.flatnonzero(matched_by == 0)
    if len(unmatched):
        index = unmatched[0]
        raise ValueError(f"Fluid interface node {int(fluid_side_nodes[index])} at "
                         f"{_format_point(coordinates_f[index])} is not matched by any solid interface node; "
                         "missing or extra interface segmentation.")
    duplicated = np.flatnonzero(matched_by > 1)
    if len(duplicated):
        index = duplicated[0]
        raise ValueError(f"Fluid interface node {int(fluid_side_nodes[index])} at "
                         f"{_format_point(coordinates_f[index])} is matched by {int(matched_by[index])} "
                         "solid interface nodes; ambiguous matching is not allowed.")
    matched_fluid = {int(solid_side_nodes[i]): int(fluid_side_nodes[solid_to_fluid[i]])
                     for i in range(len(solid_side_nodes))}

    fluid_edge_index = {tuple(sorted(int(v) for v in row)): j for j, row in enumerate(fluid.nodes)}
    claims = {}
    for i, row in enumerate(solid.nodes):
        pair = tuple(sorted((matched_fluid[int(row[0])], matched_fluid[int(row[1])])))
        if pair not in fluid_edge_index:
            raise ValueError(f"Solid edge ({int(row[0])}, {int(row[1])}) maps to fluid endpoints {pair} "
                             "with no fluid interface edge; segmentation mismatch.")
        j = fluid_edge_index[pair]
        if j in claims:
            raise ValueError(f"Fluid edge {pair} is claimed by solid edges "
                             f"({int(solid.nodes[claims[j], 0])}, {int(solid.nodes[claims[j], 1])}) and "
                             f"({int(row[0])}, {int(row[1])}); duplicated pairing.")
        claims[j] = i
    for j, row in enumerate(fluid.nodes):
        if j not in claims:
            pair = tuple(sorted(int(v) for v in row))
            raise ValueError(f"Fluid interface edge {pair} is not covered by any solid edge; "
                             "missing or extra interface segmentation.")

    cells_s = np.asarray(solid_mesh.cells)
    cells_f = np.asarray(fluid_mesh.cells)
    edge_pairs = []
    for j, i in claims.items():
        # 用所有者单元和逆时针局部端点取出两域棱的全局端点，判定遍历方向是否一致。
        solid_a = int(cells_s[solid.owners[i], solid.local[i, 0]])
        solid_b = int(cells_s[solid.owners[i], solid.local[i, 1]])
        fluid_a = int(cells_f[fluid.owners[j], fluid.local[j, 0]])
        fluid_b = int(cells_f[fluid.owners[j], fluid.local[j, 1]])
        endpoints = (int(matched_fluid[solid_a]), int(matched_fluid[solid_b]))
        if endpoints == (fluid_a, fluid_b):
            orientation = 1
        elif endpoints == (fluid_b, fluid_a):
            orientation = -1
        else:
            raise ValueError(f"Internal matching failure for solid edge ({solid_a}, {solid_b}).")
        key_f = tuple(sorted((fluid_a, fluid_b)))
        midpoint = fluid_space.edge_nodes.get(key_f)
        if midpoint is None:
            raise ValueError(f"Fluid space has no P2 midpoint for interface edge {key_f}; "
                             "fluid_space does not match the tagged fluid mesh.")
        reference_scale = max(np.linalg.norm(points_s[[solid_a, solid_b]], axis=1).max(),
                              np.linalg.norm(points_f[[fluid_a, fluid_b]], axis=1).max())
        match_tolerance = atol + rtol * reference_scale
        length = float(solid.lengths[i])
        if abs(length - float(fluid.lengths[j])) > 2 * match_tolerance:
            raise ValueError(f"Solid edge ({solid_a}, {solid_b}) and fluid edge {key_f} have different "
                             f"lengths {length!r} and {float(fluid.lengths[j])!r}.")
        normal_s = solid.normals[i]
        normal_f = fluid.normals[j]
        # 两域外法向应相反：n_s + n_f ≈ 0，容差由角度误差 (2*tol/length) 折算。
        normal_error = float(np.linalg.norm(normal_s + normal_f))
        normal_tolerance = 1e-12 + 2 * match_tolerance / max(length, np.finfo(float).tiny)
        if normal_error > normal_tolerance:
            raise ValueError(f"Solid edge ({solid_a}, {solid_b}) and fluid edge {key_f} have normals "
                             f"{_format_point(normal_s)} and {_format_point(normal_f)} that are not "
                             f"opposite (error {normal_error:.3e}); the owner cells must lie on opposite sides.")
        edge_pairs.append(InterfaceEdge((solid_a, solid_b), (fluid_a, fluid_b), int(solid.owners[i]),
                                        int(fluid.owners[j]), int(midpoint), orientation, length,
                                        (float(normal_s[0]), float(normal_s[1])),
                                        (float(normal_f[0]), float(normal_f[1]))))
    edge_pairs.sort(key=lambda pair: (min(pair.solid_edge), max(pair.solid_edge)))

    # 流体 P2 迹节点：先按 solid_nodes 顺序排列匹配顶点，再按 edge_pairs 顺序排列棱中点。
    solid_nodes = solid_side_nodes.astype(np.int64)
    solid_index = {int(node): k for k, node in enumerate(solid_nodes)}
    fluid_vertex_nodes = np.array([matched_fluid[int(node)] for node in solid_nodes], dtype=np.int64)
    n_edges = len(edge_pairs)
    n_solid_gamma = len(solid_nodes)
    n_fluid_gamma = n_solid_gamma + n_edges
    fluid_p2_nodes = np.empty(n_fluid_gamma, dtype=np.int64)
    fluid_p2_nodes[:n_solid_gamma] = fluid_vertex_nodes
    fluid_p2_nodes[n_solid_gamma:] = [pair.fluid_midpoint for pair in edge_pairs]
    if len(np.unique(fluid_p2_nodes)) != n_fluid_gamma:
        raise ValueError("Fluid interface P2 nodes are not unique; duplicated interface nodes.")
    rows = list(range(n_solid_gamma))
    columns = list(range(n_solid_gamma))
    values = [1.0] * n_solid_gamma
    for offset, pair in enumerate(edge_pairs):
        row = n_solid_gamma + offset
        for node in pair.solid_edge:
            rows.append(row)
            columns.append(solid_index[node])
            values.append(0.5)
    # 标量映射 h：顶点行单位映射，棱中点行对两端点各取 0.5；H = kron(h, I_2) 处理二维矢量自由度。
    h = csr_matrix(coo_matrix((values, (rows, columns)), shape=(n_fluid_gamma, n_solid_gamma)))
    H = _freeze_sparse(kron(h, np.eye(2), format="csr"))
    h = _freeze_sparse(h)

    X_s_gamma = np.array(points_s[solid_nodes], dtype=float)
    X_f_gamma = np.array(fluid_space.points[fluid_p2_nodes], dtype=float)
    if not np.all(np.isfinite(X_s_gamma)) or not np.all(np.isfinite(X_f_gamma)):
        raise ValueError("Non-finite reference interface coordinates.")
    coordinate_scale = max(float(np.linalg.norm(X_s_gamma, axis=1).max()),
                           float(np.linalg.norm(X_f_gamma, axis=1).max()))
    match_tolerance = atol + rtol * coordinate_scale
    # 参照匹配残差：流体 P2 迹坐标与 h 映射后的固体 P1 迹坐标之差，应小于容差。
    residual = np.linalg.norm(X_f_gamma - np.asarray(h @ X_s_gamma), axis=1)
    if float(residual.max()) > match_tolerance:
        row = int(np.argmax(residual))
        raise ValueError(f"Reference interface mismatch: node {int(fluid_p2_nodes[row])} residual "
                         f"{float(residual[row]):.3e} exceeds tolerance {match_tolerance:.3e}; rebuild the "
                         "reference interface from shared coordinates instead of adjusting nodes here.")

    reference_lengths = np.array([pair.length for pair in edge_pairs], dtype=float)
    normals_s = np.array([pair.N_s for pair in edge_pairs], dtype=float)
    normals_f = np.array([pair.N_f for pair in edge_pairs], dtype=float)
    pair_ids = np.array([[pair.solid_edge[0], pair.solid_edge[1], pair.fluid_edge[0], pair.fluid_edge[1],
                          pair.fluid_midpoint, pair.solid_owner, pair.fluid_owner, pair.orientation]
                         for pair in edge_pairs], dtype=np.int64)
    topology_id = _digest(("solid_tag", solid_tag), ("fluid_tag", fluid_tag),
                          ("solid_nodes", solid_nodes), ("fluid_vertex_nodes", fluid_vertex_nodes),
                          ("fluid_p2_nodes", fluid_p2_nodes), ("pairs", pair_ids),
                          ("X_s", X_s_gamma), ("X_f", X_f_gamma))
    return InterfaceTopology(
        solid_mesh=solid_mesh, fluid_mesh=fluid_mesh, fluid_space=fluid_space, solid_tag=solid_tag,
        fluid_tag=fluid_tag, atol=float(atol), rtol=float(rtol), solid_nodes=solid_nodes,
        fluid_vertex_nodes=fluid_vertex_nodes, fluid_p2_nodes=fluid_p2_nodes, edge_pairs=tuple(edge_pairs),
        reference_lengths=reference_lengths, N_s=normals_s, N_f=normals_f, X_s_gamma=X_s_gamma,
        X_f_gamma=X_f_gamma, h=h, H=H, solid_dofs=vector_dofs(solid_nodes),
        fluid_p2_dofs=vector_dofs(fluid_p2_nodes), topology_id=topology_id,
        solid_mesh_id=_digest(("points", points_s), ("cells", np.asarray(solid_mesh.cells))),
        fluid_mesh_id=_digest(("points", points_f), ("cells", np.asarray(fluid_mesh.cells))))

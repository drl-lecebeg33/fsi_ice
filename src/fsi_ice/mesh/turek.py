"""Turek-Hron reference geometries; Gmsh is only a triangulation generator."""

from dataclasses import dataclass

from pathlib import Path
import shutil
import subprocess

import numpy as np

from fsi_ice.mesh.triangular import TriMesh, rectangle_mesh


@dataclass(frozen=True)
class FSIMeshPair:
    """Fluid/solid submeshes split from one shared-node Gmsh file.

    The global node arrays map local P1 vertices back to the original Gmsh
    identifiers.  They provide an independent check that the two wet traces
    really share reference vertices, instead of merely having close coordinates.
    """

    fluid: TriMesh
    solid: TriMesh
    fluid_global_nodes: np.ndarray
    solid_global_nodes: np.ndarray

    def __post_init__(self):
        for name, mesh in (("fluid_global_nodes", self.fluid),
                           ("solid_global_nodes", self.solid)):
            nodes = np.array(getattr(self, name), dtype=np.int64, copy=True)
            if nodes.shape != (mesh.n_nodes,) or len(np.unique(nodes)) != len(nodes):
                raise ValueError(f"{name} must uniquely identify every local mesh node.")
            nodes.setflags(write=False)
            object.__setattr__(self, name, nodes)


FSI1_BOUNDARIES = {
    "fluid": ("walls", "outlet", "inlet", "cylinder-rigid-wall", "fluid-interface"),
    "solid": ("solid-clamp", "solid-interface"),
}


def _gmsh2_sections(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if "$MeshFormat" not in lines or lines[lines.index("$MeshFormat")+1].split()[:2] != ["2.2", "0"]:
        raise ValueError("Require ASCII Gmsh MSH version 2.2.")
    for section in ("$PhysicalNames", "$Nodes", "$Elements"):
        if section not in lines:
            raise ValueError(f"Gmsh file is missing {section}.")
    start = lines.index("$PhysicalNames")+1
    names = {}
    for line in lines[start+1:start+1+int(lines[start])]:
        dimension, tag, name = line.split(maxsplit=2)
        key = (int(dimension), int(tag))
        if key in names:
            raise ValueError(f"Duplicate Gmsh physical tag {key}.")
        names[key] = name.strip('"')
    start = lines.index("$Nodes")+1
    node_lines = [line.split() for line in lines[start+1:start+1+int(lines[start])]]
    node_ids = np.array([int(line[0]) for line in node_lines], dtype=np.int64)
    if len(np.unique(node_ids)) != len(node_ids):
        raise ValueError("Duplicate Gmsh node identifier.")
    points = {int(line[0]): (float(line[1]), float(line[2])) for line in node_lines}
    if any(float(line[3]) != 0 for line in node_lines):
        raise ValueError("Only planar z=0 meshes are supported.")
    start = lines.index("$Elements")+1
    elements = []
    for line in lines[start+1:start+1+int(lines[start])]:
        fields = list(map(int, line.split()))
        kind, ntags = fields[1:3]
        if ntags < 1:
            raise ValueError("Every imported Gmsh element needs a physical tag.")
        nodes = fields[3+ntags:]
        expected = {1: 2, 2: 3, 15: 1}.get(kind)
        if expected is None:
            raise ValueError(f"Unsupported Gmsh element type {kind}.")
        if len(nodes) != expected or any(node not in points for node in nodes):
            raise ValueError("Invalid Gmsh element connectivity.")
        if kind == 15:
            continue  # Gmsh endpoint records are irrelevant to the P1 domain/boundary import.
        physical = names.get(({1: 1, 2: 2}[kind], fields[3]))
        if physical is None:
            raise ValueError("Gmsh element references an unnamed physical tag.")
        elements.append((kind, physical, nodes))
    return points, elements


def _submesh(points, triangle_rows, boundary_rows):
    global_nodes = np.unique(np.asarray(triangle_rows, dtype=np.int64))
    mapping = {int(node): local for local, node in enumerate(global_nodes)}
    coordinates = np.array([points[int(node)] for node in global_nodes])
    cells = np.array([[mapping[node] for node in row] for row in triangle_rows], dtype=np.int64)
    corners = coordinates[cells]
    first, second = corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0]
    clockwise = first[:, 0]*second[:, 1]-first[:, 1]*second[:, 0] < 0
    cells[clockwise] = cells[clockwise][:, [0, 2, 1]]
    boundaries = {}
    for tag, rows in boundary_rows.items():
        try:
            boundaries[tag] = np.array([[mapping[node] for node in row] for row in rows], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f"Boundary {tag} contains a node outside its physical surface.") from error
    return TriMesh(coordinates, cells, boundaries), global_nodes


def read_fsi_gmsh2(path):
    """Read one MSH2 file containing the named FSI1 fluid and solid surfaces."""
    points, elements = _gmsh2_sections(path)
    triangles = {"fluid": [], "solid": []}
    edges = {}
    for kind, physical, nodes in elements:
        if kind == 2:
            if physical not in triangles:
                raise ValueError(f"Unknown physical surface {physical!r}; require fluid/solid.")
            triangles[physical].append(nodes)
        elif kind == 1:
            edges.setdefault(physical, []).append(nodes)
    for domain in triangles:
        if not triangles[domain]:
            raise ValueError(f"Gmsh file contains no {domain} triangles.")
    for domain in triangles:
        missing = set(FSI1_BOUNDARIES[domain])-edges.keys()
        if missing:
            raise ValueError(f"Missing {domain} boundary tags: {sorted(missing)}")
    fluid, fluid_ids = _submesh(points, triangles["fluid"],
                                {tag: edges[tag] for tag in FSI1_BOUNDARIES["fluid"]})
    solid, solid_ids = _submesh(points, triangles["solid"],
                                {tag: edges[tag] for tag in FSI1_BOUNDARIES["solid"]})
    return FSIMeshPair(fluid, solid, fluid_ids, solid_ids)


def fsi1_geometry(size):
    """Return the tagged shared-geometry Gmsh script for the FSI1 benchmark."""
    if isinstance(size, bool) or not isinstance(size, (int, float, np.integer, np.floating)) \
            or not np.isfinite(size) or size <= 0:
        raise ValueError("Mesh size must be finite and positive.")
    attachment = 0.2+np.sqrt(0.05**2-0.01**2)
    return f'''Mesh.MshFileVersion = 2.2;
Mesh.Algorithm = 6;
Mesh.ElementOrder = 1;
lc = {float(size):.17g};
Point(1) = {{0,0,0,4*lc}}; Point(2) = {{2.5,0,0,4*lc}};
Point(3) = {{2.5,0.41,0,4*lc}}; Point(4) = {{0,0.41,0,4*lc}};
Line(1) = {{1,2}}; Line(2) = {{2,3}}; Line(3) = {{3,4}}; Line(4) = {{4,1}};
Point(5) = {{0.2,0.2,0,lc}};
Point(6) = {{{attachment:.17g},0.21,0,lc/2}}; Point(7) = {{0.2,0.25,0,lc}};
Point(8) = {{0.15,0.2,0,lc}}; Point(9) = {{0.2,0.15,0,lc}};
Point(10) = {{{attachment:.17g},0.19,0,lc/2}};
Point(11) = {{0.6,0.19,0,lc/2}}; Point(12) = {{0.6,0.2,0,lc/2}};
Point(13) = {{0.6,0.21,0,lc/2}}; Point(14) = {{0.25,0.2,0,lc/2}};
Circle(5) = {{6,5,7}}; Circle(6) = {{7,5,8}}; Circle(7) = {{8,5,9}}; Circle(8) = {{9,5,10}};
Line(9) = {{10,11}}; Line(10) = {{11,12}}; Line(11) = {{12,13}}; Line(12) = {{13,6}};
Circle(13) = {{6,5,14}}; Circle(14) = {{14,5,10}};
Curve Loop(1) = {{1,2,3,4}}; Curve Loop(2) = {{5,6,7,8,9,10,11,12}};
Curve Loop(3) = {{9,10,11,12,13,14}};
Plane Surface(1) = {{1,2}}; Plane Surface(2) = {{3}};
Physical Curve("walls",1) = {{1,3}}; Physical Curve("outlet",2) = {{2}};
Physical Curve("inlet",3) = {{4}}; Physical Curve("fluid-interface",4) = {{9,10,11,12}};
Physical Curve("solid-interface",5) = {{9,10,11,12}};
Physical Curve("cylinder-rigid-wall",6) = {{5,6,7,8}}; Physical Curve("solid-clamp",7) = {{13,14}};
Physical Surface("fluid",101) = {{1}}; Physical Surface("solid",102) = {{2}};
Field[1] = Distance; Field[1].CurvesList = {{5,6,7,8,9,10,11,12}}; Field[1].Sampling = 200;
Field[2] = Threshold; Field[2].InField = 1; Field[2].SizeMin = lc/2; Field[2].SizeMax = 4*lc;
Field[2].DistMin = 0.03; Field[2].DistMax = 0.25; Background Field = 2;
'''


def fsi1_mesh(size, output):
    """Generate and import a matching FSI1 fluid/solid mesh pair."""
    executable = shutil.which("gmsh")
    if executable is None:
        raise RuntimeError("FSI1 benchmark meshing requires the Gmsh executable.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    geometry_path, mesh_path = output / "fsi1.geo", output / "fsi1.msh"
    geometry_path.write_text(fsi1_geometry(size), encoding="utf-8")
    process = subprocess.run([executable, str(geometry_path), "-2", "-format", "msh2",
                              "-o", str(mesh_path), "-v", "2"], capture_output=True,
                             text=True, timeout=120)
    (output / "gmsh.log").write_text(process.stdout+process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"Gmsh failed: {process.stderr.strip()}")
    return read_fsi_gmsh2(mesh_path)


def csm_mesh(nx, ny):
    base = rectangle_mesh(nx, ny, width=0.35, height=0.02)
    points = base.points.copy()
    y = points[:, 1]+0.19
    attachment = 0.2+np.sqrt(0.05**2-(y-0.2)**2)
    points[:, 0] = attachment+(points[:, 0]/0.35)*(0.6-attachment)
    points[:, 1] = y
    return TriMesh(points, base.cells, base.boundaries)


def read_gmsh2(path):
    """Read ASCII MSH2: physical line tags and first-order triangular cells."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if "$MeshFormat" not in lines or lines[lines.index("$MeshFormat")+1].split()[:2] != ["2.2", "0"]:
        raise ValueError("Require ASCII Gmsh MSH version 2.2.")
    start = lines.index("$PhysicalNames")+1
    names = {}
    for line in lines[start+1:start+1+int(lines[start])]:
        dimension, tag, name = line.split(maxsplit=2)
        names[(int(dimension), int(tag))] = name.strip('"')
    start = lines.index("$Nodes")+1
    node_lines = [line.split() for line in lines[start+1:start+1+int(lines[start])]]
    mapping = {int(line[0]): index for index, line in enumerate(node_lines)}
    points = np.array([[float(line[1]), float(line[2])] for line in node_lines])
    if any(float(line[3]) != 0 for line in node_lines):
        raise ValueError("Only planar z=0 meshes are supported.")
    start = lines.index("$Elements")+1
    cells, boundaries = [], {}
    for line in lines[start+1:start+1+int(lines[start])]:
        fields = list(map(int, line.split()))
        kind, ntags = fields[1:3]
        nodes = [mapping[node] for node in fields[3+ntags:]]
        if kind == 2:
            cells.append(nodes)
        elif kind == 1:
            name = names[(1, fields[3])]
            boundaries.setdefault(name, []).append(nodes)
        elif kind != 15:
            raise ValueError(f"Unsupported Gmsh element type {kind}.")
    cells = np.array(cells, dtype=int)
    if cells.size == 0:
        raise ValueError("Gmsh file contains no triangles.")
    # Import policy: normalize clockwise cells explicitly before TriMesh validation.
    corners = points[cells]
    a, b = corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0]
    clockwise = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0] < 0
    cells[clockwise] = cells[clockwise][:, [0, 2, 1]]
    return TriMesh(points, cells, boundaries)


def cfd_mesh(size, output):
    if not np.isfinite(size) or size <= 0:
        raise ValueError("Mesh size must be finite and positive.")
    executable = shutil.which("gmsh")
    if executable is None:
        raise RuntimeError("CFD benchmark meshing requires the Gmsh executable.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    attach = 0.2+np.sqrt(0.05**2-0.01**2)
    # Four circle arcs plus the fixed flag define one closed obstacle boundary.
    geo = f'''Mesh.MshFileVersion = 2.2;
Mesh.Algorithm = 6;
lc = {size:.17g};
Point(1) = {{0,0,0,4*lc}}; Point(2) = {{2.5,0,0,4*lc}};
Point(3) = {{2.5,0.41,0,4*lc}}; Point(4) = {{0,0.41,0,4*lc}};
Line(1) = {{1,2}}; Line(2) = {{2,3}}; Line(3) = {{3,4}}; Line(4) = {{4,1}};
Point(5) = {{0.2,0.2,0,lc}};
Point(6) = {{{attach:.17g},0.21,0,lc}}; Point(7) = {{0.2,0.25,0,lc}};
Point(8) = {{0.15,0.2,0,lc}}; Point(9) = {{0.2,0.15,0,lc}};
Point(10) = {{{attach:.17g},0.19,0,lc}};
Point(11) = {{0.6,0.19,0,lc}}; Point(12) = {{0.6,0.2,0,lc}}; Point(13) = {{0.6,0.21,0,lc}};
Circle(5) = {{6,5,7}}; Circle(6) = {{7,5,8}}; Circle(7) = {{8,5,9}}; Circle(8) = {{9,5,10}};
Line(9) = {{10,11}}; Line(10) = {{11,12}}; Line(11) = {{12,13}}; Line(12) = {{13,6}};
Curve Loop(1) = {{1,2,3,4}}; Curve Loop(2) = {{5,6,7,8,9,10,11,12}};
Plane Surface(1) = {{1,2}};
Physical Curve("walls",1) = {{1,3}}; Physical Curve("outlet",2) = {{2}};
Physical Curve("inlet",3) = {{4}}; Physical Curve("obstacle",4) = {{5,6,7,8,9,10,11,12}};
Physical Surface("fluid",5) = {{1}};
Field[1] = Distance; Field[1].CurvesList = {{5,6,7,8,9,10,11,12}}; Field[1].Sampling = 100;
Field[2] = Threshold; Field[2].InField = 1; Field[2].SizeMin = lc;
Field[2].SizeMax = 4*lc; Field[2].DistMin = 0.03; Field[2].DistMax = 0.25;
Background Field = 2;
'''
    path = output / "mesh.geo"
    path.write_text(geo, encoding="utf-8")
    mesh_path = output / "mesh.msh"
    process = subprocess.run([executable, str(path), "-2", "-format", "msh2", "-o", str(mesh_path), "-v", "2"],
                             capture_output=True, text=True, timeout=120)
    (output / "gmsh.log").write_text(process.stdout+process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"Gmsh failed: {process.stderr.strip()}")
    return read_gmsh2(mesh_path)

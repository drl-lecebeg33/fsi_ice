"""网格子包：P1 三角形网格、矩形网格与 Turek–Hron 基准几何。

对外导出 :class:`TriMesh`（带边界标签的逆时针 P1 三角形网格）、
:func:`rectangle_mesh`（矩形区域三角化），以及 FSI1 基准所需的
:class:`FSIMeshPair`、:func:`fsi1_geometry`、:func:`fsi1_mesh`、:func:`read_fsi_gmsh2`。
"""

from .triangular import TriMesh, rectangle_mesh
from .turek import FSIMeshPair, fsi1_geometry, fsi1_mesh, read_fsi_gmsh2

__all__ = ["FSIMeshPair", "TriMesh", "fsi1_geometry", "fsi1_mesh", "read_fsi_gmsh2", "rectangle_mesh"]

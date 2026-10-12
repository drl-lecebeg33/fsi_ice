"""ALE mesh layer: harmonic extension, candidate geometry, velocity and swept flux.

本子包为 ALE 网格层，负责参考网格到候选网格的运动表示：调和（Laplace）扩展
（harmonic）、候选几何（geometry）、网格速度（kinematics）与几何守恒律扫掠通量
（gcl）。整体供流体 ALE 装配复用。
"""

from .gcl import GCLTransport, build_gcl_transport
from .geometry import ALEGeometry, build_ale_geometry
from .harmonic import HarmonicExtension, MeshDisplacement, build_harmonic_extension
from .kinematics import MeshVelocity, evaluate_mesh_velocity

__all__ = [
    "HarmonicExtension",
    "build_harmonic_extension",
    "MeshDisplacement",
    "ALEGeometry",
    "build_ale_geometry",
    "MeshVelocity",
    "evaluate_mesh_velocity",
    "GCLTransport",
    "build_gcl_transport",
]

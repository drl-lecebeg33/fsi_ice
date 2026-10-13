"""Reference interface topology, motion/load mapping, current geometry and traction.

本包汇集流固耦合湿界面的固定参照拓扑、运动与对偶载荷映射、当前构形几何以及
界面载荷（traction）装配。关键约定：参照构形只匹配一次，形变构形不再重匹配；
界面自由度按交错顺序 ``[x0, y0, x1, y1, ...]`` 存储；法向取各子域外法向、界面
载荷正向为流体作用在固体上；所有公开对象均为不可变（只读数组）。
"""

from fsi_ice.interface.checks import interface_diagnostics
from fsi_ice.interface.geometry import InterfaceGeometry, evaluate_interface_geometry
from fsi_ice.interface.mapping import InterfaceMotion, map_load, map_motion, scatter_solid_load
from fsi_ice.interface.topology import InterfaceEdge, InterfaceTopology, build_matching_interface
from fsi_ice.interface.traction import (InterfaceLoad, InterfaceLoadDerivative, assemble_interface_load,
                                        interface_load_jvp, pullback_traction)

__all__ = ["InterfaceEdge", "InterfaceTopology", "build_matching_interface", "InterfaceMotion",
           "map_motion", "map_load", "scatter_solid_load", "InterfaceGeometry", "evaluate_interface_geometry",
           "InterfaceLoad", "InterfaceLoadDerivative", "assemble_interface_load", "interface_load_jvp",
           "pullback_traction", "interface_diagnostics"]

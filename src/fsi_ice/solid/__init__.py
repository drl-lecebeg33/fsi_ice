"""Self-developed two-dimensional solid mechanics.

本子包实现自研二维固体力学：``linear`` 提供小变形线弹性（平面应力/平面应变、
常应变 P1 单元与一致体力/面力），``stvk`` 提供总拉格朗日 StVK 有限变形与一致切线，
``dynamics`` 提供基于 BDF 的固体时间步。位移自由度按节点交错排列为
``[ux0, uy0, ux1, uy1, ...]``。
"""

from .linear import ElasticMaterial, assemble_elasticity, solve_elasticity

__all__ = ["ElasticMaterial", "assemble_elasticity", "solve_elasticity"]

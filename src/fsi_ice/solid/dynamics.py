"""BDF discretization of d'=v, rho*v'=Div(P)+f on a fixed reference mesh.

本模块在固定的参考网格上对固体做 BDF 时间离散，方程组为位移微分方程 d'=v 与
动量方程 ρ v' = Div(P) + f。借助 BDF 消去速度，得到以位移为未知量的非线性方程，
交由 :func:`fsi_ice.solid.stvk.solve_stvk` 求解，并返回新的 (位移, 速度) 历史。
"""

import numpy as np

from fsi_ice.solid.stvk import assemble_mass, solve_stvk


def solve_solid_step(mesh, material, external, fixed_nodes, context, *, initial=None, **kwargs):
    """推进一个 BDF 固体时间步。

    中文：``context`` 需提供 ``history_term``（形状 (2,n_nodes,2)，依次为位移历史项
    hd 与速度历史项 hv）及 ``coefficient``（BDF 系数 α）。消去速度后加速度写作
    α²d + α·hd + hv；返回值第一项为新历史 (位移, 速度) 的堆叠数组，第二项为
    :class:`~fsi_ice.solid.stvk.NonlinearSolidSolution` 解对象。
    """
    if context.history_term.shape != (2, mesh.n_nodes, 2):
        raise ValueError("Solid history shape must be (2,n_nodes,2) for displacement and velocity.")
    alpha = context.coefficient
    hd, hv = context.history_term
    mass = assemble_mass(mesh, material)
    # Eliminating v gives acceleration alpha^2*d + alpha*hd + hv.
    # 中文：把惯性项移到残差中，符号取负构成质量项右端 mass_rhs。
    history_rhs = -(mass @ (alpha*hd+hv).ravel())
    solution = solve_stvk(mesh, material, external, fixed_nodes, initial=initial,
                          mass_coefficient=alpha**2, mass_rhs=history_rhs, **kwargs)
    # 中文：由 BDF 关系回代得到本步速度 v = α·d + hd。
    velocity = alpha*solution.displacement+hd
    return np.stack((solution.displacement, velocity)), solution

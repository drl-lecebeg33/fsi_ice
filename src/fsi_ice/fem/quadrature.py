"""Positive-weight quadrature on {(xi, eta): xi, eta >= 0, xi+eta <= 1}.

本模块提供参考三角形上的正权求积规则，用于弱形式积分。约定参考三角形顶点为
(0,0)、(1,0)、(0,1)，其面积为 1/2。支持 1、2、5 阶；返回的积分点与权重一一
对应，权重均为正，权重之和恒等于参考面积 1/2，因此物理积分需再乘以单元
Jacobian 行列式。
"""

import numpy as np


def triangle_quadrature(degree=5):
    """Return points and weights whose sum is the reference area 1/2.

    中文说明：按 ``degree`` 返回 (points, weights)。``points`` 形状 (q,2)，是参考
    三角形内的积分点；``weights`` 形状 (q,)，各点权重为正且总和为 1/2。支持的
    阶数为 1、2、5。
    """
    if degree == 1:
        # 1 阶：仅取重心 (1/3,1/3)，权重等于参考面积 1/2。
        return np.array([[1/3, 1/3]]), np.array([0.5])
    if degree == 2:
        # 2 阶：三条边中点，每点权重 1/6（三点合计 1/2）。
        return np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]]), np.full(3, 1/6)
    if degree == 5:
        # 5 阶 Dunavant 型规则：1 个重心点 + 两组各 3 个对称点；
        # 权重已乘 0.5 面积因子，括号内为归一化权重。
        a, b = 0.059715871789770, 0.470142064105115
        c, d = 0.797426985353087, 0.101286507323456
        points = np.array([[1/3, 1/3], [b, b], [a, b], [b, a],
                           [d, d], [c, d], [d, c]])
        weights = 0.5 * np.array([0.225, *([0.132394152788506] * 3),
                                 *([0.125939180544827] * 3)])
        return points, weights
    raise ValueError("Supported triangle quadrature degrees are 1, 2 and 5.")

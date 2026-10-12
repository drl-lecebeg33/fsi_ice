"""Self-developed element integration, sparse assembly and constraints.

本子包提供自研有限元基础构件：P1/P2 形函数与参考单元梯度（``p1``、``p2``）、
参考三角形上的正权求积（``quadrature``），以及 P1 Poisson 弱形式组装、Dirichlet
约束消元与误差范数（``poisson``）。约定参考三角形为 :math:`\\xi,\\eta\\ge 0`、
:math:`\\xi+\\eta\\le 1`，其面积为 1/2；稀疏矩阵统一使用 COO 组装后转 CSR。
"""

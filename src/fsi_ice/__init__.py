"""Finite-element building blocks for reproducible FSI research.

本包是自研的二维有限元流固耦合（FSI）研究平台顶层包，按领域拆分为若干子包：
``mesh`` 负责 P1 三角形网格与 Turek–Hron 基准几何；``fem`` 提供 P1/P2 形函数、
三角形求积、稀疏组装与 Dirichlet 求解；``solid`` 实现小变形线弹性与 StVK 有限变形
固体。顶层仅暴露包版本号，具体功能由各子包导入。
"""

__version__ = "0.1.0"

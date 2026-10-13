"""Self-developed Taylor-Hood incompressible-flow discretization.

本子包提供自研的 Taylor-Hood P2/P1 不可压缩流离散与其时间推进：算子装配
（taylor_hood）、ALE 流场上下文（ale）、边界力/流量（forces）、线性求解器
（linear_solver）与 BDF 时间步入口（dynamics）。
"""

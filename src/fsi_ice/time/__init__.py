"""Compatibility facade exposing the BDF time-history primitives.

中文：``fsi_ice.time`` 包的对外导出入口，仅转发 ``BDFHistory`` 与 ``BDFStencil`` 两个
时间模板/历史基元；事务层（transaction.py）通过 ``fsi_ice.time.bdf`` 直接导入，不在此重复导出。
"""
from .bdf import BDFHistory, BDFStencil

__all__ = ["BDFHistory", "BDFStencil"]

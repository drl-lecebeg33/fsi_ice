"""Interface motion and dual load mapping through the fixed reference ``H``.

Mapping accepts plain state arrays only: it never reads, advances or accepts a
time history, and it never modifies its inputs.
"""

from dataclasses import dataclass

import numpy as np

from fsi_ice.interface.topology import InterfaceTopology


def _read_only(array):
    array.setflags(write=False)
    return array


def _finite_vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array.astype(float, copy=True)


@dataclass(frozen=True)
class InterfaceMotion:
    """Mapped interface motion for one candidate solid state.

    ``vertex_values`` gathers the solid interface P1 trace, ``p2_values`` is
    the fluid P2 trace ``H`` applied to the compact solid trace, and
    ``compact_values``/``solid_values`` keep the interleaved compact and full
    solid vectors for diagnostics.  All arrays are read-only.
    """

    topology_id: str
    solid_values: np.ndarray
    compact_values: np.ndarray
    vertex_values: np.ndarray
    p2_values: np.ndarray


def map_motion(topology, solid_values):
    """Map full solid displacement/velocity ``(n_s_nodes, 2)`` to the interface.

    The output ``vertex_values`` is the gathered solid P1 trace and
    ``p2_values`` is the fluid P2 trace; every fluid mid-edge value is the
    average of its two matched solid endpoints.  Histories are neither read
    nor accepted, and ``solid_values`` is copied before use.
    """
    if not isinstance(topology, InterfaceTopology):
        raise TypeError("topology must be an InterfaceTopology.")
    values = np.asarray(solid_values, dtype=float)
    expected = (topology.n_solid_nodes, 2)
    if values.shape != expected:
        raise ValueError(f"solid_values must have shape {expected}, got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("solid_values must be finite.")
    values = np.array(values, dtype=float, copy=True)
    compact = np.array(values.ravel()[topology.solid_dofs], copy=True)
    vertex_values = np.array(values[topology.solid_nodes], copy=True)
    p2_values = np.asarray(topology.H @ compact).reshape(topology.n_fluid_interface_nodes, 2)
    return InterfaceMotion(topology_id=topology.topology_id, solid_values=_read_only(values),
                           compact_values=_read_only(compact), vertex_values=_read_only(vertex_values),
                           p2_values=_read_only(np.array(p2_values, copy=True)))


def map_load(topology, fluid_loads):
    """Map integrated fluid dual nodal forces to compact solid interface forces.

    ``fluid_loads`` must be the already integrated dual nodal force vector on
    the fluid P2 interface trace, interleaved as ``[x0, y0, x1, y1, ...]`` and
    ordered by ``topology.fluid_p2_nodes``.  It is *not* a nodal traction or
    traction-coefficient vector: the measure, quadrature weights and thickness
    must already be included, otherwise the transferred load silently changes.
    The output is ``H.T @ fluid_loads`` on ``topology.solid_nodes`` in the
    same interleaved convention.
    """
    if not isinstance(topology, InterfaceTopology):
        raise TypeError("topology must be an InterfaceTopology.")
    loads = _finite_vector(fluid_loads, 2 * topology.n_fluid_interface_nodes, "fluid_loads")
    return _read_only(np.asarray(topology.H.T @ loads).astype(float, copy=True))


def scatter_solid_load(topology, compact):
    """Scatter a compact solid interface force into the full solid vector.

    Non-interface entries are zero, so the result can be added to the body
    force to form the ``external`` vector consumed by the solid residual.
    """
    if not isinstance(topology, InterfaceTopology):
        raise TypeError("topology must be an InterfaceTopology.")
    values = _finite_vector(compact, 2 * topology.n_solid_interface_nodes, "compact")
    full = np.zeros(2 * topology.n_solid_nodes, dtype=float)
    full[topology.solid_dofs] = values
    return _read_only(full)

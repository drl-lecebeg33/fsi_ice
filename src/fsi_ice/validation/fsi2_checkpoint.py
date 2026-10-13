"""Atomic, pickle-free persistence of an accepted FSI2 root and BDF history."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from fsi_ice.time.transaction import AcceptedState


SCHEMA = "fsi2-checkpoint-v1"


def source_digest():
    """Exact package source identity; documentation and tests are excluded."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode()+b"\0"+path.read_bytes()+b"\0")
    return digest.hexdigest()


def _array_digest(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def save_checkpoint(path, state, *, config, mesh_sha256, code_sha256):
    """Save after publish. A failed write/replacement preserves the older file."""
    if not isinstance(state, AcceptedState):
        raise TypeError("Checkpoint requires an AcceptedState.")
    arrays = {"solid_history": np.stack(state.solid_history),
              "fluid_history": np.stack(state.fluid_history),
              "mesh_history": np.stack(state.mesh_history), "pressure": state.pressure,
              "mesh_current": state.mesh_current, "mesh_velocity": state.mesh_velocity}
    keys = {}
    for group in ("interface", "predictor"):
        values = getattr(state, group)
        keys[group] = sorted(values)
        arrays.update({f"{group}_{index}": values[key] for index, key in enumerate(keys[group])})
    metadata = {"schema": SCHEMA, "config": config, "mesh_sha256": mesh_sha256,
        "code_sha256": code_sha256, "time": state.time, "dt": state.dt,
        "order": state.order, "step": state.step, "epoch": state.epoch,
        "generations": dict(state.generations), "shapes": dict(state.shapes),
        "keys": keys, "monitoring": state.monitoring,
        "arrays": {key: _array_digest(value) for key, value in arrays.items()}}
    arrays["metadata"] = np.array(json.dumps(metadata, allow_nan=False, ensure_ascii=False))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", suffix=".npz",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path, *, config, mesh_sha256, code_sha256):
    """Validate everything locally; callers load the root only after shape checks."""
    with np.load(path, allow_pickle=False) as data:
        if "metadata" not in data:
            raise ValueError("Not a complete FSI2 checkpoint (solution.npz cannot resume BDF).")
        meta = json.loads(str(data["metadata"].item()))
        required = {"schema", "config", "mesh_sha256", "code_sha256", "time", "dt", "order",
                    "step", "epoch", "generations", "shapes", "keys", "monitoring", "arrays"}
        if not isinstance(meta, dict) or set(meta) != required or meta["schema"] != SCHEMA:
            raise ValueError("Unsupported or malformed FSI2 checkpoint metadata.")
        original, requested = dict(meta["config"]), dict(config)
        original_end, requested_end = original.pop("end_time"), requested.pop("end_time")
        if original != requested or requested_end < original_end:
            raise ValueError("Checkpoint config differs; only end_time extension is allowed.")
        if meta["mesh_sha256"] != mesh_sha256 or meta["code_sha256"] != code_sha256:
            raise ValueError("Checkpoint mesh/source identity does not match this run.")
        if not isinstance(meta["keys"], dict) or set(meta["keys"]) != {"interface", "predictor"} or \
                not isinstance(meta["arrays"], dict):
            raise ValueError("Malformed checkpoint array/mapping manifest.")
        if set(data.files) != {"metadata", *meta["arrays"]}:
            raise ValueError("Checkpoint arrays do not match its manifest.")
        arrays = {key: np.array(data[key], copy=True) for key in meta["arrays"]}
    for key, value in arrays.items():
        if value.dtype.kind not in "fiu" or not np.all(np.isfinite(value)) or \
                _array_digest(value) != meta["arrays"][key]:
            raise ValueError(f"Invalid checkpoint array/checksum: {key}.")
    step = meta["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0 or meta["epoch"] != step or \
            meta["dt"] != config["dt"] or meta["order"] != config["order"] or \
            not np.isfinite(meta["time"]) or \
            not np.isclose(meta["time"], step*config["dt"], rtol=0, atol=1e-10) or \
            meta["time"] > config["end_time"]+1e-10 or \
            meta["generations"] != dict.fromkeys(("solid", "fluid", "mesh"), step) or \
            any(isinstance(value, bool) or not isinstance(value, int) for value in meta["generations"].values()):
        raise ValueError("Checkpoint time/order/generations are inconsistent.")
    layers = min(step+1, config["order"])
    if any(arrays[name].ndim < 2 or arrays[name].shape[0] != layers
           for name in ("solid_history", "fluid_history", "mesh_history")):
        raise ValueError("Checkpoint is missing required accepted BDF history layers.")
    rows = meta["monitoring"]
    if not isinstance(rows, list) or len(rows) != step:
        raise ValueError("Checkpoint monitoring ledger has missing/duplicate steps.")
    for index, row in enumerate(rows, 1):
        if row["step"] != index or not np.isclose(row["time"], index*config["dt"], rtol=0, atol=1e-10) or \
                row["method"] != config["method"] or row["order"] != (1 if index == 1 else config["order"]):
            raise ValueError("Checkpoint monitoring order/time/method is inconsistent.")
    mappings = {}
    for group in ("interface", "predictor"):
        names = meta["keys"][group]
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
            raise ValueError("Invalid checkpoint mapping keys.")
        mappings[group] = {key: arrays[f"{group}_{index}"] for index, key in enumerate(names)}
    state = AcceptedState(time=meta["time"], dt=meta["dt"], order=meta["order"], step=step, epoch=meta["epoch"],
        solid_states=arrays["solid_history"], fluid_states=arrays["fluid_history"], mesh_states=arrays["mesh_history"],
        solid_generation=step, fluid_generation=step, mesh_generation=step,
        pressure=arrays["pressure"], mesh_current=arrays["mesh_current"], mesh_velocity=arrays["mesh_velocity"],
        interface=mappings["interface"], predictor=mappings["predictor"], monitoring=rows)
    if {key: list(value) for key, value in state.shapes.items()} != meta["shapes"]:
        raise ValueError("Checkpoint field shapes differ from its accepted-state manifest.")
    return state

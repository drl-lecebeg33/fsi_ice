"""Full accepted-history restart, identity guards and durable failure recovery."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import fsi_ice.validation.fsi2 as module
import fsi_ice.validation.fsi2_checkpoint as checkpoint
from fsi_ice.validation.fsi2 import load_fsi2_config, run_fsi2_cli
from test_fsi2 import small_problem, COUPLING, NEWTON

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = {"mesh_sha256": "a"*64, "code_sha256": "b"*64}


def config_for(method="iqn", order=2):
    config = load_fsi2_config(ROOT/"cases/fsi2.json")
    config.update(dt=.05, end_time=.2, method=method, order=order)
    return config


def rewrite(path, mutate):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    meta = json.loads(str(arrays["metadata"].item()))
    mutate(meta, arrays)
    arrays["metadata"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)


@pytest.mark.parametrize("method", ["iqn", "monolithic"])
@pytest.mark.parametrize("order", [1, 2])
def test_restart_preserves_bdf_histories_and_four_step_solution(tmp_path, method, order):
    direct, first, resumed = small_problem(order), small_problem(order), small_problem(order)
    config = config_for(method, order)
    for _ in range(4):
        direct.step(method, coupling=COUPLING, newton=NEWTON)
    for _ in range(2):
        first.step(method, coupling=COUPLING, newton=NEWTON)
    path = tmp_path/"checkpoint.npz"
    checkpoint.save_checkpoint(path, first.coordinator.state, config=config, **IDENTITY)
    loaded = checkpoint.load_checkpoint(path, config=config, **IDENTITY)
    resumed.coordinator.load(loaded)
    for _ in range(2):
        state = resumed.step(method, coupling=COUPLING, newton=NEWTON)
        assert state.monitoring[-1]["order"] == order
    expected = direct.coordinator.state
    np.testing.assert_array_equal(resumed._initial_vector(state), direct._initial_vector(expected))
    for field in ("solid_history", "fluid_history", "mesh_history"):
        np.testing.assert_array_equal(getattr(state, field), getattr(expected, field))
    np.testing.assert_array_equal(state.mesh_velocity, expected.mesh_velocity)
    assert state.generations == expected.generations == dict.fromkeys(("solid", "fluid", "mesh"), 4)
    assert [row["step"] for row in state.monitoring] == [1, 2, 3, 4]


@pytest.fixture
def saved(tmp_path):
    driver = small_problem()
    config = config_for()
    driver.step("iqn", coupling=COUPLING, newton=NEWTON)
    path = tmp_path/"checkpoint.npz"
    checkpoint.save_checkpoint(path, driver.coordinator.state, config=config, **IDENTITY)
    return driver, config, path


@pytest.mark.parametrize("key,value", [("mesh_sha256", "c"*64), ("code_sha256", "d"*64)])
def test_identity_mismatch_rejected_without_changing_live_state(saved, key, value):
    driver, config, path = saved
    before = driver.coordinator.state
    with pytest.raises(ValueError, match="identity"):
        checkpoint.load_checkpoint(path, config=config, **{**IDENTITY, key: value})
    assert driver.coordinator.state is before


@pytest.mark.parametrize("field", ["dt", "order", "method", "solver", "end_time"])
def test_incompatible_config_rejected(saved, field):
    _, config, path = saved
    config = copy.deepcopy(config)
    config[field] = {"dt": .1, "order": 1, "method": "aitken", "solver": {}, "end_time": .1}[field]
    with pytest.raises(ValueError, match="config differs"):
        checkpoint.load_checkpoint(path, config=config, **IDENTITY)


def test_only_end_time_extension_is_allowed(saved):
    _, config, path = saved
    config = {**config, "end_time": .4}
    assert checkpoint.load_checkpoint(path, config=config, **IDENTITY).step == 1


@pytest.mark.parametrize("mutation", ["schema", "time", "generation", "ledger", "shape", "checksum", "layers"])
def test_corrupted_checkpoint_cannot_load(saved, mutation):
    driver, config, path = saved
    before = driver.coordinator.state

    def mutate(meta, arrays):
        if mutation == "schema":
            meta["schema"] = "future-format"
        elif mutation == "time":
            meta["time"] += .01
        elif mutation == "generation":
            meta["generations"]["fluid"] += 1
        elif mutation == "ledger":
            meta["monitoring"] *= 2
        elif mutation == "shape":
            arrays["solid_history"] = arrays["solid_history"][:, :, :-1]
            meta["arrays"]["solid_history"] = checkpoint._array_digest(arrays["solid_history"])
        elif mutation == "checksum":
            arrays["pressure"][0] += 1
        else:
            arrays["mesh_history"] = arrays["mesh_history"][:1]
            meta["arrays"]["mesh_history"] = checkpoint._array_digest(arrays["mesh_history"])

    rewrite(path, mutate)
    with pytest.raises(ValueError):
        checkpoint.load_checkpoint(path, config=config, **IDENTITY)
    assert driver.coordinator.state is before


def test_atomic_replace_failure_keeps_previous_checkpoint(saved, monkeypatch):
    driver, config, path = saved
    old = path.read_bytes()
    driver.step("iqn", coupling=COUPLING, newton=NEWTON)

    def fail(*args):
        raise OSError("injected replace failure")

    monkeypatch.setattr(checkpoint.os, "replace", fail)
    with pytest.raises(OSError, match="replace failure"):
        checkpoint.save_checkpoint(path, driver.coordinator.state, config=config, **IDENTITY)
    assert path.read_bytes() == old
    assert checkpoint.load_checkpoint(path, config=config, **IDENTITY).step == 1
    assert not list(path.parent.glob(".checkpoint-*.npz"))


def test_legacy_final_solution_is_not_a_restart(saved, tmp_path):
    _, config, _ = saved
    path = tmp_path/"solution.npz"
    np.savez_compressed(path, velocity=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="complete FSI2 checkpoint"):
        checkpoint.load_checkpoint(path, config=config, **IDENTITY)


def fake_builder(config, output):
    (output/"mesh").mkdir(exist_ok=True)
    (output/"mesh"/"fsi1.msh").write_bytes(b"checkpoint tests: small FE geometry")
    return small_problem(config["order"])


def test_cli_chunks_restore_ledger_and_final_solution(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "build_fsi2_problem", fake_builder)
    config = config_for()
    path = tmp_path/"config.json"
    path.write_text(json.dumps(config))
    folder = tmp_path/"chunks"
    a = run_fsi2_cli(path, folder, max_steps=2, progress=None)
    assert a["status"] == "INCOMPLETE" and not a["checks"]["target_time_complete"]
    b = run_fsi2_cli(path, folder, resume=folder/"checkpoint.npz", max_steps=2, progress=None)
    c = run_fsi2_cli(path, tmp_path/"direct", progress=None)
    assert b["resumed_steps"] == 2 and b["accepted_steps"] == c["accepted_steps"] == 4
    assert b["checks"]["target_time_complete"]
    with np.load(folder/"solution.npz") as x, np.load(tmp_path/"direct/solution.npz") as y:
        for name in x.files:
            np.testing.assert_array_equal(x[name], y[name])
    assert len((folder/"monitor.csv").read_text().splitlines()) == 5


def test_physical_failure_preserves_last_accepted_checkpoint(tmp_path, monkeypatch):
    config = config_for()
    path = tmp_path/"config.json"
    path.write_text(json.dumps(config))
    drivers = []

    def build(config, output):
        driver = fake_builder(config, output)
        original = driver.step

        def fail_second(*args, **kwargs):
            if driver.coordinator.state.step == 1:
                raise RuntimeError("injected physical step failure")
            return original(*args, **kwargs)

        driver.step = fail_second
        drivers.append(driver)
        return driver

    monkeypatch.setattr(module, "build_fsi2_problem", build)
    with pytest.raises(RuntimeError, match="physical step"):
        run_fsi2_cli(path, tmp_path/"out", progress=None)
    record = json.loads((tmp_path/"out/failure.json").read_text())
    assert record["accepted_steps"] == 1
    assert (tmp_path/"out/checkpoint.npz").exists()
    monkeypatch.setattr(module, "build_fsi2_problem", fake_builder)
    report = run_fsi2_cli(path, tmp_path/"out", resume=tmp_path/"out/checkpoint.npz", progress=None)
    assert report["resumed_steps"] == 1 and report["accepted_steps"] == 4


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_invalid_chunk_budget_is_input_failure(tmp_path, steps):
    with pytest.raises(ValueError, match="max_steps"):
        run_fsi2_cli(ROOT/"cases/fsi2.json", tmp_path, max_steps=steps, progress=None)
    assert json.loads((tmp_path/"failure.json").read_text())["stage"] == "input"

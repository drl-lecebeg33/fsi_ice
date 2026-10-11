from copy import deepcopy
import json
from pathlib import Path

import pytest

from fsi_ice.validation.fsi1 import load_fsi1_config, run_fsi1, run_fsi1_cli


ROOT = Path(__file__).resolve().parents[1]


def test_fsi1_config_rejects_nonrefining_meshes(tmp_path):
    config = json.loads((ROOT/"cases/fsi1.json").read_text())
    config["mesh_sizes"] = [.04, .04, .02]
    path = tmp_path/"bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="strictly decreasing"):
        load_fsi1_config(path)


def test_coarse_fsi1_end_to_end_and_repeat(tmp_path):
    config = load_fsi1_config(ROOT/"cases/fsi1.json")
    coarse = deepcopy(config)
    coarse["mesh_sizes"] = [.08, .06, .04]
    for key in ("tip_y_relative_error_max", "drag_relative_error_max"):
        coarse["acceptance"][key] = 1.0
    coarse["acceptance"]["lift_absolute_error_max"] = 1.0
    first = run_fsi1(coarse, tmp_path/"first", progress=lambda *args, **kwargs: None)
    second = run_fsi1(coarse, tmp_path/"second", progress=lambda *args, **kwargs: None)
    assert first["status"] == second["status"] == "PASS"
    for a, b in zip(first["results"]["levels"], second["results"]["levels"]):
        assert a.keys() == b.keys()
        for key in a.keys()-{"seconds"}:
            assert a[key] == b[key]
    assert len(list((tmp_path/"first").glob("solution_h_*.npz"))) == 3


def test_fsi1_failure_replaces_old_success(tmp_path, monkeypatch):
    from fsi_ice.validation import fsi1
    output = tmp_path/"output"
    output.mkdir()
    (output/"report.json").write_text('{"status":"PASS"}')
    (output/"solution_h_1.npz").write_bytes(b"old")
    config = load_fsi1_config(ROOT/"cases/fsi1.json")
    monkeypatch.setattr(fsi1, "SteadyFSI1Problem", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected")))
    with pytest.raises(RuntimeError, match="injected"):
        run_fsi1(config, output)
    assert not (output/"report.json").exists()
    assert not (output/"solution_h_1.npz").exists()
    assert json.loads((output/"failure.json").read_text())["status"] == "FAIL"


def test_invalid_fsi1_cli_input_replaces_old_success(tmp_path):
    output = tmp_path/"output"
    output.mkdir()
    (output/"report.json").write_text('{"status":"PASS"}')
    (output/"solution_h_1.npz").write_bytes(b"old")
    bad = tmp_path/"bad.json"
    bad.write_text("{")
    with pytest.raises(ValueError):
        run_fsi1_cli(bad, output)
    assert not (output/"report.json").exists()
    assert not (output/"solution_h_1.npz").exists()
    failure = json.loads((output/"failure.json").read_text())
    assert failure["status"] == "FAIL" and failure["stage"] == "config"
    assert failure["config_path"] == str(bad)

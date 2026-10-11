"""Coordinator-owned integration checks for T19 configuration and public CLI."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fsi_ice.validation.partitioned import load_partitioned_config, run_partitioned_check, validate_config

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_configuration_defaults_to_fixed_and_aitken_uses_same_fe_problem(tmp_path):
    old = load_partitioned_config(ROOT / "cases/partitioned_check.json")
    accelerated = load_partitioned_config(ROOT / "cases/aitken_check.json")
    for key in ("mesh", "dt", "solid", "fluid", "scales", "acceptance"):
        assert old[key] == accelerated[key]
    baseline = run_partitioned_check(old, tmp_path / "fixed")
    aitken = run_partitioned_check(accelerated, tmp_path / "aitken")
    assert baseline["status"] == aitken["status"] == "PASS"
    assert baseline["results"]["method"] == "fixed" and aitken["results"]["method"] == "aitken"
    assert baseline["results"]["iterations"] == 27
    with np.load(tmp_path / "fixed/solution.npz") as fixed, np.load(tmp_path / "aitken/solution.npz") as other:
        for key, scale in (("solid_displacement", 0.01), ("fluid_velocity", 0.1),
                           ("fluid_pressure", 1), ("interface_load", 1)):
            assert np.max(np.abs(fixed[key]-other[key]))/scale <= old["acceptance"]["direct_difference_max"]
    with (tmp_path / "aitken/iterations.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == aitken["results"]["iterations"]
    assert all("relaxation_reason" in row and np.isfinite(float(row["relaxation"])) for row in rows)


@pytest.mark.parametrize("method", ["fixed", "aitken"])
def test_cli_method_override_records_the_effective_method_from_another_directory(tmp_path, method):
    result = subprocess.run([sys.executable, "-m", "fsi_ice", "partitioned-check", "--config",
                             str(ROOT / "cases/aitken_check.json"), "--method", method,
                             "--output", "output"], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert report["status"] == "PASS" and report["config"]["coupling"]["method"] == method
    assert report["results"]["method"] == method


@pytest.mark.parametrize("value", ["unknown", True, None, 1])
def test_invalid_algorithm_selector_is_rejected_by_config(value):
    config = json.loads((ROOT / "cases/aitken_check.json").read_text())
    config["coupling"]["method"] = value
    with pytest.raises((ValueError, TypeError)):
        validate_config(config)


def test_invalid_aitken_bounds_and_unknown_config_fields_are_rejected():
    config = json.loads((ROOT / "cases/aitken_check.json").read_text())
    config["coupling"]["minimum_relaxation"] = 2.0
    with pytest.raises(ValueError):
        validate_config(config)
    config["coupling"]["minimum_relaxation"] = 0.05
    config["coupling"]["unexpected"] = 1
    with pytest.raises(ValueError):
        validate_config(config)

import json
from pathlib import Path
import subprocess
import sys

import pytest


def command(tmp_path, *args):
    return subprocess.run([sys.executable, "-m", "fsi_ice", *args], cwd=tmp_path,
                          capture_output=True, text=True, check=False)


def config_file(tmp_path, *, threshold=1.8):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"case": "poisson_sine", "mesh_levels": [4, 8, 16],
                              "acceptance": {"l2_order_min": threshold,
                                             "h1_order_min": 0.9, "residual_max": 1e-10}}))
    return path


def test_installed_cli_from_another_working_directory_and_plot_output(tmp_path):
    config = config_file(tmp_path)
    result = command(tmp_path, "poisson", "--config", str(config), "--output", "output")
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert report["status"] == "PASS" and report["config"]["mesh_levels"] == [4, 8, 16]
    assert (tmp_path / "output/poisson.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_failed_acceptance_returns_failure_and_saves_diagnostics(tmp_path):
    config = config_file(tmp_path, threshold=3.0)
    result = command(tmp_path, "poisson", "--config", str(config), "--output", "failed", "--no-plot")
    assert result.returncode == 1
    report = json.loads((tmp_path / "failed/report.json").read_text())
    assert report["status"] == "FAIL" and not report["checks"]["l2_order"]


@pytest.mark.parametrize("arguments", [("unknown",), ("poisson", "--config", "missing.json")])
def test_cli_input_errors_are_nonzero_without_unhandled_tracebacks(tmp_path, arguments):
    result = command(tmp_path, *arguments)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_malformed_config_is_rejected(tmp_path):
    config = tmp_path / "bad.json"
    config.write_text('{"case": "poisson_sine", "mesh_levels": []}')
    result = command(tmp_path, "poisson", "--config", str(config))
    assert result.returncode == 2 and "ERROR" in result.stderr


def test_console_entry_point_exists():
    executable = Path(sys.executable).parent / "fsi-ice"
    result = subprocess.run([str(executable), "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0 and "poisson" in result.stdout and "fsi1" in result.stdout


def test_postprocess_requires_saved_complete_evidence(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = command(tmp_path, "csm3", "--config", str(root / "cases/csm3.json"),
                     "--output", "missing", "--postprocess-only")
    assert result.returncode == 2 and "ERROR" in result.stderr
    assert "Traceback" not in result.stderr

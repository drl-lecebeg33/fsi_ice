"""Saved-state chronology, exact geometry, integrity and read-only export."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

import scripts.fsi2_mesh_animation as animation
from fsi_ice.mesh import TriMesh, rectangle_mesh
from fsi_ice.time.transaction import AcceptedState
from fsi_ice.validation.fsi2 import load_fsi2_config
from fsi_ice.validation.fsi2_checkpoint import save_checkpoint


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    # Analytic rigid translations test geometry I/O; this is not an FSI solve.
    fluid = rectangle_mesh(2, 2, width=1., height=.8)
    solid0 = rectangle_mesh(2, 2, width=1., height=.3)
    solid = TriMesh(solid0.points-[0., .3], solid0.cells, solid0.boundaries)
    pair = SimpleNamespace(fluid=fluid, solid=solid)
    monkeypatch.setattr(animation, "read_fsi_gmsh2", lambda path: pair)
    dm = np.stack([np.tile([0., y], (fluid.n_nodes, 1)) for y in (.02, .01)])
    ds = np.stack([np.stack((np.tile([0., y], (solid.n_nodes, 1)), np.zeros_like(solid.points))) for y in (.02, .01)])
    rows = [{"step": index, "time": index*.01, "order": 1 if index == 1 else 2,
             "method": "iqn", "minimum_volume_ratio": 1., "minimum_quality": float(fluid.quality.min())}
            for index in (1, 2)]
    state = AcceptedState(time=.02, dt=.01, order=2, step=2, epoch=2,
        solid_states=ds, fluid_states=np.zeros((2, fluid.n_nodes, 2)), mesh_states=dm,
        solid_generation=2, fluid_generation=2, mesh_generation=2,
        pressure=np.zeros(fluid.n_nodes), mesh_current=fluid.points+dm[0],
        mesh_velocity=np.zeros_like(fluid.points), monitoring=rows)
    config = load_fsi2_config(ROOT/"cases/fsi2.json")
    config["method"] = "iqn"
    folder = tmp_path/"input"
    (folder/"mesh").mkdir(parents=True)
    (folder/"mesh/fsi1.msh").write_text("Mock mesh identity for analytic I/O tests")
    mesh_hash = hashlib.sha256((folder/"mesh/fsi1.msh").read_bytes()).hexdigest()
    save_checkpoint(folder/"checkpoint.npz", state, config=config, mesh_sha256=mesh_hash, code_sha256="b"*64)
    return folder, pair


def rewrite(folder, mutate):
    path = folder/"checkpoint.npz"
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key].copy() for key in data.files}
    metadata = json.loads(values["metadata"].item())
    mutate(metadata, values)
    values["metadata"] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **values)


def test_oldest_to_newest_exact_saved_coordinates_and_no_new_states(snapshot):
    folder, pair = snapshot
    original = (folder/"checkpoint.npz").read_bytes()
    _, frames, identity = animation.saved_frames(folder)
    assert [(f["step"], f["time"]) for f in frames] == [(1, .01), (2, .02)]
    for frame, y in zip(frames, (.01, .02)):
        np.testing.assert_array_equal(frame["fluid_points"], pair.fluid.points+[0., y])
        np.testing.assert_array_equal(frame["solid_points"], pair.solid.points+[0., y])
        assert frame["minimum_volume_ratio"] == pytest.approx(1.)
    assert identity["displacement_scale"] == 1
    assert identity["interpolated_frames"] == 0
    assert (folder/"checkpoint.npz").read_bytes() == original


def test_reference_mesh_hash_mismatch_rejected(snapshot):
    folder, _ = snapshot
    (folder/"mesh/fsi1.msh").write_text("Changed identity")
    with pytest.raises(ValueError, match="Reference mesh identity"):
        animation.saved_frames(folder)


def test_array_corruption_is_rejected(snapshot):
    folder, _ = snapshot
    def mutate(metadata, values):
        values["pressure"].flat[0] += 1
    rewrite(folder, mutate)
    with pytest.raises(ValueError, match="checksum"):
        animation.saved_frames(folder)


def test_geometric_monitor_inconsistency_is_rejected(snapshot):
    folder, _ = snapshot
    def mutate(metadata, values):
        metadata["monitoring"][0]["minimum_volume_ratio"] = .9
    rewrite(folder, mutate)
    with pytest.raises(ValueError, match="Frame geometry"):
        animation.saved_frames(folder)


def test_one_layer_is_not_fabricated_into_an_animation(snapshot):
    folder, _ = snapshot
    def mutate(metadata, values):
        metadata["config"]["order"] = metadata["order"] = 1
        for row in metadata["monitoring"]:
            row["order"] = 1
        for key in ("mesh_history", "solid_history", "fluid_history"):
            values[key] = values[key][:1]
            from fsi_ice.validation.fsi2_checkpoint import _array_digest
            metadata["arrays"][key] = _array_digest(values[key])
    rewrite(folder, mutate)
    with pytest.raises(ValueError, match="at least two"):
        animation.saved_frames(folder)


def test_stale_failure_record_is_rejected(snapshot):
    folder, _ = snapshot
    (folder/"failure.json").write_text(json.dumps({"status": "FAIL", "accepted_steps": 0}))
    with pytest.raises(ValueError, match="Failure record"):
        animation.saved_frames(folder)


def test_output_directory_cannot_modify_solver_inputs(snapshot):
    folder, _ = snapshot
    for output in (folder, folder/"animation", folder.parent):
        with pytest.raises(ValueError, match="separate"):
            animation.export(folder, output)


def test_export_two_frames_offline_html_and_preserve_inputs(snapshot, tmp_path, monkeypatch):
    folder, _ = snapshot
    before = (folder/"checkpoint.npz").read_bytes()
    def render(pair, frames, output):
        paths = []
        for index in range(len(frames)):
            path = output/f"frame_{index:02d}.png"
            Image.new("RGB", (80, 60), (255*index, 0, 255*(1-index))).save(path)
            paths.append(path)
        return paths
    monkeypatch.setattr(animation, "render_frames", render)
    monkeypatch.setattr(animation.shutil, "which", lambda name: None)
    result = animation.export(folder, tmp_path/"output")
    assert not result["video_created"]
    assert result["interpolated_frames"] == 0
    with Image.open(tmp_path/"output/mesh_motion.gif") as gif:
        assert gif.n_frames == 2
        for index in range(2):
            gif.seek(index)
            assert gif.info["duration"] == 1000
    # 中文：导出页面固定为 UTF-8，显式指定编码，避免中文 Windows 上按 GBK 读取失败。
    html = (tmp_path/"output/index.html").read_text(encoding="utf-8")
    assert "0.01 s → 0.02 s" in html
    assert "data:image/gif;base64," in html
    assert "https://" not in html
    assert (folder/"checkpoint.npz").read_bytes() == before

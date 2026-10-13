"""Animate only mesh/solid states actually retained in an FSI2 checkpoint."""

import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from fsi_ice.mesh.turek import read_fsi_gmsh2
from fsi_ice.mesh.triangular import TriMesh
from fsi_ice.validation.fsi2 import validate_fsi2_config
from fsi_ice.validation.fsi2_checkpoint import load_checkpoint


def saved_frames(folder):
    """Oldest-to-newest stored layers; no interpolation or PDE recomputation."""
    folder = Path(folder)
    blob = (folder/"checkpoint.npz").read_bytes()
    with np.load(BytesIO(blob), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
    validate_fsi2_config(metadata["config"])
    mesh_path = folder/"mesh/fsi1.msh"
    mesh_hash = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    if mesh_hash != metadata["mesh_sha256"]:
        raise ValueError("Reference mesh identity does not match the checkpoint.")
    state = load_checkpoint(BytesIO(blob), config=metadata["config"],
        mesh_sha256=mesh_hash, code_sha256=metadata["code_sha256"])
    if len(state.mesh_history) < 2:
        raise ValueError("Animation requires at least two saved mesh states.")
    pair = read_fsi_gmsh2(mesh_path)
    if not np.array_equal(state.mesh_current, pair.fluid.points+state.mesh_history[0]):
        raise ValueError("Saved current mesh does not match reference points plus displacement.")
    frames = []
    for index in reversed(range(len(state.mesh_history))):
        step = state.step-index
        if step < 0:
            raise ValueError("Invalid history layer index.")
        fluid_points = pair.fluid.points+state.mesh_history[index]
        solid_displacement = state.solid_history[index][0]
        if solid_displacement.shape != pair.solid.points.shape:
            raise ValueError("Solid history shape does not match the reference mesh.")
        current = TriMesh(fluid_points, pair.fluid.cells, pair.fluid.boundaries)
        ratios = current.determinants/pair.fluid.determinants
        row = metadata["monitoring"][step-1] if step else None
        time = row["time"] if row else 0.
        if row is not None and (not np.isclose(ratios.min(), row["minimum_volume_ratio"], rtol=1e-10, atol=1e-12) or
                not np.isclose(current.quality.min(), row["minimum_quality"], rtol=1e-10, atol=1e-12)):
            raise ValueError("Frame geometry does not match the accepted monitoring row.")
        frames.append({"step": step, "time": time, "fluid_points": fluid_points,
            "solid_points": pair.solid.points+solid_displacement, "volume_ratios": ratios,
            "minimum_volume_ratio": float(ratios.min()), "minimum_quality": float(current.quality.min()),
            "minimum_cell": int(np.argmin(ratios))})
    identity = {"checkpoint_sha256": hashlib.sha256(blob).hexdigest(),
        "mesh_sha256": mesh_hash, "solver_source_sha256": metadata["code_sha256"],
        "render_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": metadata["config"], "accepted_step": state.step, "accepted_time": state.time,
        "scope": "Saved history layers only; no intermediate states are reconstructed",
        "displacement_scale": 1, "interpolated_frames": 0}
    failure_path = folder/"failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.exists() else None
    if failure is not None and (failure.get("accepted_steps") != state.step or
            failure.get("config") != metadata["config"] or failure.get("status") != "FAIL"):
        raise ValueError("Failure record does not match the saved checkpoint.")
    identity["solver_failure"] = failure
    return pair, frames, identity


def render_frames(pair, frames, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection, LineCollection
    from matplotlib.colors import Normalize

    names = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei") if name in names), "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})
    output = Path(output)
    norm = Normalize(vmin=.2, vmax=1.2, clip=True)
    cmap = plt.get_cmap("RdYlGn")
    tip = pair.solid.points[np.argmax(pair.solid.points[:, 0])]
    min_x, max_x = pair.solid.points[:, 0].min(), pair.solid.points[:, 0].max()
    all_fluid = np.concatenate([frame["fluid_points"] for frame in frames])
    local_y_min = min(pair.solid.points[:, 1].min(), *(frame["solid_points"][:, 1].min() for frame in frames))-.04
    local_y_max = max(pair.solid.points[:, 1].max(), *(frame["solid_points"][:, 1].max() for frame in frames))+.04
    paths = []
    saved_times = "、".join(f"{frame['time']:.2f} s" for frame in frames)
    for index, frame in enumerate(frames):
        fig = plt.figure(figsize=(14, 8), layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=(.65, 1))
        axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
        domains = [(all_fluid[:, 0].min(), all_fluid[:, 0].max(), all_fluid[:, 1].min(), all_fluid[:, 1].max()),
            (min_x-.07, max_x+.07, local_y_min, local_y_max),
            (tip[0]-.08, tip[0]+.05, local_y_min, local_y_max)]
        for ax, bounds, title in zip(axes, domains, ("全域网格", "梁与周围流体网格", "梁尖局部放大")):
            x0, x1, y0, y1 = bounds
            reference_triangles = pair.fluid.points[pair.fluid.cells]
            visible = (reference_triangles[:, :, 0].max(axis=1) >= x0) & (reference_triangles[:, :, 0].min(axis=1) <= x1)
            cells = pair.fluid.cells[visible]
            # Grey reference edges are geometry data, not an additional timestep.
            ax.add_collection(PolyCollection(pair.fluid.points[cells], facecolors="none", edgecolors="#b9c2cc", linewidths=.35, alpha=.6))
            ax.add_collection(PolyCollection(frame["fluid_points"][cells],
                facecolors=cmap(norm(frame["volume_ratios"][visible])), edgecolors="#264766",
                linewidths=.45 if ax is axes[0] else .65, alpha=.53))
            ax.add_collection(PolyCollection(pair.solid.points[pair.solid.cells], facecolors="none", edgecolors="#9ba3ad", linewidths=.45, alpha=.65))
            ax.add_collection(PolyCollection(frame["solid_points"][pair.solid.cells],
                facecolors="#ffd88b", edgecolors="#7c4200", linewidths=.6))
            for mesh, points, tags in ((pair.fluid, frame["fluid_points"], ("walls", "inlet", "outlet", "cylinder-rigid-wall")),
                                      (pair.solid, frame["solid_points"], ("solid-interface", "solid-clamp"))):
                edges = np.concatenate([mesh.boundaries[tag] for tag in tags])
                ax.add_collection(LineCollection(points[edges], colors="#19212b", linewidths=1.))
            ax.set(xlim=(x0, x1), ylim=(y0, y1), xlabel="x (m)", ylabel="y (m)", title=title, aspect="equal")
            ax.grid(alpha=.12)
        cell = frame["minimum_cell"]
        center = frame["fluid_points"][pair.fluid.cells[cell]].mean(axis=0)
        axes[-1].plot(*center, "o", color="#be123c", markersize=7)
        axes[-1].annotate(f"最小面积比单元 {cell}", center, xytext=(.05, .94), textcoords="axes fraction",
                         arrowprops={"arrowstyle": "->", "color": "#be123c"}, color="#be123c", fontsize=10)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=.65, extend="max",
                     label="单元面积比 A当前 / A参考（红色接近0.2门槛；最高绿色≥1.2）", pad=.02)
        fig.suptitle(f"FSI2 动网格 · 已接受步骤 {frame['step']} · t = {frame['time']:.2f} s\n"
            f"最小面积比 {frame['minimum_volume_ratio']:.4f} · 最小质量 {frame['minimum_quality']:.4f}", fontsize=17)
        fig.supxlabel(f"仅显示 {saved_times} 已保存状态 · 位移倍率 1× · 每帧慢放 1 秒 · 循环回到较早时刻\n"
                      "灰线：初始参考网格；黑/蓝线：当前网格；黄色：固体梁。未绘制失败候选或重建中间时刻。", fontsize=10)
        path = output/f"frame_{index:02d}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def export(folder, output):
    folder, output = Path(folder).resolve(), Path(output).resolve()
    if folder == output or folder in output.parents or output in folder.parents:
        raise ValueError("Animation output must be separate from the solver directory and its parents.")
    pair, frames, identity = saved_frames(folder)
    output.mkdir(parents=True, exist_ok=True)
    paths = render_frames(pair, frames, output)
    from PIL import Image
    images = [Image.open(path).convert("RGB") for path in paths]
    try:
        images[0].save(output/"mesh_motion.gif", save_all=True, append_images=images[1:],
                       duration=1000, loop=0, disposal=2, optimize=False)
    finally:
        for image in images:
            image.close()
    encoder = shutil.which("ffmpeg")
    video_created = False
    if encoder:
        subprocess.run([encoder, "-y", "-loglevel", "error", "-framerate", "1", "-i", str(output/"frame_%02d.png"),
            "-c:v", "libx264", "-r", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output/"mesh_motion.mp4")], check=True, capture_output=True)
        video_created = True
    gif = base64.b64encode((output/"mesh_motion.gif").read_bytes()).decode()
    interval = frames[1]["time"]-frames[0]["time"]
    status_note = "求解已停止，失败候选不在动画中；完整周期尚未验证。" if identity["solver_failure"] else "动画覆盖已保存历史层，不判断完整周期精度。"
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FSI2 已保存动网格快照动画</title><style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;line-height:1.7;color:#17212b;background:#f8fafc}}.viewer{{overflow:auto;max-height:82vh}}img{{width:100%;height:auto;border:1px solid #cbd5e1;border-radius:8px;background:white;box-sizing:border-box}}.viewer.zoomed img{{width:1680px;max-width:none}}.note{{background:#fff3d7;padding:12px 18px;border-radius:8px}}button{{padding:8px 16px;margin:10px 8px 10px 0;font-size:16px}}</style>
<h1>FSI2 动网格：最后两步实际快照</h1><p class="note">仅有 <b>{frames[0]['time']:.2f} s → {frames[-1]['time']:.2f} s</b> 两帧完整网格，原时间间隔为{interval:.4g}秒，逐帧慢放、循环回放。
位移倍率为1×。灰线是初始参考网格；循环跳回{frames[0]['time']:.2f}秒不表示真实往返振动。只展示已保存的{len(frames)}帧，未重建更早或中间时刻。</p>
<button id="toggle">显示末帧</button><button id="zoom">查看原始分辨率</button><div class="viewer" id="viewer"><img id="animation" alt="FSI2 最后两步实际动网格，全域及梁尖放大" src="data:image/gif;base64,{gif}"></div>
<p>数据最后接受时刻：{identity['accepted_time']:.2f}秒。{status_note}此页离线可打开。</p>
<script>const gif=document.getElementById('animation').src;const frozen='data:image/png;base64,{base64.b64encode(paths[-1].read_bytes()).decode()}';let playing=true;document.getElementById('toggle').onclick=function(){{playing=!playing;document.getElementById('animation').src=playing?gif:frozen;this.textContent=playing?'显示末帧':'播放两帧';}};document.getElementById('zoom').onclick=function(){{const zoomed=document.getElementById('viewer').classList.toggle('zoomed');this.textContent=zoomed?'适应窗口':'查看原始分辨率';}};</script></html>'''
    (output/"index.html").write_text(html, encoding="utf-8")
    manifest = {**identity, "frames": [{key: frame[key] for key in
        ("step", "time", "minimum_volume_ratio", "minimum_quality", "minimum_cell")} for frame in frames],
        "gif_frame_duration_ms": 1000, "video_created": video_created,
        "limitations": "Only retained history states; no full trajectory, interpolation or new solve"}
    (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = export(args.input, args.output)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"Animation error: {error}")
        return 2
    print(f"Exported {len(result['frames'])} actual saved states; MP4={result['video_created']}; {args.output/'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Read-only comparison of two completed short FSI2 trajectories; not periodic QA."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(left, right, *, allow_inner_atol_difference=False):
    reports = [json.loads((folder/"report.json").read_text(encoding="utf-8")) for folder in (left, right)]
    if reports[0]["mesh_sha256"] != reports[1]["mesh_sha256"]:
        raise ValueError("Cannot compare different meshes.")
    configs = [dict(report["config"]) for report in reports]
    for config in configs:
        config.pop("method")
        if allow_inner_atol_difference:
            config["coupling"] = dict(config["coupling"])
            config["coupling"].pop("absolute_tolerance")
    if configs[0] != configs[1]:
        raise ValueError("Comparison requires the same physical/time/solver configuration.")
    if allow_inner_atol_difference:
        for folder, report in zip((left, right), reports):
            if report["status"] not in ("PASS", "INCOMPLETE"):
                raise ValueError("Only accepted trajectories may be compared.")
            with (folder/"monitor.csv").open() as stream:
                import csv
                rows = list(csv.DictReader(stream))
            gate = report["config"]["acceptance"]
            if not rows or any(float(row["residual"]) > gate["residual_max"] or
                               float(row["continuity"]) > gate["continuity_max"] or
                               float(row["minimum_quality"]) < gate["minimum_mesh_quality"] or
                               float(row["minimum_volume_ratio"]) < .2 for row in rows):
                raise ValueError("A trajectory fails the shared physical acceptance gates.")
    rows = {}
    scales = {"solid": .41, "velocity": 1., "pressure": 1000., "mesh_displacement": .41,
              "interface_g": 215.06, "mesh_velocity": 1.}
    with np.load(left/"solution.npz") as a, np.load(right/"solution.npz") as b:
        if a["step"] != b["step"] or a["time"] != b["time"]:
            raise ValueError("Trajectories have different accepted times/steps.")
        for name, scale in scales.items():
            pairs = [(name, a[name], b[name], scale)] if name != "solid" else [
                ("d_s", a[name][0], b[name][0], .41), ("v_s", a[name][1], b[name][1], 1.)]
            for label, x, y, physical_scale in pairs:
                if x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                    raise ValueError(f"Invalid field {label}.")
                threshold = 1e-8*max(float(np.max(np.abs(x))), float(np.max(np.abs(y))))+1e-10*physical_scale
                difference = float(np.max(np.abs(x-y)))
                rows[label] = {"max_difference": difference, "threshold": threshold, "ratio": difference/threshold,
                               "passed": difference <= threshold}
    return {"scope": "T24 short-trajectory final-state consistency, no periodic precision verdict",
            "inner_atol_difference": allow_inner_atol_difference,
            "inner_atol_values": [report["config"]["coupling"]["absolute_tolerance"] for report in reports],
            "passed": all(row["passed"] for row in rows.values()), "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--allow-inner-atol-difference", action="store_true",
                        help="Compare physically gated solutions with different inner absolute stop tolerances")
    args = parser.parse_args()
    report = compare(args.left, args.right,
                     allow_inner_atol_difference=args.allow_inner_atol_difference)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["passed"] else 1)

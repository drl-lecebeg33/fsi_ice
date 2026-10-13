"""Repeat an unchanged FSI2 short solve against a frozen source snapshot.

Run only on an otherwise idle machine. Each sample is a separate Python process;
paths are independent and pre-existing output directories are refused.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def worker(config, output, mesh_size, end_time):
    from fsi_ice.validation.fsi2 import run_fsi2_cli
    dt = json.loads(Path(config).read_text())["dt"]
    started = perf_counter()
    report = run_fsi2_cli(config, output, method="iqn", mesh_size=mesh_size,
                          end_time=end_time, progress=None)
    if report["status"] != "INCOMPLETE" or report["accepted_steps"] != round(end_time/dt):
        raise RuntimeError(f"Unexpected short trajectory: {report['status']}.")
    print(json.dumps({"wall_seconds": perf_counter()-started, "status": report["status"],
                      "steps": report["accepted_steps"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True,
                        help="Directory containing the frozen fsi_ice package")
    parser.add_argument("--config", type=Path, default=ROOT/"cases/fsi2.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mesh-size", type=float, default=.02)
    parser.add_argument("--end-time", type=float, default=.03)
    args = parser.parse_args()
    if args.repeats < 3 or args.end_time <= 0 or args.mesh_size <= 0:
        parser.error("Require at least three repetitions and positive mesh/time controls.")
    base = args.baseline_source.resolve()
    if not (base/"fsi_ice/__init__.py").is_file():
        parser.error("baseline-source must contain a frozen fsi_ice package.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    rows = []
    for repeat in range(args.repeats):
        for version, source in (("baseline", base), ("optimized", ROOT/"src")):
            target = output/f"{version}_{repeat}"
            worker_code = (
                "from scripts.m6_benchmark import worker; "
                "import sys; worker(sys.argv[1],sys.argv[2],float(sys.argv[3]),float(sys.argv[4]))"
            )
            sample_env = {**env, "PYTHONPATH": os.pathsep.join((str(source),str(ROOT)))}
            command = [sys.executable, "-c", worker_code, str(args.config), str(target),
                       str(args.mesh_size), str(args.end_time)]
            started = perf_counter()
            result = subprocess.run(command, cwd=ROOT, env=sample_env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (output/f"{version}_{repeat}.log").write_text(result.stdout+result.stderr)
            if result.returncode:
                raise RuntimeError(f"{version} sample {repeat} failed; see its log.")
            report = json.loads((target/"report.json").read_text())
            monitor = np.genfromtxt(target/"monitor.csv", delimiter=",", names=True,
                                    dtype=None, encoding="utf-8")
            item = {"version": version, "repeat": repeat,
                    "wall_seconds": perf_counter()-started,
                    "solve_seconds": float(np.sum(monitor["solve_seconds"])),
                    "coupling_iterations": int(np.sum(monitor["coupling_iterations"])),
                    "newton_iterations": int(np.sum(monitor["newton_iterations"])),
                    "status": report["status"], "source_sha256": report["source_sha256"]}
            rows.append(item)
            print(item, flush=True)
        from t24_compare import compare
        check = compare(output/f"baseline_{repeat}", output/f"optimized_{repeat}")
        (output/f"comparison_{repeat}.json").write_text(json.dumps(check,indent=2)+"\n")
        if not check["passed"]:
            raise RuntimeError(f"Sample {repeat} changed the short FSI2 solution beyond T24 tolerance.")
    times = {version:[r["wall_seconds"] for r in rows if r["version"] == version]
             for version in ("baseline","optimized")}
    summary = {"rows": rows, "wall_median_seconds": {key:statistics.median(v) for key,v in times.items()},
               "wall_speedup_median_ratio":statistics.median(times["baseline"])/statistics.median(times["optimized"]),
               "python":sys.version, "platform":platform.platform(),
               "threads":{"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
               "method":"iqn", "mesh_size":args.mesh_size, "end_time":args.end_time,
               "repeats":args.repeats, "scope":"short FSI2 only; no long-run throughput claim"}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()

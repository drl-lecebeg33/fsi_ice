"""Prepare T26 configurations and commands without executing any solver."""

import argparse
import copy
import json
from pathlib import Path
import shlex

from fsi_ice.validation.fsi2 import load_fsi2_config, validate_fsi2_config
from fsi_ice.validation.fsi2_checkpoint import source_digest


ROOT = Path(__file__).resolve().parents[1]


def study_cases(base):
    """Two independent refinement groups share one finest intersection."""
    validate_fsi2_config(base)
    cases, groups = [], {"space": [], "time": [], "algorithms": []}
    for name, size, dt in (("space_coarse", .02, .0025), ("space_medium", .01, .0025),
                           ("intersection_fine", .005, .0025),
                           ("time_coarse", .005, .01), ("time_medium", .005, .005)):
        config = copy.deepcopy(base)
        config.update(mesh_size=size, dt=dt, end_time=20., order=2, method="iqn")
        validate_fsi2_config(config)
        cases.append({"id": name, "config": config, "phase": "convergence", "repeat": None})
    groups["space"] = ["space_coarse", "space_medium", "intersection_fine"]
    groups["time"] = ["time_coarse", "time_medium", "intersection_fine"]
    for method in ("fixed", "aitken", "iqn", "monolithic"):
        for repeat in range(1, 4):
            name = f"cost_{method}_r{repeat}"
            config = copy.deepcopy(base)
            config.update(mesh_size=.005, dt=.0025, end_time=20., order=2, method=method)
            validate_fsi2_config(config)
            cases.append({"id": name, "config": config, "phase": "algorithms", "repeat": repeat})
            groups["algorithms"].append(name)
    return cases, groups


def prepare(base_path, output):
    base = load_fsi2_config(base_path)
    cases, groups = study_cases(base)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Study output must be an empty or new directory; existing evidence is preserved.")
    output.mkdir(parents=True, exist_ok=True)
    (output/"configs").mkdir(exist_ok=True)
    commands = ["#!/usr/bin/env bash", "set -eu", "# Prepared commands only; run deliberately, serially.",
                "# Preserve reference FAIL (1) and INCOMPLETE (3) evidence; runtime error (2) stops.",
                "# Review missing stable windows and extend the group consistently when needed.",
                'run_evidence_command() {', '  if "$@"; then return 0; else',
                '    local result_code=$?', '    case "$result_code" in',
                '      1|3) printf "Scientific status exit %s; evidence retained\\n" "$result_code" ;;',
                '      *) return "$result_code" ;;', '    esac', '  fi', '}']
    for case in cases:
        config_path = output/"configs"/(case["id"]+".json")
        run_path = output/"runs"/case["id"]
        analysis_path = output/"analysis"/case["id"]
        config_path.write_text(json.dumps(case["config"], ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        case.update(config_path=str(config_path), output_path=str(run_path), analysis_path=str(analysis_path),
            argv=[str(ROOT/".venv/bin/python"), "-m", "fsi_ice", "fsi2", "--config", str(config_path), "--output", str(run_path)],
            analysis_argv=[str(ROOT/".venv/bin/python"), str(ROOT/"scripts/fsi2_analyze.py"), str(run_path), "--output", str(analysis_path)],
            threads={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
            status="NOT_RUN", permitted_to_start=case["phase"] == "convergence")
        if case["phase"] == "convergence":
            commands += ["", "# "+case["id"], "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 run_evidence_command "+shlex.join(case["argv"]),
                         "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 run_evidence_command "+shlex.join(case["analysis_argv"])]
    manifest = {"schema": "m6-study-v1", "status": "PREPARED_NOT_RUN", "solver_started": False,
        "source_sha256": source_digest(), "base_config_path": str(Path(base_path).resolve()),
        "groups": groups, "cases": cases,
        "convergence_gate": {"stable_completed_results_per_group": 3,
            "finest_reference_pass_required": True, "finest_pair_difference_over_reference_max": .02,
            "refinement_differences_must_decrease": True,
            "quantities": ["tip_y_amplitude", "frequency", "drag_mean"]},
        "algorithm_prerequisites": ["Both independent convergence groups verified",
            "Common candidate mesh/time configuration passes stable-cycle reference gates",
            "No concurrent heavy workloads; same hardware/environment/threads; three cold-start repeats",
            "Record full wall time including setup/output, and separately accepted solve time",
            "Compare same discrete solution and physical error before publishing cost ranking"],
        "notes": ["All runs start from rest; no cross-config checkpoint reuse",
            "The current h=.02, dt=.01 run is an additional baseline, outside these refinement groups",
            "Algorithm configuration is a candidate: confirm or regenerate after convergence evidence",
            "20 seconds is a common proposed horizon; extend every member consistently if not stable",
            "No computation cost or convergence is claimed by preparing this manifest"]}
    (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (output/"convergence_commands.sh").write_text("\n".join(commands)+"\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"cases/fsi2.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = prepare(args.config, args.output)
    except (ValueError, OSError) as error:
        print(f"Study preparation error: {error}")
        return 2
    print(f"PREPARED_NOT_RUN: {len(manifest['cases'])} configurations; solver_started=false; "
          f"{args.output/'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Project entry point; failures return a nonzero exit status."""

import argparse
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fsi-ice", description="FSI research foundation and validation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    poisson = subparsers.add_parser("poisson", help="Run analytical Poisson mesh convergence")
    poisson.add_argument("--config", type=Path, help="JSON case configuration; default uses levels 8,16,32")
    poisson.add_argument("--output", type=Path, default=Path("results/poisson"))
    poisson.add_argument("--no-plot", action="store_true")
    for command in ("solid", "fluid", "single-fields", "csm3", "cfd3"):
        validation = subparsers.add_parser(command, help="Run single-field verification and benchmarks")
        validation.add_argument("--config", type=Path, required=True)
        validation.add_argument("--output", type=Path, default=Path("results") / command)
        validation.add_argument("--no-plot", action="store_true")
        if command in ("csm3", "cfd3"):
            validation.add_argument("--postprocess-only", action="store_true",
                                    help="Validate complete saved histories and regenerate report/plots; do not solve again")
    subparsers.add_parser("test", help="Run the development test suite from any working directory")
    args = parser.parse_args(argv)
    try:
        if args.command in ("csm3", "cfd3"):
            from fsi_ice.validation.transient import load_transient_config, run_transient
            config = load_transient_config(args.config)
            if config["case"] != args.command:
                raise ValueError("Transient command and case must match.")
            report = run_transient(config, args.output, plot=not args.no_plot,
                                   postprocess_only=args.postprocess_only)
            print(f"{report['status']}: report saved to {args.output.resolve() / 'report.json'}")
            return 0 if report["status"] == "PASS" else 1
        if args.command == "test":
            root = Path(__file__).resolve().parents[2]
            if not (root / "tests").is_dir() or not (root / "pyproject.toml").is_file():
                raise ValueError("Development tests require an editable project checkout.")
            return subprocess.run([sys.executable, "-m", "pytest", "-c", str(root / "pyproject.toml"),
                                   str(root / "tests")], cwd=root, check=False).returncode
        if args.command in ("solid", "fluid", "single-fields"):
            from fsi_ice.validation.common import load_single_config, write_report
            from fsi_ice.validation.solid import run_solid
            from fsi_ice.validation.fluid import run_fluid

            config = load_single_config(args.config)
            if args.command == "solid":
                report = run_solid(config, args.output, plot=not args.no_plot)
            elif args.command == "fluid":
                report = run_fluid(config, args.output, plot=not args.no_plot)
            else:
                solid_report = run_solid(config, args.output / "solid", plot=not args.no_plot)
                fluid_report = run_fluid(config, args.output / "fluid", plot=not args.no_plot)
                report = write_report(args.output, "single_fields", config,
                                      {"solid": solid_report, "fluid": fluid_report},
                                      {"solid": solid_report["status"] == "PASS", "fluid": fluid_report["status"] == "PASS"},
                                      {"scope": "linear/nonlinear solid, Stokes/Navier-Stokes, manufactured BDF convergence, CSM1/2 and CFD1/2"})
            print(f"{report['status']}: report saved to {args.output.resolve() / 'report.json'}")
            return 0 if report["status"] == "PASS" else 1
        from fsi_ice.validation.poisson import PoissonConfig, run_convergence

        config = PoissonConfig.from_file(args.config) if args.config else PoissonConfig()
        report = run_convergence(config, args.output, plot=not args.no_plot)
        for row in report["results"]:
            order = "--" if row["l2_order"] is None else f"{row['l2_order']:.3f}"
            print(f"n={row['subdivisions']:3d}  L2={row['l2']:.6e}  "
                  f"H1_semi={row['h1_seminorm']:.6e}  L2_order={order}")
        print(f"{report['status']}: report saved to {args.output.resolve() / 'report.json'}")
        return 0 if report["status"] == "PASS" else 1
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

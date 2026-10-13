"""Project entry point; failures return a nonzero exit status.

本模块是 ``fsi-ice`` 命令行入口，定义并分派各子命令：``poisson``（解析解收敛）、
``partitioned-check``（分区耦合校验）、``fsi1``/``fsi2``（Turek–Hron 基准）、
``solid``/``fluid``/``single-fields``/``csm3``/``cfd3``（单场验证与基准）以及
``test``（运行开发测试）。退出码约定：0 表示 PASS，1 表示 FAIL，2 表示参数或运行
错误（ValueError/TypeError/OSError/RuntimeError），3 表示 fsi2 的 INCOMPLETE。
"""

import argparse
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    """解析命令行并分派到对应子命令，返回进程退出码。

    中文：``argv`` 为参数列表（默认取 ``sys.argv``）。成功返回 0，基准未通过返回 1，
    捕获到参数或运行错误时打印到 stderr 并返回 2，fsi2 未完成返回 3。
    """
    parser = argparse.ArgumentParser(prog="fsi-ice", description="FSI research foundation and validation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    poisson = subparsers.add_parser("poisson", help="Run analytical Poisson mesh convergence")
    poisson.add_argument("--config", type=Path, help="JSON case configuration; default uses levels 8,16,32")
    poisson.add_argument("--output", type=Path, default=Path("results/poisson"))
    poisson.add_argument("--no-plot", action="store_true")
    partitioned = subparsers.add_parser("partitioned-check", help="Validate partitioned coupling against a linear FE direct solve")
    partitioned.add_argument("--config", type=Path, required=True)
    partitioned.add_argument("--output", type=Path, default=Path("results/partitioned_check"))
    partitioned.add_argument("--method", choices=("fixed", "aitken", "iqn"), help="Override the method stored in the configuration")
    fsi1 = subparsers.add_parser("fsi1", help="Run the steady partitioned Turek-Hron FSI1 benchmark")
    fsi1.add_argument("--config", type=Path, required=True)
    fsi1.add_argument("--output", type=Path, default=Path("results/fsi1"))
    fsi2 = subparsers.add_parser("fsi2", help="Run the transient Turek-Hron FSI2 benchmark")
    fsi2.add_argument("--config", type=Path, required=True)
    fsi2.add_argument("--output", type=Path, default=Path("results/fsi2"))
    fsi2.add_argument("--method", choices=("fixed", "aitken", "iqn", "monolithic"))
    fsi2.add_argument("--end-time", type=float, help="Override duration; short runs remain INCOMPLETE")
    fsi2.add_argument("--mesh-size", type=float)
    fsi2.add_argument("--resume", type=Path, help="Explicit full-history checkpoint; solution.npz is insufficient")
    fsi2.add_argument("--max-steps", type=int, help="Accept at most this many steps in the current chunk")
    # 中文：以下单场类子命令共用 --config/--output/--no-plot 参数。
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
        if args.command == "fsi2":
            from fsi_ice.validation.fsi2 import run_fsi2_cli
            controls = {"method": args.method, "end_time": args.end_time, "mesh_size": args.mesh_size}
            if args.resume is not None:
                controls["resume"] = args.resume
            if args.max_steps is not None:
                controls["max_steps"] = args.max_steps
            report = run_fsi2_cli(args.config, args.output, **controls)
            print(f"{report['status']}: report saved to {args.output.resolve() / 'report.json'}")
            # 中文：fsi2 还需区分 INCOMPLETE（未达终时）与失败，故映射为 0/1/3。
            return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[report["status"]]
        if args.command == "fsi1":
            from fsi_ice.validation.fsi1 import run_fsi1_cli
            report = run_fsi1_cli(args.config, args.output)
            print(f"{report['status']}: report saved to {args.output.resolve() / 'report.json'}")
            return 0 if report["status"] == "PASS" else 1
        if args.command == "partitioned-check":
            from fsi_ice.validation.partitioned import run_partitioned_check_cli
            report = run_partitioned_check_cli(args.config, args.output, method=args.method)
            print(f"{report['status']}: report saved to {args.output.resolve() / 'report.json'}")
            return 0 if report["status"] == "PASS" else 1
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
        # 中文：其余情况即 ``poisson`` 子命令，走解析解网格收敛流程。
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

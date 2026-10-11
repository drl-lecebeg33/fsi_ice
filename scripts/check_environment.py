"""Check the Python research environment without running a physics solver."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The verified baseline requires Python 3.12.")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run this check with the project's .venv interpreter.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pytest
    import scipy
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import gmres, spsolve

    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    expected = np.array([1.0, 2.0])
    rhs = matrix @ expected
    sparse_matrix = csr_matrix(matrix)
    iterative, info = gmres(sparse_matrix, rhs, rtol=1e-12, atol=0.0)
    if info != 0:
        raise RuntimeError(f"GMRES did not converge: info={info}")
    residuals = {}
    for name, solution in (
        ("dense", np.linalg.solve(matrix, rhs)),
        ("sparse_direct", spsolve(sparse_matrix, rhs)),
        ("gmres", iterative),
    ):
        np.testing.assert_allclose(solution, expected, rtol=1e-12, atol=1e-12)
        residual = float(np.linalg.norm(matrix @ solution - rhs))
        if not np.isfinite(residual) or residual > 1e-10:
            raise RuntimeError(f"{name}: invalid residual {residual}")
        residuals[name] = residual

    with tempfile.TemporaryDirectory(prefix="fsi_ice_environment_") as directory:
        output = Path(directory) / "smoke.png"
        figure, axis = plt.subplots()
        axis.plot([0, 1], [0, 1])
        figure.savefig(output)
        plt.close(figure)
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("Headless plotting did not produce an image.")
        with output.open("rb") as stream:
            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                raise RuntimeError("Headless plotting did not produce a PNG.")

    test_tool = subprocess.run(
        [sys.executable, "-m", "pytest", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(json.dumps({
        "status": "PASS",
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "pytest": pytest.__version__,
        "linear_residuals": residuals,
        "headless_plot": "PASS",
        "test_tool": test_tool,
        "scope": "Environment smoke check; no physics solver tested.",
    }, indent=2))


if __name__ == "__main__":
    main()

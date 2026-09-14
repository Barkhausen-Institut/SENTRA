#!/usr/bin/env python
"""
Test runner script for SENTRA ML Training Pipeline
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ``sentra-node/python`` (parent of ``node/`` and ``client/``)
_PY_ROOT = Path(__file__).resolve().parents[2]


def run_tests(test_type="all", verbose=False, coverage=False):
    """Run tests based on type"""
    
    cmd = [sys.executable, "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=ml_training", "--cov-report=html", "--cov-report=term"])
    
    if test_type == "unit":
        # Run only unit tests (exclude integration and benchmarks)
        cmd.extend(["-m", "not integration and not benchmark"])
    elif test_type == "integration":
        # Run only integration tests
        cmd.extend(["-m", "integration"])
    elif test_type == "benchmarks":
        # Run only benchmarks
        cmd.extend([str(_PY_ROOT / "node/tests/test_benchmarks.py"), "-m", "benchmark"])
    elif test_type == "fast":
        # Run fast tests (exclude slow)
        cmd.extend(["-m", "not slow"])
    elif test_type == "all":
        # Run all tests
        pass
    elif test_type == "batched_regression":
        # Run the two batched secure MNIST regression integration tests
        cmd.extend(
            [
                "-m",
                "integration",
                str(_PY_ROOT / "node/testing/test_batched_stability_integration.py"),
                str(_PY_ROOT / "node/testing/test_batched_accuracy_floor_integration.py"),
            ]
        )
    else:
        # Run specific test file
        cmd.append(str(_PY_ROOT / f"node/tests/test_{test_type}.py"))
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(_PY_ROOT))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run SENTRA test suite")
    parser.add_argument(
        "--type",
        choices=["all", "unit", "integration", "benchmarks", "fast",
                 "secret_sharing", "kvs", "beaver_triples", "secure_comparison",
                 "secure_division", "mpc_engine", "secure_matrix_ops",
                 "communication", "batched_regression"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    args = parser.parse_args()
    
    exit_code = run_tests(args.type, args.verbose, args.coverage)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()


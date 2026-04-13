#!/usr/bin/env python3
"""
Simple benchmarking script for run_cudss.
Runs the run_cudss executable multiple times.
"""

import os
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    executable = os.path.join(REPO_ROOT, "build", "run_cudss")
    num_runs = 6
    num_rigs = 8  # Options: 1, 2, 4, 8, 10, 25
    precision = "double"  # Options: float, double
    iterations = 1  # Use values > 1 to benchmark repeated solve timing

    if not os.path.exists(executable):
        print(f"Error: {executable} not found. Make sure the project is built.")
        return

    command = [executable, "--precision", precision, "--rigs", str(num_rigs)]
    if iterations > 1:
        command.extend(["--iterations", str(iterations)])

    print(f"Running {executable} {num_runs} times with {num_rigs} rigs using {precision} precision...")
    if iterations > 1:
        print(f"Iterations per run: {iterations}")
    print(f"Command: {' '.join(command)}")
    print("=" * 60)

    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")

        try:
            result = subprocess.run(command, text=True, capture_output=True)

            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"Run {i+1} failed with return code {result.returncode}")
                if result.stderr:
                    print(f"Error output: {result.stderr}")

        except Exception as e:
            print(f"Error in run {i+1}: {e}")

    print("\n" + "=" * 60)
    print("Benchmarking complete!")


if __name__ == "__main__":
    main()

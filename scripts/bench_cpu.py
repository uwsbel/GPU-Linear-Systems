#!/usr/bin/env python3
"""
Simple benchmarking script for run_pardiso.
Runs the run_pardiso executable multiple times.
"""

import os
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    executable = os.path.join(REPO_ROOT, "build", "run_pardiso")
    num_runs = 6

    num_threads = "16"
    precision = "double"
    num_rigs = "8"

    if not os.path.exists(executable):
        print(f"Error: {executable} not found. Make sure the project is built.")
        return

    command = [executable, "--threads", num_threads, "--precision", precision, "--rigs", num_rigs]
    print(f"Running {executable} {num_runs} times...")
    print(f"Parameters: {num_threads} threads, {precision} precision, {num_rigs} rigs")
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

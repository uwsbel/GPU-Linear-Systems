#!/usr/bin/env python3
import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple


# Configuration (hardcoded values)
BUILD_DIR = './build/'
MAX_THREADS = 1  # Default to number of CPU cores
ITERATIONS = 3
OUTPUT_FILE = 'benchmark_results.png'
PROGRAMS = ['task_eigen_pardiso', 'task_pardiso']  # Empty list means all executables in build directory


def find_executables(build_dir: str) -> List[str]:
    """Find all executable files in the build directory."""
    executables = []
    build_path = Path(build_dir)
    
    if not build_path.exists():
        print(f"Error: Build directory '{build_dir}' does not exist.")
        return []
    
    for item in build_path.iterdir():
        if item.is_file() and os.access(item, os.X_OK):
            executables.append(str(item))
    
    return sorted(executables)


def run_benchmark(executable: str, num_threads: int, iterations: int) -> Tuple[float, float]:
    """Run a benchmark for a given executable with specified number of threads."""
    times = []
    
    for i in range(iterations):
        start_time = time.time()
        try:
            result = subprocess.run(
                [executable, str(num_threads)],  # Pass num_threads as an argument
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
            print(f"Run {i+1}/{iterations} of {os.path.basename(executable)} with {num_threads} threads: {elapsed:.4f}s")
        except subprocess.CalledProcessError as e:
            print(f"Error running {executable}: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return 0, 0
    
    avg_time = np.mean(times)
    std_dev = np.std(times)
    return avg_time, std_dev


def plot_results(results: Dict[str, Dict[int, Tuple[float, float]]], output_file: str = "benchmark_results.png"):
    """Generate plots from benchmark results."""
    plt.figure(figsize=(12, 8))
    
    # Plot average execution time vs number of threads
    plt.subplot(2, 1, 1)
    for program, thread_results in results.items():
        threads = sorted(thread_results.keys())
        avg_times = [thread_results[t][0] for t in threads]
        program_name = os.path.basename(program)
        plt.plot(threads, avg_times, marker='o', label=program_name)
    
    plt.xlabel('Number of Threads')
    plt.ylabel('Average Execution Time (s)')
    plt.title('Benchmark Results: Execution Time vs Number of Threads')
    plt.grid(True)
    plt.legend()
    
    # Plot speedup vs number of threads
    plt.subplot(2, 1, 2)
    for program, thread_results in results.items():
        threads = sorted(thread_results.keys())
        if threads and 1 in thread_results and thread_results[1][0] > 0:
            base_time = thread_results[1][0]  # Time with 1 thread
            speedups = [base_time / thread_results[t][0] for t in threads]
            program_name = os.path.basename(program)
            plt.plot(threads, speedups, marker='o', label=program_name)
    
    # Add ideal speedup line
    max_threads = max(max(results[p].keys()) for p in results) if results else 1
    plt.plot([1, max_threads], [1, max_threads], 'k--', label='Ideal Speedup')
    
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.title('Benchmark Results: Speedup vs Number of Threads')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Results plotted and saved to {output_file}")


def print_results_table(results: Dict[str, Dict[int, Tuple[float, float]]]):
    """Print a formatted table of benchmark results."""
    if not results:
        print("No results to display.")
        return
    
    # Get all thread counts across all programs
    all_threads = set()
    for program_results in results.values():
        all_threads.update(program_results.keys())
    thread_counts = sorted(all_threads)
    
    # Print header
    header = "Program"
    for threads in thread_counts:
        header += f" | {threads} thread{'s' if threads > 1 else ''} (s)"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    
    # Print results for each program
    for program, thread_results in results.items():
        program_name = os.path.basename(program)
        row = program_name
        for threads in thread_counts:
            if threads in thread_results:
                avg_time, std_dev = thread_results[threads]
                row += f" | {avg_time:.4f} ± {std_dev:.4f}"
            else:
                row += " | N/A"
        print(row)
    
    print("=" * len(header) + "\n")


def main():
    # Use hardcoded configuration
    build_dir = BUILD_DIR
    max_threads = MAX_THREADS
    iterations = ITERATIONS
    output_file = OUTPUT_FILE
    programs = PROGRAMS
    
    # Find executables
    if programs:
        executables = [os.path.join(build_dir, prog) if not os.path.isabs(prog) else prog 
                      for prog in programs]
        # Verify they exist and are executable
        executables = [exe for exe in executables if os.path.isfile(exe) and os.access(exe, os.X_OK)]
    else:
        executables = find_executables(build_dir)
    
    if not executables:
        print("No executable programs found to benchmark.")
        return
    
    print(f"Found {len(executables)} executable(s) to benchmark:")
    for exe in executables:
        print(f"  - {exe}")
    
    # Run benchmarks
    results = {}
    thread_counts = [1] + [t for t in range(2, max_threads + 1, 2)]
    if max_threads > 1 and max_threads % 2 != 0:
        thread_counts.append(max_threads)
    
    for executable in executables:
        program_results = {}
        print(f"\nBenchmarking {os.path.basename(executable)}...")
        
        for threads in thread_counts:
            print(f"\nRunning with {threads} thread{'s' if threads > 1 else ''}:")
            avg_time, std_dev = run_benchmark(executable, threads, iterations)
            program_results[threads] = (avg_time, std_dev)
        
        results[executable] = program_results
    
    # Print and plot results
    print_results_table(results)
    plot_results(results, output_file)


if __name__ == "__main__":
    main()

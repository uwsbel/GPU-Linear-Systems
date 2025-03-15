import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os

IMPLEMENTATIONS = {
    'naive': {'exe': './task1_naive', 'color': 'blue', 'env_var': 'OMP'},
    'mkl': {'exe': './task1_mkl', 'color': 'red', 'env_var': 'MKL'}
}

MATRIX_SIZE = 1024
RUNS = 1
THREADS = list(range(1, 5))

def run_benchmark(exe_name, threads, impl_type):
    try:
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = "/opt/intel/oneapi/mkl/latest/lib:" + env.get('LD_LIBRARY_PATH', '')
        
        # Set appropriate environment variables based on implementation
        if impl_type == 'MKL':
            env['MKL_NUM_THREADS'] = str(threads)
        elif impl_type == 'OMP':  # For OpenMP-based implementations
            env['OMP_NUM_THREADS'] = str(threads)
        
        result = subprocess.run(
            [exe_name, str(MATRIX_SIZE), str(threads)],
            capture_output=True, text=True, timeout=300,
            env=env  # Pass the modified environment
        )
        return float(result.stdout.strip().split('\n')[-1])
    except Exception as e:
        print(f"Error running {exe_name} with {threads} threads: {str(e)}")
        return None

def main():
    # Create results dictionary
    results = {impl: [] for impl in IMPLEMENTATIONS}
    
    # Run benchmarks for each implementation and thread count
    for impl_name, impl_info in IMPLEMENTATIONS.items():
        for t in THREADS:
            print(f"Running {impl_name} with {t} threads...")
            times = []
            for r in range(RUNS):
                time = run_benchmark(impl_info['exe'], t, impl_info['env_var'])
                if time is not None:
                    times.append(time)
            if times:
                avg_time = sum(times) / len(times)
                results[impl_name].append(avg_time)
                print(f"  Average time: {avg_time:.2f} ms")
            else:
                results[impl_name].append(None)
                print(f"  Failed to run benchmark")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for impl_name, impl_info in IMPLEMENTATIONS.items():
        if any(results[impl_name]):
            plt.plot(THREADS, results[impl_name], marker='o', color=impl_info['color'], label=impl_name)
    
    plt.xlabel('Number of Threads')
    plt.ylabel('Execution Time (ms)')
    plt.title(f'LU Decomposition Performance (Matrix Size: {MATRIX_SIZE})')
    plt.legend()
    plt.grid(True)
    plt.savefig('lu_decomposition_benchmark.png')
    plt.show()

if __name__ == "__main__":
    main() 
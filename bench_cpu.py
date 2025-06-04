#!/usr/bin/env python3
"""
Simple benchmarking script for task-pardiso
Runs the task-pardiso executable multiple times
"""

import subprocess
import os

def main():
    executable = "./build/task_pardiso"
    num_runs = 6
    
    # Parameters
    num_threads = "16"
    precision = "double"
    num_spokes = "80"
    
    # Check if executable exists
    if not os.path.exists(executable):
        print(f"Error: {executable} not found. Make sure the project is built.")
        return
    
    print(f"Running {executable} {num_runs} times...")
    print(f"Parameters: {num_threads} threads, {precision} precision, {num_spokes} spokes")
    print("=" * 60)
    
    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        
        try:
            # Run the executable with parameters
            result = subprocess.run([executable, num_threads, precision, num_spokes], 
                                  text=True, capture_output=True)
            
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
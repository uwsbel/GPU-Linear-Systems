#!/usr/bin/env python3
"""
Simple benchmarking script for task_cudss
Runs the task_cudss executable 5 times and prints the output
"""

import subprocess
import os

def main():
    executable = "./build/task_cudss"
    num_runs = 6
    
    # Check if executable exists
    if not os.path.exists(executable):
        print(f"Error: {executable} not found. Make sure the project is built.")
        return
    
    print(f"Running {executable} {num_runs} times...")
    print("=" * 60)
    
    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        
        try:
            # Run the executable and print its output directly
            result = subprocess.run([executable, "80", "--float"], text=True)
            
            if result.returncode != 0:
                print(f"Run {i+1} failed with return code {result.returncode}")
                
        except Exception as e:
            print(f"Error in run {i+1}: {e}")
    
    print("\n" + "=" * 60)
    print("Benchmarking complete!")

if __name__ == "__main__":
    main()

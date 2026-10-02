#!/usr/bin/env bash
# Offline refinement sweep on the ANCF 3443 shell (edge clamped by constraints):
# cuDSS 0.8 (PIVOT_AUTO) vs oneMKL Pardiso (16 threads), 5 interleaved reps per resolution.
set -u
cd "$(dirname "$0")/../.."
OUT=sweep_logs/shell
for r in 10 20 30 40 50 70 100 140 200; do
  for i in 1 2 3 4 5; do
    ./build/run_pardiso --threads 16 --precision double --rigs $r --shell > "$OUT/pardiso_${r}_${i}.log" 2>&1
    echo "$(date +%T) pardiso grid$r rep$i exit=$?"
    ./build/run_cudss --precision double --rigs $r --shell > "$OUT/cudss_${r}_${i}.log" 2>&1
    echo "$(date +%T) cudss   grid$r rep$i exit=$?"
  done
done
echo "SWEEP COMPLETE $(date +%T)"

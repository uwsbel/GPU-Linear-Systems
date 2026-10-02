#!/usr/bin/env bash
# Phase 1 offline scaling sweep: cuDSS 0.8 (PIVOT_AUTO) vs oneMKL Pardiso (16 threads)
set -u
cd "$(dirname "$0")/.."
OUT=sweep_logs
for r in 1 2 4 8 10 25 50; do
  if [ "$r" -ge 25 ]; then REPS=3; else REPS=5; fi
  for i in $(seq 1 $REPS); do
    echo "### pardiso rigs=$r rep=$i $(date +%T)"
    ./build/run_pardiso --threads 16 --precision double --rigs $r > "$OUT/pardiso_${r}_${i}.log" 2>&1
    echo "   exit=$?"
    echo "### cudss rigs=$r rep=$i $(date +%T)"
    ./build/run_cudss --precision double --rigs $r > "$OUT/cudss_${r}_${i}.log" 2>&1
    echo "   exit=$?"
  done
done
echo "### SWEEP COMPLETE $(date +%T)"

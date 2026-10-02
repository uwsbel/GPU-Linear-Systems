#!/usr/bin/env bash
# Extra repetitions 4-10 for 25 and 50 rigs (same commands as run_sweep.sh)
set -u
cd "$(dirname "$0")/.."
OUT=sweep_logs
for r in 25 50; do
  for i in $(seq 4 10); do
    echo "### pardiso rigs=$r rep=$i $(date +%T)"
    ./build/run_pardiso --threads 16 --precision double --rigs $r > "$OUT/pardiso_${r}_${i}.log" 2>&1
    echo "   exit=$?"
    echo "### cudss rigs=$r rep=$i $(date +%T)"
    ./build/run_cudss --precision double --rigs $r > "$OUT/cudss_${r}_${i}.log" 2>&1
    echo "   exit=$?"
  done
done
echo "### REPS COMPLETE $(date +%T)"

#!/usr/bin/env bash
# cuDSS-only reruns with GPU clock/temperature/power logging, to explain the 50-rig factorization spread
set -u
cd "$(dirname "$0")/../.."
OUT=sweep_logs/clockcheck
for spec in "50 1" "50 2" "50 3" "50 4" "50 5" "25 1" "50 6" "50 7" "50 8" "25 2" "50 9" "50 10" "25 3"; do
  set -- $spec; r=$1; i=$2
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,clocks_throttle_reasons.active --format=csv,noheader,nounits -lms 20 > "$OUT/smi_${r}_${i}.csv" &
  SMI=$!
  echo "### cudss rigs=$r rep=$i $(date +%T)"
  ./build/run_cudss --precision double --rigs $r > "$OUT/cudss_${r}_${i}.log" 2>&1
  echo "   exit=$?"
  kill $SMI; wait $SMI 2>/dev/null
done
echo "### DONE $(date +%T)"

#!/usr/bin/env bash
# Export the ANCF shell systems (continuous integration, edge clamped by constraints) for every grid.
set -u
D=/home/ganesh/work/GPU-Linear-Systems/data/ancf/shell
cd /home/ganesh/work/chrono-gnsh-cudss/build || exit 1
for g in 10 20 30 40 50 70 100 140 200; do
  [ -f $D/$g/solve_4_0_Z.dat ] && { echo "grid $g already exported"; continue; }
  mkdir -p $D/$g
  /usr/bin/time -f "%es peakRSS=%MkB" ./bin/jz_FEA_3443_check --grid $g --nthreads 16 --export_step 3 --export_dir $D/$g > $D/$g/export.log 2>&1
  echo "$(date +%T) grid $g exit=$? $(tail -1 $D/$g/export.log)"
  find $D/$g -type f ! -name 'solve_4_0_Z.dat' ! -name 'solve_4_0_rhs.dat' ! -name 'solve_4_0_Dv.dat' ! -name 'solve_4_0_Dl.dat' ! -name 'export.log' -delete
done

#!/bin/bash
# File:    run_scaling.sh
# Purpose: Strong-scaling sweep: same problem, several thread counts,
#          R repetitions each.  Keeps the best (minimum) serial and
#          parallel times, then prints speedup and efficiency as CSV.
# Usage:   ./run_scaling.sh <program> "<args>" "<thread counts>" [reps]
#          e.g. ./run_scaling.sh ./bin/map_saxpy "50000000 10" "1 2 4 8" 3
# Needs:   each program prints a line  CSV,...,T_serial,T_parallel,S,E

if [ $# -lt 3 ]; then
   echo "usage: $0 <program> \"<args>\" \"<thread counts>\" [reps]" >&2
   exit 1
fi

prog=$1; args=$2; threads=$3; reps=${4:-3}

if [ ! -x "$prog" ]; then
   echo "error: '$prog' not found or not executable (did you build it?)" >&2
   exit 1
fi

echo "threads,best_T_serial_s,best_T_parallel_s,speedup,efficiency"
for p in $threads; do
   bs=""; bp=""
   for r in $(seq 1 "$reps"); do
      # $args is left unquoted on purpose so "50000000 10" splits into two arguments
      line=$("$prog" "$p" $args | grep '^CSV')
      if [ -z "$line" ]; then
         echo "error: no CSV line from '$prog $p $args'" >&2
         exit 1
      fi
      ts=$(echo "$line" | awk -F, '{print $(NF-3)}')
      tp=$(echo "$line" | awk -F, '{print $(NF-2)}')
      bs=$(awk -v a="$bs" -v b="$ts" 'BEGIN{print (a=="" || b+0<a+0) ? b : a}')
      bp=$(awk -v a="$bp" -v b="$tp" 'BEGIN{print (a=="" || b+0<a+0) ? b : a}')
   done
   awk -v p="$p" -v ts="$bs" -v tp="$bp" \
       'BEGIN{s=ts/tp; printf "%d,%.6f,%.6f,%.2f,%.2f\n", p, ts, tp, s, s/p}'
done

#!/bin/zsh
# MLA decode kernel (MJ_MLA_DECODE=kernel, mj-mlak) vs SDPA over gathered rows (mj), DeepSeek-V2-Lite, ABAB.
set -u
cd $HOME/Documents/Projects/mj-mlx
PY=.venv/bin/python; SV=results/v13_scripts/services.sh
trap 'zsh $SV start; echo "cleanup done $(date +%T)"' EXIT
zsh $SV stop; sleep 30
echo "== K mla kernel vs sdpa $(date +%T)"
$PY tools/mlx_bench_suite.py --models dsv2lite --engines mj,mj-mlak --tests tg128@d0,tg128@d2048,tg128@d8192,tg128@d16384 \
  --reps 3 --rounds 2 --out results/bench_suite/v15_mla_kernel > results/bench_suite/v15_mla_kernel.log 2>&1
echo "K exit $? $(date +%T)"

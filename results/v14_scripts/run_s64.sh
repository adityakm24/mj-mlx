#!/bin/zsh
# After the memory probe: prefill attention score budget 64 MB (mj-s64) vs 256 MB (mj), ABAB, at every context the
# README quotes and at 8k / 16k, on DeepSeek-V2-Lite (MLA 192 / 128) and Qwen3.5-9B (head size 256).
set -u
cd $HOME/Documents/Projects/mj-mlx
PY=.venv/bin/python; SV=results/v13_scripts/services.sh
trap 'zsh $SV start; echo "cleanup done $(date +%T)"' EXIT
zsh $SV stop; sleep 30
echo "== S1 dsv2 s64 vs s256 $(date +%T)"
$PY tools/mlx_bench_suite.py --models dsv2lite --engines mj,mj-s64 --tests pp512,pp2048,tg128@d2048,pp8192,pp16384 \
  --reps 3 --rounds 2 --out results/bench_suite/v14_s64_dsv2 > results/bench_suite/v14_s64_dsv2.log 2>&1
echo "S1 exit $? $(date +%T)"; sleep 30
echo "== S2 q9b s64 vs s256 $(date +%T)"
$PY tools/mlx_bench_suite.py --models q9b --engines mj,mj-s64 --tests pp512,pp2048,tg128@d2048,pp8192 \
  --reps 3 --rounds 2 --out results/bench_suite/v14_s64_q9b > results/bench_suite/v14_s64_q9b.log 2>&1
echo "S2 exit $? $(date +%T)"
echo "all done $(date +%T)"

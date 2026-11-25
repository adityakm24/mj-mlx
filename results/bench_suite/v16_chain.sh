#!/bin/zsh
# v16 after the 2026-11-23 restart: Qwen3.6 retest (adds llama-server beside llama-bench: client-overhead check),
# then the clean gpt-oss suite. Sequential; each suite stops its own servers.
cd ~/Documents/Projects/mj-mlx
.venv/bin/python tools/mlx_bench_suite.py --models q35b --engines mj,llama.cpp,llama-server --reps 5 --rounds 2 \
  --out results/bench_suite/v16_q35b > results/bench_suite/v16_q35b.log 2>&1
.venv/bin/python tools/mlx_bench_suite.py --models q35b --summary-only --out results/bench_suite/v16_q35b >> results/bench_suite/v16_q35b.log 2>&1
.venv/bin/python tools/mlx_bench_suite.py --models gptoss --reps 5 --rounds 2 \
  --out results/bench_suite/v16_gptoss > results/bench_suite/v16_gptoss.log 2>&1
.venv/bin/python tools/mlx_bench_suite.py --models gptoss --summary-only --out results/bench_suite/v16_gptoss >> results/bench_suite/v16_gptoss.log 2>&1
echo CHAIN_DONE

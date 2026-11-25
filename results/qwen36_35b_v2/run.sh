#!/bin/zsh
# Qwen3.6-35B-A3B (20.4 GB, bigger than this 16 GB Mac's RAM), 2026-11-18 morning: rerun of RESULTS 1k (3) after the
# 1p fixes. Ours: SSD expert streaming, 96 of 256 slots, prefetch, 512-token random prompt (seed 0), 64 new tokens,
# at MLX_MAX_MB_PER_BUFFER 150 (new default) and 40. llama.cpp: llama-bench, all experts on the CPU (-ncmoe 40, its
# best setting in 1k), 512-token prompt test and 64-token generation test. Two rounds, round 2 reversed.
cd $HOME/Documents/Projects/MilliJoule
Q=results/qwen36_35b_v2
M=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/*/)
STORE=~/.cache/millijoule/qwen36-35b-experts-stream
GGUF=~/.cache/millijoule/gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf
PY=.venv/bin/python
caffeinate -i -w $$ &
state() { $PY tools/mlx_machine_state.py $Q/machine_state_$1.json; }
ours() {  # round mb
  MLX_MAX_MB_PER_BUFFER=$2 $PY tools/mlx_stream_run.py --model $M --experts $STORE --slots 96 --prefetch \
    --prompt-tokens 512 --new 64 --pair r$1 --out $Q/r$1_stream96_mb$2.json 2>&1 | grep '^{' | cut -c1-400 | sed "s/^/r$1 mb$2 /"
  sleep 5
}
llama() {  # round
  llama-bench -m $GGUF -ncmoe 40 -fa 1 -p 512 -n 64 -r 1 -o json > $Q/r$1_llama-bench.json 2> $Q/r$1_llama-bench.log
  $PY -c "import json,sys; [print('r$1 llama', x['n_prompt'], x['n_gen'], round(x['avg_ts'], 2)) for x in json.load(open(sys.argv[1]))]" $Q/r$1_llama-bench.json
  sleep 5
}
echo "start $(date +%H:%M:%S)"; state start
ours 1 150; ours 1 40; llama 1
state mid
llama 2; ours 2 40; ours 2 150
state end
echo "done $(date +%H:%M:%S)"

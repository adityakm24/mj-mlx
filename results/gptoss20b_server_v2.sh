#!/bin/zsh
# v2: after the shared expert mapping and MLX_MAX_MB_PER_BUFFER (RESULTS 1p).
# gpt-oss-20b, same 512 random token ids, 128 greedy tokens: ours (mapped experts, packed attention) vs llama-server.
cd $HOME/Documents/Projects/MilliJoule
O=results/gptoss20b_server_v2; mkdir -p $O
GGUF=$HOME/.cache/huggingface/hub/models--ggml-org--gpt-oss-20b-GGUF/snapshots/41e8a94d15d62070b36cf4553a0250717ecf46b8/gpt-oss-20b-MXFP4.gguf
HF=$(ls -d ~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*)
MAP=~/.cache/millijoule/gptoss20b-experts-mapped
PY=.venv/bin/python
caffeinate -i -w $$ &
$PY tools/mlx_machine_state.py $O/machine_state_start.json
ours() {
  cat $MAP/*.bin > /dev/null
  $PY tools/mlx_engine_compare.py --engine mj-overlap --model $HF --map-experts $MAP --attn-pack --lean --wired-mb 3000 --prompt-len 512 --arm ours-pack --round $1 --weights "openai/gpt-oss-20b, bf16 attention packed losslessly, MXFP4 experts mapped" --out $O/r$1_ours-pack.json 2>&1 | grep "rep "
}
bf16() {
  cat $MAP/*.bin > /dev/null
  $PY tools/mlx_engine_compare.py --engine mj-overlap --model $HF --map-experts $MAP --lean --wired-mb 3000 --prompt-len 512 --arm ours-bf16 --round $1 --weights "openai/gpt-oss-20b, bf16 attention, MXFP4 experts mapped" --out $O/r$1_ours-bf16.json 2>&1 | grep "rep "
}
llama() {
  llama-server -m $GGUF -fa on -ngl 99 -c 4096 --port 8080 > $O/r$1_llama-server.log 2>&1 &
  local pid=$!
  for i in {1..120}; do curl -s localhost:8080/health | grep -q ok && break; sleep 1; done
  $PY tools/mlx_engine_compare.py --engine llama-server --model $HF --url http://127.0.0.1:8080 --prompt-len 512 --arm llama-server --round $1 --weights "ggml-org gpt-oss-20b MXFP4 GGUF (Q8_0 attention)" --out $O/r$1_llama-server.json 2>&1 | grep "rep "
  kill $pid; wait $pid 2>/dev/null; sleep 3
}
ours 1; bf16 1; llama 1; llama 2; bf16 2; ours 2
$PY tools/mlx_machine_state.py $O/machine_state_end.json

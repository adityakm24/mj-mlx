#!/bin/zsh
# gpt-oss-20b, chat prompt (same ids for every engine), 128 greedy tokens: ours plain, ours + EAGLE-3 (k = 1, 2),
# llama-server plain (its best on this Mac: EAGLE-3 runs out of Metal memory with the draft on the GPU and is slower
# with it on the CPU, results/gptoss20b_eagle_llama_v1). Two rounds, round 2 reversed.
cd $HOME/Documents/Projects/MilliJoule
O=results/gptoss20b_eagle_v1; mkdir -p $O
GGUF=$HOME/.cache/huggingface/hub/models--ggml-org--gpt-oss-20b-GGUF/snapshots/41e8a94d15d62070b36cf4553a0250717ecf46b8/gpt-oss-20b-MXFP4.gguf
HF=$(ls -d ~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*)
MAP=~/.cache/millijoule/gptoss20b-experts-mapped
PY=.venv/bin/python
caffeinate -i -w $$ &
COMMON=(--model $HF --map-experts $MAP --attn-pack --lean --wired-mb 3000 --prompt chat)
ours() {  # round, arm, engine, k
  cat $MAP/*.bin > /dev/null
  $PY tools/mlx_engine_compare.py --engine $3 --k ${4:-2} $COMMON --arm $2 --round $1 --weights "openai/gpt-oss-20b exact (packed bf16 attention, MXFP4 experts mapped)" --out $O/r$1_$2.json 2>&1 | grep "rep " | sed "s/^/r$1 /"
}
llama() {
  llama-server -m $GGUF -fa on -ngl 99 -c 4096 --port 8080 > $O/r$1_llama-server.log 2>&1 &
  local pid=$!
  for i in {1..120}; do curl -s localhost:8080/health | grep -q ok && break; sleep 1; done
  $PY tools/mlx_engine_compare.py --engine llama-server --model $HF --url http://127.0.0.1:8080 --prompt chat --arm llama-server --round $1 --weights "ggml-org gpt-oss-20b MXFP4 GGUF (Q8_0 attention)" --out $O/r$1_llama-server.json 2>&1 | grep "rep " | sed "s/^/r$1 /"
  kill $pid; wait $pid 2>/dev/null; sleep 3
}
$PY tools/mlx_machine_state.py $O/machine_state_start.json
ours 1 plain mj-overlap; ours 1 eagle-k1 mj-eagle 1; ours 1 eagle-k2 mj-eagle 2; llama 1
llama 2; ours 2 eagle-k2 mj-eagle 2; ours 2 eagle-k1 mj-eagle 1; ours 2 plain mj-overlap
$PY tools/mlx_machine_state.py $O/machine_state_end.json

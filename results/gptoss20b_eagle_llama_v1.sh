#!/bin/zsh
# llama.cpp on gpt-oss-20b, chat prompt (same ids as ours), plain vs EAGLE-3 (nebius head, Q5_K_M GGUF), draft n 2/3/5.
cd $HOME/Documents/Projects/MilliJoule
O=results/gptoss20b_eagle_llama_v1; mkdir -p $O
GGUF=$HOME/.cache/huggingface/hub/models--ggml-org--gpt-oss-20b-GGUF/snapshots/41e8a94d15d62070b36cf4553a0250717ecf46b8/gpt-oss-20b-MXFP4.gguf
DRAFT=~/.cache/millijoule/eagle3/EAGLE3-gpt-oss-20b-Q5_K_M.gguf
HF=$(ls -d ~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*)
PY=.venv/bin/python
caffeinate -i -w $$ &
$PY tools/mlx_machine_state.py $O/machine_state_start.json
run() {  # name, extra server args...
  local name=$1; shift
  llama-server -m $GGUF -fa on -ngl 99 -c ${CTX:-4096} --port 8080 "$@" > $O/$name.log 2>&1 &
  local pid=$!
  for i in {1..120}; do curl -s localhost:8080/health | grep -q ok && break; sleep 1; done
  $PY tools/mlx_engine_compare.py --engine llama-server --model $HF --url http://127.0.0.1:8080 --prompt chat --arm $name --weights "gpt-oss-20b MXFP4 GGUF; $*" --out $O/$name.json 2>&1 | grep "rep " | sed "s/^/$name /"
  grep -i "draft acceptance\|accept" $O/$name.log | tail -2
  kill $pid; wait $pid 2>/dev/null; sleep 3
}
if [[ -z "$RETRY" ]]; then
run plain
run eagle3-n3 --spec-type draft-eagle3 -md $DRAFT -ngld 99 --spec-draft-n-max 3
run eagle3-n5 --spec-type draft-eagle3 -md $DRAFT -ngld 99 --spec-draft-n-max 5
run eagle3-n2 --spec-type draft-eagle3 -md $DRAFT -ngld 99 --spec-draft-n-max 2
else  # after the out-of-memory failures above: smaller contexts, and the draft on the CPU
run eagle3-n3-c1024 --spec-type draft-eagle3 -md $DRAFT -ngld 99 --spec-draft-n-max 3 -c 1024
run eagle3-n3-cpu --spec-type draft-eagle3 -md $DRAFT -ngld 0 --spec-draft-n-max 3
fi
$PY tools/mlx_machine_state.py $O/machine_state_end.json

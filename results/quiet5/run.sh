#!/bin/zsh
# Quiet window 5 (2026-11-18, overnight): final numbers for gpt-oss-20b (after RESULTS 1p fixes, EAGLE-3) vs llama.cpp
# on the same ids, and a Qwen3.5-9B anchor. Two rounds, round 2 reversed. No other model sessions; Mac unattended.
cd $HOME/Documents/Projects/MilliJoule
Q=results/quiet5
GGUF=$HOME/.cache/huggingface/hub/models--ggml-org--gpt-oss-20b-GGUF/snapshots/41e8a94d15d62070b36cf4553a0250717ecf46b8/gpt-oss-20b-MXFP4.gguf
HF=$(ls -d ~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*)
M9=$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/938d8919941c6e7efd3c7150eff7fe9d12afa631
MAP=~/.cache/millijoule/gptoss20b-experts-mapped
PY=.venv/bin/python
caffeinate -i -w $$ &
state() { $PY tools/mlx_machine_state.py $Q/machine_state_$1.json; }
G=(--model $HF --map-experts $MAP --attn-pack --lean --wired-mb 3000)
ours() {  # dir round prompt arm engine k [extra]
  cat $MAP/*.bin > /dev/null
  $PY tools/mlx_engine_compare.py --engine $5 --k $6 $G --prompt $3 ${=7} --arm $4 --round $2 --weights "openai/gpt-oss-20b exact (packed bf16 attention, MXFP4 experts mapped)" --out $Q/$1/r$2_$3_$4.json 2>&1 | grep "rep " | sed "s/^/$1 r$2 $3 /"
}
llama() {  # dir round prompt [extra]
  llama-server -m $GGUF -fa on -ngl 99 -c 4096 --port 8080 > $Q/$1/r$2_$3_llama-server.log 2>&1 &
  local pid=$!
  for i in {1..120}; do curl -s localhost:8080/health | grep -q ok && break; sleep 1; done
  $PY tools/mlx_engine_compare.py --engine llama-server --model $HF --url http://127.0.0.1:8080 --prompt $3 ${=4} --arm llama-server --round $2 --weights "ggml-org gpt-oss-20b MXFP4 GGUF (Q8_0 attention)" --out $Q/$1/r$2_$3_llama-server.json 2>&1 | grep "rep " | sed "s/^/$1 r$2 $3 /"
  kill $pid; wait $pid 2>/dev/null; sleep 3
}
echo "start $(date +%H:%M:%S)"; state start
mkdir -p $Q/gptoss
for rnd in 1 2; do
  if [[ $rnd == 1 ]]; then
    ours gptoss 1 chat plain mj-overlap 2; ours gptoss 1 chat eagle-auto mj-eagle auto; ours gptoss 1 chat eagle-k2 mj-eagle 2; llama gptoss 1 chat
    ours gptoss 1 random plain mj-overlap 2 "--prompt-len 512"; ours gptoss 1 random eagle-auto mj-eagle auto "--prompt-len 512"; llama gptoss 1 random "--prompt-len 512"
  else
    llama gptoss 2 random "--prompt-len 512"; ours gptoss 2 random eagle-auto mj-eagle auto "--prompt-len 512"; ours gptoss 2 random plain mj-overlap 2 "--prompt-len 512"
    llama gptoss 2 chat; ours gptoss 2 chat eagle-k2 mj-eagle 2; ours gptoss 2 chat eagle-auto mj-eagle auto; ours gptoss 2 chat plain mj-overlap 2
  fi
done
echo "gpt-oss done $(date +%H:%M:%S)"; state after_gptoss
mkdir -p $Q/qwen9b
for rnd in 1 2; do
  arms=(mlx-lm plain mtp); [[ $rnd == 2 ]] && arms=(mtp plain mlx-lm)
  for arm in $arms; do
    case $arm in
      mlx-lm) eng=mlx-lm; k=2 ;; plain) eng=mj-overlap; k=2 ;; mtp) eng=mj-mtp; k=auto-edp ;;
    esac
    $PY tools/mlx_engine_compare.py --engine $eng --k $k --model $M9 --prompt chat --arm $arm --round $rnd --out $Q/qwen9b/r${rnd}_chat_$arm.json 2>&1 | grep "rep " | sed "s/^/qwen9b r$rnd /"
  done
done
echo "done $(date +%H:%M:%S)"; state end

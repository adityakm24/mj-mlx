#!/bin/zsh
# Quiet window 5, part 2: EAGLE-3 adaptive after the k = 0 backfill change (head runs every 8th zero-draft iteration),
# random prompt plain vs adaptive (2 rounds, ABBA) and chat adaptive.
cd $HOME/Documents/Projects/MilliJoule
Q=results/quiet5
HF=$(ls -d ~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*)
MAP=~/.cache/millijoule/gptoss20b-experts-mapped
PY=.venv/bin/python
caffeinate -i -w $$ &
G=(--model $HF --map-experts $MAP --attn-pack --lean --wired-mb 3000)
ours() {  # round prompt arm engine k [extra]
  cat $MAP/*.bin > /dev/null
  $PY tools/mlx_engine_compare.py --engine $4 --k $5 $G --prompt $2 ${=6} --arm $3 --round $1 --weights "openai/gpt-oss-20b exact (packed bf16 attention, MXFP4 experts mapped)" --out $Q/gptoss/r$1_$2_$3.json 2>&1 | grep "rep " | sed "s/^/r$1 $2 /"
}
$PY tools/mlx_machine_state.py $Q/machine_state_part2_start.json
ours 3 random plain mj-overlap 2 "--prompt-len 512"; ours 3 random eagle-auto mj-eagle auto "--prompt-len 512"
ours 4 random eagle-auto mj-eagle auto "--prompt-len 512"; ours 4 random plain mj-overlap 2 "--prompt-len 512"
ours 3 chat eagle-auto mj-eagle auto
$PY tools/mlx_machine_state.py $Q/machine_state_part2_end.json

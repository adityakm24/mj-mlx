#!/bin/zsh
# A/B of the lean host gate (MJ_LEAN_GATE): suite gpt-oss configs (exact packed attention and 8-bit), ABAB.
cd $HOME/Documents/Projects/MilliJoule
O=results/decode_state/gate_ab.jsonl
M='~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*'
for round in 0 1; do
  for cfg in a8 exact; do
    for gate in 0 1; do
      A=(--attn-bits 8); [[ $cfg == exact ]] && A=(--attn-pack)
      echo "round $round $cfg gate=$gate $(date +%H:%M:%S) $(sysctl -n vm.swapusage)"
      MJ_MLOCK=1 MJ_LEAN_GATE=$gate .venv/bin/python tools/mlx_bench_worker.py --engine mj --model "$M" \
        --map-experts ~/.cache/millijoule/gptoss20b-experts-mapped $A --lean --wired-mb 3000 --chunk 512 \
        --tests tg128@d0,tg128@d2048 --reps 3 --seed0 $((round * 10)) --label mj-$cfg-gate$gate --tag $cfg-gate$gate \
        --out $O 2>&1 | grep --line-buffered -v -i warn
    done
  done
done
echo done $(date +%H:%M:%S)

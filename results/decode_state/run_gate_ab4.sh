#!/bin/zsh
# Host gate A/B, take 4: gated decode steps switch to one command buffer each (prefill keeps MLX_MAX_MB_PER_BUFFER=150).
# Arms: gate off / on, for mj-a8 and mj (exact); ABAB; the suite's 4 tests.
cd $HOME/Documents/Projects/MilliJoule
O=results/decode_state/gate_ab4.jsonl
M='~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*'
for round in 0 1; do
  for cfg in a8 exact; do
    for gate in 0 1; do
      A=(--attn-bits 8); [[ $cfg == exact ]] && A=(--attn-pack)
      echo "round $round $cfg gate=$gate $(date +%H:%M:%S) $(sysctl -n vm.swapusage)"
      MJ_MLOCK=1 MJ_LEAN_GATE=$gate .venv/bin/python tools/mlx_bench_worker.py --engine mj --model "$M" \
        --map-experts ~/.cache/millijoule/gptoss20b-experts-mapped $A --lean --wired-mb 3000 --chunk 512 \
        --tests pp512,pp2048,tg128@d0,tg128@d2048 --reps 3 --seed0 $((round * 10)) --label mj-$cfg-gate$gate \
        --tag $cfg-gate$gate --out $O 2>&1 | grep --line-buffered -v -i warn
    done
  done
done
echo done $(date +%H:%M:%S)

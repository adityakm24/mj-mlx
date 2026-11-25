#!/bin/zsh
# Host gate A/B, take 3: MLX_MAX_MB_PER_BUFFER=2600 for every step (arms MB150, MB2600, MB2600 + gate). mj-a8, ABAB.
# gated ones cannot finish until signalled), so arms: MB150 (default), MB/ops unlimited, unlimited + gate. mj-a8, ABAB.
cd $HOME/Documents/Projects/MilliJoule
O=results/decode_state/gate_ab3.jsonl
M='~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*'
for round in 0 1; do
  for arm in mb150 mb2600 mb2600_gate; do
    E=(MJ_MLOCK=1)
    [[ $arm == mb2600* ]] && E+=(MLX_MAX_MB_PER_BUFFER=2600 MLX_MAX_OPS_PER_BUFFER=1000000)
    [[ $arm == *gate ]] && E+=(MJ_LEAN_GATE=1)
    echo "round $round $arm $(date +%H:%M:%S) $(sysctl -n vm.swapusage)"
    env $E .venv/bin/python tools/mlx_bench_worker.py --engine mj --model "$M" \
      --map-experts ~/.cache/millijoule/gptoss20b-experts-mapped --attn-bits 8 --lean --wired-mb 3000 --chunk 512 \
      --tests pp512,pp2048,tg128@d0,tg128@d2048 --reps 3 --seed0 $((round * 10)) --label mj-a8-$arm --tag $arm \
      --out $O 2>&1 | grep --line-buffered -v -i warn
  done
done
echo done $(date +%H:%M:%S)

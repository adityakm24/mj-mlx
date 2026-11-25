#!/bin/zsh
# A/B of the fused prefill MoE glue (MJ_PREFILL_GLUE), suite mj-a8 config, ABAB, 3 timed reps per test.
cd $HOME/Documents/Projects/MilliJoule
O=results/gptoss_prefill_parts/glue_ab.jsonl
M='~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*'
for round in 0 1; do
  for arm in 0 1; do
    echo "round $round glue=$arm $(date +%H:%M:%S) $(sysctl -n vm.swapusage)"
    MJ_MLOCK=1 MJ_PREFILL_GLUE=$arm .venv/bin/python tools/mlx_bench_worker.py --engine mj --model "$M" \
      --map-experts ~/.cache/millijoule/gptoss20b-experts-mapped --attn-bits 8 --lean --wired-mb 3000 --chunk 512 \
      --tests pp512,pp2048,tg128@d2048 --reps 3 --seed0 $((round * 10)) --label mj-a8-glue$arm --tag glue$arm \
      --out $O 2>&1 | grep -v -i warn
  done
done
echo done $(date +%H:%M:%S)

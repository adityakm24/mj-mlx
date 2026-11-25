#!/bin/zsh
# After run_v14: what keeps DeepSeek-V2-Lite's 16k prefill at the 2048 chunk above Metal's limit (v14_attn_blocks: worker
# killed in round 1; "required > max" only in default-chunk blocks; MLX peak 12.45 vs 11.99 GB at chunk 512).
# One fresh worker per setting, pp16384 x 2 reps; MLX peak and kernel over-limit lines per worker. A probe, not a result.
set -u
cd $HOME/Documents/Projects/mj-mlx
PY=.venv/bin/python; SV=results/v13_scripts/services.sh; O=results/dsv2_16k_mem_probe_v1
END=$(date -j -f '%H:%M' '23:30' +%s)
k=0
trap 'zsh $SV start; echo "cleanup done $(date +%T)"' EXIT
mkdir -p $O; zsh $SV stop; sleep 30
M=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V2-Lite-Chat-4bit-mlx/snapshots/* | head -1)
run() {  # tag chunk scores_mb [extra args]
  [[ $(date +%s) -lt $END ]] || { echo "past end, skip $1"; return; }
  T=$(date '+%Y-%m-%d %H:%M:%S'); echo "== $1 chunk $2 scores $3 ${@:4} $(date +%T) swap $(sysctl -n vm.swapusage | awk '{print $6}')"
  MJ_MLA_DECODE=${MJ_MLA_DECODE:-sdpa} MJ_ATTN_SCORES_MB=$3 $PY tools/mlx_bench_worker.py --engine mj --label $1 --model $M --placement auto --chunk $2 ${@:4} \
    --tests ${TESTS:-pp16384} --reps 2 --out $O/$1.jsonl > $O/$1.log 2>&1
  echo "   exit $? $(date +%T)"
  n=$(/usr/bin/log show --start "$T" --predicate 'eventMessage CONTAINS "required > max"' --style compact 2>/dev/null | grep -c 'required > max')
  echo "   required>max lines: $n"; grep "tok/s" $O/$1.log | sed 's/^/   /'
  $PY -c "import json;[print('   mlx_peak_gb', round(json.loads(l)['mlx_peak_gb'],2)) for l in open('$O/$1.jsonl') if 'mlx_peak_gb' in l]" 2>/dev/null | tail -1
  sleep 30
}
run c2048_s256 2048 256
run c2048_s64 2048 64
run c1024_s256 1024 256
run c512_s256 512 256
run c2048_s256_w10000 2048 256 --wired-mb 10000
run c2048_s256_again 2048 256
TESTS=pp8192 run c2048_s256_8k 2048 256
TESTS=pp8192 run c1024_s256_8k 1024 256
# 0.7c: MLA decode at long context, SDPA over gathered latent rows (default) vs the paged split-K kernel; chunk 512
# so the 16k prefill fits. ABBA, 2 reps per test per block.
for arm in sdpa paged paged sdpa; do
  TESTS=tg128@d2048,tg128@d8192,tg128@d16384 MJ_MLA_DECODE=$arm run mla_${arm}_$((++k)) 512 256
done
echo "probe done $(date +%T)"

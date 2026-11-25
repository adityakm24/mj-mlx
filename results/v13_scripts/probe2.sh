#!/bin/zsh
# gpt-oss exact arm memory probes (one worker per setting), with wired/swap sampling and kernel-log checks.
set -u
cd $HOME/Documents/Projects/mj-mlx
OUT=results/gptoss_mem_fix_v2; mkdir -p $OUT
M=$(ls -d ~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/* | head -1)
MAP=~/.cache/millijoule/gptoss20b-experts-mapped
TESTS=pp2048,tg128@d0,tg128@d2048
run() {  # name mlock wired chunk
  local n=$1 ml=$2 w=$3 c=$4
  echo "== $n mlock=$ml wired=$w chunk=$c $(date +%T)"
  cat $MAP/*.bin > /dev/null   # prewarm the page cache, as the suite does
  ( while true; do echo "$(date +%s) $(vm_stat | awk '/wired down/{print $4*16384/1e9}') $(sysctl -n vm.swapusage | awk '{print $6}')"; sleep 2; done ) > $OUT/${n}_vm.log &
  local smp=$!
  local t0=$(date '+%Y-%m-%d %H:%M:%S')
  MJ_MLOCK=$ml /usr/bin/time -l .venv/bin/python tools/mlx_bench_worker.py --engine mj --model $M --map-experts $MAP \
    --attn-pack --lean --wired-mb $w --chunk $c --tests $TESTS --reps 3 --label mj-$n --out $OUT/$n.jsonl \
    > $OUT/$n.log 2>&1
  echo "exit $? $(date +%T)"
  kill $smp
  /usr/bin/log show --start "$t0" --predicate 'eventMessage CONTAINS "memorystatus" OR eventMessage CONTAINS "required > max"' --style compact 2>/dev/null \
    | grep -E 'killing|required > max' | head -20 > $OUT/${n}_kernel.txt
  echo "kernel lines: $(wc -l < $OUT/${n}_kernel.txt); peak wired GB: $(awk '{print $2}' $OUT/${n}_vm.log | sort -n | tail -1); swap: $(head -1 $OUT/${n}_vm.log | awk '{print $3}') -> $(tail -1 $OUT/${n}_vm.log | awk '{print $3}')"
  grep -E 'maximum resident|peak memory' $OUT/$n.log
  sleep 30
}
run w1000 1 1000 512
run w1500nl 0 1500 512
run w1000nl 0 1000 512
echo "probes done $(date +%T)"

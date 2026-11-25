#!/bin/zsh
# Tonight's run (resumed 2026-11-21 morning): B suite v13 gpt-oss, C soak of mj-x, D 16k context. Servers stopped and
# login services restarted at the end, whatever happens.
set -u
S=/private/tmp/claude-501/-Users-user-Documents-Projects-MilliJoule/ae1ca8d8-c94f-4eb0-b984-6492edd379c3/scratchpad
cd $HOME/Documents/Projects/mj-mlx
PY=.venv/bin/python
cleanup() {
  ~/.lmstudio/bin/lms server stop >/dev/null 2>&1; osascript -e 'quit app "LM Studio"' >/dev/null 2>&1
  pkill -f 'ollama serve' 2>/dev/null; pkill -f 'ollama runner' 2>/dev/null; pkill -f vllm-mlx 2>/dev/null
  kill $VMS $TOPS 2>/dev/null
  zsh $S/services.sh start
  echo "cleanup done $(date +%T)"
}
trap cleanup EXIT
mkdir -p results/bench_suite/v13_gptoss results/bench_suite/v13_long_dsv2 results/bench_suite/v13_long_gptoss results/gptoss_soak_v1
LOGS=results/bench_suite/v13_gptoss
( while true; do echo "$(date +%T) $(vm_stat | awk '/Swapouts/{o=$2} /Swapins/{i=$2} /wired down/{w=$4} END{printf "wired_GB %.2f swapins %s swapouts %s", w*16384/1e9, i, o}') $(sysctl -n vm.swapusage | awk '{print "swap_used", $6}')"; sleep 5; done ) > $LOGS/vmstat_5s.log &
VMS=$!
( while true; do echo "== $(date +%T)"; ps axo pid,rss,%cpu,comm | sort -k3 -nr | head -8; sleep 60; done ) > $LOGS/machine_top.log &
TOPS=$!
klog() { /usr/bin/log show --start "$1" --predicate 'eventMessage CONTAINS "required > max" OR (eventMessage CONTAINS "memorystatus" AND eventMessage CONTAINS "python")' --style compact 2>/dev/null | grep -E 'required > max|killing' > $2; echo "kernel lines $(wc -l < $2) -> $2"; }

echo "== B suite v13 gpt-oss $(date +%T)"; T=$(date '+%Y-%m-%d %H:%M:%S')
$PY tools/mlx_bench_suite.py --models gptoss --engines mj-x,mj-a8,llama.cpp,ollama-gguf,lmstudio-gguf --reps 5 --rounds 2 \
  --out results/bench_suite/v13_gptoss > results/bench_suite/v13_gptoss.log 2>&1
echo "B exit $? $(date +%T)"; klog "$T" results/bench_suite/v13_gptoss/kernel_log_excerpts.txt
pkill -f 'ollama serve' 2>/dev/null; ~/.lmstudio/bin/lms server stop >/dev/null 2>&1; osascript -e 'quit app "LM Studio"' >/dev/null 2>&1
sleep 60

echo "== C soak mj-x $(date +%T)"; T=$(date '+%Y-%m-%d %H:%M:%S')
M=$(ls -d ~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/* | head -1)
cat ~/.cache/millijoule/gptoss20b-experts-mapped/*.bin > /dev/null
MJ_MLOCK=1 /usr/bin/time -l $PY tools/mlx_bench_worker.py --engine mj --label mj-x-soak --model $M \
  --map-experts ~/.cache/millijoule/gptoss20b-experts-mapped --attn-pack --lean --wired-mb 1000 --chunk 512 \
  --tests pp2048,tg128@d0,tg128@d2048 --reps 150 --gap-s 3 --out results/gptoss_soak_v1/soak.jsonl \
  > results/gptoss_soak_v1/soak.log 2>&1
echo "C exit $? $(date +%T)"; klog "$T" results/gptoss_soak_v1/kernel_log_excerpts.txt
sleep 60

echo "== D long context $(date +%T)"; T=$(date '+%Y-%m-%d %H:%M:%S')
$PY tools/mlx_bench_suite.py --models dsv2lite --engines mj,llama.cpp,vllm-mlx --tests pp16384,tg128@d16384 \
  --reps 3 --rounds 2 --out results/bench_suite/v13_long_dsv2 > results/bench_suite/v13_long_dsv2.log 2>&1
echo "D1 exit $? $(date +%T)"
pkill -f vllm-mlx 2>/dev/null; sleep 30
$PY tools/mlx_bench_suite.py --models gptoss --engines mj-x,mj-a8,llama.cpp --tests pp16384,tg128@d16384 \
  --reps 3 --rounds 2 --out results/bench_suite/v13_long_gptoss > results/bench_suite/v13_long_gptoss.log 2>&1
echo "D2 exit $? $(date +%T)"; klog "$T" results/bench_suite/v13_long_gptoss/kernel_log_excerpts_d.txt
echo "all done $(date +%T)"

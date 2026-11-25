#!/bin/zsh
# 2026-11-21 afternoon window (user: "about 8 more hours"): A blocked-attention ABAB at 8k / 16k, B the 8-bit gpt-oss
# arm at the 1000 MB wired limit, D the first full tools/reproduce.py run, claim by claim with servers stopped in
# between. Nothing starts after DEADLINE. Servers stopped and login services restarted at the end, whatever happens.
set -u
cd $HOME/Documents/Projects/mj-mlx
PY=.venv/bin/python
SV=results/v14_scripts/../v13_scripts/services.sh
DEADLINE=$(date -j -f '%H:%M' '21:45' +%s)
stop_servers() {
  ~/.lmstudio/bin/lms server stop >/dev/null 2>&1; osascript -e 'quit app "LM Studio"' >/dev/null 2>&1
  pkill -f 'ollama serve' 2>/dev/null; pkill -f 'ollama runner' 2>/dev/null; pkill -f vllm-mlx 2>/dev/null; pkill -f llama-server 2>/dev/null
}
cleanup() { stop_servers; kill $VMS $TOPS 2>/dev/null; zsh $SV start; echo "cleanup done $(date +%T)"; }
trap cleanup EXIT
in_time() { [[ $(date +%s) -lt $DEADLINE ]] || { echo "past deadline, skipping $1 $(date +%T)"; return 1; }; }
klog() { /usr/bin/log show --start "$1" --predicate 'eventMessage CONTAINS "required > max" OR (eventMessage CONTAINS "memorystatus" AND eventMessage CONTAINS "python")' --style compact 2>/dev/null | grep -E 'required > max|killing' > $2; echo "kernel lines $(wc -l < $2) -> $2"; }
zsh $SV stop
LOGS=results/v14_scripts
( while true; do echo "$(date +%T) $(vm_stat | awk '/Swapouts/{o=$2} /Swapins/{i=$2} /wired down/{w=$4} END{printf "wired_GB %.2f swapins %s swapouts %s", w*16384/1e9, i, o}') $(sysctl -n vm.swapusage | awk '{print "swap_used", $6}')"; sleep 5; done ) > $LOGS/vmstat_5s.log &
VMS=$!
( while true; do echo "== $(date +%T)"; ps axo pid,rss,%cpu,comm | sort -k3 -nr | head -8; sleep 60; done ) > $LOGS/machine_top.log &
TOPS=$!
sleep 60

echo "== A blocked attention, dsv2 8k / 16k, default vs --chunk 512 $(date +%T)"
$PY tools/mlx_bench_suite.py --models dsv2lite --engines mj,mj-c512 --tests pp8192,tg128@d8192,pp16384,tg128@d16384 \
  --reps 3 --rounds 2 --out results/bench_suite/v14_attn_blocks > results/bench_suite/v14_attn_blocks.log 2>&1
echo "A exit $? $(date +%T)"; sleep 60

if in_time B; then
  echo "== B gpt-oss 8-bit arm, wired 3000 vs 1000, with rivals $(date +%T)"; T=$(date '+%Y-%m-%d %H:%M:%S')
  $PY tools/mlx_bench_suite.py --models gptoss --engines mj-a8,mj-a8w,mj-x,llama.cpp,ollama-gguf,lmstudio-gguf \
    --reps 5 --rounds 2 --out results/bench_suite/v14_gptoss > results/bench_suite/v14_gptoss.log 2>&1
  echo "B exit $? $(date +%T)"; klog "$T" results/bench_suite/v14_gptoss/kernel_log_excerpts.txt
  stop_servers; sleep 60
fi

R=results/repro/full_20261121
for id in gptoss-decode dsv2lite-decode olmoe-decode accuracy-olmoe accuracy-dsv2lite accuracy-gptoss q36-vs-llamacpp accuracy-q30b; do
  in_time $id || continue
  echo "== D reproduce $id $(date +%T)"; T=$(date '+%Y-%m-%d %H:%M:%S')
  $PY tools/reproduce.py --only $id --out $R >> results/v14_scripts/reproduce.log 2>&1
  echo "D $id exit $? $(date +%T)"; klog "$T" $R/kernel_log_$id.txt
  stop_servers; sleep 60
done
$PY tools/reproduce.py --compare $R > /dev/null 2>&1; echo "report: $R/report.md"
echo "all done $(date +%T)"

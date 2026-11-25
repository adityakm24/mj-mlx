#!/bin/zsh
# Overnight 2026-11-21/22 (user: "keep running through the night and tomorrow morning"). Phase 0 runs, each step new
# result folders; nothing starts after DEADLINE. Servers stopped between steps; login services restarted at the end.
#   L  0.15 long-context accuracy: DeepSeek-V2-Lite WikiText at 16k context with different score-block sizes (same
#      math), and at 4k blocked vs unblocked; gpt-oss exact at 16k
#   S  suite v15: every engine on OLMoE / DeepSeek-V2-Lite / gpt-oss with per-engine peak memory (0.10), llama-server
#      energy (0.6), idle-subtracted energy (0.16), rival versions and per-row provenance (0.9, 0.17)
#   R  0.9 rival knobs: llama-bench threads x ubatch at pp512 / tg128
#   C  0.11 cold start
#   X  0.10 / 0.7 32k capacity: gpt-oss and DeepSeek-V2-Lite at 32k, ours and llama.cpp
#   H  0.12 real-chat check
#   E  0.2 MMLU / GSM8K subsets
set -u
cd $HOME/Documents/Projects/mj-mlx
PY=.venv/bin/python; SV=results/v13_scripts/services.sh; O=results/v15_scripts
DEADLINE=$(date -v+1d -j -f '%H:%M' '09:30' +%s)
H=~/.cache/huggingface/hub
DSV2=$(ls -d $H/models--mlx-community--DeepSeek-V2-Lite-Chat-4bit-mlx/snapshots/* | head -1)
OLMOE=$(ls -d $H/models--mlx-community--OLMoE-1B-7B-0125-Instruct-4bit/snapshots/* | head -1)
GPTOSS=$(ls -d $H/models--openai--gpt-oss-20b/snapshots/* | head -1)
MAP=~/.cache/millijoule/gptoss20b-experts-mapped
GPTKW_A8W='{"map_experts": "'$MAP'", "lean": true, "attn_bits": 8}'
GPTKW_X='{"map_experts": "'$MAP'", "lean": true, "attn_pack": true}'
stop_servers() {
  ~/.lmstudio/bin/lms server stop >/dev/null 2>&1; osascript -e 'quit app "LM Studio"' >/dev/null 2>&1
  pkill -f 'ollama serve' 2>/dev/null; pkill -f 'ollama runner' 2>/dev/null; pkill -f vllm-mlx 2>/dev/null; pkill -f llama-server 2>/dev/null
}
cleanup() { stop_servers; kill $VMS 2>/dev/null; zsh $SV start; echo "cleanup done $(date +%T)"; }
trap cleanup EXIT
in_time() { [[ $(date +%s) -lt $DEADLINE ]] || { echo "past deadline, skipping $1 $(date +%T)"; return 1; }; }
klog() { /usr/bin/log show --start "$1" --predicate 'eventMessage CONTAINS "required > max" OR (eventMessage CONTAINS "memorystatus" AND eventMessage CONTAINS "python")' --style compact 2>/dev/null | grep -E 'required > max|killing' > $2; echo "  kernel lines $(wc -l < $2) -> $2"; }
step() { echo "== $1 $(date +%T) swap $(sysctl -n vm.swapusage | awk '{print $6}')"; }
llama_srv() {  # gguf [args]: llama-server on 8092 for the eval / chat tools
  llama-server -m $1 -fa on -ngl 99 -c 8192 -np 1 --no-cache-prompt --port 8092 --alias bench ${@:2} >/dev/null 2>&1 &
  for i in {1..300}; do curl -sf http://127.0.0.1:8092/health >/dev/null && return 0; sleep 1; done; return 1
}
zsh $SV stop
( while true; do echo "$(date +%T) $(vm_stat | awk '/Swapouts/{o=$2} /Swapins/{i=$2} /wired down/{w=$4} END{printf "wired_GB %.2f swapins %s swapouts %s", w*16384/1e9, i, o}') $(sysctl -n vm.swapusage | awk '{print "swap_used", $6}')"; sleep 5; done ) > $O/vmstat_5s.log &
VMS=$!
sleep 30

# L: long-context accuracy (0.15)
if in_time L; then
  A=results/accuracy_long_v1; mkdir -p $A; step L; T=$(date '+%Y-%m-%d %H:%M:%S')
  for mb in 64 16 1024; do
    MJ_ATTN_SCORES_MB=$mb $PY tools/mlx_accuracy_ppl.py --engine mj --model $DSV2 --ctx 16384 --chunks 2 \
      --out $A/dsv2_ctx16k_s$mb.json > $A/dsv2_ctx16k_s$mb.log 2>&1; echo "  dsv2 16k s$mb exit $?"
  done
  for mb in 0 64; do
    MJ_ATTN_SCORES_MB=$mb $PY tools/mlx_accuracy_ppl.py --engine mj --model $DSV2 --ctx 4096 --chunks 8 \
      --out $A/dsv2_ctx4k_s$mb.json > $A/dsv2_ctx4k_s$mb.log 2>&1; echo "  dsv2 4k s$mb exit $?"
  done
  $PY tools/mlx_accuracy_ppl.py --engine mlx-lm --model $DSV2 --ctx 4096 --chunks 8 --out $A/dsv2_ctx4k_mlxlm.json \
    > $A/dsv2_ctx4k_mlxlm.log 2>&1; echo "  dsv2 4k mlx-lm exit $?"
  MJ_MLOCK=1 $PY tools/mlx_accuracy_ppl.py --engine mj --model $GPTOSS --ctx 16384 --chunks 2 --placement none \
    --mj-kw "$GPTKW_X" --out $A/gptoss_x_ctx16k.json > $A/gptoss_x_ctx16k.log 2>&1; echo "  gptoss 16k exit $?"
  klog "$T" $A/kernel_log_excerpts.txt; sleep 30
fi

# S: suite v15 (0.6, 0.9, 0.10, 0.16, 0.17)
for m in olmoe:mj,mlx-lm,llama.cpp,llama-server,ollama-gguf,lmstudio-mlx,vllm-mlx \
         dsv2lite:mj,llama.cpp,llama-server,ollama-gguf,vllm-mlx \
         gptoss:mj-a8w,mj-x,llama.cpp,llama-server,ollama-gguf,lmstudio-gguf; do
  key=${m%%:*}; eng=${m#*:}
  in_time S_$key || continue
  step "S $key"; T=$(date '+%Y-%m-%d %H:%M:%S')
  $PY tools/mlx_bench_suite.py --models $key --engines $eng --reps 5 --rounds 2 --out results/bench_suite/v15_$key \
    > results/bench_suite/v15_$key.log 2>&1; echo "  S $key exit $?"
  klog "$T" results/bench_suite/v15_$key/kernel_log_excerpts.txt; stop_servers; sleep 60
done

# R: rival knobs (0.9): llama-bench threads x ubatch, short context
if in_time R; then
  R=results/rival_knobs_v1; mkdir -p $R; step R
  for key in olmoe gptoss; do
    G=$($PY -c "import sys; sys.path.insert(0,'tools'); import glob,os; from mlx_bench_suite import MODELS; print(sorted(glob.glob(os.path.expanduser(MODELS['$key']['gguf'])))[0])")
    llama-bench -m $G -fa 1 -ngl 99 -t 4,6,10 -ub 256,512,1024 -p 512 -n 128 -r 3 -o json > $R/${key}_llama_t_ub.json 2> $R/${key}_llama_t_ub.err
    echo "  R $key exit $?"; sleep 30
  done
fi

# C: cold start (0.11)
if in_time C; then
  step C
  $PY tools/mlx_cold_start.py --models olmoe --engines mj,mlx-lm,llama.cpp,ollama-gguf,lmstudio-mlx,vllm-mlx --reps 3 --rounds 2 \
    --out results/cold_start_v1/olmoe > results/cold_start_v1_olmoe.log 2>&1; echo "  C olmoe exit $?"; stop_servers; sleep 30
  $PY tools/mlx_cold_start.py --models gptoss --engines mj-a8w,llama.cpp,ollama-gguf,lmstudio-gguf --reps 3 --rounds 2 \
    --out results/cold_start_v1/gptoss > results/cold_start_v1_gptoss.log 2>&1; echo "  C gptoss exit $?"; stop_servers; sleep 30
fi

# X: 32k capacity (0.7 / 0.10)
if in_time X; then
  step X; T=$(date '+%Y-%m-%d %H:%M:%S')
  $PY tools/mlx_bench_suite.py --models gptoss --engines mj-a8w,llama.cpp --tests pp32768,tg128@d32768 --reps 2 --rounds 1 \
    --llama-reps 2 --out results/bench_suite/v15_32k_gptoss > results/bench_suite/v15_32k_gptoss.log 2>&1; echo "  X gptoss exit $?"
  $PY tools/mlx_bench_suite.py --models dsv2lite --engines mj,mj-w9 --tests pp16384,tg128@d16384,pp32768,tg128@d32768 --reps 2 --rounds 2 \
    --out results/bench_suite/v15_32k_dsv2 > results/bench_suite/v15_32k_dsv2.log 2>&1; echo "  X dsv2 exit $?"
  klog "$T" results/bench_suite/v15_32k_gptoss/kernel_log_excerpts.txt; stop_servers; sleep 30
fi

# H: real-chat check (0.12)
if in_time H; then
  step H; CC=results/chat_check_v1; mkdir -p $CC
  $PY tools/mlx_chat_check.py --model $OLMOE --engines mj,mlx-lm --out $CC/olmoe.json > $CC/olmoe.log 2>&1; echo "  H olmoe exit $?"
  $PY tools/mlx_chat_check.py --model $DSV2 --engines mj,mlx-lm --out $CC/dsv2lite.json > $CC/dsv2lite.log 2>&1; echo "  H dsv2 exit $?"
  MJ_MLOCK=1 $PY tools/mlx_chat_check.py --model $GPTOSS --placement none --mj-kw "$GPTKW_A8W" --engines mj --out $CC/gptoss_a8w.json \
    > $CC/gptoss.log 2>&1; echo "  H gptoss exit $?"
  sleep 30
fi

# E: MMLU / GSM8K (0.2), one engine at a time; llama-server on the suite's GGUF
EV=results/lm_eval_v1; mkdir -p $EV
gguf() { $PY -c "import sys; sys.path.insert(0,'tools'); import glob,os; from mlx_bench_suite import MODELS; print(sorted(glob.glob(os.path.expanduser(MODELS['$1']['gguf'])))[0])"; }
for job in olmoe:mj olmoe:llama-server olmoe:mlx-lm dsv2lite:mj dsv2lite:llama-server dsv2lite:mlx-lm gptoss:mj gptoss:llama-server; do
  key=${job%%:*}; eng=${job#*:}
  in_time E_$job || continue
  step "E $job"
  case $key in olmoe) M=$OLMOE;; dsv2lite) M=$DSV2;; gptoss) M=$GPTOSS;; esac
  extra=(); [[ $key == gptoss ]] && extra=(--placement none --mj-kw "$GPTKW_A8W")
  if [[ $eng == llama-server ]]; then
    llama_srv $(gguf $key) || { echo "  llama-server did not start"; stop_servers; continue; }
    $PY tools/mlx_lm_eval.py --engine llama-server --model $M --out $EV/${key}_llamacpp.json > $EV/${key}_llamacpp.log 2>&1
  else
    MJ_MLOCK=1 $PY tools/mlx_lm_eval.py --engine $eng --model $M ${extra[@]} --out $EV/${key}_$eng.json > $EV/${key}_$eng.log 2>&1
  fi
  echo "  E $job exit $?"; stop_servers; sleep 30
done
echo "all done $(date +%T)"

#!/bin/zsh
# Phase 2 (clean machine from 02:30, user 2026-11-22): everything timing-sensitive, then the remaining accuracy jobs.
#   S2 fresh gpt-oss and DeepSeek-V2-Lite suites (v15's first passes ran with another agent session active: 4 of 8 gpt-oss
#      blocks flagged before the end, swap up to 3.8 GB); R rival knobs; C cold start; X 16k / 32k; L2 16k accuracy reruns;
#      then the MMLU / GSM8K jobs phase 1 did not reach. Nothing starts after 09:30.
set -u
source $HOME/Documents/Projects/mj-mlx/results/v15_scripts/common.zsh
DEADLINE=$(date -j -f '%H:%M' '09:30' +%s)
trap 'stop_servers; kill $VMS 2>/dev/null; zsh $SV start; echo "cleanup done $(date +%T)"' EXIT
zsh $SV stop; vmlog $O/vmstat_5s_e.log; sleep 30
for m in gptoss:mj-a8w,mj-x,llama.cpp,llama-server,ollama-gguf,lmstudio-gguf dsv2lite:mj,llama.cpp,ollama-gguf,vllm-mlx; do
  key=${m%%:*}; eng=${m#*:}; before $DEADLINE || break
  step "S2 $key"; T=$(date '+%Y-%m-%d %H:%M:%S')
  $PY tools/mlx_bench_suite.py --models $key --engines $eng --reps 5 --rounds 2 --out results/bench_suite/v15q_$key \
    > results/bench_suite/v15q_$key.log 2>&1; echo "  S2 $key exit $?"
  klog "$T" results/bench_suite/v15q_$key/kernel_log_excerpts.txt; stop_servers; sleep 60
done
if before $DEADLINE; then
  R=results/rival_knobs_v1; mkdir -p $R; step R
  for key in olmoe gptoss; do
    llama-bench -m $(gguf $key) -fa 1 -ngl 99 -t 4,6,10 -ub 256,512,1024 -p 512 -n 128 -r 3 -o json > $R/${key}_llama_t_ub.json 2> $R/${key}_llama_t_ub.err
    echo "  R $key exit $?"; sleep 30
  done
fi
if before $DEADLINE; then
  step C
  $PY tools/mlx_cold_start.py --models olmoe --engines mj,mlx-lm,llama.cpp,ollama-gguf,lmstudio-mlx,vllm-mlx --reps 3 --rounds 2 \
    --out results/cold_start_v1/olmoe > results/cold_start_v1_olmoe.log 2>&1; echo "  C olmoe exit $?"; stop_servers; sleep 30
  $PY tools/mlx_cold_start.py --models gptoss --engines mj-a8w,llama.cpp,ollama-gguf,lmstudio-gguf --reps 3 --rounds 2 \
    --out results/cold_start_v1/gptoss > results/cold_start_v1_gptoss.log 2>&1; echo "  C gptoss exit $?"; stop_servers; sleep 30
fi
if before $DEADLINE; then
  step X; T=$(date '+%Y-%m-%d %H:%M:%S')
  $PY tools/mlx_bench_suite.py --models dsv2lite --engines mj,mj-w9 --tests pp16384,tg128@d16384,pp32768,tg128@d32768 --reps 2 --rounds 2 \
    --out results/bench_suite/v15_32k_dsv2 > results/bench_suite/v15_32k_dsv2.log 2>&1; echo "  X dsv2 exit $?"
  $PY tools/mlx_bench_suite.py --models gptoss --engines mj-a8w,llama.cpp --tests pp32768,tg128@d32768 --reps 2 --rounds 1 \
    --llama-reps 2 --out results/bench_suite/v15_32k_gptoss > results/bench_suite/v15_32k_gptoss.log 2>&1; echo "  X gptoss exit $?"
  klog "$T" results/bench_suite/v15_32k_dsv2/kernel_log_excerpts.txt; stop_servers; sleep 30
fi
if before $DEADLINE; then
  A=results/accuracy_long_v1; step L2
  for mb in 64 16; do
    MJ_ATTN_SCORES_MB=$mb $PY tools/mlx_accuracy_ppl.py --engine mj --model $DSV2 --ctx 16384 --chunks 2 \
      --out $A/dsv2_ctx16k_s${mb}_rerun.json > $A/dsv2_ctx16k_s${mb}_rerun.log 2>&1; echo "  dsv2 16k s$mb rerun exit $?"
  done
  $PY tools/mlx_accuracy_ppl.py --engine mj --model $GPTOSS --ctx 16384 --chunks 2 --placement none --mj-kw "$GPTKW_X" \
    --out $A/gptoss_x_ctx16k_nomlock.json > $A/gptoss_x_ctx16k_nomlock.log 2>&1; echo "  gptoss 16k no-mlock exit $?"
fi
# H: real-chat check (0.12); load-insensitive, so after the timed steps
CC=results/chat_check_v1; mkdir -p $CC
if before $DEADLINE && [[ ! -e $CC/olmoe.json ]]; then step "H olmoe"; $PY tools/mlx_chat_check.py --model $OLMOE --engines mj,mlx-lm --out $CC/olmoe.json > $CC/olmoe.log 2>&1; echo "  H olmoe exit $?"; fi
if before $DEADLINE && [[ ! -e $CC/dsv2lite.json ]]; then step "H dsv2"; $PY tools/mlx_chat_check.py --model $DSV2 --engines mj,mlx-lm --out $CC/dsv2lite.json > $CC/dsv2lite.log 2>&1; echo "  H dsv2 exit $?"; fi
if before $DEADLINE && [[ ! -e $CC/gptoss_a8w.json ]]; then step "H gptoss"; MJ_MLOCK=1 $PY tools/mlx_chat_check.py --model $GPTOSS --placement none --mj-kw "$GPTKW_A8W" --engines mj --out $CC/gptoss_a8w.json > $CC/gptoss.log 2>&1; echo "  H gptoss exit $?"; fi
mkdir -p results/lm_eval_v1
for job in $EVAL_JOBS; do before $DEADLINE || { echo "past deadline, not starting $job $(date +%T)"; break; }; eval_job $job; done
echo "all done $(date +%T)"

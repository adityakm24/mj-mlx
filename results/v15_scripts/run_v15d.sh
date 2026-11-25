#!/bin/zsh
# Phase 1 (until the user's clean window at 02:30): the steps whose results do not depend on machine load.
# H real-chat check (0.12), E MMLU / GSM8K (0.2). Starts no job after 02:05.
set -u
source $HOME/Documents/Projects/mj-mlx/results/v15_scripts/common.zsh
GATE=$(date -j -f '%H:%M' '02:05' +%s)
trap 'stop_servers; kill $VMS 2>/dev/null; zsh $SV start; echo "cleanup done $(date +%T)"' EXIT
zsh $SV stop; vmlog $O/vmstat_5s_d.log; mkdir -p results/chat_check_v1 results/lm_eval_v1
CC=results/chat_check_v1
if before $GATE && [[ ! -e $CC/olmoe.json ]]; then step "H olmoe"; $PY tools/mlx_chat_check.py --model $OLMOE --engines mj,mlx-lm --out $CC/olmoe.json > $CC/olmoe.log 2>&1; echo "  H olmoe exit $?"; fi
if before $GATE && [[ ! -e $CC/dsv2lite.json ]]; then step "H dsv2"; $PY tools/mlx_chat_check.py --model $DSV2 --engines mj,mlx-lm --out $CC/dsv2lite.json > $CC/dsv2lite.log 2>&1; echo "  H dsv2 exit $?"; fi
if before $GATE && [[ ! -e $CC/gptoss_a8w.json ]]; then step "H gptoss"; MJ_MLOCK=1 $PY tools/mlx_chat_check.py --model $GPTOSS --placement none --mj-kw "$GPTKW_A8W" --engines mj --out $CC/gptoss_a8w.json > $CC/gptoss.log 2>&1; echo "  H gptoss exit $?"; fi
for job in $EVAL_JOBS; do before $GATE || { echo "gate passed, not starting $job $(date +%T)"; break; }; eval_job $job; done
echo "phase 1 done $(date +%T)"

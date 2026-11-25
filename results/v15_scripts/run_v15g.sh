#!/bin/zsh
# 2026-11-22 04:45: gpt-oss exact at a 16k context scored perplexity 5,789 in one forward (181 at 512-token chunks).
# Find where it breaks, and whether the Engine's chunked path (what the speed tests use) has the problem; llama.cpp's
# own perplexity at the same contexts is the outside reference. Then the chat check (tool fixed) and MMLU / GSM8K.
set -u
source $HOME/Documents/Projects/mj-mlx/results/v15_scripts/common.zsh
trap 'stop_servers; kill $VMS 2>/dev/null; zsh $SV start; echo "cleanup done $(date +%T)"' EXIT
zsh $SV stop; vmlog $O/vmstat_5s_g.log; sleep 10
A=results/accuracy_long_v2; mkdir -p $A
ppl() {  # tag model ctx chunks [extra args]
  [[ -e $A/$1.json ]] && return
  $PY tools/mlx_accuracy_ppl.py --engine mj --model $2 --ctx $3 --chunks $4 ${@:5} --out $A/$1.json > $A/$1.log 2>&1
  echo "  $1 exit $? $(date +%T) $($PY -c "import json;d=json.load(open('$A/$1.json'));print('ppl',round(d['ppl'],3))" 2>/dev/null)"
}
step D
GX=(--placement none --mj-kw "$GPTKW_X")
ppl gptoss_x_c2048 $GPTOSS 2048 4 $GX
ppl gptoss_x_c2048_p512 $GPTOSS 2048 4 $GX --prefill-chunk 512
ppl gptoss_x_c8192 $GPTOSS 8192 2 $GX
ppl gptoss_x_c8192_p512 $GPTOSS 8192 2 $GX --prefill-chunk 512
ppl gptoss_x_c16384 $GPTOSS 16384 1 $GX
ppl gptoss_x_c16384_p512 $GPTOSS 16384 1 $GX --prefill-chunk 512
ppl gptoss_a8_c16384_p512 $GPTOSS 16384 1 --placement none --mj-kw "$GPTKW_A8W" --prefill-chunk 512
ppl dsv2_c16384_p2048 $DSV2 16384 2 --prefill-chunk 2048
G=$(gguf gptoss)
for spec in "2048 4" "8192 2" "16384 1"; do
  c=${spec% *}; n=${spec#* }
  llama-perplexity -m $G -f results/accuracy/wiki.test.raw -c $c --chunks $n -b 512 -ub 128 > $A/gptoss_llamacpp_c$c.log 2>&1
  echo "  llama.cpp c$c exit $? $(date +%T) $(grep -o 'Final estimate: PPL = [0-9.]*' $A/gptoss_llamacpp_c$c.log | tail -1)"
done
CC=results/chat_check_v1
for j in "olmoe $OLMOE" "dsv2lite $DSV2"; do
  k=${j%% *}; m=${j#* }; [[ -e $CC/$k.json ]] && continue
  step "H $k"; $PY tools/mlx_chat_check.py --model $m --engines mj,mlx-lm --out $CC/$k.json > $CC/${k}_run2.log 2>&1; echo "  H $k exit $?"
done
if [[ ! -e $CC/gptoss_a8w.json ]]; then step "H gptoss"; $PY tools/mlx_chat_check.py --model $GPTOSS --placement none --mj-kw "$GPTKW_A8W" --engines mj --out $CC/gptoss_a8w.json > $CC/gptoss_run2.log 2>&1; echo "  H gptoss exit $?"; fi
mkdir -p results/lm_eval_v1
for job in $EVAL_JOBS; do eval_job $job; done
echo "all done $(date +%T)"

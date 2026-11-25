#!/bin/zsh
# After run_v15b: re-run the two 16k DeepSeek-V2-Lite accuracy runs that ran out of memory in the logits (fixed in
# mlx_accuracy_ppl.py: NLL in row blocks). New file names (the failed runs' logs stay).
set -u
cd $HOME/Documents/Projects/mj-mlx
PY=.venv/bin/python; SV=results/v13_scripts/services.sh; A=results/accuracy_long_v1
trap 'zsh $SV start; echo "cleanup done $(date +%T)"' EXIT
zsh $SV stop; sleep 20
DSV2=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V2-Lite-Chat-4bit-mlx/snapshots/* | head -1)
for mb in 64 16; do
  [[ -e $A/dsv2_ctx16k_s$mb.json ]] && continue
  MJ_ATTN_SCORES_MB=$mb $PY tools/mlx_accuracy_ppl.py --engine mj --model $DSV2 --ctx 16384 --chunks 2 \
    --out $A/dsv2_ctx16k_s${mb}_rerun.json > $A/dsv2_ctx16k_s${mb}_rerun.log 2>&1; echo "dsv2 16k s$mb rerun exit $? $(date +%T)"
done
# gpt-oss exact at 16k without expert mlock (with MJ_MLOCK=1 macOS killed it for 90% of wired memory in run_v15b)
G=$(ls -d ~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/* | head -1)
$PY tools/mlx_accuracy_ppl.py --engine mj --model $G --ctx 16384 --chunks 2 --placement none \
  --mj-kw '{"map_experts": "'$HOME'/.cache/millijoule/gptoss20b-experts-mapped", "lean": true, "attn_pack": true}' \
  --out $A/gptoss_x_ctx16k_nomlock.json > $A/gptoss_x_ctx16k_nomlock.log 2>&1; echo "gptoss 16k no-mlock exit $? $(date +%T)"
echo "all done $(date +%T)"

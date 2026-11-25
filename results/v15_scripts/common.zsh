# Shared by the v15 phase scripts (sourced). Paths, model locations, helpers.
cd $HOME/Documents/Projects/mj-mlx
PY=.venv/bin/python; SV=results/v13_scripts/services.sh; O=results/v15_scripts
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
klog() { /usr/bin/log show --start "$1" --predicate 'eventMessage CONTAINS "required > max" OR (eventMessage CONTAINS "memorystatus" AND eventMessage CONTAINS "python")' --style compact 2>/dev/null | grep -E 'required > max|killing' > $2; echo "  kernel lines $(wc -l < $2) -> $2"; }
step() { echo "== $1 $(date +%T) swap $(sysctl -n vm.swapusage | awk '{print $6}')"; }
before() { [[ $(date +%s) -lt $1 ]]; }  # before <epoch>
gguf() { $PY -c "import sys; sys.path.insert(0,'tools'); import glob,os; from mlx_bench_suite import MODELS; print(sorted(glob.glob(os.path.expanduser(MODELS['$1']['gguf'])))[0])"; }
llama_srv() {  # gguf [args]: llama-server on 8092 for the eval / chat tools
  llama-server -m $1 -fa on -ngl 99 -c 8192 -np 1 --no-cache-prompt --port 8092 --alias bench ${@:2} >/dev/null 2>&1 &
  for i in {1..300}; do curl -sf http://127.0.0.1:8092/health >/dev/null && return 0; sleep 1; done; return 1
}
vmlog() { ( while true; do echo "$(date +%T) $(vm_stat | awk '/Swapouts/{o=$2} /Swapins/{i=$2} /wired down/{w=$4} END{printf "wired_GB %.2f swapins %s swapouts %s", w*16384/1e9, i, o}') $(sysctl -n vm.swapusage | awk '{print "swap_used", $6}')"; sleep 5; done ) >> $1 & VMS=$!; }
eval_job() {  # key:engine -> results/lm_eval_v1/<key>_<engine>.json (skipped when it exists)
  local key=${1%%:*} eng=${1#*:} M extra out
  case $key in olmoe) M=$OLMOE;; dsv2lite) M=$DSV2;; gptoss) M=$GPTOSS;; esac
  extra=(); [[ $key == gptoss ]] && extra=(--placement none --mj-kw "$GPTKW_A8W")
  out=results/lm_eval_v1/${key}_${eng/llama-server/llamacpp}.json
  [[ -e $out ]] && { echo "  E $1 already done"; return; }
  step "E $1"
  if [[ $eng == llama-server ]]; then
    llama_srv $(gguf $key) || { echo "  llama-server did not start"; stop_servers; return; }
    $PY tools/mlx_lm_eval.py --engine llama-server --model $M --out $out > ${out%.json}.log 2>&1
  else
    MJ_MLOCK=1 $PY tools/mlx_lm_eval.py --engine $eng --model $M ${extra[@]} --out $out > ${out%.json}.log 2>&1
  fi
  echo "  E $1 exit $?"; stop_servers; sleep 20
}
EVAL_JOBS=(olmoe:mj olmoe:mlx-lm olmoe:llama-server dsv2lite:mj dsv2lite:mlx-lm dsv2lite:llama-server gptoss:mj gptoss:llama-server)

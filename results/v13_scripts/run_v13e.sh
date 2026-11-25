#!/bin/zsh
# Follow-ups to v13 long context: DeepSeek-V2-Lite at 16k with 512-token prefill chunks (ours), and at 8k on every
# engine that ran at 2k. Login services stopped for the run and started again at the end.
set -u
S=/private/tmp/claude-501/-Users-user-Documents-Projects-MilliJoule/ae1ca8d8-c94f-4eb0-b984-6492edd379c3/scratchpad
cd $HOME/Documents/Projects/mj-mlx
PY=.venv/bin/python
cleanup() { pkill -f vllm-mlx 2>/dev/null; pkill -f 'ollama serve' 2>/dev/null; ~/.lmstudio/bin/lms server stop >/dev/null 2>&1; osascript -e 'quit app "LM Studio"' >/dev/null 2>&1; kill $VMS 2>/dev/null; zsh $S/services.sh start; echo "cleanup done $(date +%T)"; }
trap cleanup EXIT
zsh $S/services.sh stop
mkdir -p results/llama_16k_check_v1 results/bench_suite/v13_long_dsv2_c512 results/bench_suite/v13_8k_dsv2
( while true; do echo "$(date +%T) $(vm_stat | awk '/Swapouts/{o=$2} /Swapins/{i=$2} /wired down/{w=$4} END{printf "wired_GB %.2f swapins %s swapouts %s", w*16384/1e9, i, o}') $(sysctl -n vm.swapusage | awk '{print "swap_used", $6}')"; sleep 5; done ) > results/bench_suite/v13_8k_dsv2/vmstat_5s.log &
VMS=$!
# Bounded rival check (industry practice: quality-preserving settings only). llama.cpp at 16k with smaller batch /
# ubatch (only speed and memory change; f16 KV, flash attention, all layers on the GPU kept). Stop at the first fit.
mkdir -p results/llama_16k_check_v1
llama16k() {  # tag gguf
  for ub in 256 128; do
    echo "== llama.cpp 16k $1 ub=$ub $(date +%T)"
    for spec in "-p 16384 -n 0 -d 0" "-p 0 -n 128 -d 16384"; do
      llama-bench -m $2 -fa 1 -ngl 99 -b $ub -ub $ub -ctk f16 -ctv f16 -r 3 -o json ${=spec} \
        > results/llama_16k_check_v1/$1_ub${ub}_${${spec// /}//-/}.json 2> results/llama_16k_check_v1/$1_ub${ub}_${${spec// /}//-/}.err
      echo "  $spec exit $?"
    done
    grep -q avg_ts results/llama_16k_check_v1/$1_ub${ub}_p0n128d16384.json 2>/dev/null && grep -q avg_ts results/llama_16k_check_v1/$1_ub${ub}_p16384n0d0.json 2>/dev/null && { echo "  fits at ub=$ub"; break; }
    sleep 20
  done
}
echo "== E0 llama.cpp 16k check $(date +%T)"
llama16k gptoss $(ls ~/.cache/huggingface/hub/models--ggml-org--gpt-oss-20b-GGUF/snapshots/*/gpt-oss-20b-MXFP4.gguf | head -1)
llama16k dsv2lite $(ls ~/.cache/huggingface/hub/models--mradermacher--DeepSeek-V2-Lite-Chat-GGUF/snapshots/*/DeepSeek-V2-Lite-Chat.Q4_K_M.gguf | head -1)
sleep 30
echo "== E1 dsv2 16k chunk 512 $(date +%T)"
$PY tools/mlx_bench_suite.py --models dsv2lite --engines mj-c512 --tests pp16384,tg128@d16384 --reps 3 --rounds 1 \
  --out results/bench_suite/v13_long_dsv2_c512 > results/bench_suite/v13_long_dsv2_c512.log 2>&1
echo "E1 exit $? $(date +%T)"; sleep 30
echo "== E2 dsv2 8k $(date +%T)"
$PY tools/mlx_bench_suite.py --models dsv2lite --engines mj,mj-c512,llama.cpp,vllm-mlx --tests pp8192,tg128@d8192 --reps 3 --rounds 2 \
  --out results/bench_suite/v13_8k_dsv2 > results/bench_suite/v13_8k_dsv2.log 2>&1
echo "E2 exit $? $(date +%T)"
echo "all done $(date +%T)"

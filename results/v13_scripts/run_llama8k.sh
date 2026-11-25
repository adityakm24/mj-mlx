#!/bin/zsh
# Bounded rival check at 8k on DeepSeek-V2-Lite: llama.cpp with batch/ubatch 256 then 128 (f16 KV, -fa 1, -ngl 99).
set -u
S=/private/tmp/claude-501/-Users-user-Documents-Projects-MilliJoule/ae1ca8d8-c94f-4eb0-b984-6492edd379c3/scratchpad
cd $HOME/Documents/Projects/mj-mlx
trap 'zsh $S/services.sh start; echo "cleanup done $(date +%T)"' EXIT
zsh $S/services.sh stop
G=$(ls ~/.cache/huggingface/hub/models--mradermacher--DeepSeek-V2-Lite-Chat-GGUF/snapshots/*/DeepSeek-V2-Lite-Chat.Q4_K_M.gguf | head -1)
mkdir -p results/llama_8k_check_v1
for ub in 256 128; do
  echo "== llama.cpp 8k dsv2lite ub=$ub $(date +%T)"
  for spec in "-p 8192 -n 0 -d 0" "-p 0 -n 128 -d 8192"; do
    f=results/llama_8k_check_v1/dsv2lite_ub${ub}_${${spec// /}//-/}
    llama-bench -m $G -fa 1 -ngl 99 -b $ub -ub $ub -ctk f16 -ctv f16 -r 3 -o json ${=spec} > $f.json 2> $f.err
    echo "  $spec exit $?"
  done
  grep -q avg_ts results/llama_8k_check_v1/dsv2lite_ub${ub}_p0n128d8192.json && grep -q avg_ts results/llama_8k_check_v1/dsv2lite_ub${ub}_p8192n0d0.json && { echo "  fits at ub=$ub"; break; }
  sleep 20
done
echo "all done $(date +%T)"

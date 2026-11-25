#!/bin/zsh
# Quiet window 4 (2026-11-17 evening): repeat of today's preliminary runs (RESULTS 1l, 1m, 1n) on a quiet Mac.
cd $HOME/Documents/Projects/MilliJoule
Q=results/quiet4
M=$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/938d8919941c6e7efd3c7150eff7fe9d12afa631
MAP=~/.cache/millijoule/gptoss20b-experts-mapped
GGUF=$HOME/.cache/huggingface/hub/models--ggml-org--gpt-oss-20b-GGUF/snapshots/41e8a94d15d62070b36cf4553a0250717ecf46b8/gpt-oss-20b-MXFP4.gguf
PY=.venv/bin/python
caffeinate -i -w $$ &  # no idle sleep while this script runs
state() { $PY tools/mlx_machine_state.py $Q/machine_state_$1.json; }
echo "start $(date +%H:%M:%S)"; state start

# (A) one request, Qwen3.5-9B: plain, mlx-lm, MTP k=2, adaptive energy, adaptive edp; chat and random; 2 rounds
mkdir -p $Q/b1
for rnd in 1 2; do
  arms=(mlx-lm plain k2 auto-energy auto-edp); [[ $rnd == 2 ]] && arms=(auto-edp auto-energy k2 plain mlx-lm)
  for prompt in chat random; do
    for arm in $arms; do
      case $arm in
        plain) $PY tools/mlx_engine_compare.py --engine mj-overlap --model $M --prompt $prompt --arm plain --round $rnd --out $Q/b1/r${rnd}_${prompt}_plain.json ;;
        mlx-lm) $PY tools/mlx_engine_compare.py --engine mlx-lm --model $M --prompt $prompt --arm mlx-lm --round $rnd --out $Q/b1/r${rnd}_${prompt}_mlx-lm.json ;;
        *) k=${arm#k}; $PY tools/mlx_engine_compare.py --engine mj-mtp --model $M --prompt $prompt --k $k --arm $arm --round $rnd --out $Q/b1/r${rnd}_${prompt}_${arm}.json ;;
      esac
    done
  done
done 2>&1 | grep -v Warning | grep "rep 2\|Error\|Traceback"
echo "A done $(date +%H:%M:%S)"; state after_b1

# (B) batched, Qwen3.5-9B, 1-8 requests: plain, k1, k2, adaptive edp; 2 rounds, round 2 in reverse arm order
mkdir -p $Q/batch
$PY tools/mlx_spec_batch.py --model $M --levels 1,2,4,8 --arms plain,mtp-k1,mtp-k2,mtp-auto --policy edp --batch-cost results/spec_batch_9b/batch_cost_v1.json --round 1 --out $Q/batch/r1.json 2>&1 | grep "^B="
$PY tools/mlx_spec_batch.py --model $M --levels 1,2,4,8 --arms mtp-auto,mtp-k2,mtp-k1,plain --policy edp --batch-cost results/spec_batch_9b/batch_cost_v1.json --round 2 --out $Q/batch/r2.json 2>&1 | grep "^B="
echo "B done $(date +%H:%M:%S)"; state after_batch

# (C) gpt-oss-20b: bf16 vs packed attention (mapped experts, files read right before), llama-bench; 2 rounds, reversed
mkdir -p $Q/gptoss
for rnd in 1 2; do
  arms=(bf16 pack llama); [[ $rnd == 2 ]] && arms=(llama pack bf16)
  for arm in $arms; do
    if [[ $arm == llama ]]; then
      llama-bench -m $GGUF -p 512 -n 128 -fa 1 -r 3 -o json > $Q/gptoss/r${rnd}_llama-bench.json 2> $Q/gptoss/r${rnd}_llama-bench.log
      $PY -c "import json; print('r$rnd llama-bench', [(('pp' if r['n_gen']==0 else 'tg'), round(r['avg_ts'],1), round(r['stddev_ts'],1)) for r in json.load(open('$Q/gptoss/r${rnd}_llama-bench.json'))])"
    else
      cat $MAP/*.bin > /dev/null
      extra=(); [[ $arm == pack ]] && extra=(--attn-pack)
      $PY tools/mlx_stream_run.py --map-experts $MAP --wired-mb 3000 --prompt-tokens 512 --new 128 --pair r$rnd $extra --out $Q/gptoss/r${rnd}_$arm.json 2>&1 | grep '^{' | $PY -c "import json,sys; r=json.loads(sys.stdin.read()); print('r$rnd $arm decode %.2f tok/s  ttft %.2f s (pp %.0f tok/s)  %.0f mJ/token  resident %.2f GB' % (r['decode_tok_s'], r['ttft_s'], 512 / r['ttft_s'], 1e3 * r['gpu_j_per_token'], r['resident_gb']))"
    fi
  done
done
echo "C done $(date +%H:%M:%S)"; state end

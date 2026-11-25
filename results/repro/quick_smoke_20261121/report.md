# Reproduction report (quick_smoke_20261121)

**Quick mode (1 rep, 1 round / 3 chunks): a smoke test of the commands, not a result.**

## olmoe-decode (RESULTS 1zh; README.md, What it does well: OLMoE row; Where it is behind: short-context decode, prefill)

| model | engine | test | published tok/s [95% CI] | new tok/s [95% CI] | n | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| olmoe | llama.cpp | pp2048 | 1205.73 [1204.65-1206.82] | 1206.35 [nan-nan] | 1 | same (+0.1%) |
| olmoe | llama.cpp | pp512 | 1266.19 [1264.47-1267.90] | 1267.80 [nan-nan] | 1 | same (+0.1%) |
| olmoe | llama.cpp | tg128@d0 | 115.55 [115.46-115.64] | 115.79 [nan-nan] | 1 | same (+0.2%) |
| olmoe | llama.cpp | tg128@d2048 | 88.15 [87.45-88.86] | 89.55 [nan-nan] | 1 | same (+1.6%) |
| olmoe | mj | pp2048 | 1248.75 [1235.50-1262.00] | 1274.01 [nan-nan] | 1 | same (+2.0%) |
| olmoe | mj | pp512 | 1203.85 [1189.54-1218.16] | 1253.03 [nan-nan] | 1 | faster +4.1% |
| olmoe | mj | tg128@d0 | 137.45 [136.53-138.38] | 122.19 [nan-nan] | 1 | slower -11.1% |
| olmoe | mj | tg128@d2048 | 99.79 [97.38-102.20] | 97.82 [nan-nan] | 1 | same (-2.0%) |
| olmoe | mlx-lm | pp2048 | 1318.75 [1301.54-1335.95] | 1354.90 [nan-nan] | 1 | same (+2.7%) |
| olmoe | mlx-lm | pp512 | 1244.24 [1226.90-1261.57] | 1272.71 [nan-nan] | 1 | same (+2.3%) |
| olmoe | mlx-lm | tg128@d0 | 135.55 [134.02-137.08] | 138.92 [nan-nan] | 1 | same (+2.5%) |
| olmoe | mlx-lm | tg128@d2048 | 92.53 [90.49-94.57] | 94.72 [nan-nan] | 1 | same (+2.4%) |
| olmoe | vllm-mlx | pp2048 | 1247.32 [1232.61-1262.03] | 1263.62 [nan-nan] | 1 | same (+1.3%) |
| olmoe | vllm-mlx | pp512 | 988.90 [949.08-1028.72] | 1033.43 [nan-nan] | 1 | faster +4.5% |
| olmoe | vllm-mlx | tg128@d0 | 139.29 [139.06-139.52] | 139.49 [nan-nan] | 1 | same (+0.1%) |
| olmoe | vllm-mlx | tg128@d2048 | 98.91 [97.68-100.15] | 99.82 [nan-nan] | 1 | same (+0.9%) |

- olmoe tg128@d2048: mj tied with vllm-mlx: not judged (fewer than 2 reps)
- olmoe tg128@d2048: mj faster than mlx-lm, llama.cpp: not judged (fewer than 2 reps)
- olmoe pp2048: mlx-lm faster than mj: not judged (fewer than 2 reps)
- olmoe pp512: llama.cpp faster than mj: not judged (fewer than 2 reps)

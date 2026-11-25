A = mlx-lm 0.32.2, B = millijoule mlx runner (qwen36 v1 ABAB round 1); random weights, Qwen3.6-35B-A3B dims, 8 of 40 layers (3:1 linear:full), 4-bit affine gs64 routed experts, bf16 rest
Energy: IOReport 'GPU Energy' rail (SoC energy model), whole GPU incl. idle; Mac-only. Rounds: A 3, B 3.

| level | A tok/s (spread) | B tok/s (spread) | speed B/A | A mJ/token | B mJ/token | energy B/A | A W | B W |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 49.3 (27%) | 68.5 (30%) | 1.39x | 180.6 (67%) | 140.5 (50%) | 0.78x | 8.6 | 9.4 |
| 8 | 223.0 (27%) | 225.8 (23%) | 1.01x | 68.5 (38%) | 53.0 (51%) | 0.77x | 14.9 | 11.7 |
| 32 | 336.3 (26%) | 356.4 (28%) | 1.06x | 42.6 (32%) | 36.0 (40%) | 0.84x | 14.1 | 12.6 |
| 64 | 386.3 (27%) | 397.5 (28%) | 1.03x | 36.3 (29%) | 33.2 (34%) | 0.92x | 13.8 | 13.0 |

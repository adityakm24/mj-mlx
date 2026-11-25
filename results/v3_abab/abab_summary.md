A = mlx-lm 0.32.2, B = millijoule mlx runner (ABAB round 1 v3); random weights, gpt-oss-20b dims, 8 of 24 layers, MXFP4 experts, bf16 rest
Energy: IOReport 'GPU Energy' rail (SoC energy model), whole GPU incl. idle; Mac-only. Rounds: A 2, B 2.

| level | A tok/s (spread) | B tok/s (spread) | speed B/A | A mJ/token | B mJ/token | energy B/A | A W | B W |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 33.1 (6%) | 49.2 (29%) | 1.49x | 342.3 (5%) | 220.5 (46%) | 0.64x | 11.3 | 10.5 |
| 8 | 91.6 (3%) | 124.6 (27%) | 1.36x | 150.2 (2%) | 118.4 (31%) | 0.79x | 13.8 | 14.4 |
| 32 | 178.6 (5%) | 217.5 (24%) | 1.22x | 95.7 (6%) | 76.8 (23%) | 0.80x | 17.1 | 16.5 |
| 64 | 290.5 (23%) | 303.7 (24%) | 1.05x | 59.6 (24%) | 55.1 (25%) | 0.92x | 17.1 | 16.5 |

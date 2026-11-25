A = mlx-lm 0.32.2, B = millijoule mlx runner (v4 ABAB round 1); random weights, gpt-oss-20b dims, 8 of 24 layers, MXFP4 experts, bf16 rest
Energy: IOReport 'GPU Energy' rail (SoC energy model), whole GPU incl. idle; Mac-only. Rounds: A 3, B 3.

| level | A tok/s (spread) | B tok/s (spread) | speed B/A | A mJ/token | B mJ/token | energy B/A | A W | B W |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 41.2 (30%) | 63.9 (34%) | 1.55x | 241.6 (58%) | 177.0 (59%) | 0.73x | 9.6 | 10.9 |
| 8 | 113.9 (27%) | 133.3 (31%) | 1.17x | 128.0 (30%) | 103.0 (42%) | 0.80x | 14.3 | 13.3 |
| 32 | 201.6 (28%) | 242.6 (24%) | 1.20x | 82.1 (21%) | 70.1 (27%) | 0.85x | 16.3 | 16.8 |
| 64 | 300.0 (24%) | 380.7 (3%) | 1.27x | 56.8 (25%) | 43.8 (3%) | 0.77x | 16.8 | 16.7 |

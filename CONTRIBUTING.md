# Contributing

Thanks for trying mj-mlx. It is in beta, so the most useful things right now are bug reports and measurements from
Macs other than ours (an M4 with 16 GB).

## Reporting a bug

Open an issue with:

- your Mac (chip and memory: Apple menu > About This Mac), macOS version, Python version and `pip show mlx`;
- the model (Hugging Face repo name) and the code or command you ran;
- the full error, or what you expected and what you got.

If the Mac swapped or slowed down, say how much swap was in use (`sysctl vm.swapusage`) before and after.

## Sharing results from your Mac

Results from other chips and memory sizes are the biggest gap. The benchmark method is in
[docs/benchmarks.md](docs/benchmarks.md); `tools/reproduce.py --quick` is a smoke test that checks the tools run;
without `--quick` it re-runs the published numbers (it needs the other engines installed). Please include the
machine details above, and whether other apps were running. Numbers from a machine that swapped during the run are
still useful, but say so.

## Sending changes

1. Open an issue first for anything larger than a small fix, so we can agree on the approach.
2. Keep changes focused, and add or update tests (`tests/` compares against transformers on tiny checkpoints).
3. Run the tests before sending: `.venv/bin/python -m pytest tests -q`.
4. A change that claims a speed-up needs a measurement: interleaved A/B rounds on the same machine, as in
   [docs/benchmarks.md](docs/benchmarks.md), with the numbers in the pull request.
5. A change that touches the math (kernels, attention, quantization) needs a correctness check against a reference:
   the tests, or a same-weights comparison as in `results/RESULTS.md`.
6. New files start with the license header used across the repository:

   ```
   # Copyright 2026 the MilliJoule authors
   # SPDX-License-Identifier: Apache-2.0
   ```

By contributing you agree that your contribution is licensed under the Apache License 2.0.

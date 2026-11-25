# Energy on a Mac

mj-mlx reads Apple's IOReport "Energy Model" counters (`mj_mlx/energy.py`). No sudo is needed.

## What the number is

- **An estimate.** IOReport reports energy for SoC rails from the chip's own power model, not from an external meter.
- **GPU rail for claims.** Published mJ/token figures use the "GPU Energy" channel over a test window divided by its
  tokens. On this M4 the IOReport CPU and DRAM rails read 0 even under load (checked 2026-11-22), so earlier "SoC"
  columns were the GPU rail alone. The rest of the system (CPU, DRAM, SSD, display, fans) is not in the published
  figures.
- **Whole machine, in the harness, not yet published.** Since suite v15 the worker also reads the SMC's system total
  power (key PSTR, `mj_mlx/smc_power.py`, no sudo) and subtracts a 10 s idle baseline per test. PSTR is a ~1 s average
  that trails the load by ~1.75 s, so windows are integrated shifted by that lag. Decode windows of 1-3.5 s are too short
  for it (RESULTS 1zo), so a longer energy test comes first; no whole-machine number is published until then.
- **Comparable across engines on one Mac.** Every engine is measured the same way on the same machine. It is not
  comparable with NVML numbers from NVIDIA GPUs or with wall-power meters, and the project never puts the two side by
  side.

- **Not yet checked against an independent source.** The standard is an external power analyser at the wall
  (MLPerf Power). The planned checks (`powermetrics` for the GPU rail, a wall meter for the whole machine) are listed
  in [benchmarks.md](benchmarks.md#measurement-method-what-is-standard-and-what-is-ours).

## Pitfalls seen

- **Stale first sample.** After a long idle gap, the first counter sample can be stale, so a whole prefill's energy
  lands in the next window (RESULTS 1zb). A window that reads 0 J while the GPU was busy is invalid.
- **Extra rows cost energy even when time is flat.** On Apple GPUs, energy follows arithmetic. A speculative verify
  step or an unpacked weight path can match a baseline in tok/s and still cost more mJ/token. Energy is reported
  next to speed for that reason.

## Using it

```python
from mj_mlx.energy import Meter

with Meter() as m:
    ...                        # the work to measure
print(m.window.seconds, m.window.gpu_j, m.window.gpu_active_pct)
```

`tools/mlx_bench_worker.py` shows how the benchmarks window decode: from the first token to the last.

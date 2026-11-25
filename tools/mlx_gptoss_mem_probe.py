"""gpt-oss-20b memory footprint vs prefill speed (16 GB Mac): MLX buffer-cache handling around load and the prefill
chunk size, with page-ins per test (vm_stat) so paging of the memory-mapped experts shows directly.

    .venv/bin/python tools/mlx_gptoss_mem_probe.py --cache-mb 256 --clear-after-load --chunk 512 --out x.jsonl
"""

import argparse
import glob
import json
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))


def pageins():
    s = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return int(re.search(r"Pageins:\s+(\d+)", s).group(1)) * 16384


def wired_gb():
    s = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return int(re.search(r"Pages wired down:\s+(\d+)", s).group(1)) * 16384 / 1e9


def swap_mb():
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(re.search(r"used = ([\d.]+)M", s).group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-mb", type=int, default=1024, help="MLX buffer cache limit")
    ap.add_argument("--cache-before-load", action="store_true", help="set the cache limit before loading")
    ap.add_argument("--clear-after-load", action="store_true")
    ap.add_argument("--chunk", type=int, default=2048, help="Engine max_batch_tokens (prefill chunk)")
    ap.add_argument("--wired-mb", type=int, default=3000)
    ap.add_argument("--tests", default="pp512,pp2048,tg128@d0")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import mlx.core as mx

    from mj_mlx import Engine, load_model
    from mlx_bench_worker import ids_for

    HF = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*"))[0]
    MAP = os.path.expanduser("~/.cache/millijoule/gptoss20b-experts-mapped")
    for f in sorted(glob.glob(MAP + "/*.bin")):  # prewarm, as the suite does
        with open(f, "rb") as fh:
            while fh.read(64 << 20):
                pass
    if a.cache_before_load:
        mx.set_cache_limit(a.cache_mb << 20)
    mx.set_wired_limit(a.wired_mb << 20)
    sw0 = swap_mb()
    model = load_model(HF, map_experts=MAP, attn_pack=True, lean=True)
    load_peak = mx.get_peak_memory()
    locked_gb = model.map.locked / 1e9 if getattr(model, "map", None) else 0.0
    print(f"loaded; mlock {os.environ.get('MJ_MLOCK')}: {locked_gb:.2f} GB pinned, system wired {wired_gb():.2f} GB",
          flush=True)
    cache_after_load = mx.get_cache_memory()
    mx.set_cache_limit(a.cache_mb << 20)
    if a.clear_after_load:
        mx.clear_cache()
    f = open(a.out, "a")
    for test in a.tests.split(","):
        for r in range(a.reps + 1):
            mx.reset_peak_memory()
            n = int(test[2:]) if test.startswith("pp") else max(int(test.split("@d")[1]), 1)
            new = 1 if test.startswith("pp") else 129
            eng = Engine(model=model, num_blocks=(n + new) // 16 + 16, max_running=1, max_batch_tokens=a.chunk,
                         overlap=True, wired_limit_bytes=a.wired_mb << 20)
            eng.add_request(ids_for(n, 500 + r), new, ignore_eos=True)
            p0, t0, times = pageins(), time.perf_counter(), []
            while eng.has_work():
                before = eng.tokens_out
                eng.step()
                if eng.tokens_out > before:
                    times.append(time.perf_counter())
            pi = pageins() - p0
            tok_s = n / (times[0] - t0) if test.startswith("pp") else 128 / (times[-1] - times[0])
            row = dict(tag=a.tag, test=test, rep=r, warmup=r == 0, tok_s=tok_s, pagein_mb=pi / 1e6,
                       peak_gb=mx.get_peak_memory() / 1e9, active_gb=mx.get_active_memory() / 1e9,
                       cache_gb=mx.get_cache_memory() / 1e9, load_peak_gb=load_peak / 1e9,
                       cache_after_load_gb=cache_after_load / 1e9, swap_mb=swap_mb(), swap0_mb=sw0, wired_gb=wired_gb(),
                       mlock=os.environ.get("MJ_MLOCK"), locked_gb=locked_gb, args=vars(a))
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{a.tag} {test} rep {r}{' (warm-up)' if r == 0 else ''}: {tok_s:.1f} tok/s, page-ins "
                  f"{pi / 1e6:.0f} MB, peak {row['peak_gb']:.2f} GB, swap {row['swap_mb']:.0f} MB", flush=True)
            del eng
            mx.clear_cache()
            time.sleep(3)


if __name__ == "__main__":
    main()

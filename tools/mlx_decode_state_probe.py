"""Catch the exact gpt-oss decode "slow state" (RESULTS 1za/1zb: 33-34 vs 28-30 tok/s, same code and command) with
the machine state of each window, so fast and slow windows can be compared side by side.

Per rep (tg128@d0 by default, the suite's exact mj config): tok/s; the GPU's DVFS residency (IOReport "GPU Stats"
GPUPH: OFF, P1..P15) and CPU cluster residency (ECPU / PCPU: idle and per-state); per-step host timings from
wrapped runner calls (build of step N+1, wait for step N's candidates, host rescore, arm = embedding row + submit);
page-ins, swap-ins, swap used and free pages (vm_stat / sysctl) before and after.

    MJ_MLOCK=1 .venv/bin/python tools/mlx_decode_state_probe.py --reps 40 --out results/decode_state/p1.jsonl
"""

import argparse
import ctypes
import glob
import json
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from mj_mlx import energy as E  # noqa: E402

_io, _cf, _cfstr, _pystr = E._io, E._cf, E._cfstr, E._pystr


class States:
    """Residency per state for the channels of one IOReport group/subgroup."""

    def __init__(self, group, sub, names):
        chans = _io.IOReportCopyChannelsInGroup(_cfstr(group), _cfstr(sub), 0, 0, 0)
        self._ch = _cf.CFDictionaryCreateMutableCopy(None, 0, chans)
        self._subbed = E._vp()
        self._s = _io.IOReportCreateSubscription(None, self._ch, ctypes.byref(self._subbed), 0, None)
        self._key = _cfstr("IOReportChannels")
        self.names = names

    def snap(self):
        _cf.CFRelease(_io.IOReportCreateSamples(self._s, self._subbed, None))  # stale-first-sample guard (1zb)
        return _io.IOReportCreateSamples(self._s, self._subbed, None)

    def delta(self, a, b):
        d = _io.IOReportCreateSamplesDelta(a, b, None)
        arr = _cf.CFDictionaryGetValue(d, self._key)
        out = {}
        for i in range(_cf.CFArrayGetCount(arr)):
            c = _cf.CFArrayGetValueAtIndex(arr, i)
            nm = _pystr(_io.IOReportChannelGetChannelName(c))
            if nm in self.names:
                out[nm] = {_pystr(_io.IOReportStateGetNameForIndex(c, j)): int(_io.IOReportStateGetResidency(c, j))
                           for j in range(_io.IOReportStateGetCount(c))}
        _cf.CFRelease(d)
        _cf.CFRelease(a)
        _cf.CFRelease(b)
        return out


def summarize(st):
    """{channel: {active_pct, mean_state (index over active time, 1 = lowest), dist {state: pct of active}}}"""
    out = {}
    for nm, s in st.items():
        idle = s.get("OFF", 0) + s.get("IDLE", 0)
        tot = sum(s.values())
        act = {k: v for k, v in s.items() if k not in ("OFF", "IDLE")}
        at = sum(act.values())
        keys = list(act)
        mean = sum((i + 1) * act[k] for i, k in enumerate(keys)) / at if at else None
        out[nm] = dict(active_pct=100 * (1 - idle / tot) if tot else None, mean_state=mean,
                       dist={k: round(100 * v / at, 1) for k, v in act.items() if at and v})
    return out


def vm():
    t = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    g = lambda k: int(re.search(rf"{k}:\s+(\d+)", t).group(1))  # noqa: E731
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    used = float(re.search(r"used = ([\d.]+)M", sw).group(1))
    return dict(pageins=g("Pageins"), swapins=g("Swapins"), swapouts=g("Swapouts"), free=g("Pages free"),
                compressed=g("Pages occupied by compressor"), swap_used_mb=used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--gap-s", type=float, default=3.0)
    ap.add_argument("--attn-bits", type=int, default=None, help="8 = the mj-a8 arm; default exact packed attention")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import mlx_bench_worker as W
    from mj_mlx import runner as R

    path = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*"))[0]
    wa = SimpleNamespace(stream_experts=None, map_experts="~/.cache/millijoule/gptoss20b-experts-mapped",
                         attn_bits=a.attn_bits, attn_pack=a.attn_bits is None, lean=True, map_head=None,
                         wired_mb=3000, chunk=512, slots=128)
    run, mx = W.make_mj(path, wa)
    mx.set_cache_limit(1 << 30)

    rec = dict(build=[], wait=[], rescore=[], arm=[])
    of, oa, ol = R.MlxRunner.finish, R.MlxRunner.arm, R.MlxRunner.launch_deferred

    def finish(self, d):
        if d.cand is None:
            return of(self, d)
        t0 = time.perf_counter()
        c = np.array(d.cand)
        t1 = time.perf_counter()
        r = self.model.rescore_host(c, np.array(d.x32))
        t2 = time.perf_counter()
        rec["wait"].append(t1 - t0)
        rec["rescore"].append(t2 - t1)
        return r

    def arm(self, d, prev):
        t0 = time.perf_counter()
        oa(self, d, prev)
        rec["arm"].append(time.perf_counter() - t0)

    def launch(self, s):
        t0 = time.perf_counter()
        d = ol(self, s)
        rec["build"].append(time.perf_counter() - t0)
        return d

    R.MlxRunner.finish, R.MlxRunner.arm, R.MlxRunner.launch_deferred = finish, arm, launch
    gpu = States("GPU Stats", "GPU Performance States", {"GPUPH"})
    cpu = States("CPU Stats", "CPU Complex Performance States", {"ECPU", "PCPU"})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    f = open(a.out, "a")
    W.one(run, mx, f"tg128@d{a.depth}", 10_000)  # warm-up
    for r in range(a.reps):
        time.sleep(a.gap_s)
        for v in rec.values():
            v.clear()
        v0 = vm()
        g0, c0 = gpu.snap(), cpu.snap()
        row = W.one(run, mx, f"tg128@d{a.depth}", r)
        g, c = summarize(gpu.delta(g0, gpu.snap())), summarize(cpu.delta(c0, cpu.snap()))
        v1 = vm()
        ms = {k: dict(p50=1e3 * float(np.median(v)), p90=1e3 * float(np.percentile(v, 90)), n=len(v))
              for k, v in rec.items() if v}
        out = dict(rep=r, time=time.strftime("%H:%M:%S"), tok_s=row["tok_s"], itl_p50_ms=row["itl_p50_ms"],
                   itl_p99_ms=row["itl_p99_ms"], gpu=g, cpu=c, host_ms=ms, vm0=v0,
                   vm_delta={k: v1[k] - v0[k] for k in v0}, gpu_j=row["energy"]["gpu_j"],
                   gpu_w=row["energy"]["gpu_j"] / row["energy"]["seconds"], output_ids=row["output_ids"],
                   attn_bits=a.attn_bits)
        f.write(json.dumps(out) + "\n")
        f.flush()
        gp = g.get("GPUPH", {})
        print(f"{out['time']} rep {r}: {row['tok_s']:.2f} tok/s | GPU act {gp.get('active_pct', 0):.1f}% "
              f"mean P{gp.get('mean_state') or 0:.2f} {out['gpu_w']:.1f} W | wait {ms['wait']['p50']:.2f} "
              f"rescore {ms['rescore']['p50']:.2f} arm {ms['arm']['p50']:.2f} build {ms['build']['p50']:.2f} ms | "
              f"P-clu act {c['PCPU']['active_pct']:.0f}% P{c['PCPU']['mean_state'] or 0:.1f} | "
              f"pageins +{out['vm_delta']['pageins']} swapins +{out['vm_delta']['swapins']} "
              f"swap {v1['swap_used_mb']:.0f} MB free {v1['free'] * 16 // 1024} MB", flush=True)
        mx.clear_cache()


if __name__ == "__main__":
    main()

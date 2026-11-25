"""Benchmark suite to results/BENCH_PROTOCOL.md: the micro grid for every engine on every model, interleaved.

Per model, R rounds; each round runs every engine once (order reversed on even rounds), each engine block a fresh
process doing an untimed warm-up and --reps timed reps per test. Between blocks: a cool-down, then a machine-state
snapshot (swap, memory pressure, top processes, thermal). A block whose swap grew by more than 256 MB is flagged.

Engines:
  mj        tools/mlx_bench_worker.py (MilliJoule, engine overlap mode; streamed experts for models bigger than RAM)
  mlx-lm    tools/mlx_bench_worker.py (mlx_lm.generate.generate_step), where it can run the model
  llama.cpp llama-bench, native timing (llama-bench pp / tg definitions; -fa on, all layers on the GPU, -ncmoe for
            models bigger than RAM). No J/token here: an IOReport window per invocation is recorded raw, but it
            includes llama-bench's own model load (smoke run: 728 vs ~440 mJ/token). llama.cpp energy comes from
            the chat block (llama-server, first-to-last-token window).
  ollama-gguf  Ollama's llama.cpp runner on the same GGUF (native /api/generate, streamed, our clock)
  lmstudio-mlx / lmstudio-gguf  LM Studio's MLX / llama.cpp engines (its OpenAI server, streamed, our clock)
  vllm-mlx  vllm-mlx's OpenAI server on the mlx checkpoint (one request at a time, prefix cache off)
  Servers get the checkpoint tokenizer's text of the test ids (re-tokenizing to the test length) and, for decode
  tests, a short narrative opener (EOS cannot be ignored everywhere); short replies are marked in the rows.
Pre-warm (per model, per engine): the engine's own weight files read once before its block when the model's
memory-mapped weights are paged on demand (gpt-oss: ours' mapped experts, llama.cpp's GGUF), so neither engine's
first reps measure a page cache the previous block left cold.
Tests: pp512, pp2048 (prefill tok/s), tg128@d0 and tg128@d2048 (decode tok/s at context 0 / 2048).
Output dir: blocks/*.jsonl|json, state/*.json, summary.json, summary.md (mean +- sd, median, 95% CI, CV, n).

    .venv/bin/python tools/mlx_bench_suite.py --models q9b --reps 5 --rounds 2 --out results/bench_suite/v1
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from swap_guard import add_tripped, cfg_key, tripped  # noqa: E402
from mlx_proc_mem import PeakSampler, pids_matching  # noqa: E402  (per-engine memory, checklist 0.10)
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "bin", "python")
sys.path.insert(0, os.path.join(ROOT, "python"))

H = os.path.expanduser("~/.cache/huggingface/hub")
MODELS = {
    "q9b": dict(
        name="Qwen3.5-9B",
        mlx=f"{H}/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*",
        gguf=f"{H}/models--lmstudio-community--Qwen3.5-9B-GGUF/snapshots/*/Qwen3.5-9B-Q4_K_M.gguf",
        weights="ours / mlx-lm: mlx-community 4-bit g64 (~4.5 bpw); llama.cpp: lmstudio-community Q4_K_M (~4.8 bpw)",
        engines=["mj", "mj-s64", "mlx-lm", "llama.cpp"], mj_args=[], llama_args=[],
        variant_args={"mj-s64": []}, variant_env={"mj-s64": {"MJ_ATTN_SCORES_MB": "64"}}),
    "gptoss": dict(
        name="gpt-oss-20b",
        mlx=f"{H}/models--openai--gpt-oss-20b/snapshots/*",
        gguf=f"{H}/models--ggml-org--gpt-oss-20b-GGUF/snapshots/*/gpt-oss-20b-MXFP4.gguf",
        weights="mj: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention, experts mlocked, 512-token prefill chunks as llama.cpp's ubatch); mj-a8: the same with 8-bit affine attention weights (lossy, matched "
                "to the GGUF); llama.cpp / Ollama / LM Studio (its llama.cpp engine): ggml-org MXFP4 GGUF (Q8_0 attention)",
        engines=["mj", "mj-x", "mj-a8", "mj-a8w", "mj-a8wb", "mj-a8wbc", "mj-a8wbn", "mj-a8wbf", "mj-a8wb24",
                 "mj-a8wb16", "llama.cpp", "ollama-gguf", "lmstudio-gguf"],
        mj_args=["--map-experts", "~/.cache/millijoule/gptoss20b-experts-mapped", "--attn-pack", "--lean",
                 "--wired-mb", "3000", "--chunk", "512"], llama_args=[],
        # mj-x: the exact arm with MLX's wired limit at 1000 MB (gptoss_mem_fix_v2: no Metal "required > max" at 1000;
        # ~1000 events per probe at 3000, which preceded the v12 memorystatus kill)
        variant_args={"mj-x": ["--map-experts", "~/.cache/millijoule/gptoss20b-experts-mapped", "--attn-pack", "--lean",
                               "--wired-mb", "1000", "--chunk", "512"],
                      "mj-a8": ["--map-experts", "~/.cache/millijoule/gptoss20b-experts-mapped", "--attn-bits", "8",
                                "--lean", "--wired-mb", "3000", "--chunk", "512"],
                      # mj-a8w: the 8-bit arm at the 1000 MB wired limit (v13: every Metal over-limit line fell in mj-a8 blocks)
                      "mj-a8w": ["--map-experts", "~/.cache/millijoule/gptoss20b-experts-mapped", "--attn-bits", "8",
                                 "--lean", "--wired-mb", "1000", "--chunk", "512"],
                      # mj-a8wb: mj-a8w with the experts pinned in a background thread (MJ_MLOCK=bg; cold start, 0.11)
                      "mj-a8wb": ["--map-experts", "~/.cache/millijoule/gptoss20b-experts-mapped", "--attn-bits", "8",
                                  "--lean", "--wired-mb", "1000", "--chunk", "512"],
                      # The prepared-weights cache is on by default (~/Library/Caches/mj-mlx/prep; RESULTS 1zzc), so
                      # every mj variant reads it. For cold start: mj-a8wbc = mj-a8wb with it (v17f's name), mj-a8wbn
                      # without it, mj-a8wbf the first-ever load (an empty cache each start, which builds it)
                      "mj-a8wbc": ["--map-experts", "~/.cache/millijoule/gptoss20b-experts-mapped", "--attn-bits", "8",
                                   "--lean", "--wired-mb", "1000", "--chunk", "512"],
                      "mj-a8wbn": ["--map-experts", "~/.cache/millijoule/gptoss20b-experts-mapped", "--attn-bits", "8",
                                   "--lean", "--wired-mb", "1000", "--chunk", "512"],
                      "mj-a8wbf": ["--map-experts", "~/.cache/millijoule/gptoss20b-experts-mapped", "--attn-bits", "8",
                                   "--lean", "--wired-mb", "1000", "--chunk", "512"],
                      # mj-a8wb24 / 16: mj-a8wb pinning only each layer's 24 / 16 most-used experts (MJ_MLOCK_TOP,
                      # expert_rank.json in the store; checklist 0.11b, RESULTS 1zzd)
                      "mj-a8wb24": ["--map-experts", "~/.cache/millijoule/gptoss20b-experts-mapped", "--attn-bits", "8",
                                    "--lean", "--wired-mb", "1000", "--chunk", "512"],
                      "mj-a8wb16": ["--map-experts", "~/.cache/millijoule/gptoss20b-experts-mapped", "--attn-bits", "8",
                                    "--lean", "--wired-mb", "1000", "--chunk", "512"]},
        variant_env={"mj-a8wb": {"MJ_MLOCK": "bg"}, "mj-a8wbc": {"MJ_MLOCK": "bg"},
                     "mj-a8wbn": {"MJ_MLOCK": "bg", "MJ_PREP_CACHE": "0"},
                     "mj-a8wbf": {"MJ_MLOCK": "bg", "MJ_PREP_CACHE": "~/.cache/millijoule/prep-fresh",
                                  "MJ_PREP_CACHE_FRESH": "1"},
                     "mj-a8wb24": {"MJ_MLOCK": "bg", "MJ_MLOCK_TOP": "24"},
                     "mj-a8wb16": {"MJ_MLOCK": "bg", "MJ_MLOCK_TOP": "16"}},
        mj_env={"MJ_MLOCK": "1"},  # experts pinned in RAM, as llama.cpp's weights are (RESULTS 1y)
        prewarm={"mj": ["~/.cache/millijoule/gptoss20b-experts-mapped/*.bin",
                        "~/Library/Caches/mj-mlx/prep/*/*.safetensors"],
                 "mj-x": ["~/.cache/millijoule/gptoss20b-experts-mapped/*.bin",
                         "~/Library/Caches/mj-mlx/prep/*/*.safetensors"],
                 "mj-a8": ["~/.cache/millijoule/gptoss20b-experts-mapped/*.bin",
                          "~/Library/Caches/mj-mlx/prep/*/*.safetensors"],
                 "mj-a8w": ["~/.cache/millijoule/gptoss20b-experts-mapped/*.bin",
                           "~/Library/Caches/mj-mlx/prep/*/*.safetensors"],
                 "mj-a8wb": ["~/.cache/millijoule/gptoss20b-experts-mapped/*.bin",
                             "~/Library/Caches/mj-mlx/prep/*/*.safetensors"],
                 "mj-a8wbc": ["~/.cache/millijoule/gptoss20b-experts-mapped/*.bin",
                              "~/Library/Caches/mj-mlx/prep/*/*.safetensors"],
                 "mj-a8wbn": ["~/.cache/millijoule/gptoss20b-experts-mapped/*.bin"],
                 "mj-a8wb24": ["~/.cache/millijoule/gptoss20b-experts-mapped/*.bin",
                               "~/Library/Caches/mj-mlx/prep/*/*.safetensors"],
                 "mj-a8wb16": ["~/.cache/millijoule/gptoss20b-experts-mapped/*.bin",
                               "~/Library/Caches/mj-mlx/prep/*/*.safetensors"],
                 "mj-a8wbf": ["~/.cache/millijoule/gptoss20b-experts-mapped/*.bin"], "llama.cpp": ["GGUF"]}),
    "q35b": dict(
        name="Qwen3.6-35B-A3B (20.4 GB, bigger than RAM)",
        mlx=f"{H}/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/*",
        gguf="~/.cache/millijoule/gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf",
        weights="ours: mlx-community 4-bit g64, experts streamed from the SSD (96 slots/layer, C++ streamer; 128 ran out of Metal memory in v1); "
                "llama.cpp: Q4_K_M, experts on the CPU (-ncmoe 40)",
        engines=["mj", "mj-u96", "llama.cpp"],
        mj_args=["--stream-experts", "~/.cache/millijoule/qwen36-35b-experts-stream", "--slots", "96"],
        # mj-u96: the same memory (96 slots per layer on average), split per layer by the store's slot profile
        # (RESULTS 1zt; tools/mlx_slot_profile.py)
        variant_args={"mj-u96": ["--stream-experts", "~/.cache/millijoule/qwen36-35b-experts-stream", "--slots", "96",
                                 "--slot-profile"]},
        mj_env={"MJ_GPU_STREAM": "1"}, llama_args=["-ncmoe", "40"], llama_reps=3),
    "olmoe": dict(
        name="OLMoE-1B-7B-0125-Instruct (general MoE model)",
        mlx=f"{H}/models--mlx-community--OLMoE-1B-7B-0125-Instruct-4bit/snapshots/*",
        gguf=f"{H}/models--allenai--OLMoE-1B-7B-0125-Instruct-GGUF/snapshots/*/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf",
        weights="ours / mlx-lm: mlx-community 4-bit g64 (float16 scales; run in float16); llama.cpp / Ollama (its GGUF "
                "runner; its MLX runner rejects OlmoeForCausalLM): allenai Q4_K_M; LM Studio (its MLX engine) and "
                "vllm-mlx: the mlx-community 4-bit",
        engines=["mj", "mlx-lm", "llama.cpp", "ollama-gguf", "lmstudio-mlx", "vllm-mlx"], mj_args=["--placement", "auto"],
        llama_args=[]),
    "dsv2lite": dict(
        name="DeepSeek-V2-Lite-Chat (MLA, general MoE model)",
        mlx=f"{H}/models--mlx-community--DeepSeek-V2-Lite-Chat-4bit-mlx/snapshots/*",
        gguf=f"{H}/models--mradermacher--DeepSeek-V2-Lite-Chat-GGUF/snapshots/*/DeepSeek-V2-Lite-Chat.Q4_K_M.gguf",
        weights="ours / mlx-lm: mlx-community 4-bit g64; ours caches MLA latents, mlx-lm expanded K/V; "
                "llama.cpp / Ollama: mradermacher Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit",
        engines=["mj", "mj-c512", "mj-s64", "mj-mlak", "mj-w9", "mlx-lm", "llama.cpp", "ollama-gguf", "lmstudio-mlx", "vllm-mlx"],
        mj_args=["--placement", "auto"],
        # mj-c512: 512-token prefill chunks (llama.cpp's default ubatch); at 2048 the 16k prefill ran out of Metal memory (v13)
        variant_args={"mj-c512": ["--placement", "auto", "--chunk", "512"], "mj-s64": ["--placement", "auto"],
                      "mj-mlak": ["--placement", "auto"],  # mj-mlak: MLA decode through mla_decode.py's kernel
                      # mj-w9: MLX wired limit 9000 MB instead of the recommended working set (~12.7 GB): v15 had macOS
                      # kill a 16k worker for holding 76% of the system's wired memory
                      "mj-w9": ["--placement", "auto", "--wired-mb", "9000"]},
        # mj-s64: prefill attention score blocks of 64 MB instead of 256 (probe v1: at 16k, 2 Metal over-limit lines
        # instead of 274-283, same prefill speed)
        variant_env={"mj-s64": {"MJ_ATTN_SCORES_MB": "64"}, "mj-mlak": {"MJ_MLA_DECODE": "kernel"}},
        llama_args=[]),
    "q30b": dict(
        name="Qwen3-30B-A3B (17.2 GB 4-bit, bigger than RAM; general MoE model)",
        mlx=f"{H}/models--mlx-community--Qwen3-30B-A3B-4bit/snapshots/*",
        gguf=f"{H}/models--Qwen--Qwen3-30B-A3B-GGUF/snapshots/*/Qwen3-30B-A3B-Q4_K_M.gguf",
        weights="ours: mlx-community 4-bit g64, experts placed by memory (streamed from the SSD, slots per layer from memory: 76 on 16 GB); "
                "llama.cpp: Qwen Q4_K_M, experts on the CPU (-ncmoe 48)",
        engines=["mj", "llama.cpp", "ollama-gguf"], mj_args=["--placement", "auto"], llama_args=["-ncmoe", "48"],
        llama_reps=3),
    "glm_air": dict(
        name="GLM-4.5-Air (106B, 4-bit 65.4 GB, four times RAM)",
        mlx=f"{H}/models--mlx-community--GLM-4.5-Air-4bit/snapshots/*",
        gguf=None,  # no rival on it without the user's OK: every one would run far above RAM (swap)
        weights="ours: mlx-community 4-bit g64, experts streamed from the SSD (~/.cache/mj-mlx store), slots per "
                "layer from memory; the MTP layer (46) not loaded",
        engines=["mj", "mj-u"], mj_args=["--placement", "auto"],
        # mj-u: the same memory split per layer by the store's slot profile (RESULTS 1zt / 1zze)
        variant_args={"mj-u": ["--placement", "auto"]},
        variant_env={"mj-u": {"MJ_GPU_STREAM": "1", "MJ_SLOT_PROFILE": "1"}},
        mj_env={"MJ_GPU_STREAM": "1"}, llama_args=[]),
}
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 11: 2.20,
       12: 2.18, 13: 2.16, 14: 2.14, 15: 2.13, 19: 2.09, 24: 2.06, 29: 2.05}


def t95(df):
    if df in T95:
        return T95[df]
    return T95[max(k for k in T95 if k <= df)] if df > 1 else 12.71


def g(p):
    m = glob.glob(os.path.expanduser(p))
    return m[0] if m else None


def vm_swap_pages():
    """(swap-ins, swap-outs) since boot, in pages (16 KB): a block's paging, which net swap growth can miss (1zk)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    get = lambda k: int((re.search(rf"{k}:\s+(\d+)", out) or [0, 0])[1])
    return get("Swapins"), get("Swapouts")


SWAP_ABORT_GB = float(os.environ.get("MJ_SWAP_ABORT_GB", "20"))


class SwapGuard:
    """Abort a block whose swap-outs pass SWAP_ABORT_GB (default 20 GB, env MJ_SWAP_ABORT_GB): such a block is flagged
    anyway and only wears the SSD (suite v16: llama-server on Qwen3.6-35B wrote about 900 GB of swap per block, 1zu).
    Polls vm_stat every `every` s; on a trip, kills every process `pids()` returns and sets `tripped` (GB swapped out)."""

    def __init__(self, pids, limit_gb=None, every=2.0):
        import threading

        self.pids, self.limit, self.every = pids, SWAP_ABORT_GB if limit_gb is None else limit_gb, every
        self.tripped, self._stop, self._t = None, threading.Event(), threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.base = vm_swap_pages()[1]
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(10)

    def out_gb(self):
        return (vm_swap_pages()[1] - self.base) * 16384 / 1e9

    def _run(self):
        import signal

        while not self._stop.wait(self.every):
            gb = self.out_gb()
            if gb > self.limit:
                self.tripped = gb
                for pid in self.pids():
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                print(f"  ABORTED: swap-outs {gb:.1f} GB passed {self.limit:.0f} GB", flush=True)
                return


def engine_pids(eng, worker_pid, server=None):
    """The processes that hold an engine's model: our worker (mj, mlx-lm), or the server's model process."""
    if eng.startswith("ollama"):  # v15 OLMoE: no "runner" process matched (the server may hold the model itself)
        return pids_matching("ollama")
    if eng.startswith("lmstudio"):  # the app and its helpers; the model sits in one of them
        return sorted(set(pids_matching("LM Studio") + pids_matching("lmstudio") + pids_matching(".lmstudio")))
    if eng == "vllm-mlx":
        return [server.pid] + pids_matching("vllm-mlx", "multiprocessing") if server else pids_matching("vllm-mlx")
    if eng.startswith("llama-server") and server:
        return [server.pid]
    return [worker_pid]


def swap_mb():
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", s)
    return float(m.group(1)) if m else float("nan")


def state(path):
    subprocess.run([PY, os.path.join(HERE, "mlx_machine_state.py"), path], capture_output=True)
    th = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    try:
        d = json.load(open(path))
    except Exception:
        d = {}
    d["pmset_therm"] = th.strip()
    d["swap_used_mb"] = swap_mb()
    json.dump(d, open(path, "w"), indent=1)
    return d


def llama_block(m, gguf, reps, out_json, tests):
    """llama-bench, one invocation per test (an IOReport window each). Returns True when the swap guard stopped it."""
    from mj_mlx.energy import Meter

    rows, tripped = [], False
    for test in tests:
        if test.startswith("pp"):
            spec = ["-p", test[2:], "-n", "0", "-d", "0"]
            toks = int(test[2:]) * (reps + 1)
        else:
            d = int(test.split("@d")[1])
            n = int(test.split("@")[0][2:])
            spec = ["-p", "0", "-n", str(n), "-d", str(d)]
            toks = n * reps + 1 if d == 0 else None
        cmd = ["llama-bench", "-m", gguf, "-fa", "1", "-ngl", "99", "-r", str(reps), "-o", "json", *spec,
               *m["llama_args"]]
        sw0 = vm_swap_pages()
        with Meter() as mt:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            with PeakSampler(lambda: [p.pid]) as mem, SwapGuard(lambda: [p.pid]) as guard:
                out, err = p.communicate()
            p = subprocess.CompletedProcess(cmd, p.returncode, out, err)
        sw1 = vm_swap_pages()
        swap = dict(swapins_gb=(sw1[0] - sw0[0]) * 16384 / 1e9, swapouts_gb=(sw1[1] - sw0[1]) * 16384 / 1e9)
        print(f"llama.cpp {test}: swapped in {swap['swapins_gb']:.1f} GB, out {swap['swapouts_gb']:.1f} GB", flush=True)
        if guard.tripped:
            open(out_json + ".flagged", "a").write(f"aborted: swap-outs {guard.tripped:.1f} GB passed "
                                                   f"{guard.limit:.0f} GB ({test})\n")
            json.dump(dict(engine="llama.cpp", test=test, aborted="swap", cmd=" ".join(cmd), **swap),
                      open(out_json + ".aborted.json", "w"), indent=1)
            tripped = True
            break  # the rest of the block would swap the same way
        try:
            res = json.loads(p.stdout)
        except Exception:
            print("llama-bench failed:", p.stderr[-800:], flush=True)
            continue
        for x in res:
            w = mt.window
            e = dict(seconds=w.seconds, gpu_j=w.gpu_j, cpu_j=w.rail("CPU Energy"), ane_j=w.rail("ANE"),
                     dram_j=w.rail("DRAM"))
            rows.append(dict(engine="llama.cpp", test=test, samples_ts=x.get("samples_ts"), avg_ts=x.get("avg_ts"),
                             stddev_ts=x.get("stddev_ts"), n_prompt=x.get("n_prompt"), n_gen=x.get("n_gen"),
                             n_depth=x.get("n_depth"), build_commit=x.get("build_commit"),
                             build_number=x.get("build_number"), n_batch=x.get("n_batch"),
                             n_ubatch=x.get("n_ubatch"), flash_attn=x.get("flash_attn"), cmd=" ".join(cmd),
                             energy_window_incl_load=e, tokens_in_window=toks,
                             peak_footprint_gb=mem.peak / 1e9, peak_resident_gb=mem.peak_resident / 1e9, **swap,
                             j_per_token_gpu=None, j_per_token_soc=None))  # window includes model load: not J/token
            print(f"llama.cpp {test}: {x.get('avg_ts'):.2f} +- {x.get('stddev_ts'):.2f} tok/s", flush=True)
        time.sleep(5)
    json.dump(rows, open(out_json, "w"), indent=1)
    return tripped


OLLAMA = "http://127.0.0.1:11434"


def ollama_prepare(key, eng, src):
    """Start `ollama serve` if it is not up and create the model from `src` (GGUF file or mlx dir) once."""
    import urllib.request

    try:
        urllib.request.urlopen(OLLAMA + "/api/version", timeout=3)
    except Exception:
        subprocess.Popen(["ollama", "serve"], stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT)
        for _ in range(60):
            time.sleep(1)
            try:
                urllib.request.urlopen(OLLAMA + "/api/version", timeout=3)
                break
            except Exception:
                pass
    name = f"mj-{key}-{eng}".lower()
    have = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    if name + ":" not in have:
        mf = os.path.join(os.path.expanduser("~/.cache/millijoule"), f"Modelfile.{name}")
        open(mf, "w").write(f"FROM {src}\n")
        p = subprocess.run(["ollama", "create", name, "-f", mf], capture_output=True, text=True)
        print(f"  ollama create {name}: {(p.stdout + p.stderr).strip().splitlines()[-1:]}", flush=True)
    return name


# Engines any model can run when asked for by name (--engines), not in the default lists.
#   llama-server  llama.cpp's server on the suite's GGUF, timed by our clock through its OpenAI endpoint, so it gets
#                 a first-to-last-token energy window (llama-bench's window includes its model load): checklist 0.6
EXTRA_ENGINES = ["llama-server"]
LLAMA_URL = "http://127.0.0.1:8092"


def llama_server_extra(eng):
    """llama-server variants by name: `llama-server-mc<N>` adds llama.cpp's GPU expert cache, `--moe-cache-mib N`
    (merged Oct 2026; for MoE experts kept on the CPU with -ncmoe / -cmoe). Plain `llama-server`: nothing."""
    m = re.fullmatch(r"llama-server-mc(\d+)", eng)
    return ["--moe-cache-mib", m.group(1)] if m else []


def llama_server_start(m, gguf, extra=()):
    """llama-server as llama-bench runs it (-fa on, all layers on the GPU, the model's llama_args), one slot, an 8k
    context, prompt cache off (each rep's prompt is new anyway)."""
    p = subprocess.Popen(["llama-server", "-m", gguf, "-fa", "on", "-ngl", "99", "-c", "8192", "-np", "1",
                          "--no-cache-prompt", "--port", LLAMA_URL.rsplit(":", 1)[1], "--alias", "bench",
                          *m["llama_args"], *extra], stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT)
    ok = wait_url(LLAMA_URL + "/health", 600)
    print(f"  llama-server: {'up' if ok else 'did not come up'}", flush=True)
    return p


LMS = os.path.expanduser("~/.lmstudio/bin/lms")
LMS_URL = "http://127.0.0.1:1234"
VLLM_MLX = os.path.expanduser("~/.venvs/vllm-mlx/bin/vllm-mlx")
VLLM_URL = "http://127.0.0.1:8011"


def wait_url(url, tries=180):
    import urllib.request

    for _ in range(tries):
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(1)
    return False


def lmstudio_prepare(key, eng, src):
    """LM Studio: link the weights under ~/.lmstudio/models/mj-bench/ (symlinks, nothing copied; a GGUF through
    `lms import -l`), start its server, load with full GPU offload, an 8k context (as Ollama's num_ctx) and one
    request at a time. Returns the API identifier."""
    name = f"mj-{key}-{eng}".lower()
    d = os.path.expanduser(f"~/.lmstudio/models/mj-bench/{name}")
    if not os.path.exists(d):
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if os.path.isdir(src):
            os.symlink(os.path.realpath(src), d)
        else:  # keep the .gguf name (the HF snapshot's link), not the blob's
            subprocess.run([LMS, "import", "-l", "--user-repo", f"mj-bench/{name}", src, "-y"], capture_output=True)
    subprocess.run([LMS, "server", "start"], capture_output=True, text=True)
    wait_url(LMS_URL + "/v1/models")
    listed = json.loads(subprocess.run([LMS, "ls", "--json"], capture_output=True, text=True).stdout or "[]")
    mk = next((m["modelKey"] for m in listed if m.get("path", "").startswith(f"mj-bench/{name}")), name)
    p = subprocess.run([LMS, "load", mk, "--gpu", "max", "-c", "8192", "--parallel", "1", "--identifier", name, "-y"],
                       capture_output=True, text=True)
    print(f"  lms load {mk} as {name}: {(p.stdout + p.stderr).strip().splitlines()[-1:]}", flush=True)
    return name


def vllm_mlx_start(src):
    """vllm-mlx's OpenAI server on the mlx checkpoint: one request at a time, prefix cache off (each rep is a new
    prompt anyway; the d2048 test's prompt repeats pp2048's), a token per stream event."""
    p = subprocess.Popen([VLLM_MLX, "serve", os.path.realpath(src), "--port", VLLM_URL.rsplit(":", 1)[1],
                          "--served-model-name", "bench", "--disable-prefix-cache", "--stream-interval", "1",
                          "--max-num-seqs", "1"], stdout=open(os.devnull, "w"), stderr=subprocess.STDOUT)
    ok = wait_url(VLLM_URL + "/v1/models", 600)
    print(f"  vllm-mlx serve: {'up' if ok else 'did not come up'}", flush=True)
    return p


def rival_versions():
    """Versions of the other engines, for meta.json (checklist 0.9: versions pinned in every published table)."""
    def run(cmd):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return (p.stdout + p.stderr).strip()[-200:]
        except Exception as e:
            return f"unavailable ({type(e).__name__})"

    v = dict(ollama=run(["ollama", "--version"]),
             lmstudio_app=run(["defaults", "read", "/Applications/LM Studio.app/Contents/Info",
                               "CFBundleShortVersionString"]),
             vllm_mlx=run([os.path.join(os.path.dirname(VLLM_MLX), "python"), "-c",
                           "import importlib.metadata as m; print(m.version('vllm-mlx'), 'mlx', m.version('mlx'), "
                           "'mlx-lm', m.version('mlx-lm'))"]))
    try:  # the engines LM Studio selects for GGUF and MLX weights
        v["lmstudio_engines"] = json.load(open(os.path.expanduser("~/.lmstudio/.internal/backend-preferences-v1.json")))
    except Exception:
        v["lmstudio_engines"] = None
    return v


def stats(xs):
    xs = np.asarray([x for x in xs if x is not None], float)
    n = len(xs)
    if n == 0:
        return None
    mean, sd = float(xs.mean()), float(xs.std(ddof=1)) if n > 1 else 0.0
    half = t95(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return dict(n=n, mean=mean, sd=sd, median=float(np.median(xs)), ci95=[mean - half, mean + half],
                cv=sd / mean if mean else float("nan"), min=float(xs.min()), max=float(xs.max()))


WINDOW_SWAP_MB = 64.0  # a timed window that swaps out more than this is flagged (BENCH_PROTOCOL, swap rule)


def summarize(out, models):
    summ = {}
    rep_flags = {}
    for key in models:
        m = MODELS[key]
        rows = []
        for f in sorted(glob.glob(os.path.join(out, "blocks", f"{key}_*.jsonl"))):
            flag = os.path.exists(f + ".flagged")
            blk = [dict(json.loads(l), flagged=flag) for l in open(f)]
            # window swap rule (BENCH_PROTOCOL, agreed with the PM 2026-11-23): a rep whose timed window swapped out
            # more than WINDOW_SWAP_MB is flagged; a block with more than half its reps flagged is flagged whole
            for r in blk:
                so = (r.get("energy") or {}).get("swapouts_mb")
                if so is not None and so > WINDOW_SWAP_MB:
                    r["flagged"], r["flag_why"] = True, f"window swap-out {so:.0f} MB"
            n_rep = sum(1 for r in blk if r.get("flag_why"))
            if n_rep * 2 > len(blk):
                for r in blk:
                    r["flagged"] = True
            rep_flags[os.path.basename(f)] = dict(reps=len(blk), window_flagged=n_rep, block_flagged=n_rep * 2 > len(blk))
            rows += blk
        for f in sorted(glob.glob(os.path.join(out, "blocks", f"{key}_*_llama.json"))):
            flag = os.path.exists(f + ".flagged")
            for x in json.load(open(f)):
                for s in x["samples_ts"] or []:
                    rows.append(dict(engine="llama.cpp", test=x["test"], tok_s=s, j_per_token_gpu=x["j_per_token_gpu"],
                                     j_per_token_soc=x["j_per_token_soc"], flagged=flag))
        res = {}
        seen = list(dict.fromkeys(r["engine"] for r in rows))  # also by-name variants such as llama-server-mc4096
        for eng in m["engines"] + [e for e in EXTRA_ENGINES + seen if e not in m["engines"]]:
            for test in sorted({r["test"] for r in rows}):
                sel = [r for r in rows if r["engine"] == eng and r["test"] == test and not r["flagged"]]
                if not sel and not any(r["engine"] == eng and r["test"] == test for r in rows):
                    continue  # (an engine whose blocks were all flagged still gets its flagged row)
                res[f"{eng}|{test}"] = dict(
                    engine=eng, test=test, tok_s=stats([r["tok_s"] for r in sel]),
                    j_per_token_gpu=stats([r.get("j_per_token_gpu") for r in sel]),
                    j_per_token_soc=stats([r.get("j_per_token_soc") for r in sel]),
                    j_per_token_sys=stats([r.get("j_per_token_sys") for r in sel]),
                    j_per_token_sys_net=stats([r.get("j_per_token_sys_net") for r in sel]),
                    j_per_token_sys_steady_net=stats([r.get("j_per_token_sys_steady_net") for r in sel]),
                    j_per_token_gpu_net=stats([r.get("j_per_token_gpu_net") for r in sel]),
                    ttft_s=stats([r.get("ttft_s") for r in sel]) if any("ttft_s" in r for r in sel) else None,
                    window_swapouts_mb=stats([(r.get("energy") or {}).get("swapouts_mb") for r in sel]),
                    window_swapouts_max_mb=max([(r.get("energy") or {}).get("swapouts_mb") or 0 for r in sel] or [0])
                    if any((r.get("energy") or {}).get("swapouts_mb") is not None for r in sel) else None,
                    short=sum(1 for r in sel if r.get("short")),
                    no_rate=sum(1 for r in sel if r.get("tok_s") is None and not r.get("invalid")),
                    invalid=sum(1 for r in sel if r.get("invalid")),
                    invalid_why=sorted({str(r["invalid"])[:200] for r in sel if r.get("invalid")}),
                    excluded_flagged=sum(1 for r in rows if r["engine"] == eng and r["test"] == test and r["flagged"]),
                    flagged_tok_s=stats([r["tok_s"] for r in rows
                                         if r["engine"] == eng and r["test"] == test and r["flagged"]]))
        # correctness across in-process engines: greedy ids of the same decode seed
        agree = {}
        a = {(r["test"], r["seed"]): r["output_ids"] for r in rows if r["engine"] == "mj" and "output_ids" in r}
        b = {(r["test"], r["seed"]): r["output_ids"] for r in rows if r["engine"] == "mlx-lm" and "output_ids" in r}
        for k in sorted(set(a) & set(b)):
            x, y = a[k], b[k]
            fd = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), None)
            agree[f"{k[0]} seed {k[1]}"] = dict(identical=x == y, first_diff=fd, n=min(len(x), len(y)))
        summ[key] = dict(model=m["name"], weights=m["weights"], results=res, mj_vs_mlxlm_greedy=agree,
                         window_swap_flags={k: v for k, v in rep_flags.items() if k.startswith(key + "_")})
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    lines = ["# Benchmark suite summary", "",
             "Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks "
             "excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench "
             "(its window includes model load).", ""]
    for key, s in summ.items():
        lines += [f"## {s['model']}", "", f"Weights: {s['weights']}", "",
                  "| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token | machine mJ/token above idle | "
                  "machine, steady state | swap-out MB in window (mean / max) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        order = {"pp512": 0, "pp2048": 1, "tg128@d0": 2, "tg128@d2048": 3}
        for k, r in sorted(s["results"].items(), key=lambda kv: (order.get(kv[1]["test"], 9), kv[1]["engine"])):
            t, e = r["tok_s"], r["j_per_token_gpu"]
            if t is None:
                why = (f"invalid: {r['invalid']} server failures ({'; '.join(r['invalid_why'])})" if r.get("invalid")
                       else f"no rate: {r['no_rate']} replies of at most one token" if r.get("no_rate")
                       else "every block flagged (see below)")
                lines.append(f"| {r['engine']} | {r['test']} | {why} | | | 0 | - | - | - | - |")
                continue
            lines.append(f"| {r['engine']} | {r['test']} | {t['mean']:.2f} +- {t['sd']:.2f} | "
                         f"{t['ci95'][0]:.2f}-{t['ci95'][1]:.2f} | {100 * t['cv']:.1f}% | {t['n']} | "
                         + (f"{1e3 * e['mean']:.0f}" if e else "-")
                         + (f" | {1e3 * r['j_per_token_sys_net']['mean']:.0f}" if r.get("j_per_token_sys_net") else " | -")
                         + (f" | {1e3 * r['j_per_token_sys_steady_net']['mean']:.0f}"
                            f" +- {1e3 * r['j_per_token_sys_steady_net']['sd']:.0f}"
                            if r.get("j_per_token_sys_steady_net") else " | -")
                         + (f" ({r['short']} replies under the requested length; tok/s over the tokens produced)" if r.get("short") else "")
                         + (f" | {r['window_swapouts_mb']['mean']:.0f} / {r['window_swapouts_max_mb']:.0f}"
                            if r.get("window_swapouts_mb")
                            else " | block swap rule only (no window counters)" if r["engine"] == "llama.cpp"
                            else " | -")
                         + " |")
        wf = {k: v for k, v in (s.get("window_swap_flags") or {}).items() if v["window_flagged"]}
        if wf:
            lines += ["", f"Window swap flags (swap-out > {WINDOW_SWAP_MB:.0f} MB in the timed window): "
                      + "; ".join(f"{k}: {v['window_flagged']} of {v['reps']} reps"
                                  + (" (whole block flagged)" if v["block_flagged"] else "") for k, v in wf.items())]
        fl = [r for r in s["results"].values() if r.get("flagged_tok_s")]
        if fl:  # shown, never mixed into the table above
            lines += ["", "Flagged reps and blocks (excluded above; block reasons in blocks/*.flagged; a rep is also "
                      f"flagged when its timed window swapped out more than {WINDOW_SWAP_MB:.0f} MB, and a block "
                      "when more than half its reps are):", "",
                      "| engine | test | tok/s (flagged reps) | n |", "| --- | --- | --- | --- |"]
            for r in sorted(fl, key=lambda r: (order.get(r["test"], 9), r["engine"])):
                f = r["flagged_tok_s"]
                lines.append(f"| {r['engine']} | {r['test']} | {f['mean']:.2f} +- {f['sd']:.2f} "
                             f"({f['min']:.2f}-{f['max']:.2f}) | {f['n']} |")
        ag = s["mj_vs_mlxlm_greedy"]
        if ag:
            same = sum(v["identical"] for v in ag.values())
            fds = [v["first_diff"] for v in ag.values() if v["first_diff"] is not None]
            lines += ["", f"Greedy decode ids, ours vs mlx-lm (same seeds): {same}/{len(ag)} identical"
                      + (f"; first divergence at token {min(fds)}-{max(fds)} otherwise" if fds else "")]
        mem = []
        for f in sorted(glob.glob(os.path.join(out, "state", f"{key}_r*_mem.json"))):
            d = json.load(open(f))
            mem.append((os.path.basename(f)[len(key) + 1:-9], d))
        for f in sorted(glob.glob(os.path.join(out, "blocks", f"{key}_*_llama.json"))):
            xs = [x for x in json.load(open(f)) if x.get("peak_footprint_gb")]
            if xs:
                mem.append((os.path.basename(f)[len(key) + 1:-11], dict(
                    engine="llama.cpp", peak_footprint_gb=max(x["peak_footprint_gb"] for x in xs),
                    peak_resident_gb=max(x.get("peak_resident_gb") or 0 for x in xs))))
        if mem:  # checklist 0.10: memory per engine (macOS phys_footprint of the model's process, sampled each 1 s)
            lines += ["", "Peak memory per block, GB (footprint: macOS's figure, GPU buffers included, mapped files not; "
                      "resident: pages of the process in RAM that its page tables map, which counts a mapped weight file only where the CPU touched it: a file the GPU reads through a no-copy buffer, as llama.cpp's Metal backend does, can read near 0 or near its full size depending on how it was paged in, so resident is not comparable across engines); swap-ins / -outs over the whole block (load included):", "",
                      "| block | engine | peak footprint | peak resident | swap-in GB | swap-out GB |",
                      "| --- | --- | --- | --- | --- | --- |"]
            for tag, d in mem:
                si, so = d.get("swapins_gb"), d.get("swapouts_gb")
                pr = d.get("peak_resident_gb")
                lines.append(f"| {tag} | {d['engine']} | {d['peak_footprint_gb']:.2f} | {'-' if pr is None else f'{pr:.2f}'} | "
                             f"{'-' if si is None else f'{si:.2f}'} | {'-' if so is None else f'{so:.2f}'} |")
        lines.append("")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="q9b,gptoss")
    ap.add_argument("--engines", default="", help="comma list to restrict (default: each model's list)")
    ap.add_argument("--tests", default="pp512,pp2048,tg128@d0,tg128@d2048")
    ap.add_argument("--reps", type=int, default=5, help="timed reps per test per block")
    ap.add_argument("--llama-reps", type=int, default=0, help="llama.cpp reps (default --reps)")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--round0", type=int, default=1, help="first round number (reruns get new block files)")
    ap.add_argument("--cooldown", type=int, default=30)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    models = a.models.split(",")
    os.makedirs(os.path.join(a.out, "blocks"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "state"), exist_ok=True)
    if a.summary_only:
        return summarize(a.out, models)
    meta = dict(start=time.strftime("%Y-%m-%dT%H:%M:%S"), args=vars(a), git=subprocess.run(
        ["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        llama_bench=subprocess.run(["llama-bench", "--version"], capture_output=True, text=True).stderr.strip()[-200:],
        mlx=subprocess.run([PY, "-c", "import importlib.metadata as m; print(m.version('mlx'), m.version('mlx-lm'))"],
                           capture_output=True, text=True).stdout.strip(),
        macos=subprocess.run(["sw_vers"], capture_output=True, text=True).stdout.strip(),
        chip=subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
        power=subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.strip(),
        rivals=rival_versions())
    json.dump(meta, open(os.path.join(a.out, f"meta_{time.strftime('%H%M%S')}.json"), "w"), indent=1)
    caff = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    tests = a.tests.split(",")
    any_trip = False
    direct = os.environ.get("MJ_RUN_STEPS") != "1"  # started by hand, not by a run script: own the daily accounting
    if direct:
        from swap_guard import DAILY_GB, daily_total, swapouts_gb

        today, _ = daily_total()
        if today > DAILY_GB and os.environ.get("MJ_SWAP_DAILY_OK") != "1":
            print(f"not run: {today:.1f} GB of swap-outs today > {DAILY_GB} GB daily budget (results/swap_daily.json); "
                  "MJ_SWAP_DAILY_OK=1 only on the user's word", flush=True)
            sys.exit(3)
        so0 = swapouts_gb()
    for key in models:
        m = MODELS[key]
        want = a.engines.split(",") if a.engines else m["engines"]
        engines = [e for e in m["engines"] + EXTRA_ENGINES if e in want]
        engines += [e for e in want if e.startswith("llama-server-mc") and e not in engines]
        mlx_dir, gguf = g(m["mlx"]), (g(m["gguf"]) if m.get("gguf") else None)  # no GGUF: ours only
        for rnd in range(a.round0, a.round0 + a.rounds):
            order = engines if rnd % 2 else engines[::-1]
            for eng in order:
                tag = f"{key}_r{rnd}_{eng}"
                ck = cfg_key(eng, mlx_dir)
                if ck in tripped():  # a swap stop before, here or in an earlier run: only the user re-enables it
                    print(f"== {tag} skipped: {ck} tripped a swap stop ({tripped()[ck]})", flush=True)
                    any_trip = True
                    continue
                s0 = state(os.path.join(a.out, "state", f"{tag}_before.json"))
                print(f"== {tag} {time.strftime('%H:%M:%S')} swap {s0['swap_used_mb']:.0f} MB", flush=True)
                pw = [gguf if x == "GGUF" else x for x in m.get("prewarm", {}).get(eng, [])]
                files = [f for x in pw for f in glob.glob(os.path.expanduser(x))]
                if files:  # the engine's own weight files read once (page cache), as its warm-up would
                    t0 = time.perf_counter()
                    for f in files:
                        with open(f, "rb") as fh:
                            while fh.read(64 << 20):
                                pass
                    print(f"  prewarm {len(files)} files {time.perf_counter() - t0:.1f} s", flush=True)
                if eng == "llama.cpp":
                    blk = os.path.join(a.out, "blocks", f"{tag}_llama.json")
                    if llama_block(m, gguf, m.get("llama_reps") or a.llama_reps or a.reps, blk, tests):
                        add_tripped(ck, f"suite {a.out} {tag}")
                        any_trip = True
                else:
                    blk = os.path.join(a.out, "blocks", f"{tag}.jsonl")
                    base = "mj" if eng.startswith("mj") else eng  # mj-<variant>: our engine, other options
                    args = m.get("variant_args", {}).get(eng, m["mj_args"] if base == "mj" else [])
                    if eng.startswith("ollama"):  # ollama-gguf: its llama.cpp runner on the GGUF; ollama-mlx: MLX
                        args = ["--ollama-model", ollama_prepare(key, eng, gguf if eng == "ollama-gguf" else mlx_dir)]
                        base = "ollama"
                    server = None
                    if eng.startswith("lmstudio"):  # lmstudio-mlx: its MLX engine; lmstudio-gguf: its llama.cpp
                        args = ["--api-url", LMS_URL, "--api-model",
                                lmstudio_prepare(key, eng, gguf if eng == "lmstudio-gguf" else mlx_dir)]
                        base = "openai"
                    if eng == "vllm-mlx":
                        server = vllm_mlx_start(mlx_dir)
                        args, base = ["--api-url", VLLM_URL, "--api-model", "bench"], "openai"
                    if eng.startswith("llama-server"):
                        server = llama_server_start(m, gguf, llama_server_extra(eng))
                        args, base = ["--api-url", LLAMA_URL, "--api-model", "bench"], "openai"
                    cmd = [PY, os.path.join(HERE, "mlx_bench_worker.py"), "--engine", base, "--label", eng,
                           "--model", mlx_dir, "--tests", ",".join(tests), "--reps", str(a.reps),
                           "--seed0", str(100 * rnd), "--tag", tag, "--out", blk] + args
                    # mj variants may override the model's mj_env (variant_env), e.g. the exact gpt-oss arm without mlock
                    venv = m.get("variant_env", {}).get(eng, m.get("mj_env", {}))
                    env = dict(os.environ, **(venv if base == "mj" else {}))
                    sw0 = vm_swap_pages()
                    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    with PeakSampler(lambda: engine_pids(eng, p.pid, server)) as mem, \
                            SwapGuard(lambda: [p.pid] + engine_pids(eng, p.pid, server)) as guard:
                        out, err = p.communicate()
                    p = subprocess.CompletedProcess(cmd, p.returncode, out, err)
                    sw1 = vm_swap_pages()
                    json.dump(dict(engine=eng, peak_footprint_gb=mem.peak / 1e9, peak_resident_gb=mem.peak_resident / 1e9,
                                   samples=mem.samples,
                                   per_pid_gb={str(k): v / 1e9 for k, v in mem.per_pid.items()},
                                   swapins_gb=(sw1[0] - sw0[0]) * 16384 / 1e9, swapouts_gb=(sw1[1] - sw0[1]) * 16384 / 1e9),
                              open(os.path.join(a.out, "state", f"{tag}_mem.json"), "w"), indent=1)
                    if eng.startswith("ollama"):  # unload, so the next engine gets the memory back
                        subprocess.run(["ollama", "stop", args[1]], capture_output=True)
                    if eng.startswith("lmstudio"):
                        subprocess.run([LMS, "unload", args[3]], capture_output=True)
                    if server is not None:
                        server.terminate()
                        server.wait(60)
                    for line in p.stdout.splitlines():
                        print("  " + line, flush=True)
                    if guard.tripped:
                        open(blk + ".flagged", "a").write(f"aborted: swap-outs {guard.tripped:.1f} GB passed "
                                                          f"{guard.limit:.0f} GB\n")
                        add_tripped(ck, f"suite {a.out} {tag}")  # its later blocks, here and in later runs, skip
                        any_trip = True
                    elif p.returncode:
                        print(f"  FAILED ({p.returncode}): {p.stderr[-1500:]}", flush=True)
                        err = (p.stderr.strip().splitlines() or ["?"])[-1]
                        open(blk + ".flagged", "a").write(f"block failed (exit {p.returncode}): {err}\n")
                s1 = state(os.path.join(a.out, "state", f"{tag}_after.json"))
                grow = s1["swap_used_mb"] - s0["swap_used_mb"]
                if grow > 256:
                    open(blk + ".flagged", "a").write(f"swap grew {grow:.0f} MB during the block\n")
                    print(f"  FLAGGED: swap grew {grow:.0f} MB", flush=True)
                time.sleep(a.cooldown)
        summarize(a.out, models)
    caff.terminate()
    if direct:
        from swap_guard import daily_add

        daily_add(f"suite {a.out}", swapouts_gb() - so0, "machine counter over the suite run")
    if any_trip:  # a swap stop (or a config skipped for one): exit 3, as tools/swap_guard.py's tools do
        sys.exit(3)


if __name__ == "__main__":
    main()

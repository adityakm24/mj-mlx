"""Campaign v17c (2026-11-23, user's standing OK until ~16:00 IST), after v17b. Same rules as run_v17.py.

- qa_llama: gpt-oss long-context retrieval on llama-server, which failed at defaults in v17 (20k context) and v17b
  (17k context, 87% of memory free): Metal "Insufficient Memory" in its warm-up. As in RESULTS 1zi, llama.cpp fits
  gpt-oss at 16k on this Mac only with batch = ubatch 128 (the capacity rule: same precision, smaller ubatch); 256
  is tried first.

    .venv/bin/python results/v17c_scripts/run_v17c.py
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v17_scripts"))
sys.path.insert(0, os.path.join(HERE, "..", "v17b_scripts"))
import run_v17 as V  # noqa: E402
import run_v17b as B  # noqa: E402

LOG = os.path.join(V.ROOT, "results", "v17c_scripts", "log")


class ServerUb(B.Server8k):
    def __init__(self, gguf, log, ctx, ub):
        import subprocess
        import urllib.request

        B.wait_free(log)
        self.p = subprocess.Popen(["llama-server", "-m", gguf, "-fa", "on", "-ngl", "99", "-c", str(ctx), "-np", "1",
                                   "-b", str(ub), "-ub", str(ub), "--no-cache-prompt", "--port", V.PORT, "--alias",
                                   "bench"], stdout=open(log, "a"), stderr=__import__("subprocess").STDOUT)
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{V.PORT}/health", timeout=3)
                break
            except Exception:
                time.sleep(1)


def warm_ok():
    r = B.post("/v1/completions", dict(model="bench", prompt="Hello", max_tokens=4, temperature=0))
    return r.get("status") == 200


def step_qa_llama(log):
    for ub in (256, 128):
        s = ServerUb(V.GGUF["gptoss"], log + ".server", 17152, ub)
        try:
            ok = warm_ok()
            with open(log, "a") as f:
                f.write(f"\n[ubatch {ub}] short request {'ok' if ok else 'failed'}\n")
            if not ok:
                continue
            return V.qa(log, "gptoss", "llama-server", f"results/longctx_qa_v1/gptoss_llama_ub{ub}.json",
                        "4096,8192,16384")
        finally:
            s.stop()
            time.sleep(20)
    return 1


STEPS = [("qa_llama", step_qa_llama)]


def main():
    os.makedirs(LOG, exist_ok=True)
    status = {}
    for name, fn in STEPS:
        if time.strftime("%H:%M") >= V.STOP_AT and time.strftime("%H:%M") < "23:00":
            status[name] = "not started (past STOP_AT)"
            continue
        t = time.time()
        print(f"{time.strftime('%H:%M:%S')} {name} start", flush=True)
        try:
            rc = fn(os.path.join(LOG, f"{name}.log"))
            status[name] = f"rc {rc}, {time.time() - t:.0f} s"
        except Exception as e:
            status[name] = f"error {type(e).__name__}: {e}"[:300]
        print(f"{time.strftime('%H:%M:%S')} {name}: {status[name]}", flush=True)
    json.dump(status, open(os.path.join(LOG, "status.json"), "w"), indent=1)
    print("V17C_DONE", json.dumps(status), flush=True)


if __name__ == "__main__":
    main()

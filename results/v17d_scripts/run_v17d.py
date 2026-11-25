"""Campaign v17d (2026-11-23, user's standing OK until ~16:00 IST), after v17c. Same rules as run_v17.py.

PM review of RESULTS 1zz / 1zzb (peer, adopted):
- items_mj / items_llama: gpt-oss chat GSM8K, ours (exact arm, experts unpinned) and llama-server, on the 7 items
  behind 1zz's difference (243, 694, 931, 1167 llama-server only; 337, 452, 1182 "#### 1" endings) and 5 both got
  right (1, 8, 25, 47, 84), whole replies kept (v17b's rows kept only the last 600 characters).
- refcheck: our replies and llama-server's, teacher-forced through tools/mlx_real_eval.py: our float32 reference,
  our bf16 paths, and mlx-lm's own gpt-oss layers in bf16 and float32 (mlx-lm cannot load the checkpoint whole on
  16 GB, so its layers are streamed). Per sequence: where each config's greedy pick first leaves the reply, and its
  top 5 at the token after "####". If the references pick the same, it is the model; if not, it is us.
- margin_llama: llama-server's top 5 at the same prefixes (token ids sent as is).
- dsv2srv2: DeepSeek-V2-Lite on llama-server, per the rival-capacity rule: -ub 256, then 128, all layers on the GPU,
  same precision; only if both fail, -ngl left at llama.cpp's default (a partial offload, labelled so). The first
  that serves a short request runs the worker check, long-context QA at 4k / 8k and 1zr's MMLU / GSM8K.

    .venv/bin/python results/v17d_scripts/run_v17d.py
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v17_scripts"))
sys.path.insert(0, os.path.join(HERE, "..", "v17b_scripts"))
import run_v17 as V  # noqa: E402
import run_v17b as B  # noqa: E402

ROOT, PY = V.ROOT, V.PY
LOG = os.path.join(ROOT, "results", "v17d_scripts", "log")
D = "results/lm_eval_chat_v1"
IDX = "243,337,452,694,931,1167,1182,1,8,25,47,84"
NO_MLOCK = {"MJ_MLOCK": "0"}


class Srv:
    """llama-server with given extra arguments (no -ngl unless given), started once memory is free."""

    def __init__(self, gguf, log, ctx, extra):
        B.wait_free(log)
        self.p = subprocess.Popen(["llama-server", "-m", gguf, "-fa", "on", "-c", str(ctx), "-np", "1",
                                   "--no-cache-prompt", "--port", V.PORT, "--alias", "bench", *extra],
                                  stdout=open(log, "a"), stderr=subprocess.STDOUT)
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{V.PORT}/health", timeout=3)
                return
            except Exception:
                time.sleep(1)

    def stop(self):
        self.p.terminate()
        try:
            self.p.wait(30)
        except subprocess.TimeoutExpired:
            self.p.kill()
        time.sleep(15)


def warm_ok():
    r = B.post("/v1/completions", dict(model="bench", prompt="The capital of France is", max_tokens=8, temperature=0))
    return r.get("status") == 200


def step_items_mj(log):
    return V.sh([PY, "tools/mlx_lm_eval.py", "--engine", "mj", "--model", V.MLX["gptoss"], "--mj-kw", V.GPTOSS_X,
                 "--placement", "none", "--tasks", "gsm8k", "--chat", "--every", "4", "--effort", "low", "--idx", IDX,
                 "--full-text", "--out", f"{D}/gptoss_mj_x_items_full.json"], log, env=NO_MLOCK)


def step_items_llama(log):
    s = Srv(V.GGUF["gptoss"], log + ".server", 8192, ["-ngl", "99"])
    try:
        return V.sh([PY, "tools/mlx_lm_eval.py", "--engine", "llama-server", "--model", V.MLX["gptoss"], "--tasks",
                     "gsm8k", "--chat", "--every", "4", "--effort", "low", "--idx", IDX, "--full-text", "--out",
                     f"{D}/gptoss_llama_items_full.json"], log)
    finally:
        s.stop()


def build_seqs(out):
    """Prompt + whole reply as token ids, the reply's start, and the position of the space after the last '####'
    (its logits pick the answer number)."""
    import glob

    from transformers import AutoTokenizer

    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from mlx_lm_eval import CHAT_GSM, chat_enc

    tok = AutoTokenizer.from_pretrained(glob.glob(V.MLX["gptoss"])[0])
    qs = {}
    for line in open(os.path.join(ROOT, "results", "accuracy", "gsm8k_subset.jsonl")):
        it = json.loads(line)
        qs[it["idx"]] = it["q"]["question"]
    seqs = []
    for eng, f in (("mj", f"{D}/gptoss_mj_x_items_full.json"), ("llama", f"{D}/gptoss_llama_items_full.json")):
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            continue
        for r in json.load(open(p))["tasks"]["gsm8k"]["rows"]:
            prompt = chat_enc(tok, CHAT_GSM.format(q=qs[r["idx"]]), "low")
            reply = tok.encode(r["text"], add_special_tokens=False)
            k = r["text"].rfind("####")
            probe = []
            if k >= 0:
                pre = tok.encode(r["text"][:k + 4] + " ", add_special_tokens=False)
                if reply[:len(pre)] == pre:
                    probe = [len(prompt) + len(pre) - 1]
            seqs.append(dict(name=f"{eng}:{r['idx']}:{'ok' if r['ok'] else 'wrong'}:{r['pred']}", ids=prompt + reply,
                             reply_start=len(prompt), probe=probe))
    json.dump(seqs, open(os.path.join(ROOT, out), "w"))
    return len(seqs)


def step_refcheck(log):
    seqs = "results/lm_eval_chat_v1/gptoss_items_seqs.json"
    n = build_seqs(seqs)
    with open(log, "a") as f:
        f.write(f"{n} sequences\n")
    return V.sh([PY, "tools/mlx_real_eval.py", "--seqs-json", seqs, "--configs",
                 "mj_precise,mj_precise_pack,mlxlm_bf16,mlxlm_fp32", "--out",
                 "results/lm_eval_chat_v1/gptoss_items_refcheck.json"], log)


def step_margin_llama(log):
    s = Srv(V.GGUF["gptoss"], log + ".server", 8192, ["-ngl", "99"])
    try:
        rc = 0
        for eng in ("mj_x", "llama"):
            rc |= V.sh([PY, "tools/mlx_answer_margin.py", "--engine", "llama-server", "--model", V.MLX["gptoss"],
                        "--run", f"{D}/gptoss_{eng}_items_full.json", "--out",
                        f"{D}/gptoss_items_margin_llama_on_{eng}.json"], log)
        return rc
    finally:
        s.stop()


def step_dsv2srv2(log):
    d = "results/dsv2_llama_server_debug_v1"
    for label, extra in (("ub256", ["-ngl", "99", "-b", "256", "-ub", "256"]),
                         ("ub128", ["-ngl", "99", "-b", "128", "-ub", "128"]),
                         ("ngl-default-partial-offload", [])):
        s = Srv(V.GGUF["dsv2lite"], f"{ROOT}/{d}/server_{label}.log", 8192, extra)
        try:
            ok = warm_ok()
            with open(log, "a") as f:
                f.write(f"\n[{label}] short request {'ok' if ok else 'failed'}\n")
            if not ok:
                continue
            rc = V.sh([PY, "tools/mlx_bench_worker.py", "--engine", "openai", "--label", "llama-server", "--model",
                       V.MLX["dsv2lite"], "--tests", "pp512,tg128@d0", "--reps", "2", "--tag", label, "--api-url",
                       f"http://127.0.0.1:{V.PORT}", "--api-model", "bench", "--out", f"{d}/worker_{label}.jsonl"], log)
            rc |= V.qa(log, "dsv2lite", "llama-server", f"results/longctx_qa_v1/dsv2lite_llama_{label}.json",
                       "4096,8192")
            rc |= V.sh([PY, "tools/mlx_lm_eval.py", "--engine", "llama-server", "--model", V.MLX["dsv2lite"],
                        "--tasks", "mmlu,gsm8k", "--out", f"results/lm_eval_v1/dsv2lite_llama_{label}.json"], log)
            return rc
        finally:
            s.stop()
    return 1


STEPS = [("items_mj", step_items_mj), ("items_llama", step_items_llama), ("refcheck", step_refcheck),
         ("margin_llama", step_margin_llama), ("dsv2srv2", step_dsv2srv2)]


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
    print("V17D_DONE", json.dumps(status), flush=True)


if __name__ == "__main__":
    main()

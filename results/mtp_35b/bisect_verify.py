import os, sys, glob, json
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
sys.path.insert(0, os.path.expanduser("~/Documents/Projects/MilliJoule/python"))
import mlx.core as mx, numpy as np
from mj_mlx import load_model
from mj_mlx.spec_decode import SpecGenerator
M = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/*"))[0]
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
model = load_model(M, stream_experts=os.path.expanduser("~/.cache/millijoule/qwen36-35b-experts-stream"), expert_slots=96,
                   lean=True, io_threads=8, io_nocache=True, prefetch=True)
ids = [int(t) for t in np.random.RandomState(5).randint(1000, 100000, 40)]
N = 12
rec = {}
def cb(li, xr, idx):
    if xr.shape[0] > 4: return  # prefill
    rec.setdefault(li, []).append((np.array(xr.astype(mx.float32)), np.array(idx)))
model.route_trace = cb
g0 = SpecGenerator(model, os.path.expanduser("~/.cache/millijoule/Qwen3.6-35B-mtp/mtp.safetensors"), k=0, max_len=128)
ref = g0.generate(ids, N)
rec0 = rec; rec = {}
g1 = SpecGenerator(model, os.path.expanduser("~/.cache/millijoule/Qwen3.6-35B-mtp/mtp.safetensors"), k=1, max_len=128)
full = ids + ref + [0] * 4
def oracle(hid32, next_ids, start, kk):
    p = start + hid32.shape[0] + 1
    return mx.array(np.array(full[p:p + kk], np.int32))
g1._drafts = oracle
out = g1.generate(ids, N)
rec1 = rec
print("ref ", ref); print("spec", out, "equal", out == ref)
# rows in order: k0 one row per decode step; k1 two rows per verify step (all accepted)
for li in sorted(rec0):
    r0 = np.concatenate([x for x, _ in rec0[li]]); r1 = np.concatenate([x for x, _ in rec1[li]])
    n = min(len(r0), len(r1))
    d = np.abs(r0[:n] - r1[:n]).max(axis=-1)
    i0 = np.concatenate([i for _, i in rec0[li]]); i1 = np.concatenate([i for _, i in rec1[li]])
    first = int(np.argmax(d > 0)) if (d > 0).any() else None
    print(f"layer {li:2d}: rows {len(r0)}/{len(r1)}, MoE input first differing row {first}, max diff {d.max():.3g}, "
          f"routing sets equal {[set(a) == set(b) for a, b in zip(i0[:n], i1[:n])].count(False)} differ")
    if first is not None:
        break

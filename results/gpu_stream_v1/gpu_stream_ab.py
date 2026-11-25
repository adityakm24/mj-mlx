import os, sys, time, glob, json
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
sys.path.insert(0, os.path.expanduser("~/Documents/Projects/MilliJoule/python"))
import mlx.core as mx, numpy as np
from mj_mlx import Engine, load_model
slots, overlap = int(sys.argv[1]), sys.argv[2] == "1"
SPIN = int(os.environ.get("MJ_SPIN_US", "1000"))
M = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/*"))[0]
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
model = load_model(M, stream_experts=os.path.expanduser("~/.cache/millijoule/qwen36-35b-experts-stream"),
                   expert_slots=slots, lean=True, io_threads=8, io_nocache=True, prefetch=True)
ids = [int(t) for t in np.random.RandomState(0).randint(0, 200000, 512)]
eng = Engine(model=model, num_blocks=256, max_running=1, max_batch_tokens=2048, overlap=overlap)
eng.add_request(ids, 64, ignore_eos=True)
times, out, first = [], [], None
t_prev = None
while eng.has_work():
    before = eng.tokens_out
    for f in eng.step():
        out = list(f[2])
    if eng.tokens_out > before:
        t = time.perf_counter()
        if first is None:
            first = t; gs = model.__dict__.get("_gpu_stream")
            if gs is not None and gs.cpp is not None: gs.cpp.reset_stats(); gs.cpp.spin_us = SPIN; gs.cpp.keepalive = os.environ.get("MJ_KEEPALIVE") == "1"; gs.cpp.keepalive_gbps = float(os.environ.get("MJ_KA_GBPS", "2.5"))
        else:
            times.append((t - t_prev) / (eng.tokens_out - before))
        t_prev = t
n = len(times)
st = model.expert_stats().get("gpu_stream") or {}
print(f"slots {slots} overlap {overlap} GPU_STREAM {os.environ.get('MJ_GPU_STREAM')}: decode {1e3*np.mean(times):.1f} ms/token "
      f"({1/np.mean(times):.2f} tok/s), median {1e3*np.median(times):.1f}; IO thread per token: route wait "
      f"{1e3*st.get('route_wait_s',0)/max(n,1):.1f} ms, read wait {1e3*st.get('read_wait_s',0)/max(n,1):.1f} ms, "
      f"misses/token {st.get('misses',0)/max(n,1):.1f}, keepalive {os.environ.get('MJ_KEEPALIVE')} launched {st.get('keepalive_launched',0)/max(n,1):.1f}/token")
json.dump(dict(tokens=out), open(sys.argv[3], "w"))

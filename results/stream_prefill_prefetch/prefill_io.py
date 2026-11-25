import os, sys, time, glob
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
sys.path.insert(0, os.path.expanduser("~/Documents/Projects/MilliJoule/python"))
import mlx.core as mx, numpy as np
from mj_mlx import Engine, load_model
slots = int(sys.argv[1])
M = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/*"))[0]
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
model = load_model(M, stream_experts=os.path.expanduser("~/.cache/millijoule/qwen36-35b-experts-stream"),
                   expert_slots=slots, lean=True, io_threads=8, io_nocache=True, prefetch=True)
for rep in range(2):
    ids = [int(t) for t in np.random.RandomState(rep).randint(0, 200000, 512)]
    eng = Engine(model=model, num_blocks=256, max_running=1, max_batch_tokens=2048)
    eng.add_request(ids, 1, ignore_eos=True)
    b0 = model.store.bytes_read; t = time.perf_counter()
    while eng.has_work():
        eng.step()
    dt = time.perf_counter() - t; gb = (model.store.bytes_read - b0) / 1e9
    st = model.expert_stats()
    print(f"slots {slots} rep {rep}: prefill+1 token {dt:.2f} s, SSD {gb:.2f} GB, {gb/dt:.2f} GB/s effective")

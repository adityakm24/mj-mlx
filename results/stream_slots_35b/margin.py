import os, sys, json, glob
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
sys.path.insert(0, os.path.expanduser("~/Documents/Projects/MilliJoule/python"))
import mlx.core as mx, numpy as np
from mj_mlx import Engine, load_model
slots = int(sys.argv[1])
M = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/*"))[0]
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
model = load_model(M, stream_experts=os.path.expanduser("~/.cache/millijoule/qwen36-35b-experts-stream"),
                   expert_slots=slots, lean=True, io_threads=8, io_nocache=True, prefetch=True)
rec = []
orig = model.greedy_from
def g(x32):
    lg = model.full_logits(x32).astype(mx.float32)
    top = mx.argsort(-lg, axis=-1)[:, :3]
    v = mx.take_along_axis(lg, top, axis=-1)
    rec.append(dict(top=np.array(top).tolist()[0], logits=np.array(v).tolist()[0]))
    return orig(x32)
model.greedy_from = g
ids = [int(t) for t in np.random.RandomState(0).randint(0, 200000, 512)]
eng = Engine(model=model, num_blocks=256, max_running=1, max_batch_tokens=2048)
rid = eng.add_request(ids, 14, ignore_eos=True)
out = []
while eng.has_work():
    for f in eng.step():
        out = list(f[2])
json.dump(dict(slots=slots, tokens=out, steps=rec), open(sys.argv[2], "w"))
for i, r in enumerate(rec):
    print(i, r["top"][:2], "gap %.4f" % (r["logits"][0] - r["logits"][1]))

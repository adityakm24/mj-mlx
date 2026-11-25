import os, sys, time, glob, threading
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
sys.path.insert(0, os.path.expanduser("~/Documents/Projects/MilliJoule/python"))
import mlx.core as mx, numpy as np
import concurrent.futures as cf
from mj_mlx import Engine, load_model
from mj_mlx import expert_stream as es
slots = int(sys.argv[1])
M = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/*"))[0]
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
model = load_model(M, stream_experts=os.path.expanduser("~/.cache/millijoule/qwen36-35b-experts-stream"),
                   expert_slots=slots, lean=True, io_threads=8, io_nocache=True, prefetch=True)
T = dict(sync=0.0, reads=0.0, run_moe=0.0, n_moe=0)
real_eval = mx.eval
main = threading.main_thread()
def ev(*a):
    t = time.perf_counter(); real_eval(*a); dt = time.perf_counter() - t
    if sys._getframe(1).f_code.co_name == "run_moe":
        T["sync"] += dt
mx.eval = ev
real_np_array = np.array
real_result = cf.Future.result
def res(self, *a, **k):
    t = time.perf_counter(); r = real_result(self, *a, **k)
    if threading.current_thread() is main:
        T["reads"] += time.perf_counter() - t
    return r
cf.Future.result = res
real_run = es.run_moe
def rm(*a, **k):
    t = time.perf_counter(); r = real_run(*a, **k); T["run_moe"] += time.perf_counter() - t; T["n_moe"] += 1
    return r
es.run_moe = rm
import mj_mlx.qwen3_5_moe as q
# run_moe is imported inside _moe from .expert_stream at call time -> patched attribute is used
ids = [int(t) for t in np.random.RandomState(0).randint(0, 200000, 512)]
eng = Engine(model=model, num_blocks=256, max_running=1, max_batch_tokens=2048)
eng.add_request(ids, 48, ignore_eos=True)
eng.step()
for _ in range(4): eng.step()
for k in T: T[k] = 0 if k == "n_moe" else 0.0
n = 0; t0 = time.perf_counter()
while eng.has_work():
    eng.step(); n += 1
wall = time.perf_counter() - t0
# ids routing readback np.array(idx) happens right after the eval (already synced)
per = lambda x: 1e3 * x / n
print(f"slots {slots}: {n} steps, {per(wall):.1f} ms/step | in run_moe {per(T['run_moe']):.1f} (routing sync wait {per(T['sync']):.1f}, "
      f"SSD read waits {per(T['reads']):.1f}, other run_moe python {per(T['run_moe']-T['sync']-T['reads']):.1f}) | "
      f"outside run_moe (graph building, attention, sampling) {per(wall-T['run_moe']):.1f} ms; layers/step {T['n_moe']/n:.0f}")

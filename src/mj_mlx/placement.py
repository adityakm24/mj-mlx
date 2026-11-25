# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""Where a MoE model's routed experts live, chosen from the model's size and this Mac's memory (docs/moe-design.md).

- "resident": everything in MLX buffers (model fits in the GPU working set with room for KV and activations).
- "mapped": routed experts are zero-copy views of part-major files in the page cache (expert_map.py), optionally
  mlocked; for models that fit in RAM but not in the GPU working set (gpt-oss-20b on 16 GB: RESULTS 1p, 1y).
- "stream": routed experts on the SSD, a fixed number of slots per layer resident (expert_stream.py); for models
  bigger than RAM (Qwen3.6-35B-A3B on 16 GB: RESULTS 1d.2, 1s).
The repacked files are written once under ~/.cache/millijoule/<model>-experts[-mapped] (the checkpoint is not
modified). Layer files exist only for MoE layers ("moe_layers" in index.json).
"""

from __future__ import annotations

import glob
import json
import os
import subprocess

import mlx.core as mx
import numpy as np

ALIGN = 16384
CACHE = os.path.expanduser("~/.cache/mj-mlx")  # new expert stores are written here
# Older stores (written before 2026-11-23 under the MilliJoule name) are still found and used, not moved.
LEGACY_CACHES = [os.path.expanduser("~/.cache/millijoule")]


def checkpoint_bytes(path: str) -> tuple[int, int]:
    """(total bytes, routed-expert bytes) of a safetensors checkpoint, from the headers only."""
    import struct

    total = experts = 0
    for f in glob.glob(os.path.join(path, "*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            b = v["data_offsets"][1] - v["data_offsets"][0]
            total += b
            if ".experts." in k or ".switch_mlp." in k or ".block_sparse_moe.experts." in k:
                experts += b
    return total, experts


def experts_quantized(path: str) -> bool:
    """Whether the routed experts carry quantization scales (mlx affine / MXFP4), from the headers only."""
    import struct

    for f in glob.glob(os.path.join(path, "*.safetensors")):
        with open(f, "rb") as fh:
            hdr = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        if any(("scales" in k) and (".experts." in k or ".switch_mlp." in k) for k in hdr):
            return True
    return False


def checkpoint_dtype(path: str):
    """The activation dtype to run a checkpoint in: its quantization scales' dtype when it is quantized (an mlx
    float16-scaled checkpoint run in bf16 made gather_qmm read every expert each step: OLMoE-1B-7B decode 48 vs 135
    tok/s, RESULTS 1zd), else bf16 (also for a float32 checkpoint: a deliberate departure from mlx-lm, which keeps
    float32; docs/models.md)."""
    import struct

    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with open(f, "rb") as fh:
            hdr = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for k, v in hdr.items():
            if k.endswith(".scales") and isinstance(v, dict):
                return {"F16": mx.float16, "BF16": mx.bfloat16, "F32": mx.float32}.get(v["dtype"], mx.bfloat16)
    return mx.bfloat16


def memory() -> dict:
    """GPU working set and RAM of this Mac (bytes)."""
    info = mx.device_info()
    ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
    return dict(working_set=int(info.get("max_recommended_working_set_size", ram * 0.66)), ram=ram)


def choose(path: str, reserve: int = 2 << 30, mem: dict | None = None) -> str:
    """resident if the whole checkpoint plus `reserve` (KV, activations) fits the GPU working set; mapped if it
    fits in RAM with ~3 GB left for the system; else stream."""
    total, experts = checkpoint_bytes(path)
    mem = mem or memory()
    if total + reserve <= mem["working_set"]:
        return "resident"
    gb = lambda b: f"{b / 1e9:.1f} GB"
    if not experts:
        raise MemoryError(f"{path}: a dense model of {gb(total)} does not fit this Mac's GPU working set "
                          f"({gb(mem['working_set'])}, with {gb(reserve)} kept for the KV cache). mj-mlx streams only "
                          "MoE experts from the SSD; use a smaller or more quantized checkpoint, or mlx-lm.")
    if total - experts + reserve > mem["working_set"]:
        raise MemoryError(f"{path}: even with every routed expert on the SSD, the other weights ({gb(total - experts)}) "
                          f"plus {gb(reserve)} for the KV cache exceed this Mac's GPU working set "
                          f"({gb(mem['working_set'])}). Use a more quantized checkpoint or a Mac with more memory.")
    if total + (3 << 30) <= mem["ram"]:
        return "mapped"
    return "stream"


def _raw(a: mx.array) -> tuple[np.ndarray, str]:
    name = {mx.uint8: "uint8", mx.uint32: "uint32", mx.bfloat16: "bfloat16", mx.float16: "float16",
            mx.float32: "float32"}[a.dtype]
    return (np.array(a.view(mx.uint16)) if a.dtype == mx.bfloat16 else np.array(a)), name


def expert_parts(w: dict, i: int, num_experts: int):
    """Layer i's routed experts in the engine's layout (keys normalised by moe_lm.normalize_keys): mlx affine
    (gu_w/gu_s/gu_b gate rows then up rows, dn_w/dn_s/dn_b) or dense (gu [H, 2I], dn [I, H], transposed for
    gather_mm). Returns (parts, expert_format)."""
    from .moe_lm import stack_switch_mlp

    p = f"model.layers.{i}.mlp"
    sw = f"{p}.switch_mlp"
    stack_switch_mlp(w, p, num_experts)
    if f"{sw}.gate_proj.scales" in w:
        cat = lambda t: mx.concatenate([w[f"{sw}.gate_proj.{t}"], w[f"{sw}.up_proj.{t}"]], axis=1)
        H = w[f"{sw}.down_proj.weight"].shape[1]  # down_proj [E, H, I * bits / 32]
        bits = 32 * w[f"{sw}.gate_proj.weight"].shape[-1] // H  # gate_proj [E, I, H * bits / 32]
        gs = H // w[f"{sw}.gate_proj.scales"].shape[-1]
        return ([("gu_w", cat("weight")), ("gu_s", cat("scales")), ("gu_b", cat("biases")),
                 ("dn_w", w[f"{sw}.down_proj.weight"]), ("dn_s", w[f"{sw}.down_proj.scales"]),
                 ("dn_b", w[f"{sw}.down_proj.biases"])], {"format": "affine", "bits": bits, "group_size": gs})
    if f"{p}.experts.gate_up_proj" in w:
        gu, dn = w[f"{p}.experts.gate_up_proj"], w[f"{p}.experts.down_proj"]  # [E, 2I, H], [E, H, I]
    elif f"{sw}.gate_proj.weight" in w:
        gu = mx.concatenate([w[f"{sw}.gate_proj.weight"], w[f"{sw}.up_proj.weight"]], axis=1)
        dn = w[f"{sw}.down_proj.weight"]
    else:
        gu = mx.stack([mx.concatenate([w[f"{p}.experts.{e}.gate_proj.weight"], w[f"{p}.experts.{e}.up_proj.weight"]],
                                      axis=0) for e in range(num_experts)])
        dn = mx.stack([w[f"{p}.experts.{e}.down_proj.weight"] for e in range(num_experts)])
    return [("gu", mx.contiguous(gu.swapaxes(-1, -2))), ("dn", mx.contiguous(dn.swapaxes(-1, -2)))], {"format": "dense"}


def repack(path: str, out: str, log=print) -> str:
    """Write the routed experts of a checkpoint as one fixed-stride file per MoE layer (ExpertStore layout). gpt-oss
    and Qwen3.5 / 3.6 go through expert_repack; the general MoE families are done here. Returns `out`."""
    from . import expert_repack

    if json.load(open(os.path.join(path, "config.json"))).get("model_type") in expert_repack.FAMILIES:
        return expert_repack.repack(path, out, log)
    from .moe_lm import MoeLMConfig, normalize_keys

    cfgd = json.load(open(os.path.join(path, "config.json")))
    cfg = MoeLMConfig.from_hf(cfgd)
    w = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        w.update(mx.load(f))  # lazy: only each layer's expert tensors are read
    w = normalize_keys(w)
    moe_layers = [i for i in range(cfg.num_hidden_layers) if i not in cfg.dense_layers]
    E = cfg.num_experts
    os.makedirs(out, exist_ok=True)
    index = None
    for i in moe_layers:
        parts, fmt = expert_parts(w, i, E)
        parts = [(n, *_raw(a)) for n, a in parts]
        rec = sum(a[0].nbytes for _, a, _ in parts)
        stride = (rec + ALIGN - 1) // ALIGN * ALIGN
        desc = [dict(name=n, dtype=dt, shape=list(a.shape[1:]), bytes=int(a[0].nbytes)) for n, a, dt in parts]
        if index is None:
            index = dict(model_type=cfg.model_type, expert_format=fmt, stride=stride, record_bytes=rec,
                         num_layers=cfg.num_hidden_layers, moe_layers=moe_layers, num_experts=E, align=ALIGN,
                         parts=desc)
        assert stride == index["stride"] and desc == index["parts"], f"layer {i} differs"
        with open(os.path.join(out, f"layer_{i:02d}.bin"), "wb") as fh:
            for e in range(E):
                fh.seek(e * stride)
                for _, a, _ in parts:
                    fh.write(np.ascontiguousarray(a[e]).tobytes())
            fh.truncate(E * stride)
        for k in [k for k in w if f".layers.{i}." in k]:
            del w[k]
        log(f"layer {i}: {E} experts x {rec / 1e6:.2f} MB")
    index["source"] = os.path.realpath(path)
    json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=1)
    return out


PROFILE = "slot_profile.json"  # per-layer LRU miss curves, written next to an expert store by tools/mlx_slot_profile.py


def lru_stack_distances(seq: np.ndarray) -> np.ndarray:
    """LRU stack distance (1 = most recent) of each access of a [T, k] routing trace, the k experts of a token in
    order; 10**9 for a first access. An access misses in an LRU cache of S slots iff its distance > S (Mattson et
    al., 1970)."""
    order, out = [], []
    for row in seq:
        for e in dict.fromkeys(int(x) for x in row):
            try:
                d = order.index(e)
                out.append(d + 1)
                del order[d]
            except ValueError:
                out.append(10 ** 9)
            order.insert(0, e)
    return np.asarray(out, np.int64)


def miss_curve(dist: np.ndarray, num_experts: int) -> np.ndarray:
    """misses[S] for S = 0 .. num_experts."""
    E = num_experts
    h = np.bincount(np.minimum(dist, E + 1), minlength=E + 2)
    ge = np.cumsum(h[::-1])[::-1]
    return np.array([ge[S + 1] if S + 1 < len(ge) else 0 for S in range(E + 1)], np.int64)


def allocate_slots(curves: dict, budget: int, floor: int = 16, step: int = 4) -> dict:
    """{layer: slots} summing to at most `budget`: every layer starts at `floor`, then `step` slots at a time go to the
    layer whose misses fall most per slot (RESULTS 1zt)."""
    layers = sorted(curves)
    cap = {li: len(curves[li]) - 1 for li in layers}
    floor = max(1, min(floor, budget // max(1, len(layers))))  # never more than an equal share
    alloc = {li: min(floor, cap[li]) for li in layers}
    left = budget - sum(alloc.values())
    while left >= step:
        gains = {li: curves[li][a] - curves[li][a + step] for li, a in alloc.items() if a + step <= cap[li]}
        if not gains:
            break
        li = max(gains, key=gains.get)
        if gains[li] <= 0:
            break
        alloc[li] += step
        left -= step
    return alloc


# Calibration text for the slot profile (calibrate_slot_profile): written for this purpose, mixed on purpose (prose,
# a list, code, numbers, a dialogue) so the routing it records is not one kind of text. RESULTS 1zze: 512 tokens of
# routing gave 93% of a full 4,096-token trace's miss reduction on Qwen3.6-35B, offline.
CALIB_TEXT = """The harbour town woke slowly. Fishing boats came back before dawn, their crews sorting the night's catch
on the quay while gulls argued over what fell between the planks. By seven the bakery on the corner had sold out of
rye bread, and the ferry to the island was already half full of walkers, two school classes and a man carrying a
crate of tomato plants. Nobody remembered when the lighthouse had last been painted, but everyone agreed it needed
it.

Things to check before a long bicycle ride:
1. Tyre pressure, front and back, and a spare inner tube that fits.
2. Brakes: pads not worn to the line, cables not frayed.
3. Water, two bottles in summer, and something salty to eat.
4. Lights, even if you plan to be home before dark.
5. A note of the route left with someone at home.

def merge_intervals(intervals):
    intervals = sorted(intervals)
    out = []
    for start, end in intervals:
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out

print(merge_intervals([(1, 3), (2, 6), (8, 10), (9, 12)]))  # [[1, 6], [8, 12]]

A small shop buys 240 notebooks at 1.85 each and sells them at 2.60. After a week it has sold 175, and it marks the
rest down by 20 percent. If all of them sell, the profit is 175 x 0.75 + 65 x (2.08 - 1.85), which is 131.25 plus
14.95, or 146.20 in total. The shop owner rounds it to 146 in her notebook and wonders whether next month she should
order 300.

"Is the museum open on Mondays?" asked the visitor.
"Only in August," said the guard. "The rest of the year it opens Tuesday to Sunday, ten until six, and until nine on
Thursdays. The café stays open an hour later."
"And the roof garden?"
"Weather permitting. If it rains, they close the stairs."

In the cell, proteins fold into shapes set by their sequence of amino acids, and a misfolded protein can stop working
or clump with others. Chaperone proteins help many of them fold, and the cell breaks down those that fail. Heat,
some chemicals and a few mutations make misfolding more likely, which is one reason a fever above about 41 degrees
Celsius is dangerous.

The committee met for three hours and agreed on four points: the budget for the new library wing stays at 4.2
million; construction starts in March, not January; the reading room keeps its south windows; and the public will be
asked to vote on the name. A fifth point, about parking, was left for the next meeting, as it has been for two
years.
"""


def calibrate_slot_profile(model, tokenizer, store: str, tokens: int = 512, log=print) -> str:
    """Run CALIB_TEXT (its first `tokens` tokens, at least 512 recommended) through a streamed model, record every
    MoE layer's routed experts (the model's `route_trace` hook), and write the store's slot profile from those
    routings, as tools/mlx_slot_profile.py does from a long trace. The profile is used from the next load on
    (MJ_SLOT_PROFILE=1 / uneven=True)."""
    from .runner import Engine

    ids = tokenizer(CALIB_TEXT, add_special_tokens=False)["input_ids"][:tokens]
    seqs = {}

    def cb(li, xr, idx):
        seqs.setdefault(li, []).append(np.array(idx).astype(np.int64))

    model.route_trace = cb
    try:
        eng = Engine(model=model, num_blocks=len(ids) // 16 + 8, max_running=1, max_batch_tokens=512)
        eng.add_request(ids, 1)
        while eng.has_work():
            eng.step()
    finally:
        model.route_trace = None
    idx = json.load(open(os.path.join(store, "index.json")))
    E = idx["num_experts"]
    curves = {li: miss_curve(lru_stack_distances(np.concatenate(v)), E).tolist() for li, v in seqs.items()}
    out = dict(trace="calibration", trace_meta=dict(text="placement.CALIB_TEXT", tokens=len(ids)), tokens=len(ids),
               num_experts=E, source=idx.get("source"), curves={str(k): v for k, v in curves.items()})
    f = os.path.join(store, PROFILE)
    tmp = f + ".tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, f)
    log(f"wrote slot profile {f} from {len(ids)} calibration tokens ({len(curves)} layers)")
    return f


def uneven_slots(store: str, avg_slots: int) -> dict | int:
    """Per-layer slots with the same total as `avg_slots` in every MoE layer, from the store's slot profile; plain
    `avg_slots` when the store has none."""
    f = os.path.join(store, PROFILE)
    if not os.path.exists(f):
        return avg_slots
    prof = json.load(open(f))
    curves = {int(k): np.asarray(v, np.int64) for k, v in prof["curves"].items()}
    return allocate_slots(curves, avg_slots * len(curves))


def stream_slots(path: str, store: str, reserve: int = 2 << 30, mem: dict | None = None) -> int:
    """Resident expert slots per MoE layer for SSD streaming: what fits in the GPU working set after the
    non-expert weights and `reserve` (KV, activations, MLX temporaries). Qwen3-30B-A3B 4-bit on 16 GB: 76 (80
    ran, 96 ran out of Metal memory; RESULTS 1zd)."""
    idx = json.load(open(os.path.join(store, "index.json")))
    total, experts = checkpoint_bytes(path)
    mem = mem or memory()
    per = idx["stride"] * len(idx.get("moe_layers", range(idx["num_layers"])))  # one slot in every MoE layer
    n = (mem["working_set"] - (total - experts) - reserve) // per
    return int(max(2, min(idx["num_experts"], n)))  # at least 2 (models whose experts dwarf the memory: Mixtral)


def _existing(real: str, layout: str) -> str | None:
    """A directory under CACHE (or a legacy root) already holding this checkpoint's experts in `layout` ("store" or "part_major"),
    found by the source path its index.json records (stores written by hand or by older versions keep other names)."""
    roots = [CACHE] + [r for r in LEGACY_CACHES if r != CACHE]
    for f in [f for r in roots for f in sorted(glob.glob(os.path.join(r, "*", "index.json")))]:
        try:
            idx = json.load(open(f))
        except (OSError, ValueError):
            continue
        if idx.get("source") == real and idx.get("layout", "store") == layout:
            return os.path.dirname(f)
    return None


def cache_dirs(path: str) -> tuple[str, str]:
    """(stream store, part-major mapped) directories for a checkpoint: existing ones made from it if any, else
    <CACHE>/<name>-experts and <CACHE>/<name>-experts-mapped."""
    real = os.path.realpath(path)
    found = _existing(real, "store"), _existing(real, "part_major")
    name = os.path.basename(real.rstrip("/"))
    if "snapshots" in real.split(os.sep):  # HF cache: models--org--name/snapshots/<hash>
        name = real.split(os.sep)[real.split(os.sep).index("snapshots") - 1].replace("models--", "")
    base = os.path.join(CACHE, f"{name}-experts")
    return found[0] or base, found[1] or base + "-mapped"


def check_disk(where: str, need: int, margin: int = 2 << 30) -> None:
    """Raise before a repack that would not fit: the expert files take about the routed experts' size per layout
    (streamed, and mapped on top for placement="mapped"), plus a margin."""
    if need <= 0:
        return
    os.makedirs(where, exist_ok=True)
    free = __import__("shutil").disk_usage(where).free
    if free < need + margin:
        raise OSError(f"not enough disk for the expert files: {need / 1e9:.1f} GB needed (+{margin / 1e9:.0f} GB "
                      f"margin) in {where}, {free / 1e9:.1f} GB free. Free space (python -m mj_mlx cache list / rm "
                      "shows what mj-mlx keeps), or use placement='resident' if the model fits in memory.")


def prepare(path: str, placement: str, log=print, uneven: bool | None = None) -> dict:
    """Model kwargs for `placement`, writing the repacked expert files on first use. uneven (default: env
    MJ_SLOT_PROFILE=1): stream with per-layer slot counts from the store's slot profile (RESULTS 1zt), same memory."""
    if placement == "resident":
        return {}
    store, mapped = cache_dirs(path)
    if placement == "mapped" and os.path.exists(os.path.join(mapped, "index.json")):
        return dict(map_experts=mapped)  # the maps alone are enough; the stream store is not needed
    need = 0
    if not os.path.exists(os.path.join(store, "index.json")):
        need += checkpoint_bytes(path)[1]
    if placement == "mapped" and not os.path.exists(os.path.join(mapped, "index.json")):
        need += checkpoint_bytes(path)[1]
    check_disk(os.path.dirname(store), need)
    if not os.path.exists(os.path.join(store, "index.json")):
        log(f"repacking experts into {store} (once, about {checkpoint_bytes(path)[1] / 1e9:.1f} GB)")
        repack(path, store, log)
    if placement == "stream":
        slots = stream_slots(path, store)
        if uneven if uneven is not None else os.environ.get("MJ_SLOT_PROFILE") == "1":
            slots = uneven_slots(store, slots)
        return dict(stream_experts=store, expert_slots=slots)
    from .expert_map import repack_part_major

    log(f"writing part-major expert maps into {mapped} (once)")
    repack_part_major(store, mapped)
    return dict(map_experts=mapped)

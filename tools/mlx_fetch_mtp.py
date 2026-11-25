"""Fetch only the MTP (multi-token prediction) tensors of a Hugging Face safetensors checkpoint with HTTP range
requests, and write them to one local safetensors file. mlx-community conversions drop `mtp.*`; this gets them back
without downloading the whole checkpoint (Qwen3.5-9B: ~0.5 GB of 19 GB).

    python tools/mlx_fetch_mtp.py --repo Qwen/Qwen3.5-9B --out ~/.cache/millijoule/Qwen3.5-9B-mtp/mtp.safetensors
"""

import argparse
import json
import os
import struct
import urllib.request

from huggingface_hub import hf_hub_url, get_token


def get(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    tok = get_token()
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--prefix", default="mtp.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    idx_url = hf_hub_url(a.repo, "model.safetensors.index.json", revision=a.revision)
    wmap = json.loads(urllib.request.urlopen(idx_url).read())["weight_map"]
    shards = sorted({f for k, f in wmap.items() if k.startswith(a.prefix)})
    header, blobs, off = {}, [], 0
    for shard in shards:
        url = hf_hub_url(a.repo, shard, revision=a.revision)
        n = struct.unpack("<Q", get(url, 0, 7))[0]
        meta = json.loads(get(url, 8, 8 + n - 1))
        base = 8 + n
        for k, m in sorted(meta.items()):
            if not k.startswith(a.prefix):
                continue
            s, e = m["data_offsets"]
            data = get(url, base + s, base + e - 1)
            assert len(data) == e - s, (k, len(data), e - s)
            header[k] = {"dtype": m["dtype"], "shape": m["shape"], "data_offsets": [off, off + len(data)]}
            blobs.append(data)
            off += len(data)
            print(f"{k} {m['dtype']} {m['shape']} {len(data) / 1e6:.1f} MB", flush=True)
    header["__metadata__"] = {"source": f"{a.repo}@{a.revision}", "format": "pt"}
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(a.out))), exist_ok=True)
    with open(os.path.expanduser(a.out), "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)
    print(f"wrote {len(blobs)} tensors, {off / 1e6:.1f} MB -> {a.out}")


if __name__ == "__main__":
    main()

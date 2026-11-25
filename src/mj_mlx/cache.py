# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""What mj-mlx keeps on disk, and how to clear it (checklist 1.5).

Two places:
- ~/.cache/mj-mlx (and ~/.cache/millijoule for stores written before 2026-11-23, still used): repacked expert stores
  (streamed: expert-major records; mapped: part-major files), each with an index.json naming its source checkpoint,
  written once per model (placement.prepare, tools/mlx_repack_experts.py). ~/.cache/millijoule also holds files of
  the MilliJoule project (GGUFs, traces); those list as "other" and are never deleted without --force;
- ~/Library/Caches/mj-mlx/prep: the prepared-weights cache (gpt_oss.PrepCache), one folder per checkpoint and option
  set, with a key.json.

    python -m mj_mlx cache list            # every entry: kind, size, source checkpoint, whether that still exists
    python -m mj_mlx cache rm <name> ...   # delete entries by the name `list` prints (recognised entries only)
    python -m mj_mlx cache rm --orphans    # delete the recognised entries whose source checkpoint is gone

Entries mj-mlx did not write (anything without its index.json / key.json) are listed as "other" and never deleted
unless --force is given.
"""

from __future__ import annotations

import json
import os
import shutil

ROOTS = {"mj-mlx": "~/.cache/mj-mlx", "millijoule": "~/.cache/millijoule", "prep": "~/Library/Caches/mj-mlx/prep"}


def _size(p: str) -> int:
    if os.path.isfile(p):
        return os.path.getsize(p)
    n = 0
    for d, _, fs in os.walk(p):
        for f in fs:
            fp = os.path.join(d, f)
            if not os.path.islink(fp):
                n += os.path.getsize(fp)
    return n


def entries(roots: dict | None = None) -> list[dict]:
    """[{name, path, kind, bytes, source, source_exists}] for every entry under the cache roots."""
    out = []
    for tag, root in (roots or ROOTS).items():
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            continue
        for n in sorted(os.listdir(root)):
            p = os.path.join(root, n)
            kind, source = "other", None
            idx, key = os.path.join(p, "index.json"), os.path.join(p, "key.json")
            try:
                if os.path.isdir(p) and os.path.exists(idx):
                    meta = json.load(open(idx))
                    kind = "experts-mapped" if meta.get("layout") == "part_major" else "experts-stream"
                    source = meta.get("source") or meta.get("source_store")
                elif os.path.isdir(p) and os.path.exists(key):
                    kind, source = "prep", json.load(open(key)).get("checkpoint")
            except (OSError, ValueError):
                kind = "other"
            out.append(dict(name=f"{tag}/{n}", path=p, kind=kind, bytes=_size(p), source=source,
                            source_exists=None if source is None else os.path.exists(os.path.expanduser(source))))
    return out


def remove(names: list[str], orphans: bool = False, force: bool = False, roots: dict | None = None, log=print) -> int:
    """Delete the named entries (or every recognised orphan); returns bytes freed. Unrecognised entries need force."""
    freed = 0
    for e in entries(roots):
        want = e["name"] in names or (orphans and e["kind"] != "other" and e["source_exists"] is False)
        if not want:
            continue
        if e["kind"] == "other" and not force:
            log(f"skip {e['name']}: not an mj-mlx cache entry (no index.json / key.json); --force to delete it")
            continue
        shutil.rmtree(e["path"]) if os.path.isdir(e["path"]) else os.remove(e["path"])
        freed += e["bytes"]
        log(f"deleted {e['name']} ({e['bytes'] / 1e9:.2f} GB)")
    return freed


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m mj_mlx cache", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    rm = sub.add_parser("rm")
    rm.add_argument("names", nargs="*")
    rm.add_argument("--orphans", action="store_true")
    rm.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        es = entries()
        for e in es:
            src = "" if e["source"] is None else f"  from {e['source']}" + ("" if e["source_exists"] else "  (gone)")
            print(f"{e['bytes'] / 1e9:8.2f} GB  {e['kind']:15s} {e['name']}{src}")
        print(f"{sum(e['bytes'] for e in es) / 1e9:8.2f} GB  total")
        if any(e["name"].startswith("millijoule/") for e in es):
            print("millijoule/ = ~/.cache/millijoule: stores written before 2026-11-23 (still used) and other projects' "
                  "files (listed as other)")
        return 0
    if not a.names and not a.orphans:
        ap.error("rm needs names (from `list`) or --orphans")
    known = {e["name"] for e in entries()}
    for n in a.names:
        if n not in known:
            print(f"no entry {n!r} (see `python -m mj_mlx cache list`)")
    print(f"freed {remove(a.names, a.orphans, a.force) / 1e9:.2f} GB")
    return 0

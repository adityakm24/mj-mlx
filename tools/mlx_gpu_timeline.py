"""GPU timeline from an Instruments Metal System Trace: when the GPU was busy for one process, and where it waited.

Record (xctrace ships with Xcode; nothing is installed):
    xcrun xctrace record --template 'Metal System Trace' --output run.trace --launch -- .venv/bin/python <script> ...
Then:
    .venv/bin/python tools/mlx_gpu_timeline.py run.trace --process python --out timeline.json

Reads the `metal-gpu-intervals` table (one interval per GPU command buffer / encoder, with GPU start and duration),
keeps the intervals of the named process on compute channels, merges overlaps, and reports the busy share and the
gaps (GPU idle while the process had work outstanding: host work, syncs, SSD waits) over the chosen window.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET


def export(trace: str, schema: str) -> ET.Element:
    xp = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
    out = subprocess.run(["xcrun", "xctrace", "export", "--input", trace, "--xpath", xp], check=True,
                         capture_output=True).stdout
    return ET.fromstring(out)


def rows(root: ET.Element) -> list[dict]:
    """Rows as {column mnemonic: (value text, fmt)}, resolving xctrace's id / ref de-duplication."""
    node = root.find("node")
    cols = [c.findtext("mnemonic") for c in node.find("schema").findall("col")]
    seen: dict[str, ET.Element] = {}

    def index(e):
        if "id" in e.attrib:
            seen[e.attrib["id"]] = e
        for c in e:
            index(c)

    out = []
    for r in node.findall("row"):
        index(r)
        vals = {}
        for name, e in zip(cols, list(r)):
            e = seen.get(e.attrib["ref"], e) if "ref" in e.attrib else e
            vals[name] = (e.text, e.attrib.get("fmt", ""))
        out.append(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--process", default="python")
    ap.add_argument("--skip-s", type=float, default=0.0, help="ignore intervals before this many seconds after the "
                    "process's first GPU interval (load and prefill)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    iv = []
    for r in rows(export(a.trace, "metal-gpu-intervals")):
        proc = r.get("process", ("", ""))[1]
        chan = r.get("channel-name", ("", ""))[1]
        if a.process.lower() not in proc.lower() or "compute" not in chan.lower():
            continue
        if r.get("event-depth", ("0",))[0] not in ("0", None):
            continue  # nested encoder rows duplicate their command buffer's span
        t0, d = int(r["start"][0]), int(r["duration"][0])
        iv.append((t0, t0 + d, r.get("cmdbuffer-id", (None,))[0]))
    iv.sort()
    if not iv:
        raise SystemExit("no compute intervals for that process")
    start = iv[0][0] + int(a.skip_s * 1e9)
    iv = [x for x in iv if x[0] >= start]
    merged = []
    for s, e, _ in iv:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    span = merged[-1][1] - merged[0][0]
    busy = sum(e - s for s, e in merged)
    gaps = sorted((merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)), reverse=True)
    res = dict(trace=a.trace, process=a.process, intervals=len(iv), busy_spans=len(merged), span_ms=span / 1e6,
               busy_ms=busy / 1e6, busy_pct=100 * busy / span, gap_count=len(gaps),
               gap_ms_total=sum(gaps) / 1e6, gap_ms_largest=[g / 1e6 for g in gaps[:10]],
               gap_ms_median=(gaps[len(gaps) // 2] / 1e6) if gaps else 0,
               timeline=[[(s - merged[0][0]) / 1e6, (e - merged[0][0]) / 1e6] for s, e in merged])
    print(json.dumps({k: v for k, v in res.items() if k != "timeline"}, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()

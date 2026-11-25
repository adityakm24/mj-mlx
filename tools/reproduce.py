"""Reproduce every published number: one command (checklist 0.8).

Reads results/published.json (each claim the README and docs make, the tool and settings of the run behind it, and
that run's own output file), runs every claim again into a new folder, and writes report.md / report.json there:
each published number next to the new one, and whether each README sentence ("faster than", "tied with") still
holds.

    .venv/bin/python tools/reproduce.py --list
    .venv/bin/python tools/reproduce.py --dry-run                      # print the commands, run nothing
    .venv/bin/python tools/reproduce.py                                # everything (hours; quiet machine)
    .venv/bin/python tools/reproduce.py --only gptoss-decode,accuracy-olmoe
    .venv/bin/python tools/reproduce.py --compare results/repro/20261121_1500   # re-write a report from its runs
    .venv/bin/python tools/reproduce.py --quick --only olmoe-decode    # 1 rep, 1 round: a smoke test, not a result

Output: results/repro/<YYYYMMDD_HHMM>/<claim id>/ (never an existing folder). Speed claims run
tools/mlx_bench_suite.py (ABAB rounds, swap-flagged blocks excluded, results/BENCH_PROTOCOL.md); accuracy claims
run tools/mlx_accuracy_ppl.py with the settings recorded in the published JSON, and llama-perplexity with the
recorded arguments on the suite's GGUF. Benchmarks need a quiet machine: close other heavy apps and servers first.

Verdicts. Speed cell: "same" when the published mean is inside the new 95% interval or within 3%; otherwise the
change in percent. Perplexity: "same" within 0.2%; KL within 2%. Relations: "faster" holds when the new intervals do
not overlap in that direction, "tie" holds when they overlap.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "bin", "python")
MANIFEST = os.path.join(ROOT, "results", "published.json")
REFS = os.path.expanduser("~/.cache/millijoule/accuracy_refs")
SAME_SPEED, SAME_PPL, SAME_KL = 0.03, 0.002, 0.02


def load_manifest(path=MANIFEST):
    with open(path) as f:
        return json.load(f)["claims"]


def suite_gguf(model_key):
    sys.path.insert(0, HERE)
    from mlx_bench_suite import MODELS  # the suite's own paths: same GGUF as the speed runs

    hits = sorted(glob.glob(os.path.expanduser(MODELS[model_key]["gguf"])))
    return hits[0] if hits else MODELS[model_key]["gguf"]


# ---- commands -------------------------------------------------------------------------------------------------

def suite_cmd(c, out, quick):
    s = c["suite"]
    reps, rounds = (1, 1) if quick else (s["reps"], s["rounds"])
    return [PY, os.path.join(HERE, "mlx_bench_suite.py"), "--models", s["models"], "--engines", s["engines"],
            "--tests", s["tests"], "--reps", str(reps), "--rounds", str(rounds), "--out", out]


def ppl_cmd(run, out_dir, quick):
    """(argv, output file). Our tool: the settings recorded in the published JSON. llama.cpp: recorded args."""
    out = os.path.join(out_dir, run["name"] + (".log" if "llama" in run else ".json"))
    if "llama" in run:
        args = list(run["llama"]["args"])
        if quick:
            args[args.index("--chunks") + 1] = "3"
        raw = os.path.join(ROOT, "results", "accuracy", "wiki.test.raw")
        return ["llama-perplexity", "-m", suite_gguf(run["llama"]["model"]), "-f", raw, *args], out
    with open(os.path.join(ROOT, run["published"])) as f:
        pub = json.load(f)
    argv = [PY, os.path.join(HERE, "mlx_accuracy_ppl.py"), "--engine", pub["engine"], "--model", pub["model"],
            "--ctx", str(pub["ctx"]), "--chunks", "3" if quick else str(pub["chunks"]),
            "--mj-kw", json.dumps(pub.get("mj_kw") or {}), "--out", out]
    if pub.get("mj_kw"):  # gpt-oss / Qwen3.5 take their own kwargs, not placement= (checklist 1.0b)
        argv += ["--placement", "none"]
    if run.get("ref"):
        argv += ["--ref", os.path.join(REFS, run["ref"])]
    if run.get("save_ref") and not os.path.isdir(os.path.join(REFS, run["save_ref"])):
        argv += ["--save-ref", os.path.join(REFS, run["save_ref"])]
    return argv, out


def plan(claims, out_root, quick):
    """[(claim, [(argv, stdout file or None)])]"""
    steps = []
    for c in claims:
        d = os.path.join(out_root, c["id"])
        if c["kind"] == "suite":
            steps.append((c, [(suite_cmd(c, d, quick), os.path.join(d + ".log"))]))
        else:
            cmds = []
            for r in c["runs"]:
                argv, out = ppl_cmd(r, d, quick)
                cmds.append((argv, out if "llama" in r else out[:-5] + ".log"))
            steps.append((c, cmds))
    return steps


# ---- reading results ------------------------------------------------------------------------------------------

def suite_cells(path):
    with open(path) as f:
        d = json.load(f)
    cells = {}
    for model, v in d.items():
        for key, r in v["results"].items():
            t = r.get("tok_s")
            if t:
                cells[(model, key)] = dict(mean=t["mean"], ci=t["ci95"], n=t["n"])
    return cells


def llama_ppl(path):
    with open(path, errors="replace") as f:
        m = re.findall(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", f.read())
    return dict(ppl=float(m[-1][0]), ppl_se=float(m[-1][1])) if m else None


def ours_ppl(path):
    with open(path) as f:
        d = json.load(f)
    return {k: d[k] for k in ("ppl", "ppl_se", "kld_mean", "kld_p99", "same_top", "chunks") if k in d}


def speed_verdict(pub, new):
    if new is None:
        return "missing (every block flagged or failed)"
    r = new["mean"] / pub["mean"] - 1
    if (_judged(new) and new["ci"][0] <= pub["mean"] <= new["ci"][1]) or abs(r) <= SAME_SPEED:
        return f"same ({r:+.1%})"
    return f"{'faster' if r > 0 else 'slower'} {r:+.1%}"


def _judged(c):
    return c is not None and c["n"] >= 2 and c["ci"][0] == c["ci"][0]  # an interval needs 2 reps (nan otherwise)


def relation(rel, cells):
    m, t = rel["model"], rel["test"]
    get = lambda e: cells.get((m, f"{e}|{t}"))
    names = rel["tie"] if "tie" in rel else [rel["faster"], *rel["than"]]
    if any(get(e) is not None and not _judged(get(e)) for e in names):
        return "not judged (fewer than 2 reps)"
    if "tie" in rel:
        a, b = (get(e) for e in rel["tie"])
        if not (a and b):
            return "unknown (missing cell)"
        if a["ci"][0] <= b["ci"][1] and b["ci"][0] <= a["ci"][1]:
            return "holds"
        x, y = rel["tie"]
        hi, lo = (x, y) if a["mean"] > b["mean"] else (y, x)
        return f"does not hold: {hi} faster ({max(a['mean'], b['mean']) / min(a['mean'], b['mean']):.2f}x, intervals apart)"
    a = get(rel["faster"])
    out = []
    for e in rel["than"]:
        b = get(e)
        if not (a and b):
            out.append(f"{e}: unknown (missing cell)")
        else:
            if a["ci"][0] > b["ci"][1]:
                v = "holds"
            elif b["ci"][0] > a["ci"][1]:
                v = f"does not hold: {e} faster"
            else:
                v = "does not hold: tied (intervals overlap)"
            out.append(f"{e}: {v} ({a['mean'] / b['mean']:.2f}x)")
    return "; ".join(out)


def rel_text(rel):
    if "tie" in rel:
        return f"{rel['model']} {rel['test']}: {' tied with '.join(rel['tie'])}"
    return f"{rel['model']} {rel['test']}: {rel['faster']} faster than {', '.join(rel['than'])}"


def compare(claims, out_root, quick):
    lines = [f"# Reproduction report ({os.path.basename(out_root)})", ""]
    if quick:
        lines += ["**Quick mode (1 rep, 1 round / 3 chunks): a smoke test of the commands, not a result.**", ""]
    report = []
    for c in claims:
        d = os.path.join(out_root, c["id"])
        lines += [f"## {c['id']} ({c['source']}; {c['where']})", ""]
        entry = dict(id=c["id"], source=c["source"], rows=[], relations=[])
        if c["kind"] == "suite":
            pub = suite_cells(os.path.join(ROOT, c["published"]))
            new_path = os.path.join(d, "summary.json")
            new = suite_cells(new_path) if os.path.exists(new_path) else {}
            models = c["suite"]["models"].split(",")
            engines = c["suite"]["engines"].split(",")
            lines += ["| model | engine | test | published tok/s [95% CI] | new tok/s [95% CI] | n | verdict |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for (m, key), p in sorted(pub.items()):
                e, t = key.split("|")
                if m not in models or e not in engines or t not in c["suite"]["tests"].split(","):
                    continue
                n = new.get((m, key))
                v = speed_verdict(p, n)
                ns = f"{n['mean']:.2f} [{n['ci'][0]:.2f}-{n['ci'][1]:.2f}]" if n else "-"
                lines.append(f"| {m} | {e} | {t} | {p['mean']:.2f} [{p['ci'][0]:.2f}-{p['ci'][1]:.2f}] | {ns} | "
                             f"{n['n'] if n else 0} | {v} |")
                entry["rows"].append(dict(model=m, engine=e, test=t, published=p, new=n, verdict=v))
            lines.append("")
            for rel in c.get("relations", []):
                res = relation(rel, new)
                lines.append(f"- {rel_text(rel)}: {res}")
                entry["relations"].append(dict(claim=rel_text(rel), result=res))
        else:
            lines += ["| run | published | new | verdict |", "| --- | --- | --- | --- |"]
            for r in c["runs"]:
                if "llama" in r:
                    p, newf = llama_ppl(os.path.join(ROOT, r["published"])), os.path.join(d, r["name"] + ".log")
                    n = llama_ppl(newf) if os.path.exists(newf) else None
                else:
                    p, newf = ours_ppl(os.path.join(ROOT, r["published"])), os.path.join(d, r["name"] + ".json")
                    n = ours_ppl(newf) if os.path.exists(newf) else None
                if n is None:
                    v = "missing"
                elif quick:  # 3 chunks against 80 (or 40): the number is not comparable
                    v = "ran (quick: not compared)"
                else:
                    dp = n["ppl"] / p["ppl"] - 1
                    v = f"ppl {'same' if abs(dp) <= SAME_PPL else 'differs'} ({dp:+.2%})"
                    if "kld_mean" in p and "kld_mean" in n:
                        dk = n["kld_mean"] / p["kld_mean"] - 1
                        v += f"; KL {'same' if abs(dk) <= SAME_KL else 'differs'} ({dk:+.1%})"
                fmt = lambda x: "-" if not x else (f"ppl {x['ppl']:.3f} +- {x['ppl_se']:.3f}"
                                                  + (f", KL {x['kld_mean']:.4f}" if "kld_mean" in x else ""))
                lines.append(f"| {r['name']} | {fmt(p)} | {fmt(n)} | {v} |")
                entry["rows"].append(dict(run=r["name"], published=p, new=n, verdict=v))
            lines.append("")
        report.append(entry)
    with open(os.path.join(out_root, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_root, "report.json"), "w") as f:
        json.dump(dict(quick=quick, claims=report), f, indent=1)
    return "\n".join(lines)


# ---- main -----------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="list the claims and exit")
    ap.add_argument("--only", default="", help="comma list of claim ids (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--quick", action="store_true", help="1 rep / 1 round / 3 chunks: smoke test, not a result")
    ap.add_argument("--out", default=None, help="new output folder (default results/repro/<YYYYMMDD_HHMM>)")
    ap.add_argument("--compare", default=None, help="only (re)write the report for this existing output folder")
    ap.add_argument("--manifest", default=MANIFEST)
    a = ap.parse_args(argv)

    claims = load_manifest(a.manifest)
    if a.only:
        want = a.only.split(",")
        unknown = set(want) - {c["id"] for c in claims}
        if unknown:
            ap.error(f"unknown claim ids: {', '.join(sorted(unknown))}")
        claims = [c for c in claims if c["id"] in want]
    if a.list:
        for c in claims:
            print(f"{c['id']:20s} {c['kind']:5s} {c['source']:12s} {c['where']}")
        return 0
    if a.compare:
        meta_path = os.path.join(a.compare, "repro_meta.json")
        quick = json.load(open(meta_path)).get("quick", False) if os.path.exists(meta_path) else False
        print(compare(claims, a.compare, quick))
        return 0

    out_root = a.out or os.path.join(ROOT, "results", "repro", time.strftime("%Y%m%d_%H%M"))
    steps = plan(claims, out_root, a.quick)
    if a.dry_run:
        for c, cmds in steps:
            print(f"# {c['id']} ({c['source']})")
            for argv_, log in cmds:
                print(" ".join(argv_), f"> {log} 2>&1")
        return 0
    # results are never overwritten: each claim runs into a folder of its own that must not exist yet (claims may be
    # run one invocation at a time into the same --out, e.g. to stop servers in between; then --compare the folder)
    taken = [c["id"] for c in claims if os.path.exists(os.path.join(out_root, c["id"]))]
    if taken:
        sys.exit(f"{out_root} already has {', '.join(taken)}; pass a new --out or other --only")
    os.makedirs(out_root, exist_ok=True)
    git = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "src", "tools"], capture_output=True,
                                text=True).stdout.strip())
    meta = os.path.join(out_root, "repro_meta.json")
    if os.path.exists(meta):
        meta = os.path.join(out_root, f"repro_meta_{time.strftime('%H%M%S')}.json")
    with open(meta, "w") as f:
        json.dump(dict(start=time.strftime("%Y-%m-%dT%H:%M:%S"), git=git, dirty_src=dirty, quick=a.quick,
                       claims=[c["id"] for c in claims]), f, indent=1)
    failed = []
    for c, cmds in steps:
        os.makedirs(os.path.join(out_root, c["id"]), exist_ok=True)
        for argv_, log in cmds:
            print(f"== {c['id']}: {os.path.basename(log)} {time.strftime('%H:%M:%S')}", flush=True)
            with open(log, "w") as f:
                rc = subprocess.run(argv_, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
            if rc:
                failed.append(f"{c['id']}: {os.path.basename(log)} exit {rc}")
                print(f"   exit {rc} (see {log})", flush=True)
    print(compare(claims, out_root, a.quick))
    if failed:
        print("\nFailed steps:\n" + "\n".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

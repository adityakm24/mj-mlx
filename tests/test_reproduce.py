# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""tools/reproduce.py: the manifest points at real files, and a reproduction equal to the published runs reports every
number "same" and every README sentence "holds" (so the sentences in results/published.json are true of the data)."""
import json
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import reproduce  # noqa: E402


def _fake_repro(tmp_path):
    claims = reproduce.load_manifest()
    for c in claims:
        d = tmp_path / c["id"]
        d.mkdir()
        if c["kind"] == "suite":
            shutil.copy(os.path.join(ROOT, c["published"]), d / "summary.json")
        else:
            for r in c["runs"]:
                ext = ".log" if "llama" in r else ".json"
                shutil.copy(os.path.join(ROOT, r["published"]), d / (r["name"] + ext))
    return claims


def test_manifest_files_exist():
    for c in reproduce.load_manifest():
        paths = [c["published"]] if c["kind"] == "suite" else [r["published"] for r in c["runs"]]
        for p in paths:
            assert os.path.exists(os.path.join(ROOT, p)), p


def test_identical_run_reproduces(tmp_path):
    claims = _fake_repro(tmp_path)
    reproduce.compare(claims, str(tmp_path), quick=False)
    rep = json.load(open(tmp_path / "report.json"))
    for c in rep["claims"]:
        assert c["rows"], c["id"]
        for r in c["rows"]:
            assert r["verdict"].startswith(("same", "ppl same")), (c["id"], r)
        for rel in c["relations"]:
            assert "does not hold" not in rel["result"] and "unknown" not in rel["result"], (c["id"], rel)


def test_dry_run_prints_every_claim(capsys):
    reproduce.main(["--dry-run", "--out", "/nonexistent/repro"])
    out = capsys.readouterr().out
    for c in reproduce.load_manifest():
        assert f"# {c['id']} " in out

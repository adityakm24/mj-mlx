"""tools/swap_guard.py: a tool stops itself (exit 3) once this run's swap-outs pass the limit."""

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE = """
import sys, time
sys.path.insert(0, %r)
from swap_guard import guard_self
guard_self(lambda gb: print("saved", flush=True), limit_gb=-1, every=0.1)
time.sleep(30)
"""


def test_guard_self_exits_with_partial_save():
    p = subprocess.run([sys.executable, "-c", CODE % os.path.join(ROOT, "tools")], capture_output=True, text=True,
                       timeout=20)
    assert p.returncode == 3
    assert "saved" in p.stdout and "ABORTED" in p.stderr

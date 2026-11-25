"""Stop a tool whose run swaps past a limit (after suite v16, RESULTS 1zu: a run above RAM can write hundreds of GB of
swap to the SSD). `guard_self()` starts a daemon thread that reads macOS's swap-out counter every few seconds and,
once this run's swap-outs pass MJ_SWAP_ABORT_GB (default 20), calls `on_trip` (to save partial results) and exits
with code 3. The swap-out counter is machine-wide, so another process's paging counts too: on a shared machine the
guard errs on the side of stopping."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading

LIMIT_GB = float(os.environ.get("MJ_SWAP_ABORT_GB", "20"))


def swapouts_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    m = re.search(r"Swapouts:\s+(\d+)", out)
    return int(m.group(1)) * 16384 / 1e9 if m else 0.0


def guard_self(on_trip=None, limit_gb: float | None = None, every: float = 3.0, exit_code: int = 3):
    limit = LIMIT_GB if limit_gb is None else limit_gb
    base, stop = swapouts_gb(), threading.Event()

    def run():
        while not stop.wait(every):
            gb = swapouts_gb() - base
            if gb > limit:
                print(f"ABORTED: swap-outs {gb:.1f} GB passed {limit:.0f} GB (MJ_SWAP_ABORT_GB)", file=sys.stderr,
                      flush=True)
                try:
                    if on_trip is not None:
                        on_trip(gb)
                finally:
                    os._exit(exit_code)

    threading.Thread(target=run, daemon=True).start()
    return stop


# Engine+model configs that tripped a swap stop, shared by the suite and the run scripts (results/v17_scripts/run_v17.py
# keys tools the same way). A config listed here is skipped until the user deletes its entry.
TRIPPED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "swap_tripped.json")


def cfg_key(engine: str, model_path: str) -> str:
    """'<engine>:<HF repo>' for an HF-cache path (the repo, not the snapshot hash), else '<engine>:<dir name>'."""
    hub = [part for part in model_path.split(os.sep) if part.startswith("models--")]
    name = hub[0][len("models--"):].replace("--", "/") if hub else os.path.basename(os.path.normpath(model_path))
    return f"{engine}:{name}"


def tripped() -> dict:
    import json

    try:
        return json.load(open(TRIPPED))
    except (OSError, ValueError):
        return {}


def add_tripped(key: str, note: str) -> None:
    import json
    import time

    t = tripped()
    t[key] = f"{time.strftime('%Y-%m-%dT%H:%M')} {note}"
    tmp = TRIPPED + ".tmp"
    json.dump(t, open(tmp, "w"), indent=1)
    os.replace(tmp, TRIPPED)


# Daily swap-out budget (results/swap_daily.json): every run adds its machine swap-out delta under its date, and no
# run starts once the day passes DAILY_GB. results/v17_scripts/run_v17.py's run_steps and the suite (when started
# directly) both use it; runs started by run_steps (MJ_RUN_STEPS=1) leave the accounting to it.
DAILY = os.path.join(os.path.dirname(TRIPPED), "swap_daily.json")
DAILY_GB = 40.0


def daily_total(day: str | None = None):
    import json
    import time

    try:
        d = json.load(open(DAILY))
    except (OSError, ValueError):
        d = {}
    return sum(r["gb"] for r in d.get(day or time.strftime("%Y-%m-%d"), [])), d


def daily_add(run: str, gb: float, note: str = "") -> None:
    import json
    import time

    day = time.strftime("%Y-%m-%d")
    _, d = daily_total(day)
    d.setdefault(day, []).append(dict(run=run, gb=round(gb, 2), end=time.strftime("%H:%M"), **({"note": note} if note else {})))
    tmp = DAILY + ".tmp"
    json.dump(d, open(tmp, "w"), indent=1)
    os.replace(tmp, DAILY)


"""Snapshot of machine state for a benchmark run directory: top CPU processes, swap, memory pressure, load average.

    python tools/mlx_machine_state.py results/<run>/machine_state_start.json
"""

import json
import subprocess
import sys
import time


def main():
    top = subprocess.run(["top", "-l", "2", "-n", "8", "-o", "cpu", "-stats", "pid,command,cpu"], capture_output=True,
                         text=True).stdout
    blk = top.strip().split("\n\n")[-1].splitlines()[1:]
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    mp = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    la = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout.strip()
    d = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), top_cpu=[" ".join(l.split()) for l in blk], swap=swap,
             memory_pressure=mp, loadavg=la)
    json.dump(d, open(sys.argv[1], "w"), indent=1)
    print("state:", swap, "|", mp, "| top:", "; ".join(" ".join(l.split()[1:]) for l in blk[:3]), flush=True)


if __name__ == "__main__":
    main()

"""Run tools/mlx_bench_worker.py with the main thread at a chosen QoS class (threads it creates later inherit it).

Lean gpt-oss decode waits on the host every step (deferred step: token -> embedding row -> submit), so a host thread
moved to the efficiency cores would slow decode but not prefill, which is what the slow state looks like (RESULTS
1za / 1zb). pthread_set_qos_class_self_np is a per-thread request, not a system setting.

    .venv/bin/python tools/mlx_qos_run.py --qos interactive -- --engine mj ... (worker args)
"""

import ctypes
import os
import runpy
import sys

QOS = {"interactive": 0x21, "initiated": 0x19, "default": 0x15, "utility": 0x11, "background": 0x09}


def main():
    args = sys.argv[1:]
    qos = "default"
    if args[:1] == ["--qos"]:
        qos, args = args[1], args[2:]
    if args[:1] == ["--"]:
        args = args[1:]
    if qos != "none":
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        rc = libc.pthread_set_qos_class_self_np(QOS[qos], 0)
        if rc:
            raise OSError(rc, "pthread_set_qos_class_self_np failed")
    here = os.path.dirname(os.path.abspath(__file__))
    sys.argv = [os.path.join(here, "mlx_bench_worker.py")] + args
    sys.path.insert(0, here)
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()

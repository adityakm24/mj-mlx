"""Per-process memory on macOS, for memory-per-engine tables (checklist 0.10).

`phys_footprint` is the number macOS itself uses for a process's memory (Activity Monitor's "Memory", the jetsam
limit): dirty and compressed pages plus IOKit / GPU allocations charged to it, so Metal buffers on unified memory count.
Read with `proc_pid_rusage` (one system call, no subprocess), cheap enough to sample during a benchmark.

    from mlx_proc_mem import rusage, PeakSampler
    rusage(pid)  # {'phys_footprint': bytes, 'lifetime_max_phys_footprint': bytes, 'resident': bytes, 'wired': bytes}
    with PeakSampler(lambda: [pid1, pid2]) as s: ...   # s.peak: max over samples of the summed footprint
"""

from __future__ import annotations

import ctypes
import subprocess
import threading
import time

_FIELDS = ["user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size",
           "phys_footprint", "proc_start_abstime", "proc_exit_abstime", "child_user_time", "child_system_time",
           "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins", "child_elapsed_abstime",
           "diskio_bytesread", "diskio_byteswritten", "cpu_time_qos_default", "cpu_time_qos_maintenance",
           "cpu_time_qos_background", "cpu_time_qos_utility", "cpu_time_qos_legacy", "cpu_time_qos_user_initiated",
           "cpu_time_qos_user_interactive", "billed_system_time", "serviced_system_time", "logical_writes",
           "lifetime_max_phys_footprint", "instructions", "cycles", "billed_energy", "serviced_energy",
           "interval_max_phys_footprint", "runnable_time"]


class _RusageInfoV4(ctypes.Structure):  # <sys/resource.h> struct rusage_info_v4
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f, ctypes.c_uint64) for f in _FIELDS]


_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
_RUSAGE_INFO_V4 = 4


def rusage(pid: int) -> dict | None:
    """Memory figures of a live process (bytes), or None when it is gone or not readable."""
    info = _RusageInfoV4()
    if _libc.proc_pid_rusage(int(pid), _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        return None
    return dict(phys_footprint=info.phys_footprint, lifetime_max_phys_footprint=info.lifetime_max_phys_footprint,
                resident=info.resident_size, wired=info.wired_size)


def pids_matching(*needles: str, exclude=("mlx_bench", "mlx_cold_start")) -> list[int]:
    """PIDs whose command line contains every needle and none of `exclude` (our own benchmark processes name the
    engines in their arguments). One `ps`; call it seldom."""
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if all(n in cmd for n in needles) and not any(x in cmd for x in exclude):
            pids.append(int(pid))
    return pids


class PeakSampler:
    """Samples the summed phys_footprint of `pids_fn()` every `period` s in a thread; `pids_fn` is re-run every
    `refresh` s (servers start their worker processes late). `peak` (bytes) and `per_pid` (max per process)."""

    def __init__(self, pids_fn, period: float = 1.0, refresh: float = 10.0):
        self.pids_fn, self.period, self.refresh = pids_fn, period, refresh
        self.peak, self.peak_resident, self.per_pid, self.samples = 0, 0, {}, 0
        self._stop = threading.Event()

    def _run(self):
        pids, t_pids = [], 0.0
        while not self._stop.is_set():
            if time.monotonic() - t_pids > self.refresh:
                try:
                    pids = list(self.pids_fn())
                except Exception:
                    pids = []
                t_pids = time.monotonic()
            total = res = 0
            for p in pids:
                r = rusage(p)
                if r:
                    total += r["phys_footprint"]
                    res += r["resident"]  # includes mapped weight files (llama.cpp mmaps its GGUF)
                    self.per_pid[p] = max(self.per_pid.get(p, 0), r["phys_footprint"])
            self.peak = max(self.peak, total)
            self.peak_resident = max(self.peak_resident, res)
            self.samples += 1
            self._stop.wait(self.period)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(5)


class _VmStats64(ctypes.Structure):  # <mach/vm_statistics.h> struct vm_statistics64
    _fields_ = [("free_count", ctypes.c_uint32), ("active_count", ctypes.c_uint32),
                ("inactive_count", ctypes.c_uint32), ("wire_count", ctypes.c_uint32),
                ("zero_fill_count", ctypes.c_uint64), ("reactivations", ctypes.c_uint64),
                ("pageins", ctypes.c_uint64), ("pageouts", ctypes.c_uint64), ("faults", ctypes.c_uint64),
                ("cow_faults", ctypes.c_uint64), ("lookups", ctypes.c_uint64), ("hits", ctypes.c_uint64),
                ("purges", ctypes.c_uint64), ("purgeable_count", ctypes.c_uint32),
                ("speculative_count", ctypes.c_uint32), ("decompressions", ctypes.c_uint64),
                ("compressions", ctypes.c_uint64), ("swapins", ctypes.c_uint64), ("swapouts", ctypes.c_uint64),
                ("compressor_page_count", ctypes.c_uint32), ("throttled_count", ctypes.c_uint32),
                ("external_page_count", ctypes.c_uint32), ("internal_page_count", ctypes.c_uint32),
                ("total_uncompressed_pages_in_compressor", ctypes.c_uint64)]


_libc.mach_host_self.restype = ctypes.c_uint32
_HOST = _libc.mach_host_self()
_PAGE = 16384


def swap_pages() -> tuple[int, int] | None:
    """(swap-ins, swap-outs) since boot, in 16 KB pages, from host_statistics64 (one Mach call, no subprocess: cheap
    enough to read at a timed window's edges). The same counters as `vm_stat`'s Swapins / Swapouts."""
    st = _VmStats64()
    n = ctypes.c_uint32(ctypes.sizeof(_VmStats64) // 4)
    if _libc.host_statistics64(_HOST, 4, ctypes.byref(st), ctypes.byref(n)) != 0:  # HOST_VM_INFO64
        return None
    return int(st.swapins), int(st.swapouts)

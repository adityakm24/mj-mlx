# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""Whole-machine power from the SMC (checklist 0.16), no sudo.

The SMC key PSTR ("system total power", watts, float) is what tools such as mactop report as total power. It is a
~1 s average that trails the load by ~1.5-2 s (2026-11-21, 4096x4096 matmul burst: full power showed ~2 s after the
start and stayed ~2 s after the end), so energy integrates the samples over the window shifted by LAG and needs
sampling to continue for a few seconds after the window. Windows under ~3 s are coarse. IOReport's
Energy Model group gives this Mac's GPU rail but reads 0 for the CPU and DRAM rails (checked 2026-11-21 under CPU load),
so whole-machine energy comes from here: `PowerSampler` reads PSTR at `hz` in a thread; `.joules()` integrates.

    with PowerSampler() as p: work()
    p.mean_w, p.joules(), p.samples
"""

from __future__ import annotations

import ctypes
import struct
import threading
import time

_iokit = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")


class _Vers(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint8), ("minor", ctypes.c_uint8), ("build", ctypes.c_uint8),
                ("reserved", ctypes.c_uint8), ("release", ctypes.c_uint16)]


class _PLimit(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint16), ("length", ctypes.c_uint16), ("cpu", ctypes.c_uint32),
                ("gpu", ctypes.c_uint32), ("mem", ctypes.c_uint32)]


class _KeyInfo(ctypes.Structure):
    _fields_ = [("dataSize", ctypes.c_uint32), ("dataType", ctypes.c_uint32), ("dataAttributes", ctypes.c_uint8)]


class _KeyData(ctypes.Structure):  # SMCKeyData_t, 80 bytes
    _fields_ = [("key", ctypes.c_uint32), ("vers", _Vers), ("pLimitData", _PLimit), ("keyInfo", _KeyInfo),
                ("result", ctypes.c_uint8), ("status", ctypes.c_uint8), ("data8", ctypes.c_uint8),
                ("data32", ctypes.c_uint32), ("bytes", ctypes.c_uint8 * 32)]


_iokit.IOServiceMatching.restype = ctypes.c_void_p
_iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
_iokit.IOServiceGetMatchingService.restype = ctypes.c_uint32
_iokit.IOServiceGetMatchingService.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
_iokit.IOServiceOpen.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
_iokit.IOConnectCallStructMethod.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t,
                                             ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
_libc.mach_task_self.restype = ctypes.c_uint32

LAG = 1.75  # seconds PSTR trails the load (see the module doc)
_KERNEL_INDEX_SMC = 2
_READ_BYTES, _READ_KEYINFO = 5, 9


def _fourcc(s: str) -> int:
    return struct.unpack(">I", s.encode())[0]


class SMC:
    def __init__(self):
        svc = _iokit.IOServiceGetMatchingService(0, _iokit.IOServiceMatching(b"AppleSMC"))
        if not svc:
            raise RuntimeError("AppleSMC not found")
        self.conn = ctypes.c_uint32()
        kr = _iokit.IOServiceOpen(svc, _libc.mach_task_self(), 0, ctypes.byref(self.conn))
        if kr != 0:
            raise RuntimeError(f"IOServiceOpen: {kr}")
        self._info: dict[str, tuple[int, int]] = {}

    def _call(self, inp: _KeyData) -> _KeyData:
        out = _KeyData()
        size = ctypes.c_size_t(ctypes.sizeof(_KeyData))
        kr = _iokit.IOConnectCallStructMethod(self.conn.value, _KERNEL_INDEX_SMC, ctypes.byref(inp),
                                              ctypes.sizeof(_KeyData), ctypes.byref(out), ctypes.byref(size))
        if kr != 0 or out.result != 0:
            raise RuntimeError(f"SMC call failed: kr {kr}, result {out.result}")
        return out

    def read(self, key: str) -> float:
        if key not in self._info:
            inp = _KeyData(key=_fourcc(key), data8=_READ_KEYINFO)
            o = self._call(inp)
            self._info[key] = (o.keyInfo.dataSize, o.keyInfo.dataType)
        size, typ = self._info[key]
        inp = _KeyData(key=_fourcc(key), data8=_READ_BYTES)
        inp.keyInfo.dataSize = size
        o = self._call(inp)
        raw = bytes(o.bytes[:size])
        if typ == _fourcc("flt "):
            return struct.unpack("<f", raw[:4])[0]
        raise ValueError(f"{key}: unsupported SMC type {typ:#x}")


class PowerSampler:
    """Samples whole-machine watts (SMC PSTR) at `hz` while open."""

    def __init__(self, hz: float = 10.0, key: str = "PSTR"):
        self.smc, self.key, self.period = SMC(), key, 1.0 / hz
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append((time.perf_counter(), self.smc.read(self.key)))
            except Exception:
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self.t0 = time.perf_counter()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self.t1 = time.perf_counter()
        self._stop.set()
        self._t.join(2)

    @property
    def mean_w(self) -> float | None:
        return sum(w for _, w in self.samples) / len(self.samples) if self.samples else None

    def joules(self) -> float | None:
        m = self.mean_w
        return None if m is None else m * (self.t1 - self.t0)

    def joules_between(self, a: float, b: float, lag: float = None) -> float | None:
        """Integral of the sampled watts over [a + lag, b + lag] (perf_counter times), each sample held until the next."""
        lag = LAG if lag is None else lag
        a, b = a + lag, b + lag
        pts = self.samples
        if len(pts) < 2 or pts[-1][0] < b:
            return None
        j = 0.0
        for (t, w), (t2, _) in zip(pts, pts[1:]):
            lo, hi = max(t, a), min(t2, b)
            if hi > lo:
                j += w * (hi - lo)
        return j


if __name__ == "__main__":
    s = SMC()
    print("PSTR now:", round(s.read("PSTR"), 2), "W")

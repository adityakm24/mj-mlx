# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""Apple-silicon energy counters through IOReport (no sudo).

Reads the cumulative "Energy Model" channels (GPU, CPU, ANE, DRAM ...) that `powermetrics` also uses.
The numbers are **whole-chip, per-rail estimates from the SoC's energy model**, not a GPU-board measurement.
They are for comparing engines on this Mac only and are never put next to an NVML/B200 number.
Without sudo only the "GPU Energy" rail (nJ) updates on this M4 (CPU, DRAM and the rest read 0), so the
meter reports the GPU rail: the nearest analogue of NVML's GPU-board boundary, idle floor included.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import time
from dataclasses import dataclass

_cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
_io = ctypes.CDLL("/usr/lib/libIOReport.dylib")

_vp = ctypes.c_void_p
_cf.CFStringCreateWithCString.restype = _vp
_cf.CFStringCreateWithCString.argtypes = [_vp, ctypes.c_char_p, ctypes.c_uint32]
_cf.CFStringGetCString.restype = ctypes.c_bool
_cf.CFStringGetCString.argtypes = [_vp, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
_cf.CFDictionaryGetValue.restype = _vp
_cf.CFDictionaryGetValue.argtypes = [_vp, _vp]
_cf.CFArrayGetCount.restype = ctypes.c_long
_cf.CFArrayGetCount.argtypes = [_vp]
_cf.CFArrayGetValueAtIndex.restype = _vp
_cf.CFArrayGetValueAtIndex.argtypes = [_vp, ctypes.c_long]
_cf.CFDictionaryCreateMutableCopy.restype = _vp
_cf.CFDictionaryCreateMutableCopy.argtypes = [_vp, ctypes.c_long, _vp]
_cf.CFRelease.argtypes = [_vp]

_io.IOReportCopyChannelsInGroup.restype = _vp
_io.IOReportCopyChannelsInGroup.argtypes = [_vp, _vp, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
_io.IOReportCreateSubscription.restype = _vp
_io.IOReportCreateSubscription.argtypes = [_vp, _vp, ctypes.POINTER(_vp), ctypes.c_uint64, _vp]
_io.IOReportCreateSamples.restype = _vp
_io.IOReportCreateSamples.argtypes = [_vp, _vp, _vp]
_io.IOReportCreateSamplesDelta.restype = _vp
_io.IOReportCreateSamplesDelta.argtypes = [_vp, _vp, _vp]
for _f in ("IOReportChannelGetGroup", "IOReportChannelGetChannelName", "IOReportChannelGetUnitLabel"):
    getattr(_io, _f).restype = _vp
    getattr(_io, _f).argtypes = [_vp]
_io.IOReportSimpleGetIntegerValue.restype = ctypes.c_int64
_io.IOReportSimpleGetIntegerValue.argtypes = [_vp, ctypes.c_int32]
_io.IOReportStateGetCount.restype = ctypes.c_int32
_io.IOReportStateGetCount.argtypes = [_vp]
_io.IOReportStateGetNameForIndex.restype = _vp
_io.IOReportStateGetNameForIndex.argtypes = [_vp, ctypes.c_int32]
_io.IOReportStateGetResidency.restype = ctypes.c_int64
_io.IOReportStateGetResidency.argtypes = [_vp, ctypes.c_int32]

_UTF8 = 0x08000100


def _cfstr(s: str) -> int:
    return _cf.CFStringCreateWithCString(None, s.encode(), _UTF8)


def _pystr(ref) -> str:
    if not ref:
        return ""
    buf = ctypes.create_string_buffer(256)
    _cf.CFStringGetCString(ref, buf, 256, _UTF8)
    return buf.value.decode()


_UNIT_TO_J = {"nJ": 1e-9, "uJ": 1e-6, "µJ": 1e-6, "mJ": 1e-3, "J": 1.0}


class EnergyCounter:
    """Cumulative energy per rail since construction. `read()` returns {rail: joules}."""

    def __init__(self) -> None:
        chans = _io.IOReportCopyChannelsInGroup(_cfstr("Energy Model"), None, 0, 0, 0)
        if not chans:
            raise RuntimeError("IOReport 'Energy Model' group not available")
        self._chans = _cf.CFDictionaryCreateMutableCopy(None, 0, chans)
        self._subbed = _vp()
        self._sub = _io.IOReportCreateSubscription(None, self._chans, ctypes.byref(self._subbed), 0, None)
        if not self._sub:
            raise RuntimeError("IOReportCreateSubscription failed")
        self._key = _cfstr("IOReportChannels")
        self._start = self._sample()

    def _sample(self) -> int:
        return _io.IOReportCreateSamples(self._sub, self._subbed, None)

    def _delta(self, a: int, b: int) -> dict[str, float]:
        d = _io.IOReportCreateSamplesDelta(a, b, None)
        arr = _cf.CFDictionaryGetValue(d, self._key)
        out: dict[str, float] = {}
        for i in range(_cf.CFArrayGetCount(arr)):
            ch = _cf.CFArrayGetValueAtIndex(arr, i)
            name = _pystr(_io.IOReportChannelGetChannelName(ch))
            unit = _pystr(_io.IOReportChannelGetUnitLabel(ch)).strip()
            val = _io.IOReportSimpleGetIntegerValue(ch, 0)
            out[name] = out.get(name, 0.0) + val * _UNIT_TO_J.get(unit, 0.0)
        _cf.CFRelease(d)
        return out

    def snapshot(self) -> int:
        # The first sample after a long gap can be stale: it returned the value from 9 s earlier (a whole
        # prefill's energy then landed in the next window; suite v4 decode windows read ~4x or 0 J). An
        # immediate second sample was current every time (RESULTS 1zb, tools/mlx_energy_lag_probe.py).
        first = self._sample()
        _cf.CFRelease(first)
        return self._sample()

    def between(self, a: int, b: int) -> dict[str, float]:
        return self._delta(a, b)


class GpuResidency:
    """GPU hardware active residency from IOReport "GPU Stats" / "GPU Performance States" (channel GPUPH), the
    counter powermetrics reports as "GPU HW active residency". No sudo. active % = 1 - OFF / total over a window."""

    def __init__(self) -> None:
        chans = _io.IOReportCopyChannelsInGroup(_cfstr("GPU Stats"), _cfstr("GPU Performance States"), 0, 0, 0)
        if not chans:
            raise RuntimeError("IOReport GPU performance states not available")
        self._chans = _cf.CFDictionaryCreateMutableCopy(None, 0, chans)
        self._subbed = _vp()
        self._sub = _io.IOReportCreateSubscription(None, self._chans, ctypes.byref(self._subbed), 0, None)
        self._key = _cfstr("IOReportChannels")

    def snapshot(self) -> int:
        return _io.IOReportCreateSamples(self._sub, self._subbed, None)

    def active_pct(self, a: int, b: int) -> float | None:
        d = _io.IOReportCreateSamplesDelta(a, b, None)
        arr = _cf.CFDictionaryGetValue(d, self._key)
        res = None
        for i in range(_cf.CFArrayGetCount(arr)):
            ch = _cf.CFArrayGetValueAtIndex(arr, i)
            if _pystr(_io.IOReportChannelGetChannelName(ch)) != "GPUPH":
                continue
            st = {_pystr(_io.IOReportStateGetNameForIndex(ch, j)): _io.IOReportStateGetResidency(ch, j)
                  for j in range(_io.IOReportStateGetCount(ch))}
            total = sum(st.values())
            if total:
                res = 100.0 * (1.0 - st.get("OFF", 0) / total)
        _cf.CFRelease(d)
        return res


@dataclass
class Window:
    seconds: float
    joules: dict[str, float]
    gpu_active_pct: float | None = None

    def rail(self, *names: str) -> float:
        return sum(v for k, v in self.joules.items() if any(k.startswith(n) for n in names))

    @property
    def gpu_j(self) -> float:
        return self.rail("GPU Energy")



class Meter:
    """`with Meter() as m: ...` then `m.window` holds seconds and joules per rail."""

    _counter: EnergyCounter | None = None
    _residency: GpuResidency | None = None

    def __enter__(self) -> "Meter":
        if Meter._counter is None:
            Meter._counter = EnergyCounter()
            try:
                Meter._residency = GpuResidency()
            except RuntimeError:
                Meter._residency = None
        self._a = Meter._counter.snapshot()
        self._ra = Meter._residency.snapshot() if Meter._residency else None
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        b = Meter._counter.snapshot()
        rb = Meter._residency.snapshot() if Meter._residency else None
        dt = time.perf_counter() - self._t
        act = Meter._residency.active_pct(self._ra, rb) if rb else None
        self.window = Window(dt, Meter._counter.between(self._a, b), act)
        _cf.CFRelease(self._a)
        _cf.CFRelease(b)
        if rb:
            _cf.CFRelease(self._ra)
            _cf.CFRelease(rb)


if __name__ == "__main__":
    with Meter() as m:
        time.sleep(1.0)
    print(f"idle 1 s: {m.window.seconds:.3f} s")
    for k, v in sorted(m.window.joules.items()):
        if v:
            print(f"  {k:24s} {v:9.4f} J")

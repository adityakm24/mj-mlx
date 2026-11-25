# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""mj-mlx: MilliJoule's MoE inference engine for Apple silicon, on MLX.

Its energy numbers are whole-chip GPU-rail estimates from IOReport (docs/energy.md); they are never compared with
NVML / datacenter GPU numbers.

MLX_MAX_MB_PER_BUFFER: MLX starts a new Metal command buffer once the arrays referenced by the current one exceed this
many MB. A memory-mapped expert array is one ~400 MB buffer per layer and part, so with MLX's default nearly every MoE
op got its own command buffer: gpt-oss-20b decode 45-50 ms/token instead of 29-30 (RESULTS 1p). Too large a value keeps
more temporaries alive per buffer: a 2048-token Qwen3.5-9B prefill ran out of Metal memory at 400 and above (works at
40-250). 150 keeps both. MLX reads the variable once, when mlx.core is first imported; set it before that (the tools
here do; importing this package first also does).
"""

import os as _os
import sys as _sys

__version__ = "0.1.0.dev0"

MAX_MB_PER_BUFFER = "150"
# MLX re-requests residency of its wired set every this many ms. At MLX's default the set lapsed during a 5 s idle and
# the next request paid to re-wire it: OLMoE-1B-7B pp512 1018-1048 vs 1130-1140 tok/s at 200 (RESULTS 1zd).
RESIDENCY_REFRESH_MS = "200"
if "mlx.core" not in _sys.modules:
    _os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", MAX_MB_PER_BUFFER)
    _os.environ.setdefault("MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS", RESIDENCY_REFRESH_MS)
elif "MLX_MAX_MB_PER_BUFFER" not in _os.environ:
    import warnings as _warnings

    _warnings.warn("mlx.core was imported before mj_mlx without MLX_MAX_MB_PER_BUFFER set: "
                   "memory-mapped expert models decode ~1.6x slower. Set MLX_MAX_MB_PER_BUFFER=150 before "
                   "importing mlx.", stacklevel=2)

from .gpt_oss import GptOss, GptOssConfig  # noqa: E402
from .qwen3_5_moe import Qwen35Moe, Qwen35MoeConfig  # noqa: E402
from .runner import Engine, MlxRunner, load_model, step_arrays  # noqa: E402

__all__ = ["Engine", "GptOss", "GptOssConfig", "MlxRunner", "Qwen35Moe", "Qwen35MoeConfig", "__version__",
           "load_model", "step_arrays"]

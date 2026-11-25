# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""python -m mj_mlx <command>: `cache` (list / rm what mj-mlx keeps on disk, mj_mlx.cache)."""

import sys


def main() -> int:
    cmds = {"cache": "mj_mlx.cache"}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(f"usage: python -m mj_mlx {{{','.join(cmds)}}} ...", file=sys.stderr)
        return 2
    import importlib

    return importlib.import_module(cmds[sys.argv[1]]).main(sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())

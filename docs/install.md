# Installing mj-mlx (from source)

mj-mlx is not on PyPI yet. These steps build it from a checkout of this repository.

## Requirements

- A Mac with Apple silicon (M1 or later). It is tested on an M4 MacBook Pro with 16 GB of memory.
- macOS with the Xcode command-line tools: `xcode-select --install`.
- Python 3.11. Only 3.11 is tested so far.
- A Rust toolchain ([rustup](https://rustup.rs)) for the scheduler.
- MLX 0.32.x. The 4-bit small-batch kernel (`qmm_small.py`) compiles against the installed MLX headers, so the
  package pins the tested minor version.

## Steps

From the repository root:

```bash
python3.11 -m venv .venv
```

```bash
.venv/bin/pip install "maturin>=1.7,<2"
```

Build the scheduler (Rust, `crates/`) and install mj-mlx in editable mode, with its dependencies (mlx, numpy):

```bash
source ~/.cargo/env && .venv/bin/maturin develop --release
```

Check the install:

```bash
.venv/bin/python -c "import mj_mlx, mj_mlx.scheduler as s; print(mj_mlx.__version__, s.Scheduler)"
```

## Optional: the C++ streaming extension

The extension is only needed for `MJ_GPU_STREAM=1`. That mode routes the decode steps of models bigger than memory
through a C++ streamer, with no host sync per step. It is a prototype, and the Python streamer is the default. To
build it you also need CMake 3.27 or later and nanobind (`pip install nanobind`):

```bash
cmake -S src/mj_mlx/ext -B src/mj_mlx/ext/build -DPython_EXECUTABLE=$PWD/.venv/bin/python -DMJ_NB_DOMAIN=mlx
```

```bash
cmake --build src/mj_mlx/ext/build
```

`MJ_NB_DOMAIN` must match the nanobind domain your MLX build uses (`mlx` for the PyPI wheels).

## Tests

The test extra adds `pytest`, `torch`, `transformers>=5.23` and `safetensors`:

```bash
.venv/bin/pip install -e ".[test]"
```

```bash
.venv/bin/python -m pytest tests -q
```

The scheduler's own Rust tests run with `cargo test`.

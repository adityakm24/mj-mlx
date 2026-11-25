"""Test-wide settings."""

import os

# The prepared-weights cache is on by default (src/mj_mlx/gpt_oss.py, PrepCache); tests that build models would
# otherwise write into the user's ~/Library/Caches. tests/test_prep_cache.py turns it on for itself in a tmp dir.
os.environ.setdefault("MJ_PREP_CACHE", "0")

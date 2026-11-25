"""Draft vocabulary for MTP speculative decoding: the V most frequent token ids over local text, plus every added /
special token (chat markers, EOS), saved as int32 .npy (spec_decode.MtpHead draft_vocab).

The draft head only proposes tokens; the main model checks every one, so a token outside the set only costs a
rejected draft, never a wrong output.

    python tools/mlx_draft_vocab.py --tokenizer <model dir> --size 65536 --out ~/.cache/millijoule/Qwen3.5-9B-mtp/vocab65k.npy
"""

import argparse
import glob
import os
import random
import sys

import numpy as np

DEFAULT_GLOBS = [
    "/Library/Frameworks/Python.framework/Versions/*/lib/python3*/**/*.py",
    "/usr/share/man/man1/*.1",
    os.path.join(sys.prefix, "lib/python3*/site-packages/**/*.py"),
    os.path.join(sys.prefix, "lib/python3*/site-packages/**/*.md"),
    os.path.join(sys.prefix, "lib/python3*/site-packages/**/METADATA"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--size", type=int, default=65536)
    ap.add_argument("--globs", nargs="*", default=DEFAULT_GLOBS)
    ap.add_argument("--max-files", type=int, default=6000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    files = [f for g in a.globs for f in glob.glob(g, recursive=True)]
    random.Random(0).shuffle(files)
    n = max(len(tok), max(tok.added_tokens_decoder, default=0) + 1)
    cnt = np.zeros(n, np.int64)
    total = 0
    for f in files[:a.max_files]:
        try:
            t = open(f, errors="ignore").read()[:200000]
        except OSError:
            continue
        ids = np.array(tok(t)["input_ids"], np.int64)
        np.add.at(cnt, ids, 1)
        total += len(ids)
    special = np.array(sorted(tok.added_tokens_decoder), np.int64)
    order = [i for i in np.argsort(-cnt, kind="stable") if i not in set(special.tolist())]
    ids = np.unique(np.concatenate([special, np.array(order[:a.size - len(special)], np.int64)]))
    os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(a.out))), exist_ok=True)
    np.save(os.path.expanduser(a.out), ids.astype(np.int32))
    print(f"{len(ids)} ids ({len(special)} special) from {total} tokens in {min(len(files), a.max_files)} files "
          f"-> {a.out}")


if __name__ == "__main__":
    main()

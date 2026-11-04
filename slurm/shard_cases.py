#!/usr/bin/env python3
"""Write one shard of a JSON Lines case file.

The evaluation harness has no built-in sharding flag: a full run is sharded by
splitting the case list, giving each batch task its own output directory, and
merging the per-case files afterwards.  Splitting by line index keeps the shards
disjoint and stable, so a resubmission recomputes exactly the same rows.

Usage:
    python slurm/shard_cases.py data/longctx_max.jsonl.gz out/cases_s3.jsonl 3 32

    <input>   case file, .jsonl or .jsonl.gz
    <output>  shard file, written as plain .jsonl
    <index>   zero-based shard index (SLURM_ARRAY_TASK_ID)
    <count>   number of shards (SLURM_ARRAY_TASK_COUNT)
"""
import gzip
import io
import os
import sys


def main(argv):
    if len(argv) != 5:
        sys.exit(__doc__)
    src, dst, idx, count = argv[1], argv[2], int(argv[3]), int(argv[4])
    if count < 1 or not 0 <= idx < count:
        sys.exit("shard index %d out of range for %d shards" % (idx, count))

    op = gzip.open if src.endswith(".gz") else io.open
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)

    n_in = n_out = 0
    with op(src, "rt", encoding="utf-8") as fin, \
            io.open(dst, "w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            if not line.strip():
                continue
            if n_in % count == idx:
                fout.write(line if line.endswith("\n") else line + "\n")
                n_out += 1
            n_in += 1

    print("shard %d/%d: %d of %d cases -> %s" % (idx, count, n_out, n_in, dst))


if __name__ == "__main__":
    main(sys.argv)

#!/usr/bin/env python3
"""Tiny reader for inference/checkpoints.json (stdlib, python >= 3.6).

    manifest.py --list             -> arm keys, one per line
    manifest.py KEY FIELD [DFLT]   -> that field (lists join with spaces;
                                      missing field prints DFLT or nothing)
"""
import json
import os
import sys
from pathlib import Path

M = {k: v for k, v in json.loads(
    (Path(__file__).resolve().parent / "checkpoints.json").read_text()
).items() if not k.startswith("_")}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--list":
        print("\n".join(M))
        return
    key, field = sys.argv[1], sys.argv[2]
    dflt = sys.argv[3] if len(sys.argv) > 3 else ""
    if key not in M:
        sys.exit("unknown arm %r -- see: manifest.py --list" % key)
    v = M[key].get(field, dflt)
    if isinstance(v, list):
        v = " ".join(str(x) for x in v)
    # Registry paths may be written against ${ASL2LSC_*} roots.
    print(os.path.expandvars(str(v)))


if __name__ == "__main__":
    main()

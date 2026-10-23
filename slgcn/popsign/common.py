"""Shared helpers: run cards, hashing, logging."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from data_config import REPO_ROOT


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_run_card(out_dir: Path, card: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_card.json", "w") as f:
        json.dump(card, f, indent=2, default=str)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

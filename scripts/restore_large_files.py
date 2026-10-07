"""Restore large artifacts from Git-friendly parts: python scripts/restore_large_files.py."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "large_files_manifest.json").read_text(encoding="utf-8"))
for item in manifest:
    target = ROOT / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with target.open("wb") as output:
        for part in item["parts"]:
            with (ROOT / part).open("rb") as source:
                while block := source.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
    if digest.hexdigest() != item["sha256"]:
        raise RuntimeError(f"Checksum mismatch: {target}")
    print(f"Restored and verified: {item['path']}")

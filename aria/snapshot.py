"""Frozen benchmark snapshot utilities."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aria.library.store import Library
from aria.program_store import ProgramStore


def write_snapshot(
    snapshot_dir: str | Path,
    *,
    program_store: ProgramStore,
    library: Library,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a frozen runtime snapshot directory."""
    root = Path(snapshot_dir)
    root.mkdir(parents=True, exist_ok=True)

    program_store_path = root / "program_store.json"
    library_path = root / "library.json"
    manifest_path = root / "manifest.json"

    program_store.save_json(program_store_path)
    library.save_json(library_path)

    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "program_count": len(program_store),
        "library_entry_count": len(library.all_entries()),
        "files": {
            "program_store": {
                "path": program_store_path.name,
                "sha256": _sha256(program_store_path),
            },
            "library": {
                "path": library_path.name,
                "sha256": _sha256(library_path),
            },
        },
        "metadata": metadata or {},
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

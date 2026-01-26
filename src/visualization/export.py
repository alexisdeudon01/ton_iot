from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable


def copy_graphs(src: Path, dest: Path, exts: Iterable[str] = (".png", ".svg", ".pdf", ".md")) -> int:
    count = 0
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        rel = path.relative_to(src)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        count += 1
    return count

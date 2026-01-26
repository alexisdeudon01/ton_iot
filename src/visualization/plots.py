from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable


def _copy_tree(src: Path, dest: Path, exts: Iterable[str]) -> int:
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


def generate_results_figures(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from graph.results.results_figures import Results

        results = Results(output_dir)
        results.generate_all()
    except Exception as exc:
        print(f"[visualization] Warning: results figures generation skipped: {exc}")


def generate_all_plots(output_dir: str | Path = "graphs", include_existing: bool = True) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if include_existing:
        src_graph = Path("graph")
        if src_graph.exists():
            _copy_tree(src_graph, output_dir, exts={".png", ".svg", ".pdf", ".md"})

    results_dir = output_dir / "results"
    generate_results_figures(results_dir)

    return output_dir

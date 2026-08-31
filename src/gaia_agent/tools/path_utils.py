from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

# Directories that must never be scanned when looking for
# evaluation attachments (they contain vendored dependencies).
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "site-packages",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)

# Paths that are clearly LLM-invented placeholders.
_PLACEHOLDER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<[^>]*>"),
    re.compile(r"\[[^\]]*path[^\]]*\]", re.IGNORECASE),
    re.compile(r"(file|path|your|actual|real|attached|training)[_\s\\/]*(path|file|name)", re.IGNORECASE),
    re.compile(r"placeholder", re.IGNORECASE),
    re.compile(r"^(?:\d+_)?path(_to|_of)?[\w-]*$", re.IGNORECASE),
    re.compile(r"^[fp]ile_path$", re.IGNORECASE),
)


def is_placeholder_path(path: str) -> bool:
    """
    Return True when `path` looks like an LLM-invented placeholder
    (e.g. ``<file_path_to_sales_file>``, ``[FILE_PATH]``).
    """
    if not path or not isinstance(path, str):
        return True

    normalized = path.strip()

    if not normalized:
        return True

    if normalized in {".", "..", "/"}:
        return True

    return any(
        pattern.search(normalized)
        for pattern in _PLACEHOLDER_PATTERNS
    )


def _iter_search_roots(base_dir: str | Path) -> list[Path]:
    """
    Candidate roots: the configured base directory plus the
    standard GAIA evaluation locations (cwd and cwd/src).
    """
    base = Path(base_dir).resolve()
    cwd = Path.cwd().resolve()

    roots: list[Path] = []

    for candidate in (base, base / "gaia_agent", cwd, cwd / "src"):
        if candidate.is_dir() and candidate not in roots:
            roots.append(candidate)

    return roots


def _walk_files(
    root: Path,
    max_depth: int,
    depth: int = 0,
) -> Iterable[Path]:
    if depth >= max_depth:
        return

    try:
        entries = sorted(
            root.iterdir(),
            key=lambda p: p.name.lower(),
        )
    except OSError:
        return

    for entry in entries:
        if entry.is_dir():
            if entry.name in _SKIP_DIRS:
                continue
            yield from _walk_files(
                root=entry,
                max_depth=max_depth,
                depth=depth + 1,
            )
        elif entry.is_file():
            yield entry


def resolve_file(
    base_dir: str | Path,
    file_path: str,
    max_depth: int = 4,
) -> Path | None:
    """
    Resolve a file reference to an existing file.

    Lookup order:
      1. direct candidates (base_dir/reference, cwd/reference,
         cwd/src/reference, and the reference itself),
      2. recursive search by exact basename inside every search root
         (so an attachment living in a sub-folder or in
         ``src/gaia_agent`` is still found).

    Returns the resolved Path or None when the file does not exist
    anywhere. Never invents a path.
    """
    if is_placeholder_path(file_path):
        return None

    reference = file_path.strip().lstrip(
        "./\\"
    )
    stripped = reference.strip('"\'')
    if stripped != reference:
        reference = stripped

    filename = Path(reference).name

    roots = _iter_search_roots(base_dir)

    # 1) Direct candidates.
    direct_candidates: list[Path] = []
    for root in roots:
        direct_candidates.append(root / reference)
        if filename and filename != reference:
            direct_candidates.append(root / filename)

    try:
        direct_candidates.append(Path(reference))
        if filename and filename != reference:
            direct_candidates.append(Path(filename))
    except (OSError, ValueError):
        pass

    seen: set[Path] = set()
    for candidate in direct_candidates:
        try:
            resolved = candidate.resolve()
        except (OSError, ValueError):
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved

    # 2) Recursive search by exact basename.
    if filename:
        for root in roots:
            for found in _walk_files(root, max_depth=max_depth):
                if (
                    found.name.lower() == filename.lower()
                    and found.resolve() not in seen
                ):
                    seen.add(found.resolve())
                    return found.resolve()

    return None


def list_available_files(
    base_dir: str | Path,
    max_depth: int = 3,
    max_items: int = 100,
) -> list[str]:
    """
    List the real data files that an agent could legitimately read,
    for the planner prompt. Only data-like files are listed (Excel,
    CSV, PDF, images, media); source code, logs and project artifacts
    are excluded so the planner never mistakes them for task
    attachments.
    """
    DATA_SUFFIXES = {
        ".csv", ".tsv", ".xlsx", ".xls", ".xlsm",
        ".pdf", ".png", ".jpg", ".jpeg", ".webp",
        ".bmp", ".gif", ".mp3", ".wav", ".mp4",
        ".webm", ".m4a", ".docx", ".doc", ".pptx",
    }

    roots = _iter_search_roots(base_dir)
    found: list[str] = []

    for root in roots:
        for candidate in _walk_files(root, max_depth=max_depth):
            parts = candidate.parts
            if "HuggingFace_Results" in parts or ".venv" in parts:
                continue
            if candidate.suffix.lower() not in DATA_SUFFIXES:
                continue
            relative = candidate.relative_to(root)
            found.append(str(relative))

    deduplicated: list[str] = []
    seen: set[str] = set()

    for path in found:
        if path in seen:
            continue
        seen.add(path)
        deduplicated.append(path)

    return sorted(deduplicated)[:max_items]
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "pycache",
        "site-packages",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)


_PLACEHOLDER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<[^>]*>"),
    re.compile(r"\[[^\]]*path[^\]]*\]", re.IGNORECASE),
    re.compile(
        r"(file|path|your|actual|real|attached|training)"
        r"[_\s\\/]*(path|file|name)",
        re.IGNORECASE,
    ),
    re.compile(r"placeholder", re.IGNORECASE),
    re.compile(
        r"^(?:\d+_)?path(_to|_of)?[\w-]*$",
        re.IGNORECASE,
    ),
    re.compile(r"^[fp]ile_path$", re.IGNORECASE),
)


def is_placeholder_path(path: str) -> bool:
    if not isinstance(path, str):
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
    base = Path(base_dir).resolve()
    cwd = Path.cwd().resolve()

    roots: list[Path] = []

    for candidate in (
        base,
        base / "gaia_agent",
        cwd,
        cwd / "src",
    ):
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

    if is_placeholder_path(file_path):
        return None

    reference = file_path.strip()

    reference = reference.strip("\"'")

    while reference.startswith(("./", ".\\", "/")):
        reference = reference[1:]

    if not reference:
        return None

    roots = _iter_search_roots(base_dir)

    filename = Path(reference).name

    direct_candidates: list[Path] = []

    for root in roots:
        direct_candidates.append(root / reference)

    try:
        raw = Path(reference)
        direct_candidates.append(raw)
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

    if not filename:
        return None

    matches: list[Path] = []

    for root in roots:
        for found in _walk_files(
            root,
            max_depth=max_depth,
        ):
            try:
                resolved = found.resolve()
            except (OSError, ValueError):
                continue

            if resolved in seen:
                continue

            if found.name.lower() == filename.lower():
                matches.append(resolved)
                seen.add(resolved)

    if not matches:
        return None

    unique_matches = sorted(
        set(matches),
        key=lambda p: str(p).lower(),
    )

    if len(unique_matches) > 1:
        raise FileExistsError(
            "Ambiguous file reference. Multiple files match "
            f"'{file_path}': "
            + ", ".join(str(p) for p in unique_matches)
        )

    return unique_matches[0]


def list_available_files(
    base_dir: str | Path,
    max_depth: int = 3,
    max_items: int = 100,
) -> list[str]:
    
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
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4


@dataclass(slots=True)
class ArtifactInfo:
    """
    Explicit artifact descriptor.

    AGENTS MUST REFERENCE ARTIFACTS BY artifact_id, NEVER by
    guessing a filename. This replaces the failure mode where the
    agent assumed 'search_results.png' or 'image.png' existed.
    """

    artifact_id: str
    artifact_type: str = "text"
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        *,
        artifact_type: str = "text",
        path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ArtifactInfo":
        return cls(
            artifact_id=str(uuid4()),
            artifact_type=artifact_type,
            path=path,
            metadata=dict(metadata or {}),
        )


@dataclass(slots=True)
class ToolResultRecord:
    """
    Normalized, evidence-grade record of one tool execution.

    Recorded on EVERY step so the verifier can check the final
    answer against the real evidence trail.
    """

    step_id: int | None
    tool_name: str
    arguments: dict[str, Any]
    result: Any
    succeeded: bool
    error: str | None = None
    artifacts: list[ArtifactInfo] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ArtifactRegistry:
    """
    Tracks produced artifacts and lets tools find evidence by
    artifact_id or a resolved path (with forgiving lookups).
    """

    def __init__(self) -> None:
        self._artifacts: dict[str, ArtifactInfo] = {}

    def register(
        self,
        artifact: ArtifactInfo,
    ) -> ArtifactInfo:
        self._artifacts[artifact.artifact_id] = artifact
        return artifact

    def get(
        self,
        artifact_id: str,
    ) -> ArtifactInfo | None:
        return self._artifacts.get(artifact_id)

    def resolve_path(
        self,
        reference: str,
        *,
        base_dir: str = ".",
    ) -> str | None:
        """
        Resolve a file reference to an absolute path if it exists.

        Used by file/image tools to find GAIA-provided assets that
        live in any of the standard evaluation locations.
        """
        if not reference:
            return None

        base = Path(base_dir).resolve()

        filename = Path(reference).name

        candidates = [
            base / reference,
            base / filename,
            Path.cwd() / filename,
            Path.cwd() / "src" / filename,
            Path(reference),
        ]

        for candidate in candidates:

            try:
                if candidate.exists() and candidate.is_file():
                    return str(candidate.resolve())
            except OSError:
                continue

        return None

    def all(self) -> list[ArtifactInfo]:
        return list(self._artifacts.values())

    def clear(self) -> None:
        self._artifacts.clear()
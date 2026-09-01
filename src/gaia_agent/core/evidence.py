from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class ArtifactInfo:
  
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
    step_id: int | None
    tool_name: str
    arguments: dict[str, Any]
    result: Any
    succeeded: bool
    error: str | None = None
    artifacts: list[ArtifactInfo] = field(default_factory=list)
    evidence_type: str = "unknown"
    source: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
class ArtifactRegistry:
    def __init__(self) -> None:
        self._artifacts: dict[str, ArtifactInfo] = {}

    def register(self, artifact: ArtifactInfo) -> ArtifactInfo:
        if not artifact.artifact_id:
            raise ValueError("artifact_id must not be empty")

        self._artifacts[artifact.artifact_id] = artifact
        return artifact

    def register_many(
        self,
        artifacts: list[ArtifactInfo],
    )-> list[ArtifactInfo]:
        for artifact in artifacts:
            self.register(artifact)

        return artifacts

    def get(self, artifact_id: str) -> ArtifactInfo | None:
        if not artifact_id:
            return None

        return self._artifacts.get(artifact_id)

    def require(self, artifact_id: str) -> ArtifactInfo:
        artifact = self.get(artifact_id)
        if artifact is None:
            raise KeyError(
                f"Unknown artifact_id: {artifact_id}"
            )
        return artifact
    def resolve_artifact_path(
        self,
        artifact_id: str,
    ) -> str | None:
        artifact = self.get(artifact_id)
        if artifact is None:
            return None
        if not artifact.path:
            return None
        path = Path(artifact.path)
        try:
            if path.exists() and path.is_file():
                return str(path.resolve())
        except OSError:
            return None
        return None
    def require_artifact_path(self, artifact_id: str) -> str:
        artifact = self.require(artifact_id)
        if not artifact.path:
            raise FileNotFoundError(
                f"Artifact '{artifact_id}' has no registered path"
            )

        path = Path(artifact.path)

        try:
            if not path.exists():
                raise FileNotFoundError(
                    f"Artifact '{artifact_id}' path does not exist: "
                    f"{artifact.path}"
                )
            if not path.is_file():
                raise FileNotFoundError(
                    f"Artifact '{artifact_id}' path is not a file: "
                    f"{artifact.path}"
                )

            return str(path.resolve())

        except OSError as exc:
            raise FileNotFoundError(
                f"Unable to access artifact '{artifact_id}': "
                f"{artifact.path}"
            ) from exc

    def resolve_path(
        self,
        reference: str,
        *,
        base_dir: str = ".",
    ) -> str | None:
        if not reference:
            return None

        base = Path(base_dir).resolve()
        reference_path = Path(reference)

        candidates: list[Path] = []

        if not reference_path.is_absolute():
            candidates.append(base / reference_path)

        candidates.append(reference_path)

        filename = reference_path.name
        candidates.extend(
            [
                base / filename,
                Path.cwd() / filename,
                Path.cwd() / "src" / filename,
            ]
        )

        seen: set[str] = set()

        for candidate in candidates:
            try:
                resolved = str(candidate.resolve())

                if resolved in seen:
                    continue

                seen.add(resolved)

                if candidate.exists() and candidate.is_file():
                    return resolved

            except OSError:
                continue

        return None

    def all(self) -> list[ArtifactInfo]:
        return list(self._artifacts.values())

    def contains(self, artifact_id: str) -> bool:
        return artifact_id in self._artifacts

    def remove(self, artifact_id: str) -> ArtifactInfo | None:
        return self._artifacts.pop(artifact_id, None)

    def clear(self) -> None:
        self._artifacts.clear()

    def len(self) -> int:
        return len(self._artifacts)
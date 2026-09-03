from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from smolagents import Tool

from gaia_agent.tools.path_utils import (
    is_placeholder_path,
    resolve_file,
)


SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".webm",
    ".aac",
    ".wma",
}


class STTBackend(Protocol):
    def transcribe(
        self,
        audio_path: Path,
    ) -> str:
        ...


class FasterWhisperBackend:
    """
    Local speech-to-text backend using faster-whisper.

    The Whisper model is loaded lazily on first transcription.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root

        self._model: Any = None
        self._lock = Lock()

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "Local audio transcription requires "
                    "'faster-whisper'. Install project "
                    "dependencies with: pip install -e ."
                ) from exc

            kwargs: dict[str, Any] = {
                "device": self.device,
                "compute_type": self.compute_type,
            }

            if self.download_root is not None:
                kwargs["download_root"] = self.download_root

            self._model = WhisperModel(
                self.model_size,
                **kwargs,
            )

            return self._model

    def transcribe(
        self,
        audio_path: Path,
    ) -> str:
        model = self._get_model()

        segments, _info = model.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=True,
        )

        parts: list[str] = []

        for segment in segments:
            text = str(segment.text).strip()

            if text:
                parts.append(text)

        transcript = " ".join(parts).strip()

        if not transcript:
            raise ValueError(
                "The speech-to-text backend returned "
                "an empty transcript."
            )

        return transcript


class TranscribeAudioTool(Tool):
    """
    Transcribe a local audio file into text.
    """

    name = "transcribe_audio"

    description = (
        "Transcribe a local audio file into text using a "
        "local speech-to-text model. Supports WAV, MP3, "
        "M4A, FLAC, OGG, WEBM, AAC and WMA."
    )

    inputs = {
        "audio_path": {
            "type": "string",
            "description": (
                "Path to the audio file relative to the "
                "allowed base directory or filename."
            ),
        },
    }

    output_type = "string"

    def __init__(
        self,
        stt_backend: STTBackend,
        base_dir: str = ".",
    ) -> None:
        super().__init__()

        if stt_backend is None:
            raise ValueError(
                "TranscribeAudioTool requires an STT backend."
            )

        self.stt_backend = stt_backend
        self.base_dir = Path(base_dir).resolve()

    def forward(
        self,
        audio_path: str,
    ) -> str:
        try:
            if (
                not isinstance(audio_path, str)
                or not audio_path.strip()
            ):
                return (
                    "Error: audio_path must be a "
                    "non-empty string."
                )

            if is_placeholder_path(audio_path):
                return (
                    f"Error: Audio path '{audio_path}' "
                    "is a placeholder or invalid."
                )

            path = resolve_file(
                self.base_dir,
                audio_path,
            )

            if path is None:
                return (
                    f"Error: Audio '{audio_path}' "
                    "was not found in the allowed "
                    "search locations."
                )

            if not path.exists():
                return (
                    f"Error: Audio '{audio_path}' "
                    "does not exist."
                )

            if not path.is_file():
                return (
                    f"Error: '{audio_path}' is not a file."
                )

            extension = path.suffix.lower()

            if extension not in SUPPORTED_AUDIO_EXTENSIONS:
                return (
                    f"Error: Unsupported audio format "
                    f"'{extension}'. Supported formats: "
                    f"{', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}."
                )

            transcript = self.stt_backend.transcribe(path)

            transcript = str(transcript).strip()

            if not transcript:
                return (
                    "Error: Speech-to-text returned "
                    "an empty transcript."
                )

            return transcript

        except Exception as exc:
            return (
                "Error transcribing audio: "
                f"{type(exc).__name__}: {exc}"
            )


class AudioTools:
    def __init__(
        self,
        stt_backend: STTBackend | None = None,
        base_dir: str = ".",
        *,
        stt_model_size: str = "base",
        stt_device: str = "cpu",
        stt_compute_type: str = "int8",
    ) -> None:
        self.base_dir = Path(base_dir).resolve()

        self.stt_backend = stt_backend or FasterWhisperBackend(
            model_size=stt_model_size,
            device=stt_device,
            compute_type=stt_compute_type,
        )

    def get_tools(self) -> list[Tool]:
        tools: list[Tool] = [
            TranscribeAudioTool(
                stt_backend=self.stt_backend,
                base_dir=str(self.base_dir),
            )
        ]

        try:
            from smolagents import YoutubeTranscriptTool
        except ImportError:
            YoutubeTranscriptTool = None

        if YoutubeTranscriptTool is not None:
            tools.append(
                YoutubeTranscriptTool()
            )

        return tools
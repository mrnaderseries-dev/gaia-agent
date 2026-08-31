from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class TaskIntent(str, Enum):
    ARITHMETIC = "arithmetic"
    TEXT_TRANSFORMATION = "text_transformation"
    LOCAL_FILE = "local_file"
    IMAGE = "image"
    AUDIO_VIDEO = "audio_video"
    URL_PAGE = "url_page"
    FACTUAL_SEARCH = "factual_search"
    SELF_CONTAINED = "self_contained"
    UNKNOWN = "unknown"


_ARITHMETIC_KEYWORDS = (
    "calculate", "compute", "factorial", "divided by", "divide",
    "multiply", "multiplied", "times", "subtract", "subtracted",
    "added", "add", "sum of", "difference between", "modulo",
    "prime", "largest prime", "smallest prime", "square root",
    "cubed", "squared", "plus", "minus",
)

_TEXT_TRANSFORM_KEYWORDS = (
    "reverse", "reversed", "backwards", "anagram", "sort",
    "alphabetical", "alphabetize", "letter", "letters", "word count",
    "count the", "encode", "decode", "cipher", "opposite of ",
    "rewrite", "repeat", "caesar", "character", "uppercase",
    "lowercase", "capitalize", "palindrome", "transposition",
)

_FILE_KEYWORDS = (
    "file", "attachment", "attached", "spreadsheet", "excel", "csv",
    "workbook", "xlsx", "dataset", "data file", "table in the file",
    "these files",
)

_IMAGE_KEYWORDS = (
    "image", "picture", "photo", "photograph", "diagram", "chart",
    "graph", "chess board", "chessboard", "screenshot", "figure",
    "logo", "map",
)

_AUDIO_VIDEO_KEYWORDS = (
    "video", "audio", "mp3", "mp4", "recording", "podcast", "listen",
    "voice memo", "song", "youtube", "youtu.be", "transcript",
)


@dataclass(frozen=True, slots=True)
class TaskAnalysis:
    """Deterministic task-type analysis used to guide the planner."""

    intent: TaskIntent
    needs_external_info: bool
    recommended_first_tool: str | None
    forbidden_tools: tuple[str, ...]
    analysis_text: str


_FACTORIAL_RATIO_RE = re.compile(
    r"(\d+)(?:\s*!|\s+factorial)\s*(?:divided\s+by|/|over|by)\s*"
    r"(\d+)(?:\s*!|\s+factorial)",
    re.IGNORECASE,
)

_OPERATION_RE = re.compile(
    r"what\s+is\s+([-+]?\d+(?:[.,]\d+)?)\s*([+\-*/x×])\s*"
    r"([-+]?\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)

_VERBAL_OPERATION_RE = re.compile(
    r"([-+]?\d+(?:[.,]\d+)?)\s+(multiplied\s+by|times|divided\s+by|"
    r"plus|minus|added\s+to|subtracted\s+from)\s+"
    r"([-+]?\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)


def detect_factorial_ratio(question: str) -> tuple[int, int] | None:
    """Return (a, b) when the question asks for a!/b!."""
    match = _FACTORIAL_RATIO_RE.search(question)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def detect_simple_operation(question: str) -> str | None:
    """Return a safe python expression for simple two-operand arithmetic."""
    match = _OPERATION_RE.search(question)
    if match:
        op = match.group(2)
        if op in ("x", "×"):
            op = "*"
        return _build_expression(
            match.group(1), op, match.group(3)
        )

    match = _VERBAL_OPERATION_RE.search(question)
    if match:
        verb = match.group(2).lower()
        if "multiplied" in verb or verb == "times":
            op = "*"
        elif "divided" in verb:
            op = "/"
        elif "plus" in verb or "added" in verb:
            op = "+"
        else:
            op = "-"
        return _build_expression(
            match.group(1), op, match.group(3)
        )

    return None


def _build_expression(
    left: str,
    op: str,
    right: str,
) -> str | None:
    try:
        left_value = int(left)
    except ValueError:
        left_value = float(left)
    try:
        right_value = int(right)
    except ValueError:
        right_value = float(right)
    if op == "/" and right_value == 0:
        return None
    expression = f"{left_value} {op} {right_value}"
    try:
        compile(expression, "<classifier>", "eval")
    except SyntaxError:
        return None
    return expression


class TaskClassifier:
    """Deterministic, keyword-driven task classification."""

    def classify(
        self,
        question: str,
        *,
        available_files: Sequence[str] | None = None,
        available_tools: Sequence[str] | None = None,
    ) -> TaskAnalysis:
        text = (question or "").strip().lower()
        available = set(available_tools or [])
        files = list(available_files or [])

        has_python = "python_interpreter" in available
        has_web = "web_search" in available
        has_visit = "visit_webpage" in available
        has_excel = "analyze_excel" in available
        has_reader = "file_reader" in available
        has_image = "analyze_image" in available

        # 1) Arithmetic
        if (
            detect_factorial_ratio(question) is not None
            or detect_simple_operation(question) is not None
            or any(keyword in text for keyword in _ARITHMETIC_KEYWORDS)
        ) and _has_digit_or_math(text):
            return TaskAnalysis(
                intent=TaskIntent.ARITHMETIC,
                needs_external_info=False,
                recommended_first_tool=(
                    "python_interpreter" if has_python else None
                ),
                forbidden_tools=("web_search", "visit_webpage"),
                analysis_text=(
                    "The task is ARITHMETIC. Compute the exact result. "
                    "Prefer a python_interpreter step that calculates "
                    "the value in code (standard library only). Never "
                    "use web_search for arithmetic."
                ),
            )

        # 2) Text transformation
        #
        # ROOT-CAUSE FIX: GAIA presents some tasks ENTIRELY REVERSED
        # (".rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ...").
        # The transformation keywords ("opposite of", "write the word",
        # ...) are then invisible to the normal keyword scan, the task
        # was misclassified as factual and the planner fired a useless
        # web_search. Also test the REVERSED text so a reversal task is
        # recognized as self-contained text transformation.
        if (
            any(
                keyword in text
                for keyword in _TEXT_TRANSFORM_KEYWORDS
            )
            or any(
                keyword in text[::-1]
                for keyword in _TEXT_TRANSFORM_KEYWORDS
            )
        ):
            return TaskAnalysis(
                intent=TaskIntent.TEXT_TRANSFORMATION,
                needs_external_info=False,
                recommended_first_tool=(
                    "python_interpreter" if has_python else None
                ),
                forbidden_tools=("web_search", "visit_webpage"),
                analysis_text=(
                    "The task is a TEXT TRANSFORMATION (possibly "
                    "presented in reversed form). Perform the exact "
                    "transformation (reverse, count, sort, encode, "
                    "decode, ...) with code or deterministic "
                    "reasoning. Web search is useless here."
                ),
            )

        # 3) Local file / spreadsheet / table data
        if any(keyword in text for keyword in _FILE_KEYWORDS):
            if files:
                excel_hit = any(
                    suffix in name.lower()
                    for name in files
                    for suffix in (".xlsx", ".xls", ".xlsm", ".csv")
                )
                preferred = (
                    "analyze_excel"
                    if (excel_hit and has_excel)
                    else ("file_reader" if has_reader else None)
                )
                return TaskAnalysis(
                    intent=TaskIntent.LOCAL_FILE,
                    needs_external_info=False,
                    recommended_first_tool=preferred,
                    forbidden_tools=(),
                    analysis_text=(
                        "The task refers to LOCAL FILES. Real file "
                        "paths are listed in AVAILABLE LOCAL FILES. "
                        "Never invent file paths; use an existing file."
                    ),
                )
            return TaskAnalysis(
                intent=TaskIntent.LOCAL_FILE,
                needs_external_info=True,
                recommended_first_tool=None,
                forbidden_tools=(),
                analysis_text=(
                    "The task mentions a file/dataset, but no matching "
                    "file exists in the environment. Do NOT invent a "
                    "file path and do NOT retry a missing file. Use an "
                    "alternative strategy (reasoning or web evidence)."
                ),
            )

        return _classify_media_and_web(
            question=question,
            text=text,
            has_web=has_web,
            has_visit=has_visit,
            has_image=has_image,
        )


def _classify_media_and_web(
    *,
    question: str,
    text: str,
    has_web: bool,
    has_visit: bool,
    has_image: bool,
) -> TaskAnalysis:
    # 4) Image
    if any(keyword in text for keyword in _IMAGE_KEYWORDS):
        return TaskAnalysis(
            intent=TaskIntent.IMAGE,
            needs_external_info=True,
            recommended_first_tool=(
                "analyze_image" if has_image else None
            ),
            forbidden_tools=(),
            analysis_text=(
                "The task involves an IMAGE. Use analyze_image "
                "with a real existing image path when available; "
                "otherwise reason from the question or search for "
                "the underlying facts."
            ),
        )

    # 5) Audio / video
    if any(keyword in text for keyword in _AUDIO_VIDEO_KEYWORDS):
        return TaskAnalysis(
            intent=TaskIntent.AUDIO_VIDEO,
            needs_external_info=True,
            recommended_first_tool=None,
            forbidden_tools=(),
            analysis_text=(
                "The task involves AUDIO/VIDEO. Use a transcript "
                "tool or local media file if available; otherwise "
                "the alternative is a focused web search for the "
                "specific facts."
            ),
        )

    # 6) URL navigation
    url_match = re.search(
        r"https?://[^\s<>]+",
        question or "",
        re.IGNORECASE,
    )
    if url_match:
        target = url_match.group(0)
        if (
            "youtube.com" in target.lower()
            or "youtu.be" in target.lower()
        ):
            return TaskAnalysis(
                intent=TaskIntent.AUDIO_VIDEO,
                needs_external_info=True,
                recommended_first_tool=(
                    "web_search" if has_web else None
                ),
                forbidden_tools=(),
                analysis_text=(
                    "The task references a YouTube video. Visiting "
                    "YouTube via visit_webpage usually fails with "
                    "403/consent walls. Prefer a focused web_search "
                    "for the specific fact asked about this video."
                ),
            )
        if has_visit:
            return TaskAnalysis(
                intent=TaskIntent.URL_PAGE,
                needs_external_info=True,
                recommended_first_tool="visit_webpage",
                forbidden_tools=(),
                analysis_text=(
                    "The task contains a URL. The first evidence "
                    f"step should visit_webpage for '{target}'."
                ),
            )

    # 7) Wikipedia / encyclopedic facts
    if any(
        keyword in text
        for keyword in (
            "wikipedia", "wiki", "encyclopedic", "encyclopedia",
        )
    ):
        return TaskAnalysis(
            intent=TaskIntent.FACTUAL_SEARCH,
            needs_external_info=True,
            recommended_first_tool=(
                "visit_webpage" if has_visit
                else ("web_search" if has_web else None)
            ),
            forbidden_tools=(),
            analysis_text=(
                "The task asks for encyclopedic facts. Prefer "
                "visit_webpage on the en.wikipedia.org article or a "
                "keyword web_search."
            ),
        )

    # 8) Factual / knowledge
    if _looks_factual(text):
        return TaskAnalysis(
            intent=TaskIntent.FACTUAL_SEARCH,
            needs_external_info=True,
            recommended_first_tool=(
                "web_search" if has_web else None
            ),
            forbidden_tools=(),
            analysis_text=(
                "The task asks for external factual information. "
                "Use a SHORT keyword web_search query (2-8 words) "
                "built from the key entities only, never the raw "
                "question."
            ),
        )

    # 9) Self-contained / reasoning only
    return TaskAnalysis(
        intent=TaskIntent.SELF_CONTAINED,
        needs_external_info=False,
        recommended_first_tool=None,
        forbidden_tools=("web_search",),
        analysis_text=(
            "The task can be answered from reasoning and the "
            "available context WITHOUT web search. If no tool "
            "adds capability, use ONLY the final LLM answer step."
        ),
    )


def _has_digit_or_math(text: str) -> bool:
    return (
        any(char.isdigit() for char in text)
        or "!" in text
        or any(
            symbol in text
            for symbol in ("+", "-", "*", "x", "×", "÷", "/")
        )
    )


def _looks_factual(text: str) -> bool:
    factual_markers = (
        "who", "what", "when", "where", "which", "how many",
        "how much", "how old", "who was", "who is", "what is",
        "how did", "in which year", "capital", "population",
        "largest", "smallest", "tallest", "highest", "country",
        "city", "president", "prime minister", "born", "died",
        "discover", "invent", "album", "song", "release",
        "election", "treaty", "war", "company", "founded",
        "census", "known for", "based on",
    )
    return any(marker in text for marker in factual_markers)
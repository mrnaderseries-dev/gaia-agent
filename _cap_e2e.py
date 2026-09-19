"""End-to-end capability gate via the real agent (PDF/word/web/vision/audio)."""
import asyncio
import shutil
import sys
from pathlib import Path

from gaia_agent.context.attachments import Attachment
from gaia_agent.main import create_agent
from gaia_agent.core.agent_state import AgentState

ONLY = sys.argv[1] if len(sys.argv) > 1 else None

CASES = [
    (
        "pdf_sum",
        "The attached PDF lists quantities 120, 45, 7, and 300 units. "
        "What is their total? Reply with digits only.",
        [("_cap_note.pdf", "_cap_pdf_case.pdf")],
    ),
    (
        "word_avg_speed",
        "A train travels 120 km at 60 km/h then returns the same 120 km "
        "at 40 km/h. What is the average speed in km/h? Reply with digits only.",
        [],
    ),
    (
        "word_apples",
        "A store sells 3 apples for 2 dollars. How much would 27 apples "
        "cost in dollars? Reply with digits only.",
        [],
    ),
    (
        "word_area",
        "A rectangle has length 12 and width 7. What is its area? "
        "Reply with digits only.",
        [],
    ),
    (
        "web_capital",
        "What is the capital of France? Reply with the city name only.",
        [],
    ),
    (
        "vision_shape",
        "Look at the attached image. What shape and color is drawn on the "
        "right side of the image? Reply with two words only, such as "
        "'red circle'.",
        [("_cap_vision_shape.png", "_cap_vision_case.png")],
    ),
    (
        "audio_tones",
        "Transcribe the attached audio file. Reply with the transcript only.",
        [("_cap_audio_two_tones.wav", "_cap_audio_case.wav")],
    ),
]


async def run_case(name: str, question: str, attachments: list[tuple[str, str]]):
    staged: list[Attachment] = []
    for index, (source, dest) in enumerate(attachments):
        shutil.copyfile(source, dest)
        staged.append(
            Attachment(
                attachment_id=f"{name}-{index}",
                filename=Path(dest).name,
                path=str(Path(dest).resolve()),
            )
        )
    agent = await create_agent()
    state = AgentState(user_request=question, attachments=staged)
    try:
        result = await agent.run(state)
        print("=" * 70)
        print("CASE:", name)
        print("QUESTION:", question)
        print("FINAL:", repr(result.final_answer))
        print("VERIFIED:", result.final_answer_verified)
        print("STATUS:", result.status if hasattr(result, "status") else result.phase)
        print("TERMINATION:", result.termination_reason)
        for record in result.evidence:
            tool = getattr(record, "tool_name", "?")
            output = str(getattr(record, "result", ""))[:300]
            print(f"EVIDENCE [{tool}]:", repr(output))
    finally:
        for _, dest in attachments:
            Path(dest).unlink(missing_ok=True)


async def main() -> None:
    for name, question, attachments in CASES:
        if ONLY is not None and name != ONLY:
            continue
        await run_case(name, question, attachments)


asyncio.run(main())

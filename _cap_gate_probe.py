"""Capability-gate probes: PDF + word-problem planner + web tool (real runtime)."""
import asyncio

from gaia_agent.llm.provider.ollama import OllamaClient
from gaia_agent.planner.task_classifier import TaskClassifier
from gaia_agent.planner.strategy_selector import (
    StrategyContext,
    StrategySelector,
)
from gaia_agent.tools.files import FileReaderTool
from gaia_agent.tools.python import PythonInterpreterTool
from smolagents import DuckDuckGoSearchTool

QUANT_QUESTIONS = (
    "A train travels 120 km at 60 km/h then returns the same 120 km "
    "at 40 km/h. What is the average speed in km/h?",
    "A store sells 3 apples for 2 dollars. How much would 27 apples "
    "cost in dollars?",
    "A rectangle has length 12 and width 7. What is its area?",
)


async def main() -> None:
    tools = ("python_interpreter", "web_search", "file_reader")
    selector = StrategySelector()
    classifier = TaskClassifier()

    # PDF via production tool contract
    reader = FileReaderTool(base_dir=".")
    pdf_text = reader.forward("_cap_note.pdf")
    print("PDF-TOOL:", repr(pdf_text[:160]))

    # Classifier + strategy probes (deterministic path, no LLM needed)
    for question, files in [
        (
            "The attached PDF lists quantities 120, 45, 7, and 300 units. "
            "What is their total?",
            ["_cap_note.pdf"],
        ),
        (QUANT_QUESTIONS[0], []),
        (QUANT_QUESTIONS[1], []),
        (QUANT_QUESTIONS[2], []),
    ]:
        analysis = classifier.classify(
            question, available_files=files, available_tools=list(tools)
        )
        decision = selector.select(
            analysis,
            StrategyContext(
                available_tools=frozenset(tools), available_files=tuple(files)
            ),
        )
        print("Q:", question[:70])
        print(
            "  intent:", analysis.intent,
            "| strategy:", decision.strategy,
            "| tool:", decision.primary_tool,
        )

    # Real web tool (infrastructure check, not agent E2E)
    search = DuckDuckGoSearchTool()
    web_text = str(search("capital of France"))
    print("WEB-TOOL-HAS-PARIS:", "Paris" in web_text)

    # Real python computation over the PDF numbers
    python_tool = PythonInterpreterTool()
    print("PYTHON-SUM:", python_tool.forward("result = 120+45+7+300"))

    # Real Ollama client sanity (text path used by planner/verifier)
    client = OllamaClient()
    print("OLLAMA-CLIENT:", type(client).__name__)


asyncio.run(main())

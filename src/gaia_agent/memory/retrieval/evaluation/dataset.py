
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    relevant_memories: Sequence[str]


EVALUATION_DATASET: Sequence[EvaluationCase] = [

    EvaluationCase(
        query="What programming language does the user prefer?",
        relevant_memories=(
            "The user prefers Python for backend development.",
        ),
    ),

    EvaluationCase(
        query="What framework does the user use for APIs?",
        relevant_memories=(
            "The user uses FastAPI for backend APIs.",
        ),
    ),

    EvaluationCase(
        query="What database does the user use for persistence?",
        relevant_memories=(
            "The user is using PostgreSQL for persistence.",
        ),
    ),

    EvaluationCase(
        query="What is the user building?",
        relevant_memories=(
            "The user is building the GAIA AI agent.",
            "The user is building a reliability layer.",
            "The user is building a GitHub repository monitoring system.",
        ),
    ),

    EvaluationCase(
        query="What does the user use for deployment?",
        relevant_memories=(
            "The user is learning Docker for deployment.",
        ),
    ),

    EvaluationCase(
        query="Which programming language does the user favor for server-side work?",
        relevant_memories=(
            "The user prefers Python for backend development.",
        ),
    ),

    EvaluationCase(
        query="What technology is the user studying for asynchronous code?",
        relevant_memories=(
            "The user is studying asynchronous programming with async and await.",
            "The user is learning Python async programming.",
        ),
    ),

    EvaluationCase(
        query="Which database technology handles the project's persistent data?",
        relevant_memories=(
            "The user is using PostgreSQL for persistence.",
            "The user is implementing PostgreSQL repositories.",
        ),
    ),

    EvaluationCase(
        query="What container technology is the user studying for production?",
        relevant_memories=(
            "The user is learning containerization.",
            "The user is learning Docker for deployment.",
        ),
    ),

    EvaluationCase(
        query="How does the user make the agent resilient to failures?",
        relevant_memories=(
            "The user is building a reliability layer.",
            "The user is studying transient and permanent failures.",
            "The user is studying failure classification.",
        ),
    ),

    EvaluationCase(
        query="How does the system find previously stored information?",
        relevant_memories=(
            "The user is implementing memory retrieval.",
            "The user is studying Top-K retrieval.",
        ),
    ),

    EvaluationCase(
        query="How is the user making AI systems observable in production?",
        relevant_memories=(
            "The user is studying observability for AI agents.",
        ),
    ),

    EvaluationCase(
        query="How does the agent decide when it should stop?",
        relevant_memories=(
            "The user is designing termination policies.",
        ),
    ),

    EvaluationCase(
        query="What mechanism determines whether an action requires permission?",
        relevant_memories=(
            "The user is implementing approval policies.",
        ),
    ),

    EvaluationCase(
        query="How is semantic search implemented in the memory system?",
        relevant_memories=(
            "The user is implementing memory retrieval.",
            "The user is implementing semantic similarity.",
            "The user is studying embeddings and vector search.",
            "The user is studying Top-K retrieval.",
        ),
    ),
    EvaluationCase(
        query="What technique is used to select the best matching memories?",
        relevant_memories=(
            "The user is implementing memory retrieval.",
            "The user is learning about cosine similarity.",
            "The user is studying Top-K retrieval.",
        ),
    ),

    EvaluationCase(
        query="What does the user use to monitor repositories?",
        relevant_memories=(
            "The user is building a GitHub repository monitoring system.",
            "The user is analyzing GitHub repositories.",
        ),
    ),

    EvaluationCase(
        query="What does the user use to analyze software structure?",
        relevant_memories=(
            "The user is studying software architecture analysis.",
            "The user is analyzing code dependencies.",
        ),
    ),

    EvaluationCase(
        query="What is the user learning to trace distributed operations?",
        relevant_memories=(
            "The user is learning distributed tracing.",
        ),
    ),

    EvaluationCase(
        query="How does the user make tool execution safer?",
        relevant_memories=(
            "The user is learning safe tool execution.",
            "The user is designing execution policies.",
        ),
    ),
]
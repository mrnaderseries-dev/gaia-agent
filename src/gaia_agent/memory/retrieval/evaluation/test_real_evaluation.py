from __future__ import annotations

from uuid import uuid4

import pytest

from gaia_agent.memory.models import Memory
from gaia_agent.memory.retrieval.embedding import OllamaEmbeddingProvider
from gaia_agent.memory.retrieval.evaluation.dataset import EVALUATION_DATASET
from gaia_agent.memory.retrieval.models import RetrievalCandidate
from gaia_agent.memory.retrieval.dense_retriver import CandidateRetriever
from gaia_agent.memory.retrieval.lexical_retriever import LexicalRetriever
from gaia_agent.memory.retrieval.hybrid import HybridScorer
from gaia_agent.memory.retrieval.ranker import Ranker
from gaia_agent.memory.retrieval.retriever import MemoryRetriever


def create_memory(content: str) -> Memory:
    return Memory(
        user_id=str(uuid4()),
        content=content,
        memory_id=uuid4(),
        memory_type="fact",
        source="test",
        importance=1.0,
        confidence=1.0,
        metadata={},
    )


def build_test_memories() -> list[Memory]:
    return [
        create_memory("The user prefers Python for backend development."),
        create_memory("The user is learning Python async programming."),
        create_memory("The user uses FastAPI for backend APIs."),
        create_memory("The user is building the GAIA AI agent."),
        create_memory("The user is building a reliability layer."),
        create_memory("The user is learning LangGraph."),
        create_memory("The user is learning RAG and vector databases."),
        create_memory("The user is studying embeddings and vector search."),
        create_memory("The user is using PostgreSQL for persistence."),
        create_memory("The user is implementing PostgreSQL repositories."),
        create_memory("The user is learning Docker for deployment."),
        create_memory("The user is learning containerization."),
        create_memory("The user is studying observability for AI agents."),
        create_memory("The user is designing termination policies."),
        create_memory("The user is implementing approval policies."),
        create_memory("The user is implementing memory retrieval."),
        create_memory("The user is implementing semantic similarity."),
        create_memory("The user is studying Top-K retrieval."),
        create_memory("The user is learning about cosine similarity."),
        create_memory(
            "The user is building a GitHub repository monitoring system."
        ),
        create_memory("The user is analyzing GitHub repositories."),
        create_memory("The user is studying software architecture analysis."),
        create_memory("The user is analyzing code dependencies."),
        create_memory("The user is learning distributed tracing."),
        create_memory("The user is learning safe tool execution."),
        create_memory("The user is designing execution policies."),
    ]


def extract_memory(result: RetrievalCandidate | Memory) -> Memory:
    if isinstance(result, RetrievalCandidate):
        return result.memory

    if isinstance(result, Memory):
        return result

    raise TypeError(
        f"Unsupported retrieval result type: {type(result)}"
    )


def calculate_recall(
    retrieved: list[RetrievalCandidate],
    relevant_memories: tuple[str, ...],
) -> float:

    if not relevant_memories:
        return 0.0

    retrieved_contents = {
        extract_memory(result).content
        for result in retrieved
    }

    found = sum(
        1
        for memory in relevant_memories
        if memory in retrieved_contents
    )

    return found / len(relevant_memories)


def calculate_precision(
    retrieved: list[RetrievalCandidate],
    relevant_memories: tuple[str, ...],
) -> float:

    if not retrieved:
        return 0.0

    relevant_set = set(relevant_memories)

    relevant_retrieved = sum(
        1
        for result in retrieved
        if extract_memory(result).content in relevant_set
    )

    return relevant_retrieved / len(retrieved)
def calculate_reciprocal_rank(
    retrieved: list[RetrievalCandidate],
    relevant_memories: tuple[str, ...],
) -> float:

    relevant_set = set(relevant_memories)

    for rank, result in enumerate(retrieved, start=1):

        memory = extract_memory(result)

        if memory.content in relevant_set:
            return 1.0 / rank

    return 0.0


def build_retriever() -> MemoryRetriever:
    provider = OllamaEmbeddingProvider()

    return MemoryRetriever(
        candidate_retriever=CandidateRetriever(provider),
        lexical_retriever=LexicalRetriever(),
        hybrid_scorer=HybridScorer(),
        ranker=Ranker(),
    )


def print_results(
    results: list[RetrievalCandidate],
) -> None:

    print("\nTOP 5 RESULTS:")

    for rank, candidate in enumerate(results, start=1):

        print(
            f"{rank}. "
            f"memory={candidate.memory_score:.4f} "
            f"reranker={candidate.reranker_score:.4f} "
            f"hybrid={candidate.hybrid_score:.4f} "
            f"dense={candidate.dense_score:.4f} "
            f"lexical={candidate.lexical_score:.4f} "
            f"- {candidate.memory.content}"
        )


@pytest.mark.asyncio
async def test_real_embedding_evaluation():

    memories = build_test_memories()

    retriever = build_retriever()

    total_recall = 0.0
    total_precision = 0.0
    total_mrr = 0.0

    for case in EVALUATION_DATASET:

        print("\n" + "=" * 70)
        print(f"QUERY: {case.query}")
        print("=" * 70)

        results = await retriever.retrieve(
            query=case.query,
            memories=memories,
            top_k=5,
        )

        print_results(results)

        recall = calculate_recall(
            results,
            case.relevant_memories,
        )

        precision = calculate_precision(
            results,
            case.relevant_memories,
        )

        mrr = calculate_reciprocal_rank(
            results,
            case.relevant_memories,
        )

        print(f"\nRecall@5: {recall:.2f}")
        print(f"Precision@5: {precision:.2f}")
        print(f"Reciprocal Rank: {mrr:.2f}")

        total_recall += recall
        total_precision += precision
        total_mrr += mrr

    count = len(EVALUATION_DATASET)

    assert count > 0

    average_recall = total_recall / count
    average_precision = total_precision / count
    average_mrr = total_mrr / count

    print("\n")
    print("=" * 70)
    print("MEMORY RETRIEVAL EVALUATION")
    print("=" * 70)

    print(f"Recall@5:    {average_recall:.3f}")
    print(f"Precision@5: {average_precision:.3f}")
    print(f"MRR:         {average_mrr:.3f}")

    assert average_recall >= 0.50


@pytest.mark.asyncio
async def test_component_scores():

    memories = build_test_memories()

    retriever = build_retriever()

    print("\n")
    print("=" * 70)
    print("COMPONENT SCORE INSPECTION")
    print("=" * 70)

    for case in EVALUATION_DATASET:

        results = await retriever.retrieve(
            query=case.query,
            memories=memories,
            top_k=5,
        )

        print("\n")
        print(f"QUERY: {case.query}")

        for rank, candidate in enumerate(results, start=1):

            print(
                f"{rank}. "
                f"hybrid={candidate.hybrid_score:.4f} | "
                f"reranker={candidate.reranker_score:.4f} | "
                f"memory={candidate.memory_score:.4f} | "
                f"{candidate.memory.content}"
            )
@pytest.mark.asyncio
async def test_score_thresholds():

    memories = build_test_memories()
    retriever = build_retriever()

    thresholds = [
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]

    print("\n")
    print("=" * 70)
    print("MEMORY SCORE THRESHOLD CALIBRATION")
    print("=" * 70)

    for threshold in thresholds:

        total_recall = 0.0
        total_precision = 0.0
        total_mrr = 0.0
        total_returned = 0

        for case in EVALUATION_DATASET:

            results = await retriever.retrieve(
                query=case.query,
                memories=memories,
                top_k=5,
            )

            filtered_results = [
                candidate
                for candidate in results
                if candidate.memory_score >= threshold
            ]

            recall = calculate_recall(
                filtered_results,
                case.relevant_memories,
            )

            precision = calculate_precision(
                filtered_results,
                case.relevant_memories,
            )

            mrr = calculate_reciprocal_rank(
                filtered_results,
                case.relevant_memories,
            )

            total_recall += recall
            total_precision += precision
            total_mrr += mrr
            total_returned += len(filtered_results)

        count = len(EVALUATION_DATASET)

        average_recall = total_recall / count
        average_precision = total_precision / count
        average_mrr = total_mrr / count
        average_returned = total_returned / count

        print()
        print(f"Threshold: {threshold:.2f}")
        print(f"Recall@5:     {average_recall:.3f}")
        print(f"Precision@5:  {average_precision:.3f}")
        print(f"MRR:          {average_mrr:.3f}")
        print(f"Avg returned: {average_returned:.2f}")            



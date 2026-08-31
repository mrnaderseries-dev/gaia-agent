from uuid import uuid4

from gaia_agent.memory.retrieval.evaluation import (
    RetrievalCase,
    RetrievalEvaluator,
)


def test_perfect_retrieval():
    memory_1 = type(
        "Memory",
        (),
        {"memory_id": uuid4()},
    )()

    memory_2 = type(
        "Memory",
        (),
        {"memory_id": uuid4()},
    )()

    case = RetrievalCase(
        query="What language does the user prefer?",
        relevant_memory_ids={
            str(memory_1.memory_id),
            str(memory_2.memory_id),
        },
    )

    results = [
        (memory_1, 0.99),
        (memory_2, 0.95),
    ]

    metrics = RetrievalEvaluator.evaluate(
        results,
        case,
    )

    assert metrics.precision_at_k == 1.0
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 1.0


def test_relevant_memory_in_second_position():
    memory_1 = type(
        "Memory",
        (),
        {"memory_id": uuid4()},
    )()

    memory_2 = type(
        "Memory",
        (),
        {"memory_id": uuid4()},
    )()

    case = RetrievalCase(
        query="What language does the user prefer?",
        relevant_memory_ids={
            str(memory_2.memory_id),
        },
    )

    results = [
        (memory_1, 0.90),
        (memory_2, 0.80),
    ]

    metrics = RetrievalEvaluator.evaluate(
        results,
        case,
    )

    assert metrics.precision_at_k == 0.5
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 0.5


def test_no_relevant_memory_retrieved():
    memory_1 = type(
        "Memory",
        (),
        {"memory_id": uuid4()},
    )()

    relevant_memory = uuid4()

    case = RetrievalCase(
        query="What database does the user use?",
        relevant_memory_ids={
            str(relevant_memory),
        },
    )

    results = [
        (memory_1, 0.90),
    ]

    metrics = RetrievalEvaluator.evaluate(
        results,
        case,
    )

    assert metrics.precision_at_k == 0.0
    assert metrics.recall_at_k == 0.0
    assert metrics.mrr == 0.0
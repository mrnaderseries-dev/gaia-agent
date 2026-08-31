# Memory Retrieval Baseline v1

## Embedding Model

Model: nomic-embed-text
Provider: Ollama

## Metrics

Recall@5: 0.683
Precision@5: 0.213
MRR: 0.670

## Evaluation Dataset

Queries: [عدد الـqueries عندك]
Memories: [عدد الـmemories]

## Observations

### Strengths
- Good direct semantic matching.
- Good paraphrase retrieval.
- Relevant memories are often ranked highly.
- Strong performance on technical concepts.

### Weaknesses
- Some relevant memories are not retrieved in Top-5.
- Precision@5 is relatively low.
- Hard negatives cause ranking errors.
- Complex conceptual queries perform worse.
- Similar contextual memories can outrank the correct memory.

## Examples

### Good
Query:
"What programming language does the user prefer?"

Expected:
"The user prefers Python for backend development."

Rank: 1

### Failure
Query:
"What database is used by the agent?"

Expected:
"The user is using PostgreSQL for persistence."

Result:
Not retrieved in Top-5.

### Ranking Failure
Query:
"What technology is used to deploy the system?"

Expected:
"Docker"

Production deployment ranked above Docker.

## Conclusion

This is the baseline for future retrieval improvements.
Memory Retrieval Baseline v1.1

Embedding Model

Model: "nomic-embed-text"
Provider: Ollama

Metrics

Previous Baseline

Recall@5: 0.683
Precision@5: 0.213
MRR: 0.670

Threshold-Calibrated Baseline

Threshold: "0.60"

Recall@5: 0.871
Precision@5: 0.532
MRR: 1.000

Threshold Calibration

Threshold| Recall@5| Precision@5| MRR
0.50| 0.942| 0.330| 1.000
0.55| 0.887| 0.397| 1.000
0.60| 0.871| 0.532| 1.000

Observations

Strengths

- Dense semantic retrieval successfully identifies directly relevant memories.
- Paraphrased queries are generally retrieved successfully.
- MRR reached 1.000 at the tested thresholds.
- Increasing the threshold substantially improves precision.
- Threshold "0.60" provides a better precision/recall trade-off than "0.50".
- The correct memory is frequently ranked first.

Weaknesses

- Increasing the threshold reduces Recall@5.
- Precision is still limited because semantically similar memories are frequently retrieved together.
- Similar memories can outrank or accompany the intended memory.
- Dense similarity alone does not reliably distinguish between closely related concepts.
- Complex conceptual queries remain more difficult.
- The current retrieval system does not yet combine lexical relevance with semantic relevance.

Current Decision

Use "0.60" as the current calibrated threshold for the dense retrieval baseline.

This value is a baseline for comparison, not a final production value.

Next Architecture Step

The next improvement is Hybrid Retrieval:

Query
→ Dense Retrieval
→ Lexical Retrieval
→ Hybrid Scoring
→ Ranking
→ Threshold
→ Final Memories

The current dense retrieval results will be used as the baseline for evaluating the Hybrid Retrieval architecture.

Conclusion

The threshold calibration demonstrates that the retrieval system can significantly improve precision while maintaining strong recall and perfect MRR on the current evaluation dataset.

The main remaining problem is not simply finding relevant memories, but distinguishing the most relevant memory from semantically similar memories.

Therefore, the next step is to introduce lexical retrieval and combine it with dense semantic retrieval through a hybrid scoring layer.
# Memory Retrieval Baseline v1.2

## Retrieval + Memory Scoring + Reranking

This baseline evaluates the retrieval pipeline after introducing:

Query
→ Dense Retrieval
→ Lexical Retrieval
→ Hybrid Scoring
→ Memory Scoring
→ Reranking
→ Final Ranking

## Metrics

Recall@5: 0.942
Precision@5: 0.330
MRR: 0.829

## Comparison with Previous Stage

| Version | Recall@5 | Precision@5 | MRR |
|---|---:|---:|---:|
| v1 | 0.683 | 0.213 | 0.670 |
| v1.1 | 0.871 | 0.532 | 1.000 |
| v1.2 | 0.942 | 0.330 | 0.829 |

## Impact of Memory Scoring and Reranking

Compared with the earlier retrieval stage where:

memory score = 0
reranker score = 0

the introduction of memory scoring and reranking significantly improved retrieval quality.

### Recall

Recall@5 increased by approximately 13 percentage points compared with the previous stage.

This indicates that the additional ranking signals help preserve relevant memories within the Top-5 results.

### Ranking Quality

The reranking stage provides an additional signal for distinguishing memories that have similar semantic similarity scores.

This is especially useful when several memories are conceptually related but only one directly answers the query.

The earlier system relied primarily on retrieval similarity, while the current system evaluates candidates using additional memory-level signals before producing the final ranking.

## Observations

### Strengths

- Strong Top-5 recall.
- Memory scoring improves candidate prioritization.
- Reranking provides an additional ranking signal beyond semantic similarity.
- Relevant memories remain highly retrievable even for paraphrased queries.
- The system can distinguish between several semantically related memories more effectively than dense retrieval alone.
- The retrieval pipeline is now composed of multiple independent ranking signals.

### Weaknesses

- Precision@5 decreased compared with the threshold-calibrated dense baseline.
- Some semantically related memories are still ranked together with the correct memory.
- Memory scoring and reranking do not completely solve hard-negative cases.
- Some conceptual queries still produce incorrect top-ranked memories.
- The scoring weights have not yet been formally calibrated through a larger evaluation dataset.

## Important Finding

The results show that retrieval quality cannot be evaluated using semantic similarity alone.

A memory can have a high embedding similarity while still being less useful than another memory.

Therefore, the retrieval system benefits from separating:

1. Candidate retrieval
2. Relevance scoring
3. Memory-level scoring
4. Reranking
5. Final selection

This provides a stronger foundation for the Context Layer.

## Current Decision

Keep the current retrieval architecture as the target architecture for the next stage.

Do not optimize the scoring weights further yet.

The current implementation is sufficient to move forward to the Context Layer, where retrieved memories will be converted into useful context for the LLM.

## Next Architecture Step

Memory Retrieval
→ Candidate Selection
→ Hybrid Scoring
→ Memory Scoring
→ Reranking
→ Top-K Selection
→ Context Builder
→ LLM

## Conclusion

Memory retrieval has progressed from simple dense semantic search into a multi-stage retrieval and ranking pipeline.

The current results demonstrate strong Recall@5 (0.942), while memory scoring and reranking provide additional ranking signals for distinguishing relevant memories from semantically similar candidates.

Although Precision@5 is not yet optimal, further optimization is not necessary before proceeding.

The retrieval layer is now sufficiently functional to integrate with the Context Layer.

Future improvements can be evaluated incrementally through changes to scoring weights, reranking strategy, filtering, and retrieval thresholds.

The current results should therefore be treated as the retrieval baseline before Context Builder integration.
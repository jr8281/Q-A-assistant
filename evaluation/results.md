## Retrieval quality (9 questions, k=5)

| Method | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| bm25 | 0.44 | 1.00 | 1.00 | 0.72 |
| dense | 0.56 | 0.78 | 1.00 | 0.69 |
| hybrid | 0.56 | 1.00 | 1.00 | 0.76 |

## Hybrid weight sweep (alpha = weight of BM25)

| alpha | Recall@3 | MRR |
|---|---|---|
| 0.00 | 0.78 | 0.69 |
| 0.25 | 0.89 | 0.71 |
| 0.50 | 1.00 | 0.76 |
| 0.75 | 1.00 | 0.78 |
| 1.00 | 1.00 | 0.72 |

## Relevance gate (9 answerable, 5 unanswerable)

| min_similarity | Answerable passed | Unanswerable blocked |
|---|---|---|
| 0.30 | 100% | 0% |
| 0.35 | 100% | 0% |
| 0.40 | 100% | 0% |
| 0.45 | 100% | 60% |
| 0.50 | 100% | 80% |
| 0.55 | 100% | 100% |
| 0.60 | 100% | 100% |
| 0.65 | 100% | 100% |
| 0.70 | 89% | 100% |

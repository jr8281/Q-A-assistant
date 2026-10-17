"""Retrieval evaluation: compares BM25, dense and hybrid search, sweeps the hybrid weight (alpha),
and measures the relevance gate on answerable vs. unanswerable questions.

Run from the project root:   python -m evaluation.run_eval
Writes evaluation/results.md (paste the tables into the README).
"""
import json
from pathlib import Path

from app.embedding import PyTorchEmbedder
from app.indexing import load_corpus
from app.retriever import HybridRetriever

K = 5
HERE = Path(__file__).parent


def first_hit_ranks(retriever, items, **search_kwargs):
    """Rank (1-based) of the first retrieved chunk whose (file, page) is a gold page, else None."""
    ranks = []
    for item in items:
        gold = {(item["source"], p) for p in item["pages"]}
        results = retriever.search(item["question"], top_k=K, **search_kwargs)
        ranks.append(next(
            (i for i, r in enumerate(results, 1) if (r["source"], r["page"]) in gold), None
        ))
    return ranks


def summarize(ranks):
    n = len(ranks)
    recall = lambda k: sum(1 for r in ranks if r and r <= k) / n
    mrr = sum(1 / r for r in ranks if r) / n
    return recall(1), recall(3), recall(5), mrr


def top_similarity(retriever, question):
    # Same quantity the API gate uses: best similarity among the returned sources
    return max(r["similarity"] for r in retriever.search(question, top_k=3))


def main():
    data = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    answerable, unanswerable = data["answerable"], data["unanswerable"]

    _, chunks = load_corpus()
    retriever = HybridRetriever(chunks, PyTorchEmbedder())

    lines = [f"## Retrieval quality ({len(answerable)} questions, k={K})", "",
             "| Method | Recall@1 | Recall@3 | Recall@5 | MRR |", "|---|---|---|---|---|"]
    for mode in ("bm25", "dense", "hybrid"):
        r1, r3, r5, mrr = summarize(first_hit_ranks(retriever, answerable, mode=mode))
        lines.append(f"| {mode} | {r1:.2f} | {r3:.2f} | {r5:.2f} | {mrr:.2f} |")

    lines += ["", "## Hybrid weight sweep (alpha = weight of BM25)", "",
              "| alpha | Recall@3 | MRR |", "|---|---|---|"]
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        _, r3, _, mrr = summarize(first_hit_ranks(retriever, answerable, mode="hybrid", alpha=alpha))
        lines.append(f"| {alpha:.2f} | {r3:.2f} | {mrr:.2f} |")

    pos = [top_similarity(retriever, it["question"]) for it in answerable]
    neg = [top_similarity(retriever, q) for q in unanswerable]
    lines += ["", f"## Relevance gate ({len(pos)} answerable, {len(neg)} unanswerable)", "",
              "| min_similarity | Answerable passed | Unanswerable blocked |", "|---|---|---|"]
    for i in range(9):
        t = 0.30 + 0.05 * i
        lines.append(f"| {t:.2f} | {sum(s >= t for s in pos) / len(pos):.0%} "
                     f"| {sum(s < t for s in neg) / len(neg):.0%} |")

    report = "\n".join(lines)
    (HERE / "results.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
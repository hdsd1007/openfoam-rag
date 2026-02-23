"""
retrieval_metrics.py — Pure Python IR retrieval metrics.

Computes Recall@k, MRR, and NDCG@k by comparing retrieved chunk pages
against relevant_pages from the golden set.
"""

import math


def compute_recall_at_k(retrieved_pages, relevant_pages, k):
    """
    Recall@k: fraction of relevant pages found in top-k retrieved pages.

    Args:
        retrieved_pages: Ordered list of page numbers from retrieved chunks.
        relevant_pages: Set/list of ground-truth relevant page numbers.
        k: Cutoff rank.

    Returns:
        Float in [0, 1]. Returns 0.0 if relevant_pages is empty.
    """
    if not relevant_pages:
        return 0.0
    relevant_set = set(relevant_pages)
    retrieved_at_k = set(retrieved_pages[:k])
    return len(retrieved_at_k & relevant_set) / len(relevant_set)


def compute_mrr(retrieved_pages, relevant_pages):
    """
    Mean Reciprocal Rank: 1/rank of the first relevant page in retrieved list.

    Args:
        retrieved_pages: Ordered list of page numbers from retrieved chunks.
        relevant_pages: Set/list of ground-truth relevant page numbers.

    Returns:
        Float in [0, 1]. Returns 0.0 if no relevant page is found.
    """
    relevant_set = set(relevant_pages)
    for rank, page in enumerate(retrieved_pages, 1):
        if page in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(retrieved_pages, relevant_pages, k):
    """
    NDCG@k: Normalized Discounted Cumulative Gain at rank k.

    Uses binary relevance (1 if page is in relevant_pages, 0 otherwise).

    Args:
        retrieved_pages: Ordered list of page numbers from retrieved chunks.
        relevant_pages: Set/list of ground-truth relevant page numbers.
        k: Cutoff rank.

    Returns:
        Float in [0, 1]. Returns 0.0 if relevant_pages is empty.
    """
    if not relevant_pages:
        return 0.0

    relevant_set = set(relevant_pages)

    # DCG: sum of relevance / log2(rank + 1) for top-k
    dcg = 0.0
    for i, page in enumerate(retrieved_pages[:k]):
        rel = 1.0 if page in relevant_set else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because rank starts at 1, log2(1+1)

    # Ideal DCG: all relevant pages ranked first
    ideal_length = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_length))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def compute_all_retrieval_metrics(retrieved_pages, relevant_pages):
    """
    Compute all standard IR retrieval metrics.

    Args:
        retrieved_pages: Ordered list of page numbers from retrieved chunks.
        relevant_pages: Set/list of ground-truth relevant page numbers.

    Returns:
        Dict with recall@5, recall@10, recall@15, recall@20,
        MRR, NDCG@10, NDCG@20.
    """
    return {
        "recall@5": round(compute_recall_at_k(retrieved_pages, relevant_pages, 5), 4),
        "recall@10": round(compute_recall_at_k(retrieved_pages, relevant_pages, 10), 4),
        "recall@15": round(compute_recall_at_k(retrieved_pages, relevant_pages, 15), 4),
        "recall@20": round(compute_recall_at_k(retrieved_pages, relevant_pages, 20), 4),
        "MRR": round(compute_mrr(retrieved_pages, relevant_pages), 4),
        "NDCG@10": round(compute_ndcg_at_k(retrieved_pages, relevant_pages, 10), 4),
        "NDCG@20": round(compute_ndcg_at_k(retrieved_pages, relevant_pages, 20), 4),
    }

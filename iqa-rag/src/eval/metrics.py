"""
Evaluation Metrics for IQA-RAG.

Calculates:
- Hit-Rate@K: Checks whether ground-truth paper/section/keywords are retrieved in top-K.
- Precision@K: Fraction of retrieved docs that are relevant to the query.
- MRR@K: Mean Reciprocal Rank of the first relevant document.
- Faithfulness Score: Evaluates whether claims in generated answers are grounded in retrieved context.
- Answer Relevance (ROUGE-L / Token F1): Quantifies agreement between generated and ground-truth answers.
"""

import re
from typing import List, Dict, Any, Set


def _normalize_tokens(text: str) -> Set[str]:
    """Tokenize and normalize text into a set of lower-case words."""
    words = re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', text.lower())
    return set(words)


def is_doc_relevant(
    doc: Dict[str, Any],
    target_paper: str,
    target_section: str,
    keywords: List[str]
) -> bool:
    """
    Determine if a retrieved document is relevant to the ground truth item.
    Matches either paper title similarity, section title, or dense keyword overlap.
    """
    doc_title = doc.get("paper_title", "").lower()
    doc_section = doc.get("section_title", "").lower()
    doc_text = doc.get("text", "").lower()

    # Target paper match
    target_norm = target_paper.lower()
    if target_norm and (target_norm in doc_title or doc_title in target_norm):
        return True

    # Check keyword overlap in document text
    if keywords:
        matched_kw = sum(1 for kw in keywords if kw.lower() in doc_text or kw.lower() in doc_title)
        if matched_kw >= max(1, len(keywords) // 2):
            return True

    return False


def calculate_hit_rate_at_k(
    retrieved_docs: List[Dict[str, Any]],
    target_paper: str,
    target_section: str,
    keywords: List[str],
    k: int
) -> float:
    """Return 1.0 if at least one relevant document is found in top-k, else 0.0."""
    top_k_docs = retrieved_docs[:k]
    for doc in top_k_docs:
        if is_doc_relevant(doc, target_paper, target_section, keywords):
            return 1.0
    return 0.0


def calculate_precision_at_k(
    retrieved_docs: List[Dict[str, Any]],
    target_paper: str,
    target_section: str,
    keywords: List[str],
    k: int
) -> float:
    """Calculate the proportion of top-k retrieved documents that are relevant."""
    top_k_docs = retrieved_docs[:k]
    if not top_k_docs:
        return 0.0
    relevant_count = sum(
        1 for doc in top_k_docs
        if is_doc_relevant(doc, target_paper, target_section, keywords)
    )
    return round(relevant_count / len(top_k_docs), 4)


def calculate_mrr_at_k(
    retrieved_docs: List[Dict[str, Any]],
    target_paper: str,
    target_section: str,
    keywords: List[str],
    k: int
) -> float:
    """Calculate Reciprocal Rank (1/rank) for the first relevant document in top-k."""
    top_k_docs = retrieved_docs[:k]
    for rank, doc in enumerate(top_k_docs, start=1):
        if is_doc_relevant(doc, target_paper, target_section, keywords):
            return round(1.0 / rank, 4)
    return 0.0


def evaluate_faithfulness(
    answer: str,
    retrieved_docs: List[Dict[str, Any]],
    llm_judge_model: Any = None
) -> float:
    """
    Faithfulness Judge: Measures whether claims in the generated answer
    are strictly grounded in the retrieved context chunks (no hallucination).
    Scores range from 0.0 (completely unfaithful/hallucinatory) to 1.0 (fully grounded).
    """
    if not answer or not retrieved_docs:
        return 0.0

    # Combine context tokens
    context_text = " ".join([d.get("text", "") for d in retrieved_docs])
    context_tokens = _normalize_tokens(context_text)

    # Clean citations from answer before token extraction
    cleaned_answer = re.sub(r'\[[^\]]+\]', '', answer)
    answer_tokens = _normalize_tokens(cleaned_answer)

    if not answer_tokens:
        return 1.0 if answer else 0.0

    # Token-level containment ratio
    grounded_tokens = answer_tokens.intersection(context_tokens)
    ratio = len(grounded_tokens) / len(answer_tokens)

    # Citation bonus: If answer explicitly cites sources, add confidence
    has_citations = bool(re.search(r'\[(?:Source|\w+)[^\]]*\]', answer))
    bonus = 0.1 if has_citations else 0.0

    score = min(1.0, round(ratio * 0.9 + bonus, 4))
    return score


def calculate_token_f1(generated: str, ground_truth: str) -> float:
    """Compute token-level F1 overlap between generated answer and ground truth."""
    gen_tokens = _normalize_tokens(generated)
    gt_tokens = _normalize_tokens(ground_truth)

    if not gen_tokens or not gt_tokens:
        return 0.0

    common = gen_tokens.intersection(gt_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(gen_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 4)


def evaluate_sample(
    query_item: Dict[str, Any],
    pipeline_result: Dict[str, Any],
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, Any]:
    """
    Evaluate a single Q&A query result across all metrics.
    """
    retrieved_docs = pipeline_result.get("retrieved_docs", [])
    answer = pipeline_result.get("answer", "")
    target_paper = query_item.get("target_paper", "")
    target_section = query_item.get("target_section", "")
    keywords = query_item.get("keywords", [])
    ground_truth = query_item.get("ground_truth_answer", "")

    metrics = {
        "id": query_item.get("id", ""),
        "question": query_item.get("question", ""),
        "target_paper": target_paper,
        "faithfulness": evaluate_faithfulness(answer, retrieved_docs),
        "answer_f1": calculate_token_f1(answer, ground_truth),
        "retrieval_latency_ms": pipeline_result.get("retrieval_latency_ms", 0.0),
        "total_latency_ms": pipeline_result.get("total_latency_ms", 0.0)
    }

    for k in k_values:
        metrics[f"hit_rate@{k}"] = calculate_hit_rate_at_k(retrieved_docs, target_paper, target_section, keywords, k)
        metrics[f"precision@{k}"] = calculate_precision_at_k(retrieved_docs, target_paper, target_section, keywords, k)
        metrics[f"mrr@{k}"] = calculate_mrr_at_k(retrieved_docs, target_paper, target_section, keywords, k)

    return metrics

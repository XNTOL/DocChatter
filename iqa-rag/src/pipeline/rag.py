"""
Unified RAG Pipeline for IQA-RAG.

Ties together Retriever and Generator into a cohesive end-to-end interface
with latency measurement and query-level observability.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root in sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR.parent))

from src.config import default_config, RAGConfig
from src.pipeline.retriever import IQARetriever
from src.pipeline.generator import IQAGenerator


class IQARAGPipeline:
    """End-to-end Question Answering pipeline over Image Quality Assessment research papers."""

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        retriever: Optional[IQARetriever] = None,
        generator: Optional[IQAGenerator] = None
    ):
        self.config = config or default_config
        self.retriever = retriever or IQARetriever(self.config)
        self.generator = generator or IQAGenerator(self.config)

    def answer_query(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute full RAG pipeline: query -> retrieve top-k docs -> generate grounded answer.
        """
        start_time = time.perf_counter()

        # Step 1: Retrieval
        retrieval_start = time.perf_counter()
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
        retrieval_duration_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)

        # Step 2: Generation
        gen_res = self.generator.generate(query, retrieved_docs)

        total_duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "query": query,
            "answer": gen_res.get("answer", ""),
            "retrieved_docs": retrieved_docs,
            "citations": gen_res.get("citations", []),
            "model": gen_res.get("model", "unknown"),
            "retrieval_latency_ms": retrieval_duration_ms,
            "total_latency_ms": total_duration_ms,
            "num_retrieved": len(retrieved_docs)
        }

    def run(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Convenience alias for answer_query."""
        return self.answer_query(query, top_k=top_k)


# Alias
RAGPipeline = IQARAGPipeline


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    pipeline = IQARAGPipeline()
    test_q = "What is the key mechanism of LPIPS?"
    response = pipeline.run(test_q, top_k=3)
    print("=" * 60)
    print("QUESTION:", response["query"])
    print("TOTAL LATENCY:", response["total_latency_ms"], "ms")
    print("MODEL:", response["model"])
    print("ANSWER:\n", response["answer"].encode("ascii", errors="backslashreplace").decode("ascii"))
    print("=" * 60)

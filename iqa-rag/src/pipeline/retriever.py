"""
Retriever Module for IQA-RAG.

Performs top-k similarity retrieval over the vector store and formats
evidence contexts with source attribution for generator consumption.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root in sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR.parent))

from src.config import default_config, RAGConfig
from src.index.vectorstore import IQAVectorStore


class IQARetriever:
    """Retrieves relevant IQA paper chunks based on semantic similarity."""

    def __init__(self, config: Optional[RAGConfig] = None, vectorstore: Optional[IQAVectorStore] = None):
        self.config = config or default_config
        self.vectorstore = vectorstore or IQAVectorStore(self.config)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Execute top-k semantic retrieval against the indexed vector store.
        """
        k = top_k or self.config.retrieval_top_k
        raw_results = self.vectorstore.query(query, top_k=k)

        formatted_docs = []
        for rank, r in enumerate(raw_results, start=1):
            meta = r.get("metadata", {})
            formatted_docs.append({
                "rank": rank,
                "chunk_id": r.get("chunk_id", ""),
                "score": r.get("score", 0.0),
                "text": r.get("document", ""),
                "paper_title": meta.get("paper_title", "Unknown Paper"),
                "section_title": meta.get("section_title", "Unknown Section"),
                "page": meta.get("page", 1),
            })

        return formatted_docs

    @staticmethod
    def format_context_for_prompt(docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved docs into a clean prompt context block with provenance tags.
        """
        if not docs:
            return "No relevant context found."

        context_parts = []
        for d in docs:
            header = f"[Source {d['rank']}] Paper: \"{d['paper_title']}\" | Section: \"{d['section_title']}\" (p. {d['page']})"
            context_parts.append(f"{header}\n{d['text'].strip()}")

        return "\n\n".join(context_parts)


# Alias
Retriever = IQARetriever


if __name__ == "__main__":
    retriever = IQARetriever()
    query = "How do BRISQUE MSCN coefficients work in spatial domain?"
    docs = retriever.retrieve(query, top_k=3)
    print(f"Retrieved {len(docs)} documents for query: '{query}'")
    for d in docs:
        print(f"[{d['rank']}] Score: {d['score']} | {d['paper_title']} | {d['section_title']}")

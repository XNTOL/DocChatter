"""
Vector Store for IQA-RAG.

Manages persistent indexing and semantic similarity queries for document chunks.
Uses ChromaDB when available, with a persistent pure-Python vector store fallback.
"""

import sys
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root in sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR.parent))

from src.config import default_config, RAGConfig
from src.index.embed import EmbeddingGenerator


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class FallbackVectorStore:
    """Persistent in-memory vector store with cosine similarity ranking."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
            except Exception:
                self.records = []

    def _save(self):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False)

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        existing_ids = {r["id"]: idx for idx, r in enumerate(self.records)}
        for cid, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            record = {
                "id": cid,
                "embedding": emb,
                "document": doc,
                "metadata": meta
            }
            if cid in existing_ids:
                self.records[existing_ids[cid]] = record
            else:
                self.records.append(record)
                existing_ids[cid] = len(self.records) - 1
        self._save()

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        scored = []
        for r in self.records:
            score = cosine_similarity(query_embedding, r["embedding"])
            scored.append({
                "id": r["id"],
                "document": r["document"],
                "metadata": r["metadata"],
                "score": score
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self.records)

    def clear(self):
        self.records = []
        self._save()


class IQAVectorStore:
    """Unified Vector Store interface for IQA-RAG."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or default_config
        self.embedder = EmbeddingGenerator(self.config)
        self.collection_name = self.config.collection_name
        self.persist_dir = self.config.vectorstore_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._chroma_client = None
        self._chroma_collection = None
        self._fallback_store = None
        self._use_fallback = False

        self._init_store()

    def _init_store(self):
        """Initialize ChromaDB or fallback store."""
        try:
            import chromadb
            from chromadb.config import Settings
            self._chroma_client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"[VectorStore] Initialized ChromaDB collection: '{self.collection_name}' at {self.persist_dir}")
        except Exception as e:
            print(f"[VectorStore] ChromaDB not available ({e}). Using persistent fallback vector store.")
            self._use_fallback = True
            fallback_file = self.persist_dir / f"{self.collection_name}_fallback.json"
            self._fallback_store = FallbackVectorStore(fallback_file)

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Embed and index a list of chunk dictionaries.
        """
        if not chunks:
            return 0

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [{
            "paper_id": c.get("paper_id", ""),
            "paper_title": c.get("paper_title", ""),
            "section_title": c.get("section_title", ""),
            "page": int(c.get("page", 1)),
            "token_count": int(c.get("token_count", 0))
        } for c in chunks]

        print(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = self.embedder.embed_texts(documents)

        if not self._use_fallback and self._chroma_collection is not None:
            try:
                # Upsert into Chroma in batches of 100
                batch_size = 100
                for i in range(0, len(ids), batch_size):
                    end = i + batch_size
                    self._chroma_collection.upsert(
                        ids=ids[i:end],
                        embeddings=embeddings[i:end],
                        documents=documents[i:end],
                        metadatas=metadatas[i:end]
                    )
                return len(ids)
            except Exception as e:
                print(f"[Warning] Chroma upsert failed: {e}. Switching to fallback.")
                self._use_fallback = True
                fallback_file = self.persist_dir / f"{self.collection_name}_fallback.json"
                self._fallback_store = FallbackVectorStore(fallback_file)

        if self._fallback_store is not None:
            self._fallback_store.add(ids, embeddings, documents, metadatas)
            return len(ids)

        return 0

    def query(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar document chunks for a given query text.
        """
        k = top_k or self.config.retrieval_top_k
        query_vec = self.embedder.embed_text(query_text)

        if not self._use_fallback and self._chroma_collection is not None:
            try:
                results = self._chroma_collection.query(
                    query_embeddings=[query_vec],
                    n_results=min(k, max(1, self._chroma_collection.count())),
                    include=["documents", "metadatas", "distances"]
                )
                formatted = []
                if results and results.get("ids") and results["ids"][0]:
                    for i in range(len(results["ids"][0])):
                        dist = results["distances"][0][i] if "distances" in results else 0.0
                        # Convert cosine distance to similarity score
                        similarity = max(0.0, 1.0 - dist)
                        formatted.append({
                            "chunk_id": results["ids"][0][i],
                            "document": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "score": round(similarity, 4)
                        })
                return formatted
            except Exception as e:
                print(f"[Warning] Chroma query failed: {e}. Falling back.")
                self._use_fallback = True
                fallback_file = self.persist_dir / f"{self.collection_name}_fallback.json"
                self._fallback_store = FallbackVectorStore(fallback_file)

        if self._fallback_store is not None:
            raw_res = self._fallback_store.query(query_vec, top_k=k)
            return [{
                "chunk_id": r["id"],
                "document": r["document"],
                "metadata": r["metadata"],
                "score": round(r["score"], 4)
            } for r in raw_res]

        return []

    def count(self) -> int:
        """Return total indexed chunks."""
        if not self._use_fallback and self._chroma_collection is not None:
            try:
                return self._chroma_collection.count()
            except Exception:
                pass
        if self._fallback_store:
            return self._fallback_store.count()
        return 0

    def clear(self):
        """Clear all stored vectors."""
        if not self._use_fallback and self._chroma_client is not None:
            try:
                self._chroma_client.delete_collection(self.collection_name)
                self._chroma_collection = self._chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                pass
        if self._fallback_store:
            self._fallback_store.clear()


# Alias for clean naming
VectorStore = IQAVectorStore


if __name__ == "__main__":
    store = VectorStore()
    processed_json = default_config.processed_dir / "chunks.json"
    if processed_json.exists():
        with open(processed_json, "r", encoding="utf-8") as f:
            sample_chunks = json.load(f)
        count = store.add_chunks(sample_chunks)
        print(f"Indexed {count} chunks in vector store.")

    res = store.query("What does LPIPS measure?", top_k=3)
    print(f"Retrieved {len(res)} results:")
    for r in res:
        print(f"- [{r['score']}] {r['metadata']['paper_title']} ({r['metadata']['section_title']}): {r['document'][:80]}...")

"""
Embedding Generation for IQA-RAG.

Supports:
- Local HuggingFace embeddings via sentence-transformers
- Google Gemini embeddings (text-embedding-004)
- OpenAI embeddings (text-embedding-3-small)
- Deterministic pure-Python n-gram hash vectorizer fallback
"""

import os
import sys
import re
import math
import hashlib
from pathlib import Path
from typing import List, Optional

# Ensure project root in sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR.parent))

from src.config import default_config, RAGConfig


STOP_WORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "but", "if", "or", "because",
    "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "should", "now", "what", "which", "who", "whom", "this", "that"
}


def _deterministic_hash_embedding(text: str, dim: int = 384) -> List[float]:
    """
    Pure Python deterministic semantic hash embedding generator.
    Filters common stopwords, applies term frequency weighting and bigrams
    with L2 normalization.
    """
    vec = [0.0] * dim
    clean_text = re.sub(r'[^a-zA-Z0-9_\s]', ' ', text.lower())
    words = [w for w in clean_text.split() if len(w) > 1 and w not in STOP_WORDS]
    if not words:
        return vec

    word_counts = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1

    # Unigram features with sublinear term frequency
    for word, count in word_counts.items():
        tf_weight = 1.0 + math.log(count)
        # 3 independent hash probes for better bucket spread
        h1 = int(hashlib.md5(word.encode("utf-8")).hexdigest()[:8], 16)
        h2 = int(hashlib.sha1(word.encode("utf-8")).hexdigest()[:8], 16)
        
        dim1 = h1 % dim
        dim2 = h2 % dim
        
        vec[dim1] += tf_weight
        vec[dim2] += tf_weight * 0.5

    # Bigram features
    for idx in range(len(words) - 1):
        bigram = f"{words[idx]}_{words[idx+1]}"
        bh = int(hashlib.md5(bigram.encode("utf-8")).hexdigest()[:8], 16)
        bdim = bh % dim
        vec[bdim] += 1.5

    # L2 normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


class EmbeddingGenerator:
    """Universal embedding generator with local, API, and fallback backends."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or default_config
        self.provider = self.config.embedding_provider.lower()
        self.model_name = self.config.embedding_model
        self._st_model = None
        self._init_backend()

    def _init_backend(self):
        """Initialize the requested embedding backend or flag fallback."""
        if self.provider == "local":
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(self.model_name)
                print(f"[EmbeddingGenerator] Loaded local SentenceTransformer: {self.model_name}")
            except Exception as e:
                print(f"[EmbeddingGenerator] Local sentence-transformers not available ({e}). Using deterministic embedding fallback.")
                self.provider = "deterministic"
        elif self.provider == "gemini":
            if not self.config.gemini_api_key:
                print("[EmbeddingGenerator] GEMINI_API_KEY not set. Using deterministic fallback.")
                self.provider = "deterministic"
        elif self.provider == "openai":
            if not self.config.openai_api_key:
                print("[EmbeddingGenerator] OPENAI_API_KEY not set. Using deterministic fallback.")
                self.provider = "deterministic"

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings into vector representations."""
        if not texts:
            return []

        # Local SentenceTransformer
        if self.provider == "local" and self._st_model is not None:
            try:
                embeddings = self._st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                return embeddings.tolist()
            except Exception as e:
                print(f"[Warning] SentenceTransformer encode failed: {e}. Falling back.")

        # Google Gemini API
        elif self.provider == "gemini" and self.config.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config.gemini_api_key)
                res = genai.embed_content(
                    model="models/text-embedding-004",
                    content=texts,
                    task_type="retrieval_document"
                )
                if "embedding" in res:
                    return res["embedding"]
            except Exception as e:
                print(f"[Warning] Gemini API embedding error: {e}. Falling back.")

        # OpenAI API
        elif self.provider == "openai" and self.config.openai_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.config.openai_api_key)
                response = client.embeddings.create(
                    input=texts,
                    model=self.model_name or "text-embedding-3-small"
                )
                return [d.embedding for d in response.data]
            except Exception as e:
                print(f"[Warning] OpenAI API embedding error: {e}. Falling back.")

        # Deterministic pure-python fallback
        return [_deterministic_hash_embedding(t) for t in texts]


if __name__ == "__main__":
    embedder = EmbeddingGenerator()
    test_texts = [
        "LPIPS Learned Perceptual Image Patch Similarity deep features",
        "BRISQUE blind no reference spatial domain MSCN coefficients"
    ]
    vectors = embedder.embed_texts(test_texts)
    print(f"Generated {len(vectors)} embeddings of dimension {len(vectors[0])}.")

"""
Centralized Configuration for IQA-RAG.

Holds paths, model hyperparameters, chunking parameters, and API configuration.
Supports parameter sweeps for evaluation and comparison runs.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Optional dotenv loader
try:
    from dotenv import load_dotenv
    # Look for .env in project root or current directory
    _possible_env = Path(__file__).resolve().parent.parent / ".env"
    if _possible_env.exists():
        load_dotenv(_possible_env)
    else:
        load_dotenv()
except ImportError:
    pass


@dataclass
class RAGConfig:
    """Master configuration class for the IQA-RAG pipeline."""

    # Project Directories
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(init=False)
    raw_pdfs_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    vectorstore_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    eval_set_path: Path = field(init=False)

    # Ingestion & Chunking Hyperparameters
    chunk_size: int = 300
    chunk_overlap: int = 50
    min_chunk_size: int = 50
    section_aware: bool = True

    # Vector Indexing
    embedding_provider: str = "local"  # 'local', 'gemini', 'openai', 'tfidf'
    embedding_model: str = "all-MiniLM-L6-v2"
    collection_name: str = "iqa_literature"
    distance_metric: str = "cosine"  # 'cosine', 'l2', 'ip'

    # Retrieval
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.0

    # Generator / LLM
    llm_provider: str = "gemini"  # 'gemini', 'openai', 'fallback'
    llm_model: str = "gemini-1.5-flash"
    temperature: float = 0.2
    max_output_tokens: int = 1024

    # API Keys (loaded from env)
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    def __post_init__(self):
        # Resolve all child directories
        self.data_dir = self.project_root / "data"
        self.raw_pdfs_dir = self.data_dir / "raw_pdfs"
        self.processed_dir = self.data_dir / "processed"
        self.vectorstore_dir = self.data_dir / "vectorstore"
        self.results_dir = self.project_root / "results"
        self.eval_set_path = self.data_dir / "eval_set.json"

        # Ensure essential directories exist
        for d in [self.data_dir, self.raw_pdfs_dir, self.processed_dir, self.vectorstore_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Pull environment variables if not explicitly provided
        if self.gemini_api_key is None:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Allow environment variable overrides for models & providers
        self.llm_provider = os.getenv("LLM_PROVIDER", self.llm_provider)
        self.llm_model = os.getenv("LLM_MODEL", self.llm_model)
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", self.embedding_provider)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", self.embedding_model)

        # Allow numeric overrides
        if os.getenv("CHUNK_SIZE"):
            self.chunk_size = int(os.getenv("CHUNK_SIZE"))
        if os.getenv("CHUNK_OVERLAP"):
            self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))
        if os.getenv("RETRIEVAL_TOP_K"):
            self.retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K"))

    def to_dict(self) -> dict:
        """Serialize configuration to dictionary (omitting secrets)."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "section_aware": self.section_aware,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "collection_name": self.collection_name,
            "retrieval_top_k": self.retrieval_top_k,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }


# Global default configuration instance
default_config = RAGConfig()

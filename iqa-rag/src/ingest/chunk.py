"""
Section-Aware Document Chunking for IQA-RAG.

Splits parsed academic papers into semantically coherent chunks that respect
section boundaries, preserve sentence continuity, and include rich metadata
for downstream retrieval and citation generation.
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root in sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR.parent))

from src.config import default_config, RAGConfig


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex boundary detection."""
    # Matches sentence terminators followed by whitespace and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
    return [s.strip() for s in sentences if s.strip()]


def estimate_tokens(text: str) -> int:
    """Approximate token count (rule of thumb: ~4 characters per token)."""
    return max(1, len(text) // 4)


class SectionAwareChunker:
    """
    Chunks documents while strictly respecting section boundaries,
    ensuring no chunk blends two distinct paper sections.
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or default_config
        self.chunk_size = self.config.chunk_size
        self.chunk_overlap = self.config.chunk_overlap
        self.min_chunk_size = self.config.min_chunk_size
        self.output_dir = self.config.processed_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def chunk_section(
        self,
        paper_title: str,
        section_title: str,
        page_number: int,
        section_text: str,
        paper_id: str
    ) -> List[Dict[str, Any]]:
        """
        Split a single section's text into overlapping chunks respecting sentences.
        """
        # If section is shorter than min_chunk_size, retain as single chunk or skip if trivial
        if len(section_text.strip()) < self.min_chunk_size:
            return []

        # Convert chunk_size from approximate tokens to characters (~4 chars/token)
        target_char_len = self.chunk_size * 4
        overlap_char_len = self.chunk_overlap * 4

        sentences = split_into_sentences(section_text)
        if not sentences:
            sentences = [section_text]

        chunks = []
        current_sentences: List[str] = []
        current_len = 0
        chunk_idx = 0

        for sentence in sentences:
            sent_len = len(sentence)
            if current_len + sent_len > target_char_len and current_sentences:
                # Flush current chunk
                chunk_str = " ".join(current_sentences).strip()
                if len(chunk_str) >= self.min_chunk_size:
                    chunk_id = f"{paper_id}_{slugify(section_title)}_c{chunk_idx:03d}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "paper_id": paper_id,
                        "paper_title": paper_title,
                        "section_title": section_title,
                        "page": page_number,
                        "text": chunk_str,
                        "token_count": estimate_tokens(chunk_str),
                        "char_count": len(chunk_str)
                    })
                    chunk_idx += 1

                # Retain overlap from end of sentence list
                overlap_buffer = []
                overlap_len = 0
                for prev_sent in reversed(current_sentences):
                    if overlap_len + len(prev_sent) <= overlap_char_len:
                        overlap_buffer.insert(0, prev_sent)
                        overlap_len += len(prev_sent)
                    else:
                        break
                current_sentences = overlap_buffer
                current_len = overlap_len

            current_sentences.append(sentence)
            current_len += sent_len

        # Flush final sentences in section
        if current_sentences:
            chunk_str = " ".join(current_sentences).strip()
            if len(chunk_str) >= self.min_chunk_size:
                chunk_id = f"{paper_id}_{slugify(section_title)}_c{chunk_idx:03d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "paper_title": paper_title,
                    "section_title": section_title,
                    "page": page_number,
                    "text": chunk_str,
                    "token_count": estimate_tokens(chunk_str),
                    "char_count": len(chunk_str)
                })

        return chunks

    def chunk_document(self, parsed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk all sections of a single parsed document."""
        paper_title = parsed_doc.get("title", "Unknown Paper")
        paper_id = slugify(parsed_doc.get("file_name", paper_title).replace(".pdf", ""))
        sections = parsed_doc.get("sections", [])

        all_chunks = []
        for sec in sections:
            sec_chunks = self.chunk_section(
                paper_title=paper_title,
                section_title=sec["section_title"],
                page_number=sec.get("page", 1),
                section_text=sec["text"],
                paper_id=paper_id
            )
            all_chunks.extend(sec_chunks)

        return all_chunks

    def chunk_and_save(self, parsed_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk all parsed documents and save to JSON and Parquet."""
        json_path = self.output_dir / "chunks.json"
        existing_chunks_map = {}
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    old_list = json.load(f)
                    for chk in old_list:
                        existing_chunks_map[chk["chunk_id"]] = chk
            except Exception:
                pass

        for doc in parsed_docs:
            chunks = self.chunk_document(doc)
            for chk in chunks:
                existing_chunks_map[chk["chunk_id"]] = chk

        all_chunks = list(existing_chunks_map.values())

        # Save to JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(all_chunks)} total chunks to {json_path}")

        # Save to Parquet if pandas and pyarrow/fastparquet are available
        parquet_path = self.output_dir / "chunks.parquet"
        try:
            import pandas as pd
            df = pd.DataFrame(all_chunks)
            df.to_parquet(parquet_path, index=False)
            print(f"Saved chunks dataframe to {parquet_path}")
        except Exception as e:
            print(f"[Note] Parquet save skipped ({e}); JSON preserved.")

        return all_chunks


def slugify(text: str) -> str:
    """Convert string to alphanumeric slug."""
    text = re.sub(r'[^a-zA-Z0-9]', '_', text.lower())
    text = re.sub(r'_+', '_', text).strip('_')
    return text[:32] or "sec"


if __name__ == "__main__":
    from src.ingest.parse import PDFParser
    parser = PDFParser()
    docs = parser.parse_all_raw_pdfs()
    chunker = SectionAwareChunker()
    chunks = chunker.chunk_and_save(docs)
    print(f"Successfully generated {len(chunks)} chunks across {len(docs)} papers.")

"""
PDF Parser for IQA Research Papers.

Extracts text page-by-page, strips header/footer noise and arXiv stamps,
and performs heuristic section header detection to output clean, structured sections.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure project root in sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR.parent))

from src.config import default_config, RAGConfig


# Common section header patterns in computer vision & ML academic papers
SECTION_PATTERNS = [
    re.compile(r'^(?:(?:\d+|[IVXLCDM]+)\.?\s+)?(Abstract)\b', re.IGNORECASE),
    re.compile(r'^(?:(?:\d+|[IVXLCDM]+)\.?\s+)?(Introduction)\b', re.IGNORECASE),
    re.compile(r'^(?:(?:\d+|[IVXLCDM]+)\.?\s+)?(Related\s+Work|Background)\b', re.IGNORECASE),
    re.compile(r'^(?:(?:\d+|[IVXLCDM]+)\.?\s+)?(Proposed\s+Method|Methodology|Method|Architecture)\b', re.IGNORECASE),
    re.compile(r'^(?:(?:\d+|[IVXLCDM]+)\.?\s+)?(Experiments|Experimental\s+Results|Evaluations|Results)\b', re.IGNORECASE),
    re.compile(r'^(?:(?:\d+|[IVXLCDM]+)\.?\s+)?(Ablation\s+Study|Analysis|Discussion)\b', re.IGNORECASE),
    re.compile(r'^(?:(?:\d+|[IVXLCDM]+)\.?\s+)?(Conclusion|Concluding\s+Remarks)\b', re.IGNORECASE),
    re.compile(r'^(?:(?:\d+|[IVXLCDM]+)\.?\s+)?(References|Bibliography)\b', re.IGNORECASE),
]

ARXIV_STAMP_PATTERN = re.compile(r'arXiv:\d{4}\.\d{4,5}(?:v\d+)?\s+\[[\w\.-]+\]\s+\d{1,2}\s+\w+\s+\d{4}', re.IGNORECASE)
PAGE_NUMBER_PATTERN = re.compile(r'^\s*(?:Page\s+)?\d+\s*$', re.IGNORECASE)


def clean_page_text(raw_text: str) -> str:
    """Clean common artifacts from extracted PDF page text."""
    if not raw_text:
        return ""

    lines = raw_text.splitlines()
    cleaned_lines = []

    for line in lines:
        line_str = line.strip()
        # Filter out standalone page numbers
        if PAGE_NUMBER_PATTERN.match(line_str):
            continue
        # Filter out arXiv timestamp headers
        if ARXIV_STAMP_PATTERN.search(line_str):
            line_str = ARXIV_STAMP_PATTERN.sub('', line_str).strip()
        if not line_str:
            continue
        cleaned_lines.append(line_str)

    text = "\n".join(cleaned_lines)
    # Fix hyphenation across line breaks: e.g. "distor-\ntion" -> "distortion"
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Collapse multiple consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text per page using pypdf or PyMuPDF, with graceful fallback.
    Returns list of {'page': int, 'text': str}.
    """
    pages_data = []

    # Attempt 1: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        for idx, page in enumerate(reader.pages):
            raw = page.extract_text() or ""
            pages_data.append({"page": idx + 1, "text": clean_page_text(raw)})
        if pages_data:
            return pages_data
    except ImportError:
        pass
    except Exception as e:
        print(f"[Warning] pypdf extraction failed for {pdf_path}: {e}")

    # Attempt 2: PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        for idx in range(len(doc)):
            raw = doc[idx].get_text()
            pages_data.append({"page": idx + 1, "text": clean_page_text(raw)})
        doc.close()
        if pages_data:
            return pages_data
    except ImportError:
        pass
    except Exception as e:
        print(f"[Warning] PyMuPDF extraction failed for {pdf_path}: {e}")

    # Attempt 3: If it is already a text file disguised or fallback
    try:
        with open(pdf_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            # If text content contains meaningful readable words
            if len(re.findall(r'\b[a-zA-Z]{3,}\b', content)) > 20:
                pages_data.append({"page": 1, "text": clean_page_text(content)})
    except Exception:
        pass

    return pages_data


def detect_sections(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Segment the document into logical sections based on detected headings.
    """
    sections: List[Dict[str, Any]] = []
    current_section = "Abstract"
    current_page = 1
    current_buffer: List[str] = []

    for page_info in pages_data:
        p_num = page_info["page"]
        text = page_info["text"]
        lines = text.split("\n")

        for line in lines:
            trimmed = line.strip()
            # Check if line matches a known section header
            matched_header = None
            for pat in SECTION_PATTERNS:
                m = pat.match(trimmed)
                if m:
                    # Capture standardized heading name
                    matched_header = trimmed
                    break

            if matched_header and len(trimmed) < 60:
                # Save previous section if buffer is populated
                if current_buffer:
                    sec_text = "\n".join(current_buffer).strip()
                    if sec_text:
                        sections.append({
                            "section_title": current_section,
                            "page": current_page,
                            "text": sec_text
                        })
                    current_buffer = []

                current_section = matched_header
                current_page = p_num
            else:
                current_buffer.append(line)

    # Flush final section
    if current_buffer:
        sec_text = "\n".join(current_buffer).strip()
        if sec_text:
            sections.append({
                "section_title": current_section,
                "page": current_page,
                "text": sec_text
            })

    # If no section headings were detected, wrap everything as full text
    if not sections and pages_data:
        full = "\n\n".join([p["text"] for p in pages_data if p["text"]])
        if full:
            sections.append({
                "section_title": "Main Content",
                "page": 1,
                "text": full
            })

    return sections


class PDFParser:
    """End-to-end parser extracting structured sections from academic PDFs."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or default_config

    def parse_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Parse a single PDF file and return clean metadata and sections.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        pages_data = extract_text_from_pdf(pdf_path)
        sections = detect_sections(pages_data)

        # Infer title from filename or first page
        title = pdf_path.stem.replace("_", " ")

        return {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "title": title,
            "num_pages": len(pages_data),
            "sections": sections,
            "full_text": "\n\n".join([s["text"] for s in sections])
        }

    def parse_all_raw_pdfs(self) -> List[Dict[str, Any]]:
        """Parse all PDFs currently in raw_pdfs directory."""
        pdf_files = list(self.config.raw_pdfs_dir.glob("*.pdf"))
        parsed_docs = []
        for pdf_file in pdf_files:
            print(f"Parsing: {pdf_file.name}")
            parsed_docs.append(self.parse_pdf(pdf_file))
        return parsed_docs


if __name__ == "__main__":
    parser = PDFParser()
    docs = parser.parse_all_raw_pdfs()
    print(f"Parsed {len(docs)} documents.")
    for doc in docs:
        print(f"- {doc['title']} ({doc['num_pages']} pages, {len(doc['sections'])} sections)")

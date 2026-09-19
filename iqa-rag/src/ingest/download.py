"""
arXiv Paper Downloader for IQA-RAG.

Fetches research papers related to Image Quality Assessment (IQA)
from the arXiv API, downloads PDFs, and caches paper metadata.
Uses the `arxiv` library if available, with pure urllib fallback.
"""

import os
import sys
import json
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR.parent))

from src.config import default_config, RAGConfig


def sanitize_filename(title: str) -> str:
    """Convert paper title to a clean, filesystem-safe filename."""
    clean = re.sub(r'[^a-zA-Z0-9_\-\s]', '', title)
    clean = re.sub(r'\s+', '_', clean.strip())
    return clean[:80] or "arxiv_paper"


class ArxivDownloader:
    """Downloads arXiv papers matching IQA topics or specific arXiv IDs."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or default_config
        self.output_dir = self.config.raw_pdfs_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.output_dir / "metadata.json"

    def _load_metadata(self) -> Dict[str, Any]:
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_metadata(self, metadata: Dict[str, Any]):
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def fetch_via_urllib(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Fetch papers directly using arXiv Atom API."""
        encoded_query = urllib.parse.quote(query)
        url = (
            f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}"
            f"&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
        )
        papers = []
        try:
            import ssl
            ctx = ssl.create_default_context()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "DocChatter-IQA-RAG/1.0"})
                with urllib.request.urlopen(req, timeout=20, context=ctx) as response:
                    xml_data = response.read()
            except Exception:
                ctx = ssl._create_unverified_context()
                req = urllib.request.Request(url, headers={"User-Agent": "DocChatter-IQA-RAG/1.0"})
                with urllib.request.urlopen(req, timeout=20, context=ctx) as response:
                    xml_data = response.read()
            root = ET.fromstring(xml_data)
            atom_ns = "{http://www.w3.org/2005/Atom}"
            for entry in root.findall(f"{atom_ns}entry"):
                title_elem = entry.find(f"{atom_ns}title")
                summary_elem = entry.find(f"{atom_ns}summary")
                id_elem = entry.find(f"{atom_ns}id")
                published_elem = entry.find(f"{atom_ns}published")
                authors = [
                    a.find(f"{atom_ns}name").text
                    for a in entry.findall(f"{atom_ns}author")
                    if a.find(f"{atom_ns}name") is not None
                ]

                arxiv_id = id_elem.text.split("/abs/")[-1] if id_elem is not None else ""
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""

                title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else "Untitled"
                summary = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""
                published = published_elem.text if published_elem is not None else ""

                papers.append({
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": authors,
                    "summary": summary,
                    "published": published,
                    "pdf_url": pdf_url
                })
        except Exception as e:
            print(f"[Warning] Direct arXiv API fetch error: {e}")
        return papers

    def search_and_download(
        self,
        query: str = "Image Quality Assessment",
        max_results: int = 5,
        paper_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search arXiv for IQA papers and download their PDFs.
        """
        papers_meta = self._load_metadata()
        downloaded = []

        papers_to_process = []
        if paper_ids:
            for pid in paper_ids:
                papers_to_process.extend(self.fetch_via_urllib(f"id:{pid}", max_results=1))
        else:
            # Try official arxiv package first
            try:
                import arxiv
                client = arxiv.Client()
                search = arxiv.Search(
                    query=query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                for r in client.results(search):
                    papers_to_process.append({
                        "arxiv_id": r.get_short_id(),
                        "title": r.title.strip().replace("\n", " "),
                        "authors": [a.name for a in r.authors],
                        "summary": r.summary.strip().replace("\n", " "),
                        "published": str(r.published),
                        "pdf_url": r.pdf_url
                    })
            except Exception:
                papers_to_process = self.fetch_via_urllib(query, max_results=max_results)

        for paper in papers_to_process:
            pid = paper["arxiv_id"]
            title_slug = sanitize_filename(paper["title"])
            pdf_filename = f"{title_slug}.pdf"
            pdf_filepath = self.output_dir / pdf_filename

            if not pdf_filepath.exists() and paper.get("pdf_url"):
                print(f"Downloading: {paper['title'][:60]}... -> {pdf_filename}")
                try:
                    import ssl
                    ctx = ssl._create_unverified_context()
                    req = urllib.request.Request(
                        paper["pdf_url"],
                        headers={"User-Agent": "DocChatter-IQA-RAG/1.0"}
                    )
                    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp, open(pdf_filepath, "wb") as out:
                        out.write(resp.read())
                    paper["local_pdf_path"] = str(pdf_filepath)
                    papers_meta[pid] = paper
                    downloaded.append(paper)
                except Exception as ex:
                    print(f"Failed to download {paper['pdf_url']}: {ex}")
            else:
                paper["local_pdf_path"] = str(pdf_filepath)
                papers_meta[pid] = paper
                downloaded.append(paper)

        self._save_metadata(papers_meta)
        return downloaded


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch and download IQA papers from arXiv")
    parser.add_argument("--query", type=str, default="Image Quality Assessment LPIPS BRISQUE", help="Search query")
    parser.add_argument("--max-results", type=int, default=3, help="Maximum number of papers to download")
    args = parser.parse_args()

    downloader = ArxivDownloader()
    print(f"Searching arXiv for '{args.query}' (max {args.max_results})...")
    res = downloader.search_and_download(query=args.query, max_results=args.max_results)
    print(f"Successfully processed {len(res)} papers in {downloader.output_dir}")

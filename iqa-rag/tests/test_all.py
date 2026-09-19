"""
Automated Test Suite for IQA-RAG.

Tests all core subsystems:
1. Configuration & paths
2. Ingestion, PDF parsing, and section-aware chunking
3. Embedding generation and vector store operations
4. Pipeline retrieval, generation, and unified RAG flow
5. Evaluation metrics (Hit-Rate, Precision, MRR, Faithfulness, F1)
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAGConfig, default_config
from src.ingest.download import sanitize_filename
from src.ingest.parse import clean_page_text, detect_sections
from src.ingest.chunk import split_into_sentences, SectionAwareChunker
from src.index.embed import EmbeddingGenerator
from src.index.vectorstore import IQAVectorStore
from src.pipeline.retriever import IQARetriever
from src.pipeline.generator import IQAGenerator
from src.pipeline.rag import IQARAGPipeline
from src.eval.metrics import (
    calculate_hit_rate_at_k,
    calculate_precision_at_k,
    calculate_mrr_at_k,
    evaluate_faithfulness,
    calculate_token_f1
)


class TestConfig(unittest.TestCase):
    def test_paths_exist(self):
        cfg = RAGConfig()
        self.assertTrue(cfg.data_dir.exists())
        self.assertTrue(cfg.raw_pdfs_dir.exists())
        self.assertTrue(cfg.processed_dir.exists())
        self.assertTrue(cfg.vectorstore_dir.exists())
        self.assertTrue(cfg.results_dir.exists())

    def test_serialization(self):
        cfg = RAGConfig(chunk_size=400, retrieval_top_k=7)
        d = cfg.to_dict()
        self.assertEqual(d["chunk_size"], 400)
        self.assertEqual(d["retrieval_top_k"], 7)


class TestIngest(unittest.TestCase):
    def test_sanitize_filename(self):
        raw = "LPIPS: The Unreasonable Effectiveness / of Deep Features?"
        clean = sanitize_filename(raw)
        self.assertNotIn(":", clean)
        self.assertNotIn("/", clean)
        self.assertNotIn("?", clean)

    def test_clean_page_text(self):
        raw = "Header info\narXiv:2004.09058v1 [cs.CV] 20 Apr 2020\n12\nActual content here."
        cleaned = clean_page_text(raw)
        self.assertNotIn("arXiv:2004.09058v1", cleaned)
        self.assertIn("Actual content here.", cleaned)

    def test_sentence_splitting(self):
        text = "This is the first sentence. Here is the second sentence! Is this third?"
        sents = split_into_sentences(text)
        self.assertEqual(len(sents), 3)

    def test_section_aware_chunking(self):
        chunker = SectionAwareChunker(config=RAGConfig(chunk_size=100, chunk_overlap=20))
        doc = {
            "title": "Test Paper",
            "file_name": "test_paper.pdf",
            "sections": [
                {
                    "section_title": "Introduction",
                    "page": 1,
                    "text": "Image Quality Assessment is an important field. It evaluates image fidelity across distortions."
                },
                {
                    "section_title": "Methodology",
                    "page": 2,
                    "text": "We formulate our metric using deep feature distances across multiple convolutional layers."
                }
            ]
        }
        chunks = chunker.chunk_document(doc)
        self.assertGreaterEqual(len(chunks), 2)
        # Ensure section titles are preserved and distinct
        sections_in_chunks = set(c["section_title"] for c in chunks)
        self.assertIn("Introduction", sections_in_chunks)
        self.assertIn("Methodology", sections_in_chunks)


class TestIndex(unittest.TestCase):
    def setUp(self):
        self.embedder = EmbeddingGenerator()
        self.store = IQAVectorStore()

    def test_embeddings_generation(self):
        texts = ["LPIPS deep feature perceptual distance", "BRISQUE natural scene statistics"]
        vecs = self.embedder.embed_texts(texts)
        self.assertEqual(len(vecs), 2)
        self.assertEqual(len(vecs[0]), len(vecs[1]))
        self.assertGreater(len(vecs[0]), 0)

    def test_vectorstore_indexing_and_query(self):
        test_chunks = [
            {
                "chunk_id": "unit_test_chk_01",
                "paper_id": "test_paper",
                "paper_title": "Deep Perceptual Metrics",
                "section_title": "Introduction",
                "page": 1,
                "text": "Deep convolutional features correlate strongly with human visual perception and perceptual similarity.",
                "token_count": 15
            }
        ]
        count = self.store.add_chunks(test_chunks)
        self.assertGreaterEqual(count, 1)

        query_res = self.store.query("perceptual similarity deep convolutional features", top_k=1)
        self.assertGreaterEqual(len(query_res), 1)
        self.assertIn("Deep Perceptual Metrics", query_res[0]["metadata"]["paper_title"])


class TestPipeline(unittest.TestCase):
    def test_pipeline_execution(self):
        pipeline = IQARAGPipeline()
        query = "How does LPIPS compute deep feature perceptual distance?"
        res = pipeline.run(query, top_k=2)

        self.assertEqual(res["query"], query)
        self.assertIn("answer", res)
        self.assertGreater(len(res["answer"]), 10)
        self.assertGreater(len(res["retrieved_docs"]), 0)
        self.assertIn("total_latency_ms", res)


class TestEvaluationMetrics(unittest.TestCase):
    def setUp(self):
        self.retrieved_sample = [
            {
                "paper_title": "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)",
                "section_title": "Methodology",
                "text": "LPIPS computes deep feature distance using normalized activations.",
                "page": 3
            },
            {
                "paper_title": "No-Reference Image Quality Assessment in the Spatial Domain (BRISQUE)",
                "section_title": "MSCN",
                "text": "BRISQUE fits AGGD distributions to MSCN coefficients.",
                "page": 2
            }
        ]

    def test_hit_rate(self):
        hr_lpips = calculate_hit_rate_at_k(
            self.retrieved_sample,
            target_paper="LPIPS",
            target_section="Methodology",
            keywords=["deep features", "normalized"],
            k=1
        )
        self.assertEqual(hr_lpips, 1.0)

        hr_brisque_k1 = calculate_hit_rate_at_k(
            self.retrieved_sample,
            target_paper="BRISQUE",
            target_section="MSCN",
            keywords=["AGGD", "MSCN"],
            k=1
        )
        self.assertEqual(hr_brisque_k1, 0.0)

        hr_brisque_k2 = calculate_hit_rate_at_k(
            self.retrieved_sample,
            target_paper="BRISQUE",
            target_section="MSCN",
            keywords=["AGGD", "MSCN"],
            k=2
        )
        self.assertEqual(hr_brisque_k2, 1.0)

    def test_precision_at_k(self):
        prec = calculate_precision_at_k(
            self.retrieved_sample,
            target_paper="LPIPS",
            target_section="Methodology",
            keywords=["deep features"],
            k=2
        )
        self.assertEqual(prec, 0.5)

    def test_mrr(self):
        mrr = calculate_mrr_at_k(
            self.retrieved_sample,
            target_paper="BRISQUE",
            target_section="MSCN",
            keywords=["AGGD"],
            k=2
        )
        self.assertEqual(mrr, 0.5)

    def test_faithfulness(self):
        answer = "LPIPS computes deep feature distance using normalized activations. [Source 1]"
        score = evaluate_faithfulness(answer, self.retrieved_sample)
        self.assertGreater(score, 0.7)

    def test_token_f1(self):
        gt = "LPIPS evaluates perceptual distance via deep feature activations."
        gen = "LPIPS computes perceptual distance using deep feature activations."
        f1 = calculate_token_f1(gen, gt)
        self.assertGreater(f1, 0.6)


if __name__ == "__main__":
    unittest.main()

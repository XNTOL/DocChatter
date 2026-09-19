"""
Evaluation Runner for IQA-RAG.

Executes eval_set.json through the RAG pipeline, calculates Hit-Rate@K,
Precision@K, MRR, Faithfulness, and Latency, and saves detailed run results to results/.
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root in sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR.parent))

from src.config import default_config, RAGConfig
from src.pipeline.rag import IQARAGPipeline
from src.eval.metrics import evaluate_sample


class RAGEvaluator:
    """Orchestrates benchmark evaluation over curated Q&A evaluation datasets."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or default_config
        self.pipeline = IQARAGPipeline(self.config)

    def run_benchmark(
        self,
        eval_set_path: Optional[Path] = None,
        k_values: List[int] = [1, 3, 5],
        run_tag: str = "eval"
    ) -> Dict[str, Any]:
        """
        Run the full evaluation set through the pipeline and compute benchmark metrics.
        """
        target_path = Path(eval_set_path or self.config.eval_set_path)
        if not target_path.exists():
            raise FileNotFoundError(f"Evaluation set not found at: {target_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            eval_set = json.load(f)

        print(f"Starting evaluation on {len(eval_set)} questions (tag: {run_tag})...")
        sample_results = []
        start_time = time.perf_counter()

        for idx, item in enumerate(eval_set, start=1):
            q = item["question"]
            pipeline_res = self.pipeline.run(q, top_k=max(k_values))
            sample_eval = evaluate_sample(item, pipeline_res, k_values=k_values)
            sample_eval["generated_answer"] = pipeline_res.get("answer", "")
            sample_results.append(sample_eval)

            if idx % 5 == 0 or idx == len(eval_set):
                print(f"Evaluated [{idx}/{len(eval_set)}] queries...")

        total_wall_time = round(time.perf_counter() - start_time, 2)

        # Aggregate metrics
        num_samples = len(sample_results)
        summary = {
            "timestamp": datetime.now().isoformat(),
            "run_tag": run_tag,
            "total_queries": num_samples,
            "wall_time_sec": total_wall_time,
            "config": self.config.to_dict(),
            "metrics": {}
        }

        # Calculate means
        for k in k_values:
            summary["metrics"][f"hit_rate@{k}"] = round(sum(s[f"hit_rate@{k}"] for s in sample_results) / num_samples, 4)
            summary["metrics"][f"precision@{k}"] = round(sum(s[f"precision@{k}"] for s in sample_results) / num_samples, 4)
            summary["metrics"][f"mrr@{k}"] = round(sum(s[f"mrr@{k}"] for s in sample_results) / num_samples, 4)

        summary["metrics"]["mean_faithfulness"] = round(sum(s["faithfulness"] for s in sample_results) / num_samples, 4)
        summary["metrics"]["mean_answer_f1"] = round(sum(s["answer_f1"] for s in sample_results) / num_samples, 4)
        summary["metrics"]["avg_latency_ms"] = round(sum(s["total_latency_ms"] for s in sample_results) / num_samples, 2)

        # Build final report
        report = {
            "summary": summary,
            "per_query_results": sample_results
        }

        # Save to results/
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_filename = f"run_{date_str}_{run_tag}.json"
        out_path = self.config.results_dir / out_filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\nEvaluation Complete! Saved report to: {out_path}")
        self._print_summary(summary)

        return report

    @staticmethod
    def _print_summary(summary: Dict[str, Any]):
        m = summary["metrics"]
        print("=" * 60)
        print(f"BENCHMARK SUMMARY [{summary['run_tag']}]")
        print(f"Total Queries: {summary['total_queries']} | Wall Time: {summary['wall_time_sec']}s")
        print("-" * 60)
        for k, v in m.items():
            print(f"  {k:<24}: {v}")
        print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run IQA-RAG evaluation benchmark")
    parser.add_argument("--tag", type=str, default="test", help="Run label tag")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top-k")
    args = parser.parse_args()

    evaluator = RAGEvaluator()
    evaluator.run_benchmark(run_tag=args.tag, k_values=[1, 3, args.top_k])

"""
Generator Module for IQA-RAG.

Constructs grounded answers with in-text citations using retrieved paper contexts.
Supports Google Gemini, OpenAI, and an offline grounded extractor fallback.
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root in sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR.parent))

from src.config import default_config, RAGConfig
from src.pipeline.retriever import IQARetriever


SYSTEM_PROMPT = """You are an expert academic research assistant specializing in Image Quality Assessment (IQA) literature.
Your task is to answer questions accurately and thoroughly based strictly on the provided research paper contexts.

Guidelines:
1. Grounding: Answer ONLY using information explicitly supported in the provided context. Do not speculate or extrapolate beyond the text.
2. In-text Citations: Explicitly cite source papers for every technical claim using [Source N] or [Paper Title, Section].
3. Clarity & Technical Depth: Include relevant mathematical formulations, metric names (e.g., SRCC, PLCC, MSCN, LPIPS), and experimental benchmarks mentioned in the context.
4. Insufficient Context: If the provided contexts do not contain enough evidence to answer the question, state clearly: "Based on the provided research context, there is insufficient information to answer this question."
"""


def _format_rag_prompt(query: str, context_str: str) -> str:
    """Combine system directives, context, and user question."""
    return f"""{SYSTEM_PROMPT}

=== RETRIEVED CONTEXT ===
{context_str}

=== USER QUESTION ===
{query}

=== GROUNDED ANSWER (with citations) ===
"""


def _offline_grounded_generator(query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Offline grounded answer synthesizer.
    Selects the most relevant sentences from retrieved chunks, attaches source citations,
    and returns a structured response without requiring external API calls.
    """
    if not retrieved_docs:
        return {
            "answer": "Based on the provided research context, there is insufficient information to answer this question.",
            "citations": [],
            "model": "offline-grounded-fallback"
        }

    # Extract sentences from top docs
    query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
    selected_claims = []
    citations = []

    for d in retrieved_docs[:3]:
        text = d["text"]
        sentences = re.split(r'(?<=[.!?])\s+', text)
        top_sentence = None
        max_overlap = -1

        for sent in sentences:
            s_clean = sent.strip()
            if len(s_clean) < 30:
                continue
            s_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', s_clean.lower()))
            overlap = len(query_words.intersection(s_words))
            if overlap > max_overlap:
                max_overlap = overlap
                top_sentence = s_clean

        if top_sentence:
            cite_tag = f"[{d['paper_title']}, {d['section_title']}]"
            selected_claims.append(f"{top_sentence} {cite_tag}")
            citations.append(cite_tag)

    if not selected_claims:
        # Fallback to first chunk content
        first_doc = retrieved_docs[0]
        cite_tag = f"[{first_doc['paper_title']}, {first_doc['section_title']}]"
        answer = f"{first_doc['text'].strip()} {cite_tag}"
        citations = [cite_tag]
    else:
        answer = " ".join(selected_claims)

    return {
        "answer": answer,
        "citations": list(set(citations)),
        "model": "offline-grounded-fallback"
    }


class IQAGenerator:
    """Generates grounded responses using configured LLM provider."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or default_config
        self.provider = self.config.llm_provider.lower()
        self.model_name = self.config.llm_model
        self.temperature = self.config.temperature

    def generate(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate answer from query and retrieved documents.
        """
        context_str = IQARetriever.format_context_for_prompt(retrieved_docs)
        prompt = _format_rag_prompt(query, context_str)

        # 1. Google Gemini Provider
        if self.provider == "gemini" and self.config.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config.gemini_api_key)
                model = genai.GenerativeModel(self.model_name or "gemini-1.5-flash")
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.config.max_output_tokens
                    )
                )
                answer_text = response.text.strip()
                citations = re.findall(r'\[Source \d+\]|\[[^\]]+,\s*[^\]]+\]', answer_text)
                return {
                    "answer": answer_text,
                    "citations": citations,
                    "model": f"gemini:{self.model_name}",
                    "prompt": prompt
                }
            except Exception as e:
                print(f"[Warning] Gemini generation failed: {e}. Falling back to offline grounded generator.")

        # 2. OpenAI Provider
        elif self.provider == "openai" and self.config.openai_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.config.openai_api_key)
                response = client.chat.completions.create(
                    model=self.model_name or "gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.config.max_output_tokens
                )
                answer_text = response.choices[0].message.content.strip()
                citations = re.findall(r'\[Source \d+\]|\[[^\]]+,\s*[^\]]+\]', answer_text)
                return {
                    "answer": answer_text,
                    "citations": citations,
                    "model": f"openai:{self.model_name}",
                    "prompt": prompt
                }
            except Exception as e:
                print(f"[Warning] OpenAI generation failed: {e}. Falling back to offline grounded generator.")

        # 3. Offline Grounded Fallback
        res = _offline_grounded_generator(query, retrieved_docs)
        res["prompt"] = prompt
        return res


# Alias
Generator = IQAGenerator


if __name__ == "__main__":
    generator = IQAGenerator()
    sample_docs = [
        {
            "rank": 1,
            "paper_title": "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)",
            "section_title": "Methodology",
            "page": 3,
            "text": "LPIPS measures perceptual distance by extracting features from convolutional networks like AlexNet or VGG, normalizing activations across channels, and computing spatial L2 distances."
        }
    ]
    ans = generator.generate("How does LPIPS compute perceptual distance?", sample_docs)
    print("Model:", ans["model"])
    print("Answer:", ans["answer"])
    print("Citations:", ans["citations"])

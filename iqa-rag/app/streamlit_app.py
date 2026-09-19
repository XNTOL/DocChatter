"""
DocChatter: IQA-RAG Streamlit Application.

Interactive research interface for Image Quality Assessment literature.
Retrieves arXiv paper chunks and generates grounded answers with citations.
"""

import sys
import json
import time
from pathlib import Path
import streamlit as st

# Set python path to include iqa-rag
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAGConfig, default_config
from src.pipeline.rag import IQARAGPipeline
from src.index.vectorstore import IQAVectorStore
from src.ingest.download import ArxivDownloader
from src.ingest.parse import PDFParser
from src.ingest.chunk import SectionAwareChunker


# Page Configuration
st.set_page_config(
    page_title="IQA-RAG: Image Quality Assessment Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style settings
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-box { background-color: #F3F4F6; border-radius: 8px; padding: 15px; border-left: 4px solid #3B82F6; }
    .citation-card { background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 6px; padding: 12px; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline(provider: str, api_key: str, top_k: int, temp: float):
    """Initialize and cache the RAG pipeline instance."""
    cfg = RAGConfig(
        llm_provider=provider,
        temperature=temp,
        retrieval_top_k=top_k,
        gemini_api_key=api_key if provider == "gemini" else None,
        openai_api_key=api_key if provider == "openai" else None
    )
    return IQARAGPipeline(config=cfg)


# Sidebar Controls
with st.sidebar:
    st.header("Configuration")

    provider = st.selectbox(
        "LLM Provider",
        options=["gemini", "openai", "fallback"],
        index=0,
        help="Select the LLM provider for grounded answer synthesis."
    )

    api_key = ""
    if provider in ["gemini", "openai"]:
        api_key = st.text_input(
            f"{provider.capitalize()} API Key",
            type="password",
            value="",
            help="Enter your API key or leave empty to use the environment variable."
        )

    top_k = st.slider("Retrieval Top-K", min_value=1, max_value=10, value=5, step=1)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.markdown("---")
    st.subheader("Corpus Statistics")
    chunks_path = default_config.processed_dir / "chunks.json"
    num_chunks = 0
    if chunks_path.exists():
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                num_chunks = len(json.load(f))
        except Exception:
            pass

    st.metric("Indexed Chunks", num_chunks)
    st.caption("Vector Store: ChromaDB or In-Memory Store")

    st.markdown("---")
    st.caption("DocChatter IQA-RAG Version 0.1.0")


# App Header
st.markdown('<div class="main-header">IQA-RAG: Image Quality Assessment Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Retrieval-Augmented Question Answering across peer-reviewed Image Quality Assessment literature.</div>',
    unsafe_allow_html=True
)

# Main Navigation Tabs
tab_chat, tab_corpus, tab_eval = st.tabs(["Literature Q&A", "Ingestion and Corpus Explorer", "Evaluation Dashboard"])


# -----------------------------------------------------------------------------
# TAB 1: INTERACTIVE Q&A
# -----------------------------------------------------------------------------
with tab_chat:
    col_input, col_tips = st.columns([3, 1])

    with col_tips:
        st.markdown("##### Example Questions")
        example_prompts = [
            "How does LPIPS compute perceptual distance?",
            "How does BRISQUE compute quality features in the spatial domain?",
            "What is the difference between FR, RR, and NR IQA?",
            "How does CLIP-IQA adapt vision-language models for zero-shot IQA?",
            "Why is Mean Squared Error (MSE) criticized as an IQA metric?",
            "What is the difference between MOS and DMOS?"
        ]
        chosen_prompt = None
        for p in example_prompts:
            if st.button(p, key=f"btn_{p[:20]}"):
                chosen_prompt = p

    with col_input:
        query_input = st.text_input(
            "Enter your research question:",
            value=chosen_prompt if chosen_prompt else "",
            placeholder="For example: How does MUSIQ handle multi-scale patch sampling?"
        )
        ask_button = st.button("Submit Question", type="primary", use_container_width=True)

    if (ask_button or chosen_prompt) and query_input:
        with st.spinner("Retrieving document chunks and generating answer..."):
            pipeline = get_pipeline(provider, api_key, top_k, temperature)
            response = pipeline.run(query_input, top_k=top_k)

        # Display Answer Box
        st.markdown("### Grounded Answer")
        st.markdown(response["answer"])

        # Display Telemetry
        c1, c2, c3 = st.columns(3)
        c1.caption(f"**Retrieval Latency:** {response['retrieval_latency_ms']} ms")
        c2.caption(f"**Total Latency:** {response['total_latency_ms']} ms")
        c3.caption(f"**Model Engine:** `{response['model']}`")

        # Display Retrieved Contexts
        st.markdown("---")
        st.markdown("### Retrieved Evidence Sources")
        docs = response.get("retrieved_docs", [])
        if not docs:
            st.info("No documents retrieved.")
        else:
            for d in docs:
                with st.expander(f"[Source {d['rank']}] {d['paper_title']} | {d['section_title']} (Score: {d['score']})"):
                    st.markdown(f"**Section:** `{d['section_title']}` | **Page:** `{d['page']}` | **Chunk ID:** `{d['chunk_id']}`")
                    st.markdown(f"> {d['text']}")


# -----------------------------------------------------------------------------
# TAB 2: INGESTION & CORPUS EXPLORER
# -----------------------------------------------------------------------------
with tab_corpus:
    st.subheader("Fetch Papers from arXiv")
    col_dl1, col_dl2, col_dl3 = st.columns([3, 1, 1])
    with col_dl1:
        arxiv_query = st.text_input("arXiv Search Query", value="Image Quality Assessment LPIPS")
    with col_dl2:
        max_papers = st.number_input("Maximum Papers", min_value=1, max_value=5, value=2)
    with col_dl3:
        st.write("")
        st.write("")
        fetch_btn = st.button("Download and Ingest", type="secondary")

    if fetch_btn and arxiv_query:
        with st.spinner("Downloading arXiv papers, parsing text, and creating chunks..."):
            downloader = ArxivDownloader()
            downloaded = downloader.search_and_download(query=arxiv_query, max_results=max_papers)
            parser = PDFParser()
            parsed = parser.parse_all_raw_pdfs()
            chunker = SectionAwareChunker()
            chunks = chunker.chunk_and_save(parsed)
            store = IQAVectorStore()
            store.add_chunks(chunks)
            st.success(f"Ingested {len(downloaded)} papers and indexed {len(chunks)} chunks.")

    st.markdown("---")
    st.subheader("Corpus Chunk Browser")
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)

        filter_paper = st.selectbox(
            "Filter by Paper Title",
            options=["All Papers"] + list(dict.fromkeys([c["paper_title"] for c in all_chunks]))
        )

        display_chunks = all_chunks if filter_paper == "All Papers" else [c for c in all_chunks if c["paper_title"] == filter_paper]
        st.caption(f"Showing {len(display_chunks)} chunks:")

        for chk in display_chunks[:10]:
            with st.expander(f"{chk['paper_title']} - {chk['section_title']} ({chk['token_count']} tokens)"):
                st.write(chk["text"])
                st.caption(f"Chunk ID: {chk['chunk_id']} | Page: {chk.get('page', 1)}")


# -----------------------------------------------------------------------------
# TAB 3: EVALUATION & BENCHMARK DASHBOARD
# -----------------------------------------------------------------------------
with tab_eval:
    st.subheader("Benchmark Results: Baseline vs. Tuned")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Hit-Rate @ 5", "100.0%", "+4.17%")
    col_m2.metric("Precision @ 3", "0.8000", "+110.53%")
    col_m3.metric("Faithfulness Score", "0.9184", "+20.34%")
    col_m4.metric("Answer F1", "0.6232", "+41.64%")

    # Load Comparison Table Markdown
    comp_path = default_config.results_dir / "comparison_table.md"
    if comp_path.exists():
        with open(comp_path, "r", encoding="utf-8") as f:
            md_table = f.read()
        st.markdown(md_table)
    else:
        st.info("Comparison table not found. Run python -m src.eval.compare_runs to create the table.")

    st.markdown("---")
    st.subheader("Sample Evaluation Queries")
    eval_set_file = default_config.eval_set_path
    if eval_set_file.exists():
        with open(eval_set_file, "r", encoding="utf-8") as f:
            eval_pairs = json.load(f)
        for pair in eval_pairs[:5]:
            with st.expander(f"[{pair['id']}] {pair['question']}"):
                st.markdown(f"**Target Paper:** *{pair['target_paper']}* ({pair['target_section']})")
                st.markdown(f"**Ground Truth:** {pair['ground_truth_answer']}")
                st.markdown(f"**Keywords:** `{', '.join(pair['keywords'])}`")

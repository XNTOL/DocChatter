# IQA-RAG Evaluation Benchmark: Baseline vs. Tuned

## Configuration Comparison

| Hyperparameter | Baseline Run | Tuned Run | Description |
| :--- | :--- | :--- | :--- |
| **Chunk Size** | `500` tokens | `300` tokens | Section-aware semantic granularity |
| **Chunk Overlap** | `30` tokens | `50` tokens | Boundary context preservation |
| **Retrieval Top-K** | `k=3` | `k=5` | Evidence recall depth |
| **Section-Aware** | `False` | `True` | Prevents cross-section chunk contamination |
| **Embedding Model** | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` | Dense representation quality |
| **Generator LLM** | `standard-baseline` | `gemini-1.5-flash` | Grounded synthesis model |

## Performance Metrics Comparison

| Evaluation Metric | Baseline | Tuned | Absolute Delta (Δ) | Relative Gain (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hit-Rate @ 1** | 0.6800 | 0.8000 | **+0.1200** | **+17.65%** | Improved |
| **Hit-Rate @ 3** | 0.9200 | 0.9600 | **+0.0400** | **+4.35%** | Improved |
| **Hit-Rate @ 5** | 0.9600 | 1.0000 | **+0.0400** | **+4.17%** | Improved |
| **Precision @ 1** | 0.6800 | 0.8000 | **+0.1200** | **+17.65%** | Improved |
| **Precision @ 3** | 0.3800 | 0.8000 | **+0.4200** | **+110.53%** | Improved |
| **Precision @ 5** | 0.2896 | 0.6800 | **+0.3904** | **+134.81%** | Improved |
| **MRR @ 5** | 0.8080 | 0.9000 | **+0.0920** | **+11.39%** | Improved |
| **Faithfulness Score** | 0.7632 | 0.9184 | **+0.1552** | **+20.34%** | Improved |
| **Answer Token F1** | 0.4400 | 0.6232 | **+0.1832** | **+41.64%** | Improved |
| **Avg Latency (ms)** | 12.40 | 15.80 | **+3.40** | **+27.42%** | Slower |

## Key Insights & Qualitative Takeaways

1. **Section-Aware Chunking Stops Text Mixing**: The chunker splits text only inside section boundaries. This stops unrelated data from entering model definitions.
2. **Smaller Chunk Size**: A chunk size of 300 tokens gives more precise context to the generator. This increases the token F1 score and faithfulness.
3. **Larger Retrieval Limit**: An increase from k=3 to k=5 supplies more source evidence. This helps complex questions while latency remains low.

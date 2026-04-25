# Johnson-Lindenstrauss Lemma — Companion Code

Companion code for a Medium post demonstrating the Johnson-Lindenstrauss (JL) Lemma applied to compressing high-dimensional vector embeddings in a RAG-style vector database.

## What the script does

The script ([`Johnson_Lindestrauss_Lemma_v5.py`](Johnson_Lindestrauss_Lemma_v5.py)) runs five sections end-to-end:

1. **JL Bound Grid** — Computes the minimum projection dimension `k` guaranteed by the JL Lemma across a grid of distortion tolerances (`ε = 0.1 … 0.5`) and dataset sizes (`n = 1K … 100K`). Saves a table as CSV and PNG.

2. **Vector DB Setup** — Downloads 5,000 T5-XXL embeddings (4096-dimensional) from HuggingFace, L2-normalises them, and ingests two [ChromaDB](https://www.trychroma.com/) collections: one at full dimensionality (baseline) and one compressed to 512 dimensions via `SparseRandomProjection`. The fitted projection matrix is serialised to disk so queries use the exact same subspace as ingestion.

3. **ChromaDB Benchmark** — Queries both collections with 20 random vectors and reports Top-10 Recall and search speedup of the compressed collection over the baseline.

4. **k-Sweep** — Sweeps `k ∈ {64, 128, 256, 512, 1024, 2048}` using in-memory brute-force cosine search, measuring recall and speedup at each compression level and flagging whether each `k` satisfies the JL bound at `ε = 0.3`.

5. **Pitfalls Table** — Summarises six common mistakes when applying JL projection in practice (e.g. not saving the projection matrix, choosing `k` below the bound, mixing cosine and Euclidean distance) along with concrete fixes.

## Results

All outputs are saved in the [`results/`](results/) folder:

| File | Description |
|---|---|
| `jl_epsilon_n_k_table.csv / .png` | JL bound grid (ε × n → minimum k) |
| `jl_benchmark_results.csv / .png` | Recall & speedup across k values |
| `jl_pitfalls_table.csv / .png` | Common pitfalls and fixes |

## Dependencies

```
numpy
scikit-learn
chromadb
datasets (HuggingFace)
matplotlib
```

Install with:

```bash
pip install numpy scikit-learn chromadb datasets matplotlib
```

## Usage

```bash
python Johnson_Lindestrauss_Lemma_v5.py
```

On first run the script downloads the dataset from HuggingFace (~5K vectors). Subsequent runs reuse the local ChromaDB and saved projector, skipping the download.

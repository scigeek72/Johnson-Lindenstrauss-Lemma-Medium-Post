"""
Johnson-Lindenstrauss Lemma — Companion Code for Medium Post
============================================================
Demonstrates JL random projection for compressing high-dimensional
vector embeddings stored in a vector database (RAG simulation).

Outputs (saved to working directory):
  jl_epsilon_n_k_table.csv / .png  — JL bound grid (ε × n → k)
  jl_benchmark_results.csv / .png  — recall & speedup across k values
  jl_pitfalls_table.csv    / .png  — common pitfalls and fixes
  projector.pkl                    — saved fitted projection matrix
"""

import csv
import math
import os
import pickle
import textwrap
import time

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
import numpy as np
import chromadb
from datasets import load_dataset
from sklearn.random_projection import SparseRandomProjection

# ─── Configuration ─────────────────────────────────────────────────────────────
DB_PATH        = "./my_vector_db"
PROJECTOR_PATH = "./projector.pkl"
N_SAMPLES      = 5000
ORIGINAL_DIM   = 4096   # T5-XXL embedding dimension
REDUCED_DIM    = 512    # our chosen compressed dimension
BATCH_SIZE     = 1000
N_BENCHMARK    = 20     # query repetitions per benchmark condition
K_SWEEP        = [64, 128, 256, 512, 1024, 2048]  # k values to evaluate

# ─── Section 1: Pure Python JL Bound ──────────────────────────────────────────
#
# The Johnson-Lindenstrauss Lemma states:
#   For any 0 < ε < 1 and any set of n points in R^d, a random linear map
#   to R^k with k ≥ 4 ln(n) / (ε²/2 − ε³/3) preserves all pairwise
#   Euclidean distances within a factor of (1 ± ε) with high probability.
#
# This function implements that bound in pure Python — no numpy, no sklearn.

def jl_min_dimension(n: int, eps: float) -> int:
    """Return the minimum k guaranteed by the JL lemma for n points at tolerance eps."""
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    if n < 2:
        raise ValueError(f"n must be ≥ 2, got {n}")
    numerator   = 4.0 * math.log(n)
    denominator = (eps ** 2 / 2.0) - (eps ** 3 / 3.0)
    return math.ceil(numerator / denominator)


# ─── Output Helpers ────────────────────────────────────────────────────────────

def save_csv(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([headers] + rows)
    print(f"  Saved -> {path}")


def save_table_png(path, headers, rows, title, wrap_width=28):
    """Render a list-of-rows table as a styled PNG image."""
    def wrap(text):
        return "\n".join(textwrap.wrap(str(text), wrap_width))

    wrapped_rows = [[wrap(cell) for cell in row] for row in rows]
    n_cols = len(headers)
    # estimate row heights by max newlines per row
    row_heights = [max(cell.count("\n") + 1 for cell in row) for row in wrapped_rows]
    total_lines = sum(row_heights) + 2  # +2 for header + padding
    fig_h = max(2.5, 0.38 * total_lines + 1.0)
    fig_w = max(10, 2.0 * n_cols)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=wrapped_rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#1A2A3A")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(0.08)

    for i, h in enumerate(row_heights, start=1):
        for j in range(n_cols):
            cell = tbl[i, j]
            cell.set_facecolor("#EAF2FF" if i % 2 == 0 else "#FFFFFF")
            cell.set_height(max(0.06, 0.055 * h))

    ax.set_title(title, fontsize=11, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


# ─── Section 2: ε × n → k Grid ────────────────────────────────────────────────

print("\n" + "=" * 62)
print("  SECTION 1: JL Bound Grid  (ε × n  →  minimum k)")
print("=" * 62)
print(f"\n  Interpretation: each cell is the smallest k such that a random")
print(f"  projection to R^k preserves all pairwise distances within (1±ε).")
print(f"  Our chosen reduced_dim = {REDUCED_DIM}  (marked with [OK] where k ≤ {REDUCED_DIM}).\n")

EPS_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]
N_VALUES   = [1_000, 5_000, 10_000, 50_000, 100_000]

headers_grid = ["eps \\ n"] + [f"n={n:,}" for n in N_VALUES]
rows_grid = []
for eps in EPS_VALUES:
    row = [f"eps={eps}"]
    for n in N_VALUES:
        k = jl_min_dimension(n, eps)
        tag = " [OK]" if k <= REDUCED_DIM else ""
        row.append(f"{k}{tag}")
    rows_grid.append(row)

col_w = 14
print("  " + "  ".join(h.ljust(col_w) for h in headers_grid))
print("  " + "-" * (col_w * len(headers_grid) + 2 * len(headers_grid)))
for row in rows_grid:
    print("  " + "  ".join(str(c).ljust(col_w) for c in row))

save_csv("jl_epsilon_n_k_table.csv", headers_grid, rows_grid)
save_table_png(
    "jl_epsilon_n_k_table.png",
    headers_grid, rows_grid,
    title=f"JL Lemma: Minimum Projection Dimension k\n"
          f"for (1±eps) Distortion   [OK] = k <= {REDUCED_DIM} (our chosen dim)",
    wrap_width=14,
)

# ─── Section 3: Vector DB Setup ────────────────────────────────────────────────

print("\n" + "=" * 62)
print("  SECTION 2: Vector DB Setup")
print("=" * 62)

client    = chromadb.PersistentClient(path=DB_PATH)
col_std   = client.get_or_create_collection(name="real_baseline")
col_turbo = client.get_or_create_collection(name="real_turbo")
count     = col_std.count()

if count >= N_SAMPLES and os.path.exists(PROJECTOR_PATH):
    # Happy path: DB and projector both exist.
    print(f"\n  Found {count:,} embeddings and saved projector. Skipping download.")
    with open(PROJECTOR_PATH, "rb") as f:
        projector = pickle.load(f)
    print(f"  Projector loaded from {PROJECTOR_PATH}  "
          f"(matrix shape: {ORIGINAL_DIM} -> {REDUCED_DIM})")

elif count >= N_SAMPLES and not os.path.exists(PROJECTOR_PATH):
    # DB exists but no projector saved — refit deterministically and re-ingest turbo.
    print(f"\n  Found {count:,} embeddings but no saved projector.")
    print("  Refitting projector (same random_state=42) and re-ingesting turbo collection...")
    stored       = col_std.get(limit=N_SAMPLES, include=["embeddings"])
    data         = np.array(stored["embeddings"]).astype(np.float32)
    projector    = SparseRandomProjection(n_components=REDUCED_DIM, random_state=42)
    data_reduced = projector.fit_transform(data)

    ids = [f"id_{i}" for i in range(N_SAMPLES)]
    for i in range(0, N_SAMPLES, BATCH_SIZE):
        col_turbo.upsert(
            ids=ids[i : i + BATCH_SIZE],
            embeddings=data_reduced[i : i + BATCH_SIZE].tolist(),
        )
    with open(PROJECTOR_PATH, "wb") as f:
        pickle.dump(projector, f)
    print(f"  Projector saved to {PROJECTOR_PATH}")

else:
    # Cold start: download, project, ingest.
    print(f"\n  Local DB incomplete ({count}/{N_SAMPLES}). Downloading from HuggingFace...")
    print("  Dataset: JusteLeo/t5-xxl-embedding  (4096-dimensional T5-XXL embeddings)")

    ds = load_dataset("JusteLeo/t5-xxl-embedding", split="train", streaming=True)
    data_list = []
    for i, entry in enumerate(ds):
        data_list.append(entry["vector"])
        if i >= N_SAMPLES - 1:
            break

    data  = np.array(data_list, dtype=np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data /= norms  # L2-normalise so cosine sim == dot product

    ids          = [f"id_{i}" for i in range(N_SAMPLES)]
    projector    = SparseRandomProjection(n_components=REDUCED_DIM, random_state=42)
    data_reduced = projector.fit_transform(data)

    # Persist the fitted projector — MUST be the same object used at query time.
    with open(PROJECTOR_PATH, "wb") as f:
        pickle.dump(projector, f)
    print(f"  Projector saved to {PROJECTOR_PATH}")

    def batched_add(collection, embeddings, id_list):
        for i in range(0, len(id_list), BATCH_SIZE):
            collection.add(
                ids=id_list[i : i + BATCH_SIZE],
                embeddings=embeddings[i : i + BATCH_SIZE].tolist(),
            )

    batched_add(col_std,   data,         ids)
    batched_add(col_turbo, data_reduced, ids)
    print("  Ingestion complete.")

print(f"\n  Storage (baseline):   {N_SAMPLES * ORIGINAL_DIM * 4 / 1e6:.1f} MB  "
      f"({N_SAMPLES:,} x {ORIGINAL_DIM:,} float32)")
print(f"  Storage (compressed): {N_SAMPLES * REDUCED_DIM  * 4 / 1e6:.1f} MB  "
      f"({N_SAMPLES:,} x {REDUCED_DIM:,} float32)")
print(f"  Storage reduction:    {REDUCED_DIM / ORIGINAL_DIM:.2%} of original")

# ─── Section 4: ChromaDB Benchmark (k = REDUCED_DIM) ──────────────────────────

print("\n" + "=" * 62)
print(f"  SECTION 3: ChromaDB Benchmark  "
      f"({ORIGINAL_DIM}d baseline  vs  {REDUCED_DIM}d compressed)")
print("=" * 62)

# Warm up the OS file cache so timing is stable.
_dummy = np.random.randn(1, ORIGINAL_DIM).astype(np.float32)
_ = col_std.query(query_embeddings=_dummy.tolist(), n_results=1)
_ = col_turbo.query(query_embeddings=projector.transform(_dummy).tolist(), n_results=1)

# Use 50 stored vectors as our query pool (avoids redownload).
pool = np.array(col_std.get(limit=50, include=["embeddings"])["embeddings"])

recalls_db, t_std_db, t_turbo_db = [], [], []
for _ in range(N_BENCHMARK):
    q_full = pool[np.random.randint(len(pool))].reshape(1, -1)
    q_comp = projector.transform(q_full)

    t0 = time.perf_counter()
    res_std   = col_std.query(query_embeddings=q_full.tolist(), n_results=10)
    t_std_db.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    res_turbo = col_turbo.query(query_embeddings=q_comp.tolist(), n_results=10)
    t_turbo_db.append(time.perf_counter() - t0)

    overlap = len(set(res_std["ids"][0]) & set(res_turbo["ids"][0]))
    recalls_db.append(overlap / 10.0)

speedup_db  = np.mean(t_std_db) / np.mean(t_turbo_db)
recall_db   = np.mean(recalls_db) * 100
jl_k_at_03  = jl_min_dimension(N_SAMPLES, 0.3)

print(f"\n  Vectors in DB:         {col_std.count():,}")
print(f"  Original dimension:    {ORIGINAL_DIM:,}d")
print(f"  Compressed dimension:  {REDUCED_DIM:,}d")
print(f"  Storage ratio:         {REDUCED_DIM / ORIGINAL_DIM:.2%}")
print(f"  Search speedup:        {speedup_db:.2f}x")
print(f"  Mean Top-10 Recall:    {recall_db:.1f}%")
print(f"\n  JL bound at eps=0.3, n={N_SAMPLES:,}: k >= {jl_k_at_03}")
print(f"  Our k={REDUCED_DIM} is "
      f"{'ABOVE' if REDUCED_DIM >= jl_k_at_03 else 'BELOW'} the JL bound for eps=0.3")

# ─── Section 5: k-Sweep (in-memory brute force) ────────────────────────────────

print("\n" + "=" * 62)
print("  SECTION 4: k-Sweep  —  Recall & Speedup vs Compression")
print("=" * 62)
print(f"\n  Ground truth:  brute-force cosine search over {N_SAMPLES:,} x {ORIGINAL_DIM:,}d vectors")
print(f"  Sweep range:   k in {K_SWEEP}\n")

stored_all = col_std.get(limit=N_SAMPLES, include=["embeddings"])
data_full  = np.array(stored_all["embeddings"], dtype=np.float32)

def brute_force_top10(matrix: np.ndarray, query: np.ndarray) -> set:
    """Return indices of top-10 nearest neighbours by cosine similarity."""
    sims = (matrix @ query.T).squeeze()
    return set(np.argsort(sims)[-10:].tolist())

query_indices = np.random.randint(0, len(data_full), size=N_BENCHMARK)
sweep_rows = []

for k in K_SWEEP:
    proj_k   = SparseRandomProjection(n_components=k, random_state=42)
    data_k   = proj_k.fit_transform(data_full).astype(np.float32)

    # Normalise compressed vectors so cosine sim == dot product.
    norms_k  = np.linalg.norm(data_k, axis=1, keepdims=True)
    data_k  /= np.where(norms_k == 0, 1.0, norms_k)

    jl_bound = jl_min_dimension(N_SAMPLES, 0.3)
    jl_safe  = "Yes" if k >= jl_bound else f"No (bound={jl_bound})"

    recalls_k, t_full_k, t_comp_k = [], [], []
    for idx in query_indices:
        q_full  = data_full[idx].reshape(1, -1)
        q_comp  = data_k[idx].reshape(1, -1)

        t0 = time.perf_counter()
        gt_nn = brute_force_top10(data_full, q_full)
        t_full_k.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        approx_nn = brute_force_top10(data_k, q_comp)
        t_comp_k.append(time.perf_counter() - t0)

        recalls_k.append(len(gt_nn & approx_nn) / 10.0)

    speedup_k  = np.mean(t_full_k) / np.mean(t_comp_k)
    recall_k   = np.mean(recalls_k) * 100
    storage_k  = k / ORIGINAL_DIM

    row = [
        str(k),
        f"{storage_k:.2%}",
        f"{speedup_k:.2f}x",
        f"{recall_k:.1f}%",
        jl_safe,
    ]
    sweep_rows.append(row)
    print(f"  k={k:>5d}  storage={storage_k:.2%}  speedup={speedup_k:.2f}x  "
          f"recall={recall_k:.1f}%  JL-safe(eps=0.3): {jl_safe}")

headers_sweep = [
    "k (dim)",
    "Storage Ratio",
    "Search Speedup",
    "Top-10 Recall",
    f"Above JL Bound? (eps=0.3, n={N_SAMPLES:,})",
]
save_csv("jl_benchmark_results.csv", headers_sweep, sweep_rows)
save_table_png(
    "jl_benchmark_results.png",
    headers_sweep, sweep_rows,
    title=(f"JL Projection: Recall & Speedup vs Compression Level\n"
           f"n={N_SAMPLES:,} vectors, original dim={ORIGINAL_DIM:,}d  |  Ground truth = brute-force cosine"),
    wrap_width=22,
)

# ─── Section 6: Pitfalls Table ─────────────────────────────────────────────────

print("\n" + "=" * 62)
print("  SECTION 5: Common Pitfalls and Fixes")
print("=" * 62)

pitfalls = [
    [
        "1. Not saving the projection matrix",
        "Re-fitting a new random matrix at query time gives a different subspace "
        "than the one used at ingest. Query and DB vectors become incomparable — "
        "recall silently drops to near zero.",
        "Serialize the fitted projector (pickle) immediately after fit_transform. "
        "Load the same object at query time. Never re-fit on new or dummy data.",
    ],
    [
        "2. Choosing k below the JL bound",
        "JL guarantees hold only when k >= 4 ln(n) / (eps^2/2 - eps^3/3). "
        "Below this threshold, distance distortion is unbounded — you lose the "
        "mathematical guarantee entirely, not just slightly.",
        "Call jl_min_dimension(n, eps) before choosing k. "
        "If storage forces a smaller k, accept a larger eps and report it honestly.",
    ],
    [
        "3. Mixing cosine and Euclidean distance",
        "JL preserves Euclidean distances, not cosine similarities. "
        "If your vector DB uses cosine distance on un-normalised vectors, "
        "the JL guarantee does not apply.",
        "L2-normalise all vectors before projecting. After normalisation, "
        "cosine similarity equals dot product, and Euclidean distances are "
        "monotonically related — the JL bound then applies.",
    ],
    [
        "4. Forgetting n grows over time",
        "The required k scales as O(log n). A k that was JL-safe at n=5,000 "
        "may not be safe at n=500,000. Stale projectors also cannot project "
        "new documents into a consistent subspace without re-ingesting everything.",
        "Re-evaluate jl_min_dimension(n, eps) whenever n grows by an order of "
        "magnitude. Plan for periodic re-indexing if the DB grows significantly.",
    ],
    [
        "5. Structured or low-rank data",
        "JL assumes worst-case (adversarial) point sets. Real embeddings often "
        "lie on a low-dimensional manifold. Recall may be fine at k << JL bound "
        "or may collapse unpredictably — the guarantee does not apply in either "
        "direction for structured data.",
        "Always validate recall empirically across a held-out query set. "
        "Do not rely on the theoretical bound alone for production deployments.",
    ],
    [
        "6. Projecting query with a different k than the DB",
        "If the DB was indexed at k=512 but the query is projected to k=256 "
        "(or vice versa), the vectors have different lengths and the distance "
        "computation is undefined or silently wrong.",
        "Store k alongside the projector. At query time, assert that the query "
        "dimension matches the collection dimension before searching.",
    ],
]

headers_pit = ["Pitfall", "What Goes Wrong", "Fix"]
for row in pitfalls:
    print(f"\n  {row[0]}")
    print(f"    Problem: {row[1][:90]}...")
    print(f"    Fix:     {row[2][:90]}...")

save_csv("jl_pitfalls_table.csv", headers_pit, pitfalls)
save_table_png(
    "jl_pitfalls_table.png",
    headers_pit, pitfalls,
    title="Johnson-Lindenstrauss in Practice: Common Pitfalls and Fixes",
    wrap_width=32,
)

# ─── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("  OUTPUTS SAVED")
print("=" * 62)
print("  jl_epsilon_n_k_table.csv / .png   (Section 1: JL bound grid)")
print("  jl_benchmark_results.csv / .png   (Section 4: k-sweep benchmark)")
print("  jl_pitfalls_table.csv    / .png   (Section 5: pitfalls)")
print("  projector.pkl                     (fitted projection matrix)")
print("=" * 62 + "\n")

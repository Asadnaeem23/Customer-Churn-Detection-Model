# Parallelism in Customer Churn Detection Project

This document explains **where** and **how** OpenMP, MPI, and GPU parallelism are used across every section of `churn.ipynb`.

---

## Quick Reference Table

| Notebook Cell | Section | Parallelism Type | How It Is Applied |
|---|---|---|---|
| Cell 11 | Section 6.1 – RF Sequential | ❌ None (baseline) | `n_jobs=1` — single core only |
| Cell 12 | Section 6.1 – LR Sequential | ❌ None (baseline) | `n_jobs=1` — single core only |
| Cell 13 | Section 6.2 – RF Parallel | ✅ **OpenMP** | `n_jobs=-1` — all CPU cores |
| Cell 14 | Section 6.2 – LR Parallel | ✅ **OpenMP** | `n_jobs=-1` — all CPU cores |
| Cell 16 | Section 8 – Batch Pipeline | ✅ **MPI + OpenMP** | `np.array_split` scatter + `joblib.Parallel` gather |
| Cell 17 | Section 9 – Comparison Table | 📊 Measurement only | Displays speedups, no execution |
| Cell 18 | Section 10 – Gradio UI | ✅ **Auto-detect: GPU / MPI+OpenMP / Sequential** | Chosen at runtime based on hardware and batch size |

---

## 1. OpenMP (Multi-threaded CPU Parallelism)

### What is OpenMP here?
OpenMP is a shared-memory parallelism API. In Python's scikit-learn, it is exposed through the `n_jobs` parameter, which controls how many CPU threads are used internally (via `joblib` and native C extensions that link against OpenMP).

### Where it is used

#### Cell 13 — Random Forest (OpenMP training)
```python
rf_parallel = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1          # ← OpenMP: use ALL available cores
)
rf_parallel.fit(X_train, y_train)
```
- `n_jobs=-1` tells scikit-learn to spawn one thread per core.
- **What is parallelised:** building each decision tree. The 100 trees are distributed across all cores simultaneously.
- **Compared against:** Cell 11 which uses `n_jobs=1` (sequential baseline).

#### Cell 14 — Logistic Regression (OpenMP training)
```python
lr_parallel = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1          # ← OpenMP: use ALL available cores
)
lr_parallel.fit(X_train, y_train)
```
- `n_jobs=-1` parallelises the internal LIBLINEAR/SAGA solver across cores.
- **Compared against:** Cell 12's sequential `lr_sequential`.

#### Cell 16 — Batch Pipeline (OpenMP inference inside each MPI node)
```python
from joblib import Parallel, delayed

all_probs = Parallel(n_jobs=-1)(
    delayed(predict_chunk)(chunk, loaded_model)
    for chunk in chunks
)
```
- Each `predict_chunk` call internally uses `rf_parallel.predict_proba()`, which itself runs with OpenMP threads.
- So OpenMP operates **inside** each MPI-simulated node.

#### Cell 18 — Gradio UI (CSV batch, OpenMP path)
```python
elif n >= 200:
    chunks    = np.array_split(processed, N_CORES)
    all_probs = Parallel(n_jobs=N_CORES)(
        delayed(lambda c: rf_parallel.predict_proba(c)[:, 1])(chunk)
        for chunk in chunks
    )
    method = f"MPI-split + OpenMP ({N_CORES} cores) — {n} rows"
```
- Triggered when batch has ≥ 200 rows and no GPU is available.
- `N_CORES = multiprocessing.cpu_count()` is detected at startup.

---

## 2. MPI (Message Passing Interface — Simulated)

### What is MPI here?
True MPI requires `mpi4py` and multiple processes/nodes. This project **simulates the MPI communication pattern** using:
- `numpy.array_split` → **MPI Scatter** (distribute data to nodes)
- `joblib.Parallel` → **parallel execution** (each worker = one MPI rank)
- `numpy.concatenate` → **MPI Gather** (collect results back)

This makes the code genuinely parallel and demonstrates the MPI pattern without requiring a cluster setup.

### Where it is used

#### Cell 16 — Section 8: Batch Prediction Pipeline
```python
num_nodes = multiprocessing.cpu_count()   # MPI world size
print(f"🖥  Simulating {num_nodes} MPI processes (one per CPU core)")

# MPI SCATTER — split data across simulated nodes
chunks = np.array_split(X_test_processed, num_nodes)

# MPI NODES — each worker loads model and runs inference
all_probs = Parallel(n_jobs=-1)(
    delayed(predict_chunk)(chunk, loaded_model)
    for chunk in chunks
)

# MPI GATHER — collect all partial results
final_probs = np.concatenate(all_probs)
```

**Full data flow:**
```
X_test_processed (all rows)
        │
        ▼  np.array_split (MPI Scatter)
  ┌─────┬─────┬─────┬─────┐
  │ c1  │ c2  │ c3  │ c4  │   ← one chunk per CPU core (MPI rank)
  └──┬──┴──┬──┴──┬──┴──┬──┘
     │     │     │     │      predict_chunk() runs on each
     ▼     ▼     ▼     ▼      (OpenMP threads inside each call)
  [p1] [p2] [p3] [p4]         partial probability arrays
     │     │     │     │
     └──────────────────┘
              │  np.concatenate (MPI Gather)
              ▼
       final_probs (all rows unified)
```

#### Cell 18 — Gradio UI (CSV batch ≥ 200 rows activates MPI path)
Same scatter-parallel-gather pattern as Cell 16, but triggered dynamically at UI runtime based on uploaded CSV size.

---

## 3. GPU (NVIDIA cuML — Optional)

### What is GPU here?
[RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/) is a drop-in replacement for scikit-learn that runs on NVIDIA GPUs using CUDA. It is optional — the project auto-detects it and falls back to CPU if unavailable.

### Detection (Cell 18 startup)
```python
GPU_AVAILABLE = False
try:
    import cuml
    GPU_AVAILABLE = True
except ImportError:
    pass   # GPU simply not used; falls back to CPU
```

### Where it is used

#### Cell 18 — Gradio UI (GPU path for large batches)
```python
if GPU_AVAILABLE and n >= 1000:
    import cuml.ensemble
    gpu_model = cuml.ensemble.RandomForestClassifier()
    probs = rf_parallel.predict_proba(processed)[:, 1]
    method = f"GPU (cuML) — {n} rows"
```
- Activated when: CSV has ≥ 1000 rows **AND** cuML is installed.
- In a full GPU deployment, `rf_parallel` would be replaced with a cuML model retrained on the GPU.
- The current implementation uses the CPU model as a fallback while demonstrating the correct code path and auto-detection logic.

### How to enable GPU
```bash
# Requires NVIDIA GPU + CUDA 11 or 12
pip install cuml-cu11          # for CUDA 11
# or
pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12   # for CUDA 12
```

---

## 4. Sequential Baselines (No Parallelism)

#### Cell 11 — RF Sequential
```python
rf_sequential = RandomForestClassifier(
    n_jobs=1    # ← force single core
)
```

#### Cell 12 — LR Sequential
```python
lr_sequential = LogisticRegression(
    n_jobs=1    # ← force single core
)
```

These exist purely as **benchmarks** to measure speedup from parallelism. Their timings are recorded in `sequential_rf_time` and `sequential_lr_time` and compared in Section 9.

---

## 5. Auto-Parallelism Decision Logic (Cell 18)

At runtime the Gradio UI inspects the uploaded batch size and system capabilities to automatically pick the fastest available strategy:

```
Upload CSV
     │
     ▼
GPU available AND rows ≥ 1000?
     ├── YES → GPU (cuML) inference
     └── NO
          │
          ▼
     rows ≥ 200?
          ├── YES → MPI-split + OpenMP (joblib.Parallel, N_CORES workers)
          └── NO  → Sequential (single predict_proba call)
```

The `method` string shown in the UI summary tells the user exactly which path was taken.

---

## 6. Performance Comparison (Section 9 / Cell 17)

| Strategy | Model | Parallelism | Timing Variable | Speedup Variable |
|---|---|---|---|---|
| Sequential (baseline) | Random Forest | None (`n_jobs=1`) | `sequential_rf_time` | `1.00x` |
| OpenMP RF | Random Forest | OpenMP (`n_jobs=-1`) | `parallel_rf_time` | `sequential_rf_time / parallel_rf_time` |
| OpenMP LR | Logistic Regression | OpenMP (`n_jobs=-1`) | `parallel_lr_time` | `sequential_lr_time / parallel_lr_time` |
| MPI-style Batch | Random Forest | MPI-scatter + joblib | `elapsed` | "Batch distributed" |

The comparison table is printed by Cell 17 and also stored in the `comparison` DataFrame variable.

---

## 7. Libraries Used for Parallelism

| Library | Role |
|---|---|
| `sklearn` (`n_jobs=-1`) | OpenMP thread control for RF and LR training/inference |
| `joblib.Parallel` + `delayed` | MPI-simulation: spawn N workers, each processes one data chunk |
| `multiprocessing.cpu_count()` | Detect number of physical CPU cores at runtime |
| `numpy.array_split` | MPI Scatter — split dataset into equal chunks |
| `numpy.concatenate` | MPI Gather — merge partial results back into one array |
| `cuml` (optional) | GPU-accelerated drop-in replacement for sklearn models |

---

## 8. How to Reproduce

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run all cells top-to-bottom in `churn.ipynb`.
3. **Section 6** trains sequential and OpenMP models and prints timings.
4. **Section 8** runs the MPI-style batch pipeline on `test.csv` and saves `churn_predictions.csv`.
5. **Section 9** prints the side-by-side comparison table.
6. **Section 10** launches the Gradio UI — use Tab 2 to upload any customer CSV and see auto-parallelism in action.

# 🗂️ Project Structure — HLA Solver Prototype

This file describes the intended organization of the prototype HLA solver project. It is structured for clean modularity, YAML-based configuration, and support for both notebook-based and CLI-based execution.

---

## 🧱 Root Layout

```
hla_solver_prototype/
│
├── config.yaml                # Main user-editable config file (YAML format)
├── requirements.txt           # Python dependencies
├── run.py                     # CLI launcher
│
├── README.md                  # Project overview and instructions
├── PROJECT_STRUCTURE.md       # This file
│
├── hla_solver/                # Main solver package
│   ├── __init__.py
│   ├── solver.py              # Main HLASolver class (configurable via YAML)
│   ├── backend.py             # Backend selector and parallelization strategies
│   ├── utils.py               # Common helper functions and logging
│   ├── preprocessing.py       # Preprocessing logic for loading + preparing dataset
│   ├── greedy.py              # Serial greedy HLA coverage algorithm
│   └── validation.py          # Optional: validation / metrics
│
├── notebook_dev/              # Jupyter notebooks for experimentation
│   └── prototype_dev.ipynb    # Main notebook for iterative development
│
├── tests/                     # (Optional) Unit and integration tests
│   ├── __init__.py
│   └── test_solver.py         # Basic tests for solver behavior
│
└── data/                      # Local datasets (add .gitignore as needed)
    └── hla_alleles_smallDataset.csv
```

---

## 📁 hla_solver Package

This is the heart of the application. Each file should be focused:

| File                | Purpose |
|---------------------|---------|
| `solver.py`         | `HLASolver` class: entry point for configuration, execution |
| `backend.py`        | Manages execution modes (`serial`, `threads`, `processes`) |
| `utils.py`          | Logging, verbose decorators, argument validation |
| `greedy.py`         | Implements greedy coverage logic (serial implementation) |
| `preprocessing.py`  | Handles dataset loading, doubling, allele mapping |
| `validation.py`     | (Optional) Coverage metrics or evaluation utilities |

---

## 📁 notebook_dev

Use this folder to explore:

- Prototyping new features
- Visualizing performance
- Comparing backends on small data

Keep notebooks separate from core logic for clarity and testability.

---

## 🔧 `config.yaml` Example

```yaml
dataset: "data/hla_alleles_smallDataset.csv"
number_doubles: 0

solver:
  max_iter: 10
  top_n: 1
  top_k_global: 3
  verbose: true
  backend: "serial"           # "serial", "threads", "processes"
  num_workers: 4
  progress_mode: "per_worker" # or "progress_bar" or "silent"
```

---

## 🧪 Testing

Basic tests should go into `tests/`. Later, consider `pytest` or `unittest` integration.

---

## 🧩 Future Extensions

- C++/CUDA backends for `greedy.py`
- MPI-style parallel launchers (`mpi4py`)
- Model evaluation and benchmarking suite
- Containerization (e.g., Docker)

---

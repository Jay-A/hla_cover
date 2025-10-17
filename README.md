# HLA Solver Prototype

A modular, configurable, and Jupyter-safe prototype for solving greedy HLA allele coverage problems. Designed to be YAML-driven and extendable to high-performance backends (e.g., `mpi4py`, C++).

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/hla_solver_prototype.git
cd hla_solver_prototype
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Or use your preferred environment manager (e.g. `conda`, `pipenv`).

### 3. Edit configuration

Edit `config.yaml` to define:
- Dataset path
- Number of iterations
- Backend: `serial`, `threads`, or `processes`
- Verbosity, parallel workers, etc.

### 4. Run from CLI

```bash
python run.py --config config.yaml
```

### 5. Or develop in a Jupyter notebook

Inside `notebook_dev/prototype_dev.ipynb`, import the solver class and call:

```python
from hla_solver.solver import HLASolver
solver = HLASolver.from_yaml("config.yaml")
solver.run()
```

---

## Configuration Options (`config.yaml`)

Example:

```yaml
dataset: "hla_alleles_smallDataset.csv"
number_doubles: 0

solver:
  max_iter: 10
  top_n: 1
  top_k_global: 3
  verbose: true
  backend: "threads"         # "serial", "threads", or "processes"
  num_workers: 4
  progress_mode: "progress_bar"  # "silent", "per_worker", or "progress_bar"
```

---

## Project Structure

See [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) for full directory layout and responsibilities.

---

##  Notes

-  Jupyter-safe: only thread or serial modes recommended in notebooks.
-  Multiprocessing mode will warn if launched from within Jupyter/IPython.
-  Designed to scale into a future C++ or `mpi4py` implementation.

---

## 📄 License

MIT License (you can modify this section if needed)

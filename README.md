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

BSD 2-Clause License

Copyright (c) 2025, Jay M. Appleton
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

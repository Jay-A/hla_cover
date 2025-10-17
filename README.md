# HLA Solver Prototype

A modular, configurable, and Jupyter-safe prototype for solving greedy HLA allele coverage problems. Designed to be YAML-driven and extendable to high-performance backends (e.g., `mpi4py`, C++).

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Jay-A/hla_cover.git
cd hla_cover
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
- Backend: `serial`, `thread_pool`, or `process_pool`
- Verbosity, parallel workers, etc.

### 4. Or develop in a Jupyter notebook

Inside `notebook_dev/prototype_dev.ipynb`, import the solver class and instantiate:

```python
import sys, os

# Set up project root path
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hla_solver import HLASolver

# Path to config file
config_path = os.path.join(project_root, "configs", "cover_serial_config.yaml")
solver = HLASolver(config_path)
```

The solver object may now be used to freshly begin the problem defined in the config file or be used to restart from a restart\_state file:
```python
best_filters = solver.run()
# or if restarting:
best_filters = solver.run(restart_file="<path/to/restart/file.csv")
```
---

## Configuration Options (`config.yaml`)

Example:

```yaml
# configs/config.yaml

solver:
  max_iter: 15
  top_n: 3
  top_k_global: 3

data:
  dataset_path: "data/simulated_HLA_samples_20000.csv"
  number_doubles: 0

parallel:
  mode: "serial"
  num_workers: 1
  
logging:
  verbose: true
  enable_timing: true
  progress_mode: progress_bar
  log_dir: "logs"

warnings:
  warn_process_pool_in_ipynb: true  
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

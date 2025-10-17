"""
hla_solver
==========

A restartable, parallelized greedy solver for HLA allele set coverage problems.

This package provides a modular framework for solving HLA coverage problems using
a greedy strategy, with support for process/thread-based parallelism, dataset preprocessing,
logging, and experimental selection tools.

Modules
-------
- solver         : Main interface for loading configuration and running the solver
- backend        : Parallelization strategies (serial, thread, or process pool)
- utils          : Logging, timing, and general-purpose utilities
- preprocessing  : Input dataset parsing and allele set construction
- validation     : Tools for evaluating HLA coverage and verifying solution quality
- selectors      : Experimental filter discovery and sample-selection strategies

Example
-------
>>> from hla_solver import HLASolver
>>> solver = HLASolver("config.yaml")
>>> solver.run()

"""

from .solver import HLASolver
from .backend import BackendManager
from .utils import (
    verbose_logger,
    is_notebook,
    warn_once,
    chunk_indices,
    safe_print,
    tqdm_progress,
    is_file_empty,
    save_restart_state,
    load_restart_state,
    TimingLogger,
)
from .preprocessing import preprocess  
from .selectors import (
    analyze_hla_group,
    find_best_single_allele_filter,
    find_top_filter_combinations,
    find_top_covering_samples,
    find_matching_samples_by_filter,
    multi_sample_coverage_with_cache,
    multi_sample_coverage_and_remainder,
    sample_coverage_percent
)

# Package metadata
__version__ = "0.1.0"
__author__ = "Jay M. Appleton"
__license__ = "MIT"
__email__ = "jay.appleton@colorado.edu"
__url__ = "https://github.com/jay-a/parallel-hla-cover-solver"

__all__ = [
    # Core API
    'HLASolver',
    'BackendManager',

    # Utils
    'verbose_logger',
    'is_notebook',
    'warn_once',
    'chunk_indices',
    'safe_print',
    'tqdm_progress',
    'is_file_empty',
    'save_restart_state',
    'load_restart_state',
    'TimingLogger',

    'preprocess',

    # Filter selection tools
    'analyze_hla_group',
    'find_best_single_allele_filter',
    'find_top_filter_combinations',
    'find_top_covering_samples',
    'find_matching_samples_by_filter',
    'multi_sample_coverage_with_cache',
    'multi_sample_coverage_and_remainder',
    'sample_coverage_percent',
]


"""
hla_solver.backend module
=========================

This module provides the BackendManager class to manage parallelism strategies
(serial, thread pool, and process pool) for running the HLA greedy coverage solver.

It handles warnings about unsafe parallelism modes in Jupyter notebooks and
delegates execution to the appropriate backend implementation.

Imports:
- concurrent.futures: for thread/process pools
- tqdm: for progress bars
- warnings: for runtime warnings
- hla_solver.greedy: solver functions
- hla_solver.utils: utility functions including verbose logging
"""

import concurrent.futures
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from hla_solver.greedy import greedy_hla_coverage_fast
from hla_solver.utils import verbose_logger, PROJECT_ROOT
import sys
from typing import Optional, List
from datetime import datetime
from pathlib import Path

# top-level function (must be importable from module scope for multiprocessing)
def _process_partition_worker(
    df_partition,
    hla_groups,
    allele_to_id,
    allele_sets_by_sample,
    top_n,
    pid=0,
    verbose=False,
):
    """
    Worker function for process pool partition evaluation.

    Mirrors thread pool logic using `find_top_covering_samples`.
    Must be defined at top level to be pickle-able by multiprocessing.
    """
    try:
        from hla_solver.selectors import find_top_covering_samples

        if verbose:
            print(f"[Process {pid}] Starting top-{top_n} candidate search on {len(df_partition)} samples")

        # Call same logic as thread pool
        top_candidates = find_top_covering_samples(
            df=df_partition,
            hla_groups=hla_groups,
            top_n=top_n,
            allele_sets_by_sample=allele_sets_by_sample,
            allele_to_id=allele_to_id,
            coverage_type="intersection",
            verbose=False,
            enable_timing=False,
            timing=None
        )

        if verbose:
            print(f"[Process {pid}] Found {len(top_candidates)} candidates")

        return top_candidates

    except Exception as e:
        print(f"[Process {pid}] Worker failed: {e}")
        return []

class BackendManager:
    """
    Manages parallel backend strategies for the HLA greedy coverage solver.

    Supported modes:
    -----------------
    - ``serial``:       Single-threaded, sequential greedy coverage
    - ``thread_pool``:  Multi-threaded execution using ThreadPoolExecutor
    - ``process_pool``: Multi-processing using ProcessPoolExecutor
                         (with restart-safe execution and Jupyter warnings)

    Parameters
    ----------
    mode : str
        Execution mode: 'serial', 'thread_pool', or 'process_pool'
    num_workers : int
        Number of parallel workers to use (if applicable)
    verbose : bool
        Enable detailed logging during execution
    progress_mode : str
        Progress feedback style: 'progress_bar', 'per_worker', or 'silent'
    restart_file : str or Path, optional
        Path to a restart file for resuming an interrupted run
    """

    def __init__(self, mode='serial', num_workers=1, verbose=False, progress_mode='progress_bar', restart_file=None):
        """
        Initialize backend manager with given configuration.

        Parameters
        ----------
        mode : str
            Parallelism mode: 'serial', 'thread_pool', or 'process_pool'.
        num_workers : int
            Number of workers to use for parallel backends.
        verbose : bool
            Enable verbose output.
        progress_mode : str
            Progress reporting mode ('progress_bar', 'per_worker', 'silent').
        restart_file : Optional[str or Path]
            Path to a restart file (if resuming). If None, a new one will be created.
        """
        self.mode = mode.lower()
        self.num_workers = num_workers
        self.verbose = verbose
        self.progress_mode = progress_mode

        # Corrected: reference the correct variable
        self.is_restart = restart_file is not None

        if self.mode == 'process_pool' and self._is_notebook():
            warnings.warn(
                "ProcessPoolExecutor is not fully safe in Jupyter notebooks. "
                "Consider using 'serial' or 'thread_pool' modes instead.",
                RuntimeWarning
            )

        if self.is_restart:
            self.restart_path = Path(restart_file)
            if not self.restart_path.exists():
                raise FileNotFoundError(f"[Restart] File not found: {self.restart_path}")
            if self.verbose:
                print(f"[BackendManager] Resuming from restart: {self.restart_path}")
        else:
            # Fresh run — create new restart path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.restart_path = PROJECT_ROOT / "restarts" / f"greedy_restart_{self.mode}_{timestamp}.csv"
            self.restart_path.parent.mkdir(parents=True, exist_ok=True)

            # Create empty restart file to indicate fresh run start
            self.restart_path.touch(exist_ok=False)

            if self.verbose:
                print(f"[BackendManager] Starting new run. Restart path: {self.restart_path}")

    def _is_notebook(self):
        """
        Check whether execution is inside a Jupyter notebook or qtconsole.

        Returns
        -------
        bool
            True if running in a notebook/qtconsole, else False.
        """
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True  # Jupyter notebook or qtconsole
            else:
                return False  # Other type (likely terminal)
        except NameError:
            return False  # Probably standard python interpreter

    def split_dataframe(self, df: pd.DataFrame, n_parts: Optional[int] = None, min_rows_per_part: int = 1000) -> List[pd.DataFrame]:
        """
        Split a DataFrame into roughly equal partitions with a minimum number of rows.

        This prevents over-parallelization when the number of samples is small.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset to partition
        n_parts : int, optional
            Number of desired partitions (default: number of workers)
        min_rows_per_part : int
            Minimum rows per partition to avoid excessive overhead

        Returns
        -------
        List[pandas.DataFrame]
            Row-disjoint subsets of the original DataFrame
        """
        n_parts = n_parts or self.num_workers
        total_rows = len(df)
    
        # Calculate max partitions so that each partition >= min_rows_per_part
        max_parts = max(1, total_rows // min_rows_per_part)
    
        # Use the minimum of requested n_parts and max_parts to avoid overparallelization
        final_parts = min(n_parts, max_parts)
    
        return np.array_split(df, final_parts)

    def _run_serial(
        self,
        df,
        hla_groups,
        allele_sets_by_sample,
        allele_to_id,
        id_to_allele,
        max_iter,
        top_n,
        top_k_global,  # still unused here, but kept for consistent interface
        verbose=False,
        enable_timing=False,
        timing=None,
    ):
        """
        Run the greedy coverage algorithm in serial mode (single thread).
    
        Parameters
        ----------
        df : pandas.DataFrame
            Input data frame containing sample and allele information.
        hla_groups : list
            List of HLA groups (e.g., [('HLA-A', 'A1', 'A2'), ...]).
        allele_to_id : dict
            Mapping from allele string to numeric ID.
        allele_sets_by_sample : dict
            Dict of sample_id -> group -> frozenset of allele IDs.
        max_iter : int
            Maximum number of iterations in the greedy algorithm.
        top_n : int
            Number of top samples to consider at each iteration.
        top_k_global : int
            Number of global solutions to keep (not used in serial version).
        verbose : bool
            Whether to print verbose debug information.
        enable_timing : bool
            Whether to enable timing output.
        timing : Callable or None
            Timing context manager.
    
        Returns
        -------
        tuple
            Tuple of (selected_samples, filters, remainder) from the greedy algorithm.
        """
        if self.verbose:
            print("[BackendManager] Running in SERIAL mode")
    
        with (timing("serial::greedy_hla_coverage", enabled=enable_timing) if timing else nullcontext()):
    
            selected_samples, filters, remainder = greedy_hla_coverage_fast(
                df=df,
                hla_groups=hla_groups,
                allele_sets_by_sample=allele_sets_by_sample,
                allele_to_id=allele_to_id,
                id_to_allele=id_to_allele,
                max_iter=max_iter,
                top_n=top_n,
                restart_path=self.restart_path,
                verbose=self.verbose
            )

        if self.verbose:
            print( f"Selected Samples: {selected_samples}" )
            print( f"Filters: {filters}" )
            print( f"remainder: {remainder}" )
    
        return selected_samples, filters, remainder

    def _run_thread_pool(
        self,
        df,
        hla_groups,
        allele_sets_by_sample,
        allele_to_id,
        id_to_allele,
        max_iter,
        top_n,
        top_k_global,
        local_bests_count=3,
        verbose=False,
        enable_timing=False,
        timing=None,
    ):
        """
        Execute the greedy coverage algorithm using multiple threads.

        Uses `ThreadPoolExecutor` to evaluate partitions concurrently.
        At each iteration, threads return their top-N candidates, which
        are globally evaluated to select the best sample to add.

        Parameters
        ----------
        (Same as _run_serial)

        Returns
        -------
        tuple
            (selected_sample_ids, filter_sequence, remaining_uncovered_df)
        """
        if self.verbose:
            print("[BackendManager] Running in THREADED GREEDY mode")
    
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from hla_solver.selectors import (
            sample_coverage_percent,
            multi_sample_coverage_and_remainder,
            find_top_covering_samples,
        )
        from hla_solver.utils import load_restart_state, save_restart_state
    
        selected_sample_ids = []
        filter_sequence = []
        match_cache = {}
    
        # Handle restart logic
        if self.restart_path is not None:
            restart_path = Path(self.restart_path)
            print(f"[Threaded Solver] Restart path: {restart_path}")
            if restart_path.is_file():
                selected_sample_ids, filter_sequence = load_restart_state(restart_path)
                if self.verbose:
                    print(f"[Restart] Loaded {len(selected_sample_ids)} samples from: {restart_path}")
    
                _, _, match_cache, remainder_df = multi_sample_coverage_and_remainder(
                    df,
                    hla_groups,
                    selected_sample_ids,
                    allele_sets_by_sample,
                    coverage_type='intersection',
                    verbose=self.verbose
                )
            else:
                raise FileNotFoundError(f"[Restart] File not found: {restart_path}")
        else:
            # Start fresh
            remainder_df = df.copy()
    
        current_df = remainder_df.copy()
        current_sample_ids = set(current_df["Sample ID"])
    
        for iteration in range(max_iter):
            if not current_sample_ids:
                if self.verbose:
                    print("[Info] All samples covered.")
                break
    
            if self.verbose:
                print(f"\n[Iteration {iteration + 1}] Searching top-{local_bests_count} in {len(current_sample_ids)} samples...")
    
            # Partition current uncovered DataFrame
            n_parts = min(self.num_workers, max(1, len(current_sample_ids) // 2500))
            partitions = self.split_dataframe(current_df, n_parts=n_parts)
    
            iteration_candidates = []
            thread_outputs = {}
    
            with ThreadPoolExecutor(max_workers=n_parts) as executor:
                futures = {
                    executor.submit(
                        find_top_covering_samples,
                        df=part_df,
                        hla_groups=hla_groups,
                        top_n=local_bests_count,
                        allele_sets_by_sample={sid: allele_sets_by_sample[sid] for sid in part_df["Sample ID"]},
                        allele_to_id=allele_to_id,
                        coverage_type="intersection",
                        verbose=False,
                        enable_timing=False,
                        timing=None
                    ): i
                    for i, part_df in enumerate(partitions)
                }
    
                for future in as_completed(futures):
                    thread_id = futures[future]
                    results = future.result()
                    iteration_candidates.extend(results)
                    thread_outputs[thread_id] = results
    
            # Print per-thread top samples
            if self.verbose:
                for thread_id, samples in sorted(thread_outputs.items()):
                    print(f"\n[Thread {thread_id}] Top {local_bests_count} samples from partition:")
                    for i, sample in enumerate(samples, start=1):
                        sid = sample["Sample ID"]
                        cov = sample.get("Coverage Subset (%)", "?")
                        print(f"  ({i}) {sid} — Partition coverage: {cov}%")
    
            if not iteration_candidates:
                if self.verbose:
                    print("[Warning] No candidates returned from any partition.")
                break
    
            # Evaluate full dataset coverage for each candidate
            best_sample = None
            best_coverage = -1
            best_filter = None
    
            if self.verbose:
                print("\n[Global Evaluation] Computing full coverage for all candidates:")
    
            for candidate in iteration_candidates:
                sample_id = candidate["Sample ID"]
                remaining_coverage, _ = sample_coverage_percent(
                    df=current_df,
                    hla_groups=hla_groups,
                    sample_id=sample_id,
                    allele_sets_by_sample=allele_sets_by_sample,
                    allele_to_id=allele_to_id,
                    coverage_type="intersection",
                    verbose=False
                )
                candidate["Coverage Total (%)"] = remaining_coverage
    
                if self.verbose:
                    print(f"  ➤ {sample_id} — Global coverage (remaining set): {remaining_coverage}%")
    
                if remaining_coverage > best_coverage:
                    best_sample = candidate
                    best_filter = candidate["Alleles"]
                    best_coverage = remaining_coverage
    
            if best_sample is None:
                if self.verbose:
                    print("[Warning] No valid global sample found.")
                break
    
            best_sample_id = best_sample["Sample ID"]
            selected_sample_ids.append(best_sample_id)
            filter_sequence.append(best_filter)
    
            if self.verbose:
                subset_coverage = best_sample.get("Coverage Subset (%)", "?")
                print(
                    f"\n[Selected] Sample {best_sample_id} with estimated coverage "
                    f"{best_coverage}% (remaining global), {subset_coverage}% (subset)"
                )
    
            # Update uncovered set
            coverage_pct, combined_alleles, covered_ids, remainder_df = multi_sample_coverage_and_remainder(
                df=df,
                hla_groups=hla_groups,
                sample_ids=selected_sample_ids,
                sample_allele_sets=allele_sets_by_sample,
                coverage_type="intersection",
                verbose=self.verbose
            )
    
            # Save restart state (thread-safe single-process I/O)
            if self.restart_path is not None:
                hla_pairs_named = {
                    group: [id_to_allele[allele_id] for allele_id in best_filter.get(group, [])]
                    for group in best_filter
                }
    
                save_restart_state(
                    path=self.restart_path,
                    sample_id=best_sample_id,
                    hla_groups=best_filter.keys(),
                    hla_pairs=hla_pairs_named,
                    coverage=best_sample["Coverage Total (%)"],
                    cumulative_cover=coverage_pct
                )
                if self.verbose:
                    print(f"[Restart] Saved restart state to: {self.restart_path}")
    
            current_df = remainder_df
            current_sample_ids = set(remainder_df["Sample ID"])
    
            if self.verbose:
                print(f"[Info] Accumulated {len(selected_sample_ids)} Sample IDs:\n    {selected_sample_ids}")
                print(f"[Info] Total dataset coverage: {coverage_pct}%")
                print(f"[Info] Remaining uncovered samples: {len(current_sample_ids)}")
    
        return selected_sample_ids, filter_sequence, current_df

    def _run_process_pool(
        self,
        df,
        hla_groups,
        allele_sets_by_sample,
        allele_to_id,
        id_to_allele,
        max_iter,
        top_n,
        top_k_global,
        local_bests_count=3,
        verbose=False,
        enable_timing=False,
        timing=None
    ):
        """
        Execute the greedy coverage algorithm using multiple processes.

        Similar to `_run_thread_pool`, but uses `ProcessPoolExecutor`.
        Restartable via checkpoint file. Each process handles a data partition
        and returns top local candidates, which are then globally evaluated.

        Parameters
        ----------
        (Same as _run_thread_pool)

        Returns
        -------
        tuple
            (selected_sample_ids, filter_sequence, remaining_uncovered_df)
        """
        if self.verbose:
            print("[BackendManager] Running in PROCESS POOL GREEDY mode")
    
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from hla_solver.selectors import (
            sample_coverage_percent,
            multi_sample_coverage_and_remainder,
        )
        from hla_solver.utils import load_restart_state, save_restart_state
    
        selected_sample_ids = []
        filter_sequence = []
    
        # Load restart state if provided
        if self.restart_path is not None:
            restart_path = Path(self.restart_path)
            print(f"[Process Pool Solver] Restart path: {restart_path}")
            if restart_path.is_file():
                selected_sample_ids, filter_sequence = load_restart_state(restart_path)
                if self.verbose:
                    print(f"[Restart] Loaded {len(selected_sample_ids)} samples from: {restart_path}")
                _, _, _, remainder_df = multi_sample_coverage_and_remainder(
                    df,
                    hla_groups,
                    selected_sample_ids,
                    allele_sets_by_sample,
                    coverage_type="intersection",
                    verbose=self.verbose,
                )
            else:
                raise FileNotFoundError(f"[Restart] File not found: {restart_path}")
        else:
            remainder_df = df.copy()
    
        current_df = remainder_df.copy()
        current_sample_ids = set(current_df["Sample ID"])
    
        with (timing("process_pool::greedy_hla_coverage", enabled=enable_timing) if timing else nullcontext()):
            for iteration in range(max_iter):
                if not current_sample_ids:
                    if self.verbose:
                        print("[Info] All samples covered.")
                    break
    
                if self.verbose:
                    print(f"\n[Iteration {iteration + 1}] Searching top-{local_bests_count} in {len(current_sample_ids)} samples...")
    
                n_parts = min(self.num_workers, max(1, len(current_sample_ids) // 2500))
                partitions = self.split_dataframe(current_df, n_parts=n_parts)
    
                partition_inputs = []
                for i, part_df in enumerate(partitions):
                    sample_ids = part_df["Sample ID"].tolist()
                    allele_subset = {sid: allele_sets_by_sample[sid] for sid in sample_ids}
                    partition_inputs.append((part_df, hla_groups, allele_to_id, allele_subset, local_bests_count, i, self.verbose))
    
                iteration_candidates = []
                process_outputs = {}
    
                with ProcessPoolExecutor(max_workers=n_parts) as executor:
                    future_to_pid = {
                        executor.submit(_process_partition_worker, *args): args[5]  # args[5] is pid
                        for args in partition_inputs
                    }
    
                    for future in as_completed(future_to_pid):
                        pid = future_to_pid[future]
                        try:
                            results = future.result()
                            iteration_candidates.extend(results)
                            process_outputs[pid] = results
                        except Exception as e:
                            print(f"[Partition {pid}] Failed with error: {e}")
    
                if self.verbose:
                    for pid, samples in sorted(process_outputs.items()):
                        print(f"\n[Process {pid}] Top {local_bests_count} samples from partition:")
                        for i, sample in enumerate(samples, start=1):
                            sid = sample["Sample ID"]
                            cov = sample.get("Coverage Subset (%)", "?")
                            print(f"  ({i}) {sid} — Partition coverage: {cov}%")
    
                if not iteration_candidates:
                    if self.verbose:
                        print("[Warning] No candidates returned from any partition.")
                    break
    
                best_sample = None
                best_coverage = -1
                best_filter = None
    
                if self.verbose:
                    print("\n[Global Evaluation] Computing full coverage for all candidates:")
    
                for candidate in iteration_candidates:
                    sample_id = candidate["Sample ID"]
                    full_coverage, _ = sample_coverage_percent(
                        df=current_df,
                        hla_groups=hla_groups,
                        sample_id=sample_id,
                        allele_sets_by_sample=allele_sets_by_sample,
                        allele_to_id=allele_to_id,
                        coverage_type="intersection",
                        verbose=False,
                    )
                    candidate["Coverage Total (%)"] = full_coverage
    
                    if self.verbose:
                        print(f"  ➤ {sample_id} — Global coverage (remaining set): {full_coverage}%")
    
                    if full_coverage > best_coverage:
                        best_sample = candidate
                        best_filter = candidate["Alleles"]
                        best_coverage = full_coverage
    
                if best_sample is None:
                    if self.verbose:
                        print("[Warning] No valid global sample found.")
                    break
    
                best_sample_id = best_sample["Sample ID"]
                selected_sample_ids.append(best_sample_id)
                filter_sequence.append(best_filter)
    
                if self.verbose:
                    subset_coverage = best_sample.get("Coverage Subset (%)", "?")
                    print(
                        f"\n[Selected] Sample {best_sample_id} with estimated coverage "
                        f"{best_coverage}% (remaining global), {subset_coverage}% (subset)"
                    )
    
                coverage_pct, combined_alleles, covered_ids, remainder_df = multi_sample_coverage_and_remainder(
                    df=df,
                    hla_groups=hla_groups,
                    sample_ids=selected_sample_ids,
                    sample_allele_sets=allele_sets_by_sample,
                    coverage_type="intersection",
                    verbose=self.verbose,
                )
    
                # Save restart state
                if self.restart_path is not None:
                    hla_pairs_named = {
                        group: [id_to_allele[allele_id] for allele_id in best_filter.get(group, [])]
                        for group in best_filter
                    }
                    save_restart_state(
                        path=self.restart_path,
                        sample_id=best_sample_id,
                        hla_groups=best_filter.keys(),
                        hla_pairs=hla_pairs_named,
                        coverage=best_sample["Coverage Total (%)"],
                        cumulative_cover=coverage_pct,
                    )
                    if self.verbose:
                        print(f"[Restart] Saved restart state to: {self.restart_path}")
    
                current_df = remainder_df
                current_sample_ids = set(current_df["Sample ID"])
    
                if self.verbose:
                    print(f"[Info] Accumulated {len(selected_sample_ids)} Sample IDs:\n    {selected_sample_ids}")
                    print(f"[Info] Total dataset coverage: {coverage_pct}%")
                    print(f"[Info] Remaining uncovered samples: {len(current_sample_ids)}")
    
        return selected_sample_ids, filter_sequence, current_df

    def run_local_global_greedy_cover(
        self,
        df,
        hla_groups,
        allele_sets_by_sample,
        allele_to_id,
        id_to_allele,
        max_iter=10,
        top_n=1,
        top_k_global=5,
        verbose=False,
        enable_timing=False,
        timing=None,
    ):
        """
        Public method to run the greedy solver using the configured backend.

        Dispatches to the appropriate internal runner based on `mode`.

        Parameters
        ----------
        df : pandas.DataFrame
            The input dataset of samples and HLA allele columns
        hla_groups : list
            List of HLA allele column groups to cover
        allele_sets_by_sample : dict
            Mapping from Sample ID → group → frozenset of allele IDs
        allele_to_id : dict
            Allele → integer ID mapping
        id_to_allele : dict
            Reverse mapping of integer ID → allele
        max_iter : int
            Maximum greedy iterations to perform
        top_n : int
            Number of top candidates to keep per iteration (default: 1)
        top_k_global : int
            Number of top global solutions to maintain (default: 5)
        verbose : bool
            Enable verbose output
        enable_timing : bool
            Enable timing logger (if available)
        timing : context manager, optional
            Timing context to wrap stages

        Returns
        -------
        tuple
            (selected_sample_ids, filter_sequence, remaining_uncovered_df)
        """
        if self.mode == 'serial':
            print("\n[BackendManager] Running in serial mode.")
            return self._run_serial(
                df=df,
                hla_groups=hla_groups,
                allele_sets_by_sample=allele_sets_by_sample,
                allele_to_id=allele_to_id,
                id_to_allele=id_to_allele,
                max_iter=max_iter,
                top_n=top_n,
                top_k_global=top_k_global,
                verbose=verbose,
                enable_timing=enable_timing,
                timing=timing,
            )
    
        elif self.mode == 'thread_pool':
            print("\n[BackendManager] Running in thread_pool mode.")
            return self._run_thread_pool(
                df=df,
                hla_groups=hla_groups,
                allele_sets_by_sample=allele_sets_by_sample,
                allele_to_id=allele_to_id,
                id_to_allele=id_to_allele,
                max_iter=max_iter,
                top_n=top_n,
                top_k_global=top_k_global,
                verbose=verbose,
                enable_timing=enable_timing,
                timing=timing,
            )

        elif self.mode == 'process_pool':
            print("\n[BackendManager] Running in process_pool mode.")
            return self._run_process_pool(
                df=df,
                hla_groups=hla_groups,
                allele_sets_by_sample=allele_sets_by_sample,
                allele_to_id=allele_to_id,
                id_to_allele=id_to_allele,
                max_iter=max_iter,
                top_n=top_n,
                top_k_global=top_k_global,
                verbose=verbose,
                enable_timing=enable_timing,
                timing=timing,
            )
 
        else:
            raise ValueError(f"Unknown backend mode: {self.mode}")
    

# hla_solver/greedy.py

import numpy as np
import time
import pprint
import pandas as pd
# from .selectors import find_top_covering_samples, multi_sample_coverage_and_remainder

from hla_solver.selectors import find_top_covering_samples, multi_sample_coverage_and_remainder, sample_coverage_percent
from hla_solver.utils import verbose_logger, save_restart_state, load_restart_state
from contextlib import nullcontext
# import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Set
import sys

@verbose_logger(label="Greedy HLA Coverage (Fast)")
def greedy_hla_coverage_fast(
    df,
    hla_groups,
    allele_sets_by_sample,
    allele_to_id,
    id_to_allele,
    max_iter=10,
    top_n=1,
    restart_path=None,
    verbose=False,
    enable_timing=False,
    timing=None,
):
    """
    Serial greedy algorithm to find a minimal set of samples covering the alleles.

    Parameters
    ----------
    df : pd.DataFrame
        Input sample dataframe.
    hla_groups : list
        List of HLA group tuples, e.g. [('HLA-A', 'A1', 'A2'), ...]
    allele_to_id : dict
        Mapping from allele string to numeric ID.
    allele_sets_by_sample : dict
        Mapping from Sample ID -> {HLA group -> frozenset of allele IDs}.
    max_iter : int
        Max number of iterations to run.
    top_n : int
        Number of top samples to consider at each step.
    restart_path : str or Path
        Optional path to resume from a previous state.
    verbose : bool
        If True, logs detailed output.
    enable_timing : bool
        Whether to time internal steps.
    timing : context manager
        Timing context function.

    Returns
    -------
    selected_sample_ids : list
        List of Sample IDs selected by the greedy coverage algorithm.
    filter_sequence : list
        Corresponding list of allele filter dictionaries per selected sample.
    remainder_df : pd.DataFrame
        Remaining samples after selection.
    """

    selected_sample_ids = []
    filter_sequence = []
    match_cache = {}

    print( f" restart_path: {restart_path}" )
    if restart_path is not None:
        restart_path = Path(restart_path)
        if restart_path.is_file():
            selected_sample_ids, filter_sequence = load_restart_state(restart_path)
            if verbose:
                print(f"[Restart] Loaded {len(selected_sample_ids)} samples from: {restart_path}")
            
            # Recompute remainder and match_cache from selected samples
            _, _, match_cache, remainder_df = multi_sample_coverage_and_remainder(
                df,
                hla_groups,
                selected_sample_ids,
                allele_sets_by_sample,
                coverage_type='intersection',
                verbose=verbose
            )
            current_sample_ids = set(remainder_df["Sample ID"])
        else:
            raise FileNotFoundError(f"[Restart] File not found: {restart_path}")
    else:
        # Start fresh
        current_sample_ids = set(df["Sample ID"])
        remainder_df = df.copy()

    for iteration in range(max_iter):
        if not current_sample_ids:
            if verbose:
                print("[Info] All samples have been covered.")
            break

        if verbose:
            print(f"\n[Iteration {iteration + 1}] Searching top-{top_n} filters on {len(current_sample_ids)} samples...")

        subset_allele_sets = {
            sid: allele_sets_by_sample[sid]
            for sid in current_sample_ids
        }

        with (timing(f"Iteration {iteration + 1}::find_top_covering_samples", enabled=enable_timing)
              if timing else nullcontext()):
            top_samples = find_top_covering_samples(
                df=remainder_df[remainder_df["Sample ID"].isin(current_sample_ids)],
                hla_groups=hla_groups,
                top_n=top_n,
                allele_sets_by_sample=subset_allele_sets,
                allele_to_id=allele_to_id,
                verbose=verbose,
                enable_timing=enable_timing,
                timing=timing,
            )

        if not top_samples:
            raise RuntimeError("[greedy_hla_coverage_fast] No samples with valid coverage found in this iteration.")

        best_sample = top_samples[0]
        sample_id = best_sample["Sample ID"]
        allele_filter = best_sample["Alleles"]

        full_coverage_pct, _ = sample_coverage_percent(
            df=df,
            hla_groups=hla_groups,
            sample_id=sample_id,
            allele_sets_by_sample=allele_sets_by_sample,
            allele_to_id=allele_to_id,
            coverage_type='intersection',
            verbose=False
        )
        best_sample["Coverage Total (%)"] = full_coverage_pct

        if verbose:
            print(
                f"[Selected] Sample {sample_id} with estimated coverage "
                f"{best_sample.get('Coverage Total (%)', '?')}% (full), "
                f"{best_sample.get('Coverage Subset (%)', '?')}% (subset)"
            )

        selected_sample_ids.append(sample_id)
        filter_sequence.append(allele_filter)

        full_coverage_pct, _, match_cache, remainder_df = multi_sample_coverage_and_remainder(
            df,
            hla_groups,
            selected_sample_ids,
            allele_sets_by_sample,
            coverage_type='intersection',
            verbose=verbose
        )

        # print(f"hla_groups: {hla_groups}")
        # print(f"type(hla_groups): {type(hla_groups)}")
        # print(f"filter_sequence: {filter_sequence[0].keys()}")
        # print(f"type(filter_sequence): {type(filter_sequence)}")

        # Map allele IDs to string names using id_to_allele
        hla_pairs_named = {
            group: [id_to_allele[allele_id] for allele_id in alleles]
            for group, alleles in allele_filter.items()
        }

        if restart_path is not None:
            save_restart_state(
                path=restart_path,
                sample_id=sample_id,
                hla_groups=filter_sequence[0].keys(),
                hla_pairs=hla_pairs_named,
                coverage=best_sample["Coverage Total (%)"], 
                cumulative_cover=full_coverage_pct
            )
            if verbose:
                print(f"[Restart] Saved restart state to: {restart_path}")

        current_sample_ids = set(remainder_df["Sample ID"])

        if verbose:
            print(f"[Info] Accumulated {len(selected_sample_ids)} Sample IDs:\n    {selected_sample_ids}")
            print(f"[Info] Total data set coverage: {full_coverage_pct}%")
            print(f"[Info] Remaining uncovered samples: {len(current_sample_ids)}")

    return selected_sample_ids, filter_sequence, remainder_df

# @verbose_logger(label="Greedy HLA Coverage (Fast)")
# def greedy_hla_coverage_fast(
#     df,
#     hla_groups,
#     allele_to_id,
#     allele_matrix,
#     allele_id_to_positions,
#     max_iter=10,
#     top_n=1,
#     verbose=False
# ):
#     """
#     Serial greedy algorithm to find a minimal set of samples covering the alleles.

#     Returns:
#         selected_sample_ids (list): Sample IDs of selected samples.
#         filter_sequence (list): Corresponding allele filters (sets or lists).
#         remainder_df (pd.DataFrame): DataFrame of remaining uncovered samples.
#     """
#     current_indices = np.arange(len(df))
#     selected_sample_ids = []
#     filter_sequence = []
#     match_cache = {}

#     for iteration in range(max_iter):
#         if len(current_indices) == 0:
#             if verbose:
#                 print("All samples covered.")
#             break

#         if verbose:
#             print(f"\nIteration {iteration + 1}: Searching top-{top_n} filters on {len(current_indices)} samples...")

#         sub_df = df.iloc[current_indices].reset_index(drop=True)
#         sub_matrix = allele_matrix[current_indices]

#         top_samples = find_top_covering_samples(
#             df=sub_df,
#             hla_groups=hla_groups,
#             top_n=top_n,
#             allele_to_id=allele_to_id,
#             allele_matrix=sub_matrix,
#             allele_id_to_positions=allele_id_to_positions,
#             total_samples=len(df),
#             current_subset_indices=current_indices
#         )

#         if not top_samples:
#             if verbose:
#                 print("No more samples with coverage found.")
#             break

#         best_sample = top_samples[0]
#         local_sample_idx = best_sample["Sample Index"]
#         global_sample_idx = current_indices[local_sample_idx]
#         sample_id = df.iloc[global_sample_idx]["Sample ID"]
#         allele_filter = best_sample["Alleles"]

#         if verbose:
#             print(
#                 f"Selected Sample {sample_id} (index {global_sample_idx}) with estimated coverage "
#                 f"{best_sample['Coverage Total (%)']}% (full), "
#                 f"{best_sample['Coverage Subset (%)']}% (subset)"
#             )

#         selected_sample_ids.append(sample_id)
#         filter_sequence.append(allele_filter)

#         # Use updated multi_sample_coverage_and_remainder
#         try:
#             _, _, match_cache, remainder_df = multi_sample_coverage_and_remainder(
#                 df, hla_groups, selected_sample_ids, match_cache=match_cache
#             )
#         except Exception as e:
#             if verbose:
#                 print(f"[Error] Failed to compute remainder_df: {e}")
#             break

#         if remainder_df is None:
#             if verbose:
#                 print("[Warning] remainder_df is None. Stopping early.")
#             break

#         current_indices = remainder_df.index.to_numpy()

#     return selected_sample_ids, filter_sequence, remainder_df


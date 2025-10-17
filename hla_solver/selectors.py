"""
selectors.py
============

This module provides functions for analyzing HLA allele distributions and
identifying top-performing alleles or sample combinations that maximize dataset coverage.

These tools are useful for:
- Initializing or benchmarking greedy HLA coverage solvers
- Selecting effective filter alleles
- Analyzing allele frequency by group
- Estimating sample-level coverage

Functions
---------
- `analyze_hla_group`: Compute allele frequencies and matched sample counts per HLA group.
- `find_best_single_allele_filter`: Select a single best-performing allele across all groups.
- `find_top_filter_combinations`: Discover top combinations of alleles to maximize sample match.
- `find_top_covering_samples`: Identify samples that individually match the most others.
- `sample_coverage_percent`: Calculate the % of the dataset matched by one sample.
- `find_matching_samples_by_filter`: Return samples matching a custom allele filter.
- `multi_sample_coverage_and_remainder`: Compute coverage from a donor sample set.
- `multi_sample_coverage_with_cache`: Cached version for fast coverage evaluation.

Author
------
Jay M. Appleton

"""


import pandas as pd
import time
from collections import defaultdict
import numpy as np
from typing import Optional, List, Tuple, Union, Dict, Any
import itertools

import time
import pandas as pd
from typing import List, Tuple, Dict, Any, Callable
from contextlib import contextmanager

from contextlib import nullcontext
from typing import Callable, List, Dict, Tuple, Any
import pandas as pd
import time

from typing import List, Tuple, Dict, Set, Optional
import pandas as pd

from typing import Dict, Set

def is_sample_covered(
    reference_alleles: Dict[str, Set[int]],
    query_alleles: Dict[str, Set[int]],
    coverage_type: str = "intersection"
) -> bool:
    """
    Determine whether a query sample is covered by the reference sample
    based on a specific coverage logic across HLA groups.

    Parameters
    ----------
    reference_alleles : dict
        Dictionary of group -> set of allele IDs (e.g., from the reference sample).
    query_alleles : dict
        Dictionary of group -> set of allele IDs (e.g., from another sample).
    coverage_type : str
        One of "union", "intersection", "identity".

    Returns
    -------
    bool
        True if query is covered by reference, False otherwise.
    """
    if coverage_type == "identity":
        return all(reference_alleles[g] == query_alleles[g] for g in reference_alleles)
    elif coverage_type == "intersection":
        return all(reference_alleles[g] & query_alleles[g] for g in reference_alleles)
    elif coverage_type == "union":
        return any(reference_alleles[g] & query_alleles[g] for g in reference_alleles)
    else:
        raise ValueError(f"Invalid coverage_type: {coverage_type}")

def multi_sample_coverage_and_remainder(
    df: pd.DataFrame,
    hla_groups: List[Tuple[str, str, str]],
    sample_ids: List[str],
    sample_allele_sets: Dict[str, Dict[str, frozenset]],
    coverage_type: str = "intersection",
    verbose: bool = False,
) -> Tuple[float, Dict[str, Set[int]], Set[str], pd.DataFrame]:
    """
    Computes dataset coverage using a set of donor sample IDs.
    Each recipient is covered only if at least one donor fully matches
    under the specified coverage type (intersection, union, identity).
    """

    if not sample_ids:
        if verbose:
            print("[multi_sample_coverage_and_remainder] No sample IDs provided.")
        return 0.0, {}, set(), df.copy()

    # Sanitize donor sample allele sets
    donor_allele_sets = {
        sid: sample_allele_sets[sid]
        for sid in sample_ids
        if sid in sample_allele_sets
    }

    # Store covered recipient IDs
    all_sample_ids = df["Sample ID"].tolist()
    covered_sample_ids = set()

    for target_id in all_sample_ids:
        if target_id not in sample_allele_sets:
            continue
        target_alleles = sample_allele_sets[target_id]

        for donor_id, donor_alleles in donor_allele_sets.items():
            if is_sample_covered(donor_alleles, target_alleles, coverage_type):
                covered_sample_ids.add(target_id)
                break  # Stop at first matching donor

    total_samples = len(all_sample_ids)
    coverage_pct = round(100 * len(covered_sample_ids) / total_samples, 2) if total_samples else 0.0
    uncovered_df = df[~df["Sample ID"].isin(covered_sample_ids)]

    # Report alleles used (union of donor alleles per group)
    combined_allele_dict = {group: set() for group, _, _ in hla_groups}
    for donor_dict in donor_allele_sets.values():
        for group in combined_allele_dict:
            combined_allele_dict[group].update(donor_dict.get(group, set()))

    if verbose:
        logic_label = {
            "union": "at least one allele in any group",
            "intersection": "at least one allele in all groups",
            "identity": "exact allele pairs in all groups"
        }.get(coverage_type.lower(), "unknown")

        print(f"\n[multi_sample_coverage_and_remainder] Sample IDs used: {sample_ids}")
        print(f"  ➤ Coverage type: {coverage_type.upper()} ({logic_label})")
        print(f"  ➤ Combined dataset coverage: {coverage_pct}% ({len(covered_sample_ids)} of {total_samples} samples)")
        print("  ➤ Allele IDs used by group:")
        for group, alleles in combined_allele_dict.items():
            print(f"    {group}: {sorted(alleles)}")

    return coverage_pct, combined_allele_dict, covered_sample_ids, uncovered_df


def find_top_covering_samples(
    df: pd.DataFrame,
    hla_groups: List[Tuple[str, str, str]],
    top_n: int,
    allele_sets_by_sample: Dict[str, Dict[str, frozenset]],
    allele_to_id: Dict[str, int],
    coverage_type: str = "intersection",
    verbose: bool = False,
    enable_timing: bool = False,
    timing: Callable[[str, bool], Any] = None
) -> List[Dict[str, Any]]:
    """
    Identify the top N samples with the highest allele coverage based on precomputed integer sets.
    """
    # fallback if no timing function provided
    timing = timing or (lambda label, enabled=True: nullcontext())

    sample_results = []

    with timing("find_top_covering_samples::main", enabled=enable_timing):
        start_time = time.time()

        for idx, row in df.iterrows():
            sample_id = row["Sample ID"]

            try:
                with timing(f"coverage::sample_{sample_id}", enabled=False):
                    from hla_solver.selectors import sample_coverage_percent
                    coverage_pct, allele_dict = sample_coverage_percent(
                        df=df,
                        hla_groups=hla_groups,
                        sample_id=sample_id,
                        allele_sets_by_sample=allele_sets_by_sample,
                        allele_to_id=allele_to_id,
                        coverage_type=coverage_type,
                        verbose=False
                    )
            except Exception as e:
                if verbose:
                    print(f"[Warning] Skipping sample '{sample_id}' due to error: {e}")
                continue

            if coverage_pct is None:
                continue

            match_count_subset = round(coverage_pct * len(df) / 100)

            sample_results.append({
                "Sample Index": idx,
                "Sample ID": sample_id,
                "Alleles": allele_dict,
                "Matched Samples Subset": match_count_subset,
                "Coverage Subset (%)": coverage_pct
            })

            if verbose and ((idx + 1) % 100 == 0 or (idx + 1) == len(df)):
                progress = (idx + 1) / len(df)
                elapsed = time.time() - start_time
                eta = (elapsed / progress) - elapsed
                eta_h, eta_r = divmod(int(eta), 3600)
                eta_m, eta_s = divmod(eta_r, 60)
                print(f"Processed {idx + 1}/{len(df)} samples... ETA ~ {eta_h:02}:{eta_m:02}:{eta_s:02}", end='\r')

        with timing("sorting_top_samples", enabled=enable_timing):
            top_sample_results = sorted(
                sample_results,
                key=lambda x: x["Matched Samples Subset"],
                reverse=True
            )[:top_n]

    if verbose:
        total_time = int(time.time() - start_time)
        print(f"\nCompleted all samples in {total_time}s")
        for rank, result in enumerate(top_sample_results, 1):
            print(f"({rank}) Sample ID: {result['Sample ID']} — Coverage: {result['Coverage Subset (%)']}%")

    return top_sample_results

def sample_coverage_percent(
    df: pd.DataFrame,
    hla_groups: List[Tuple[str, str, str]],
    sample_id: str,
    allele_sets_by_sample: Dict[str, Dict[str, frozenset]],
    allele_to_id: Dict[str, int],
    coverage_type: str = "intersection",
    verbose: bool = False
) -> Tuple[Optional[float], Optional[Dict[str, List[int]]]]:
    """
    Calculate the percentage of the dataset that shares alleles with a given sample,
    using specified matching logic across HLA groups.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing allele columns.
    hla_groups : list of tuple(str, str, str)
        List of HLA group definitions, where each tuple is (group_name, col1, col2).
    sample_id : str
        The ID of the sample to evaluate as a reference.
    allele_sets_by_sample : dict
        Dictionary mapping sample IDs to allele sets by HLA group.
    allele_to_id : dict
        Mapping from allele strings to integer IDs.
    coverage_type : str, optional
        One of "intersection", "union", or "identity". Default is "intersection".
    verbose : bool, optional
        If True, prints detailed analysis. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
            - coverage_pct (float): Percent of dataset matched
            - allele_dict (dict): The reference alleles used, grouped by HLA type

    Raises
    ------
    ValueError
        If the sample ID is not found in the allele set dictionary.

    Examples
    --------
    >>> sample_coverage_percent(df, hla_groups, "Sample123", allele_sets, allele_to_id)
    (87.5, {"HLA-A": [1, 2], "HLA-B": [5, 7], ...})
    """
    if sample_id not in allele_sets_by_sample:
        if verbose:
            print(f"[sample_coverage_percent] Sample ID '{sample_id}' not found.")
        return None, None

    reference_alleles = allele_sets_by_sample[sample_id]
    match_count = 0
    total_samples = len(allele_sets_by_sample)

    for other_id, other_alleles in allele_sets_by_sample.items():
        if other_id == sample_id:
            continue

        if is_sample_covered(reference_alleles, other_alleles, coverage_type):
            match_count += 1

    coverage_pct = round(100 * match_count / total_samples, 2) if total_samples else 0.0

    if verbose:
        logic_label = {
            "union": "at least one allele in any group",
            "intersection": "at least one allele in all groups",
            "identity": "exact allele pairs in all groups"
        }[coverage_type.lower()]

        print(f"\n[sample_coverage_percent] Sample ID '{sample_id}' covers "
              f"{coverage_pct}% of the dataset ({match_count} / {total_samples} samples).")
        print(f"  ➤ Coverage type: {coverage_type.upper()} ({logic_label})")
        print("  ➤ Sample Alleles (by group):")
        for group, alleles in reference_alleles.items():
            print(f"    {group}: {sorted(alleles)}")

    return coverage_pct, {group: sorted(alleles) for group, alleles in reference_alleles.items()}


##############################################################################################################
##############################################################################################################
##############################################################################################################

def analyze_hla_group(
    df: pd.DataFrame,
    group_name: str,
    col1: str,
    col2: str,
    timing: bool = False,
    required_allele_filter: Optional[List[Tuple[str, str, str]]] = None,
    relative_to_full_df: bool = False
) -> Optional[pd.DataFrame]:
    """
    Analyze allele frequency and sample coverage for a specific HLA group.

    This function evaluates the occurrence and coverage of unique alleles
    from the two specified columns (representing a single HLA group),
    optionally filtered by required allele presence in other columns.

    .. note::
        This function is currently included in the ``selectors`` module,
        but may be better suited in a future ``analysis`` module.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing HLA allele columns.
    group_name : str
        Human-readable name for the group (e.g., "HLA-A").
    col1 : str
        First column containing alleles for this group.
    col2 : str
        Second column containing alleles for this group.
    timing : bool, optional
        If True, prints timing information (default: False).
    required_allele_filter : list of tuple(str, str, str), optional
        List of required allele filters applied before analysis.
        Each tuple is (col1, col2, allele).
    relative_to_full_df : bool, optional
        If True, coverage percentages are calculated relative to the
        original (unfiltered) dataset. If False, percentages are relative
        to the filtered subset (default: False).

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with allele statistics sorted by matched sample count.
        If no data remains after filtering, returns None.

        Columns:
            - "Allele ID": Internal numeric ID
            - "Allele Name": Original allele string
            - "Matched Sample Count": Number of samples containing allele
            - "Percentage Cover": Coverage as percentage (float)

    Examples
    --------
    >>> analyze_hla_group(df, "HLA-A", "HLA-A1", "HLA-A2")
    """

    if timing:
        t_start = time.time()

    print(f"\nAnalyzing HLA Group: {group_name}")

    df_original = df.copy()
    original_num_samples = len(df_original)

    # Apply required allele filtering (if provided)
    if required_allele_filter:
        for i, (f_col1, f_col2, req_allele) in enumerate(required_allele_filter):
            condition = (
                (df[f_col1].astype(str).str.strip() == req_allele) |
                (df[f_col2].astype(str).str.strip() == req_allele)
            )
            df = df[condition]
            print(f"Filter {i+1}: Required allele '{req_allele}' in '{f_col1}' or '{f_col2}' — {len(df)} samples remaining")

    if df.empty:
        print("No samples remaining after filtering. Exiting analysis.")
        return None

    num_samples = len(df)

    # Collect all alleles and map to IDs
    all_alleles = pd.concat([
        df[col1].astype(str).str.strip(),
        df[col2].astype(str).str.strip()
    ])
    unique_alleles = all_alleles.unique()

    allele_to_id = {allele: idx for idx, allele in enumerate(unique_alleles)}
    id_to_allele = {idx: allele for allele, idx in allele_to_id.items()}

    # Encode alleles numerically
    allele_matrix = np.zeros((num_samples, 2), dtype=int)
    allele_matrix[:, 0] = df[col1].astype(str).str.strip().map(allele_to_id)
    allele_matrix[:, 1] = df[col2].astype(str).str.strip().map(allele_to_id)

    # Track which samples each allele appears in
    allele_id_to_positions = defaultdict(set)
    for sample_idx in range(num_samples):
        for locus_idx in range(2):
            allele_id = allele_matrix[sample_idx, locus_idx]
            allele_id_to_positions[allele_id].add(sample_idx)

    # Compute statistics for each allele
    denominator = original_num_samples if relative_to_full_df else num_samples
    allele_id_stats = []
    for allele_id, matched_samples in allele_id_to_positions.items():
        count = len(matched_samples)
        pct_cover = count / denominator if denominator > 0 else 0.0
        allele_id_stats.append({
            "Allele ID": allele_id,
            "Allele Name": id_to_allele[allele_id],
            "Matched Sample Count": count,
            "Percentage Cover": round(100 * pct_cover, 2)
        })

    allele_df = pd.DataFrame(allele_id_stats).sort_values(
        by="Matched Sample Count", ascending=False
    ).reset_index(drop=True)

    if timing:
        t_end = time.time()
        print(f"Time taken for {group_name}: {t_end - t_start:.4f} seconds")

    return allele_df

def find_best_single_allele_filter(df, hla_groups, timing=False):
    """
    Identify the best single HLA allele (from any group) to use as a filter based on its
    ability to filter the most samples from the dataset.

    This function scans the top allele in each HLA group and evaluates its filtering
    impact, returning the one that filters the largest number of samples.

    Args:
        df (pd.DataFrame): Full dataset of samples with HLA allele columns.
        hla_groups (list of tuples): List of (group_name, col1, col2) HLA group definitions.
        timing (bool, optional): Whether to print timing/debug info. Defaults to False.

    Returns:
        tuple: 
            - best_filter (tuple): (col1, col2, allele) representing the most impactful filter.
            - best_stats (dict): Dictionary summarizing filter impact (sample count, etc.).
    """
    top_alleles_per_group = []

    # Step 1: Get top allele for each HLA group
    for group_name, col1, col2 in hla_groups:
        print(f"\n[Scanning top allele for group {group_name}]")
        all_df = analyze_hla_group(df.copy(), group_name, col1, col2, timing=timing)
        if all_df is None or all_df.empty:
            continue
        top_allele = all_df.iloc[0]["Allele Name"]
        top_alleles_per_group.append((col1, col2, top_allele))

    print(f"\nCollected top alleles from each group: {top_alleles_per_group}")

    # Step 2: Test each top allele as a single filter
    best_filter = None
    best_stats = {"Filtered Samples": 0}

    for i, filter_tuple in enumerate(top_alleles_per_group):
        f_col1, f_col2, allele = filter_tuple
        print(f"[Testing filter {i+1}/{len(top_alleles_per_group)}: {allele} in {f_col1}/{f_col2}]", end='\r', flush=True)

        filtered_df = df[
            (df[f_col1].astype(str).str.strip() == allele) |
            (df[f_col2].astype(str).str.strip() == allele)
        ]

        num_filtered_samples = len(filtered_df)
        print(f" --> {num_filtered_samples} samples match")

        if num_filtered_samples > best_stats["Filtered Samples"]:
            best_stats = {
                "Filter": filter_tuple,
                "Filtered Samples": num_filtered_samples
            }
            best_filter = filter_tuple

    print("\nBest Single Filter Allele:")
    print(best_stats)
    return best_filter, best_stats

def find_top_filter_combinations(
    df: pd.DataFrame,
    hla_groups: List[Tuple[str, str, str]],
    top_k_per_group: int = 5,
    top_n_results: int = 8,
    timing: bool = False
) -> List[Dict[str, Any]]:
    """
    Finds the top combinations of single allele filters (one per group)
    that result in the highest dataset coverage.

    This function identifies the most common alleles within each HLA group,
    then exhaustively tests all possible one-allele-per-group filter
    combinations to determine which combinations match the most samples.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing allele columns.
    hla_groups : list of tuple(str, str, str)
        Each tuple defines an HLA group as (group_name, col1, col2).
    top_k_per_group : int, optional
        Number of top alleles to select from each group (default: 5).
    top_n_results : int, optional
        Number of top filter combinations to return (default: 8).
    timing : bool, optional
        If True, enables timing and verbose analysis output (default: False).

    Returns
    -------
    list of dict
        Each result dictionary contains:
            - "Combo": list of (group_name, col1, col2, allele)
            - "Matched Samples": int
            - "Coverage (%)": float

    Example
    -------
    >>> find_top_filter_combinations(df, hla_groups=[("HLA-A", "HLA-A1", "HLA-A2")])
    """

    all_group_top_alleles = []

    for group_name, col1, col2 in hla_groups:
        print(f"\nAnalyzing top {top_k_per_group} alleles for group: {group_name}")
        allele_df = analyze_hla_group(df.copy(), group_name, col1, col2, timing=timing)

        if allele_df is None or allele_df.empty:
            print(f"Skipping group {group_name} — no alleles found.")
            continue

        top_k = allele_df.head(top_k_per_group)
        group_top = [(group_name, col1, col2, row["Allele Name"]) for _, row in top_k.iterrows()]
        all_group_top_alleles.append(group_top)

    # Generate all combinations: one allele per group
    combinations = list(itertools.product(*all_group_top_alleles))
    print(f"\nTotal combinations to test: {len(combinations)}")

    total_samples = len(df)
    combination_results = []

    for i, combo in enumerate(combinations):
        required_filter = [(col1, col2, allele) for (_, col1, col2, allele) in combo]
        filtered_df = df.copy()

        # Apply filter (AND logic across groups)
        for col1, col2, allele in required_filter:
            condition = (
                (filtered_df[col1].astype(str).str.strip() == allele) |
                (filtered_df[col2].astype(str).str.strip() == allele)
            )
            filtered_df = filtered_df[condition]

        matched = len(filtered_df)
        coverage_pct = round(100 * matched / total_samples, 2)

        combination_results.append({
            "Combo": combo,
            "Matched Samples": matched,
            "Coverage (%)": coverage_pct
        })

        if i % 50 == 0 or i == len(combinations) - 1:
            print(f"  Tested {i+1}/{len(combinations)} combinations", end='\r', flush=True)

    # Sort and return top N combinations
    top_results = sorted(
        combination_results,
        key=lambda x: x["Matched Samples"],
        reverse=True
    )[:top_n_results]

    print("\nTop Filter Combinations:\n")
    for idx, res in enumerate(top_results, 1):
        filter_list = [f"{group} {allele}" for (group, _, _, allele) in res["Combo"]]
        filter_str = ', '.join(filter_list)
        print(f"({idx})  [ {filter_str} ]")
        print(f"      Total Coverage: {res['Coverage (%)']} %\n")

    return top_results

def find_matching_samples_by_filter(
    df: pd.DataFrame,
    filter_alleles: Dict[str, List[str]],
    coverage_type: str = "intersection",
    verbose: bool = False
) -> List[str]:
    """
    Identifies Sample IDs from the DataFrame that match a given set of allele filters,
    using the specified matching logic.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing HLA allele information.
    filter_alleles : dict
        Dictionary specifying allele pairs to match per HLA group.
        Example: { "HLA-A": ["02:01", "24:02:04"], ... }
    coverage_type : str, optional
        Matching logic: "union", "intersection", or "identity". Default is "identity".
    verbose : bool, optional
        Whether to print matching sample count and summary.

    Returns
    -------
    List[str]
        List of Sample IDs matching the given filter and logic.

    Raises
    ------
    ValueError
        If no filter alleles are provided, or invalid coverage type is passed.

    Notes
    -----
    - Matching is order-independent across all types.
    - Assumes that for each HLA group, two columns exist (e.g., HLA-A_1, HLA-A_2).
    """

    if not filter_alleles:
        raise ValueError("No filter alleles provided.")

    if coverage_type.lower() not in {"union", "intersection", "identity"}:
        raise ValueError("Invalid coverage_type. Must be one of: 'union', 'intersection', 'identity'.")

    coverage_type = coverage_type.lower()

    matched_indices = pd.Series(False, index=df.index) if coverage_type == "union" else pd.Series(True, index=df.index)

    for group, filter_pair in filter_alleles.items():
        clean_alleles = sorted(str(a).strip() for a in filter_pair)

        group_cols = [col for col in df.columns if col.startswith(group)]
        if len(group_cols) != 2:
            raise ValueError(f"Expected exactly 2 columns for group '{group}', found: {group_cols}")
        col1, col2 = group_cols

        if coverage_type == "identity":
            group_match = df.apply(
                lambda row: sorted([str(row[col1]).strip(), str(row[col2]).strip()]) == clean_alleles,
                axis=1
            )
        else:
            group_match = df.apply(
                lambda row: bool(
                    set([str(row[col1]).strip(), str(row[col2]).strip()])
                    & set(clean_alleles)
                ),
                axis=1
            )

        if coverage_type == "union":
            matched_indices |= group_match
        elif coverage_type == "intersection":
            matched_indices &= group_match
        elif coverage_type == "identity":
            matched_indices &= group_match  # strict match for each group

    matched_df = df[matched_indices]
    matched_sample_ids = matched_df["Sample ID"].tolist()

    if verbose:
        print(f"\n[filter match] Found {len(matched_sample_ids)} matching sample(s) using '{coverage_type}' logic.\n")
        if not matched_df.empty:
            print(matched_df[["Sample ID"] + [col for col in df.columns if col.startswith(tuple(filter_alleles.keys()))]])

    return matched_sample_ids


def multi_sample_coverage_with_cache(
    df: pd.DataFrame,
    hla_groups: List[Tuple[str, str, str]],
    sample_indices: List[int],
    match_cache: Optional[Dict[Tuple[str, str], pd.Series]] = None
) -> Tuple[Optional[float], Optional[Dict[str, set]], Dict[Tuple[str, str], pd.Series]]:
    """
    Computes overall dataset coverage for a given list of sample indices,
    using combined allele filters across multiple HLA groups. Optionally caches
    match results to accelerate repeated evaluations.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing HLA allele columns.
    hla_groups : list of tuple
        List of (group_name, col1, col2) specifying the HLA groups and their columns.
    sample_indices : list of int
        Row indices in `df` representing the selected samples.
    match_cache : dict, optional
        Dictionary to store or reuse cached Series matching each (group, allele) combination.

    Returns
    -------
    coverage_pct : float or None
        Percentage of dataset matched by the combined allele set. Returns None if inputs are invalid.
    combined_allele_dict : dict or None
        Dictionary of alleles used for filtering, grouped by HLA group.
        Format: {group_name: set of alleles}
    updated_cache : dict
        Updated cache containing match Series for each (group, allele).

    Notes
    -----
    - Matching is based on OR logic across alleles within a group.
    - Cached results avoid repeated string comparisons for the same allele.
    """
    if match_cache is None:
        match_cache = {}

    # Check for invalid sample indices
    missing_indices = [idx for idx in sample_indices if idx not in df.index]
    if missing_indices:
        print(f"The following sample indices were not found in the DataFrame: {missing_indices}")
        return None, None, match_cache

    # Build combined allele dictionary by group from the selected samples
    combined_allele_dict: Dict[str, set] = {group: set() for group, _, _ in hla_groups}
    for idx in sample_indices:
        row = df.loc[idx]
        for group, col1, col2 in hla_groups:
            combined_allele_dict[group].update([
                str(row[col1]).strip(),
                str(row[col2]).strip()
            ])

    # Construct the overall match condition
    condition = pd.Series(False, index=df.index)

    for group, col1, col2 in hla_groups:
        group_condition = pd.Series(False, index=df.index)
        for allele in combined_allele_dict[group]:
            cache_key = (group, allele)
            if cache_key not in match_cache:
                match_series = (
                    df[col1].astype(str).str.strip().eq(allele) |
                    df[col2].astype(str).str.strip().eq(allele)
                )
                match_cache[cache_key] = match_series
            group_condition |= match_cache[cache_key]

        condition |= group_condition  # Union across all groups

    matched_df = df[condition]
    coverage_pct = round(100 * len(matched_df) / len(df), 2)

    # Logging output
    print(f"\nSample indices evaluated: {sample_indices}")
    print(f"Combined coverage: {coverage_pct}% ({len(matched_df)} of {len(df)} samples)")
    print("Combined Alleles by Group:")
    for group, alleles in combined_allele_dict.items():
        print(f"  • {group}: {sorted(alleles)}")

    return coverage_pct, combined_allele_dict, match_cache





"""
validation.py
Validation and evaluation metrics for HLA greedy coverage solutions.
"""

import numpy as np
import pandas as pd

def safe_allele(val):
    """Safely convert allele value to string or return empty string for NaN."""
    return str(val).strip() if pd.notnull(val) else ''

from typing import List, Tuple
import pandas as pd


def safe_allele(value) -> str:
    """Helper to ensure alleles are treated as stripped strings."""
    return str(value).strip()


def coverage_score(
    df: pd.DataFrame,
    hla_groups: List[Tuple[str, str, str]],
    selected_indices: List[int],
    coverage_type: str = "union",
    verbose: bool = False
) -> float:
    """
    Compute the percentage of samples covered by the selected samples.

    A selected sample 'covers' a trial sample if their alleles overlap
    in either all groups ("intersection") or any group ("union").

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset of HLA samples.
    hla_groups : list of tuple
        Each tuple specifies an HLA group and its two allele columns.
    selected_indices : list of int
        Row indices of the selected (reference) samples.
    coverage_type : str, optional
        "intersection" (default): Trial is covered if all groups share at least one allele.
        "union": Trial is covered if any group shares at least one allele.
    verbose : bool, optional
        Whether to print detailed output. Default is False.

    Returns
    -------
    float
        Percentage of samples in the dataset that are covered by the selected samples.

    Raises
    ------
    ValueError
        If an invalid coverage_type is provided.
    """
    if coverage_type.lower() not in {"intersection", "union"}:
        raise ValueError(f"Invalid coverage_type: '{coverage_type}'. Must be 'union' or 'intersection'.")

    is_union = coverage_type.lower() == "union"

    total_samples = len(df)
    covered_trials = set()

    def safe_allele(value):
        return str(value).strip()

    # Precompute alleles for each selected sample
    selected_alleles = {
        idx: {
            group: {safe_allele(df.loc[idx, col1]), safe_allele(df.loc[idx, col2])}
            for group, col1, col2 in hla_groups
        }
        for idx in selected_indices
    }

    # Iterate over all trial samples
    for trial_idx, trial_row in df.iterrows():
        trial_alleles_by_group = {
            group: {safe_allele(trial_row[col1]), safe_allele(trial_row[col2])}
            for group, col1, col2 in hla_groups
        }

        for test_alleles in selected_alleles.values():
            if is_union:
                # Match if any group overlaps
                if any(test_alleles[group] & trial_alleles_by_group[group] for group, _, _ in hla_groups):
                    covered_trials.add(trial_idx)
                    break
            else:
                # Match only if all groups overlap
                if all(test_alleles[group] & trial_alleles_by_group[group] for group, _, _ in hla_groups):
                    covered_trials.add(trial_idx)
                    break

    coverage_pct = (len(covered_trials) / total_samples) * 100 if total_samples else 0.0

    if verbose:
        print(f"\n[selectors.coverage_score] Coverage with {len(selected_indices)} selected samples:")
        print(f"  Coverage Type: {coverage_type.upper()}")
        print(f"  Samples Covered: {len(covered_trials)} / {total_samples} ({coverage_pct:.2f}%)")

    return coverage_pct


def get_alleles_for_sample(df, sample_idx, hla_groups):
    """
    Extract alleles for a single sample index from the dataframe, using the specified HLA groups.

    Args:
        df (pd.DataFrame): The full HLA dataset dataframe.
        sample_idx (int): The index of the sample to extract alleles from.
        hla_groups (list of tuples): List of (group_name, col1, col2) tuples specifying allele columns.

    Returns:
        list: List of allele strings for the sample.
    """
    sample = df.loc[sample_idx]  # safer to use loc with actual df index
    alleles = []

    for _, col1, col2 in hla_groups:
        val1 = sample.get(col1)
        val2 = sample.get(col2)

        if pd.notna(val1):
            alleles.append(str(val1).strip())
        if pd.notna(val2):
            alleles.append(str(val2).strip())

    return alleles

def print_validation_report(df, hla_groups, selected_indices):
    """
    Print a summary report of the solution quality.
    """
    coverage = coverage_score(df, hla_groups, selected_indices)
    print(f"Validation Report:")
    print(f" - Number of selected samples: {len(selected_indices)}")
    print(f" - Estimated coverage: {coverage:.2f}% of total alleles")

def validate_and_coerce_allele_columns(df, hla_groups):
    """
    Ensure HLA allele columns are string-typed and non-null.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset.

    hla_groups : list of tuple
        Format: [(group_name, col1, col2), ...]

    Returns
    -------
    pandas.DataFrame
        Modified DataFrame with corrected allele columns.
    """
    for _, col1, col2 in hla_groups:
        for col in (col1, col2):
            if not pd.api.types.is_string_dtype(df[col]):
                print(f"[Validation] Warning: Column '{col}' is not string. Coercing to string.")
                df[col] = df[col].fillna("").astype(str)
    return df


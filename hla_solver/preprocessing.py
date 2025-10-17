"""
preprocessing.py
================

Handles parsing and transformation of HLA dataset CSV files.

This module:
- Loads CSVs containing HLA allele data
- Truncates alleles to a consistent resolution (e.g., 2-field)
- Builds mappings from alleles to internal IDs
- Groups alleles by HLA loci (e.g., HLA-A, HLA-B)
- Optionally duplicates data for augmentation
- Builds per-sample allele sets for greedy solver input
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict
from contextlib import nullcontext

from .utils import verbose_logger
from hla_solver.validation import validate_and_coerce_allele_columns


@verbose_logger(label="Preprocessing Dataset")
def preprocess(csv_path, number_doubles=0, verbose=False, depth=2, timing_logger=None):
    """
    Preprocess the HLA dataset from a CSV file for input to the greedy solver.

    Parameters
    ----------
    csv_path : str or Path
        Path to the input CSV file containing HLA data.
    number_doubles : int, optional
        If > 0, duplicate the dataset this many times to simulate larger inputs.
    verbose : bool, optional
        Enable detailed console output.
    depth : int, optional
        The allele resolution depth to truncate to (e.g., 2 for 2-field alleles).
    timing_logger : TimingLogger, optional
        Logger for timing different preprocessing stages.

    Returns
    -------
    df : pd.DataFrame
        The loaded and cleaned dataset.
    hla_groups : List[Tuple[str, str, str]]
        Groupings of HLA loci with column pairs (e.g., ("HLA-A", "HLA-A1", "HLA-A2")).
    allele_to_id : Dict[str, int]
        Mapping from allele names to internal integer IDs.
    id_to_allele : Dict[int, str]
        Reverse mapping from ID to allele names.
    allele_matrix : np.ndarray (bool)
        Binary matrix of shape (num_samples, num_alleles) indicating allele presence.
    allele_id_to_positions : Dict[int, Set[int]]
        Map from allele ID to row indices in which it appears.
    allele_columns : List[str]
        List of all HLA allele column names in the dataset.
    sample_allele_sets : Dict[str, Dict[str, FrozenSet[int]]]
        Per-sample dictionary of allele IDs by HLA group.
    """
    
    def truncate_allele(allele: str, depth: int) -> str:
        """Truncate allele to the specified resolution (e.g., 2-field)."""
        return ":".join(allele.split(":")[:depth])

    timing = timing_logger.timing if timing_logger else lambda label: nullcontext()

    if verbose:
        print(f"Loading dataset from {csv_path}...")

    with timing("Load dataset"):
        df = pd.read_csv(csv_path)

    if verbose:
        print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")

    # Detect allele columns
    with timing("Detect allele columns"):
        allele_columns = [col for col in df.columns if col.startswith("HLA")]

    if verbose:
        print(f"Identified allele columns: {allele_columns}")

    # Truncate alleles to desired resolution (e.g., 2-field)
    with timing("Truncate alleles"):
        for col in allele_columns:
            df[col] = df[col].astype(str).str.strip().apply(
                lambda a: truncate_allele(a, depth) if pd.notna(a) else a
            )

    # Create allele ID mappings
    with timing("Index alleles"):
        unique_alleles = set()
        for col in allele_columns:
            unique_alleles.update(df[col].dropna().unique())

        unique_alleles = sorted(unique_alleles)
        allele_to_id = {allele: idx for idx, allele in enumerate(unique_alleles)}
        id_to_allele = {idx: allele for allele, idx in allele_to_id.items()}

    # Build binary allele matrix
    with timing("Build allele matrix"):
        allele_matrix = np.zeros((len(df), len(unique_alleles)), dtype=bool)
        allele_id_to_positions = {}

        for i, row in df.iterrows():
            for col in allele_columns:
                allele = row[col]
                if pd.notna(allele):
                    allele_id = allele_to_id.get(allele)
                    if allele_id is not None:
                        allele_matrix[i, allele_id] = True
                        allele_id_to_positions.setdefault(allele_id, set()).add(i)

    # Group HLA columns by locus (e.g., HLA-A)
    with timing("Group HLA columns"):
        group_map = defaultdict(list)
        for col in allele_columns:
            match = re.match(r"(HLA-[A-Z]+)", col)
            if match:
                group_name = match.group(1)
                group_map[group_name].append(col)

        hla_groups = []
        for group_name, cols in group_map.items():
            sorted_cols = sorted(cols)
            if len(sorted_cols) >= 2:
                hla_groups.append((group_name, sorted_cols[0], sorted_cols[1]))
            elif verbose:
                print(f"[WARN] Group {group_name} has <2 columns and was skipped.")

    if verbose:
        print(f"HLA groups identified: {hla_groups}")

    # Validate and possibly coerce allele formatting
    with timing("Validate/coerce alleles"):
        df = validate_and_coerce_allele_columns(df, hla_groups)

    # Optionally augment the dataset
    with timing("Data duplication (augmentation)"):
        if number_doubles > 0:
            df = pd.concat([df] * (number_doubles + 1), ignore_index=True)
            allele_matrix = np.vstack([allele_matrix] * (number_doubles + 1))
            if verbose:
                print(f"Doubled dataset {number_doubles} times. New size: {len(df)} samples.")

    # Build per-sample allele sets by group
    with timing("Build sample allele sets"):
        sample_allele_sets = {}

        for i, row in df.iterrows():
            sample_id = row["Sample ID"]
            sample_dict = {}

            for group_name, col1, col2 in hla_groups:
                alleles = set()
                for col in [col1, col2]:
                    allele = row[col]
                    if pd.notna(allele):
                        allele_id = allele_to_id.get(allele)
                        if allele_id is not None:
                            alleles.add(allele_id)

                sample_dict[group_name] = frozenset(sorted(alleles))

            sample_allele_sets[sample_id] = sample_dict

    return (
        df,
        hla_groups,
        allele_to_id,
        id_to_allele,
        allele_matrix,
        allele_id_to_positions,
        allele_columns,
        sample_allele_sets
    )



"""
Utility functions for logging, timing, safe printing, progress tracking,
and restartable state saving in the HLA solver framework.
"""

import time
import functools
import warnings
import datetime
import csv
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def verbose_logger(label=None):
    """
    Decorator that logs function start/end and execution time when `verbose=True` is passed.

    Useful for debugging or benchmarking specific function calls.

    Parameters
    ----------
    label : str, optional
        Custom label for logging. Defaults to the function's name.

    Returns
    -------
    function
        Wrapped function with optional logging.

    Examples
    --------
    >>> @verbose_logger(label="MyFunction")
    ... def example_func(verbose=False):
    ...     pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            verbose = kwargs.get('verbose', False)
            name = label or func.__name__
            if verbose:
                print(f"\n--- START [{name}] ---")
                start_time = time.time()

            result = func(*args, **kwargs)

            if verbose:
                end_time = time.time()
                elapsed = end_time - start_time
                print(f"--- END [{name}] | Time: {elapsed:.2f}s ---")

            return result
        return wrapper
    return decorator


def is_notebook():
    """
    Detect if the current environment is a Jupyter notebook or qtconsole.

    Returns
    -------
    bool
        True if running in a Jupyter notebook or qtconsole, False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except NameError:
        return False


def warn_once(message, category=UserWarning):
    """
    Issue a warning only once, suppressing repeated warnings.

    Parameters
    ----------
    message : str
        Warning message.
    category : Warning, optional
        Warning class to use. Default is UserWarning.
    """
    warnings.warn(message, category, stacklevel=2)


def chunk_indices(total_size, num_chunks):
    """
    Split a range [0, total_size) into `num_chunks` roughly equal parts.

    Parameters
    ----------
    total_size : int
        Total number of items to split.
    num_chunks : int
        Number of chunks to create.

    Returns
    -------
    list of np.ndarray
        List of arrays with indices for each chunk.

    Examples
    --------
    >>> chunk_indices(10, 3)
    [array([0, 1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]
    """
    import numpy as np
    indices = np.arange(total_size)
    chunks = np.array_split(indices, num_chunks)
    return chunks


def safe_print(*args, **kwargs):
    """
    Print safely, suppressing any exceptions that may occur.

    Useful in multiprocessing or unstable output environments.
    """
    try:
        print(*args, **kwargs)
    except Exception:
        pass


def tqdm_progress(iterable, total=None, desc=None, disable=False):
    """
    Wrap an iterable with a tqdm progress bar.

    Parameters
    ----------
    iterable : iterable
        The iterable to wrap.
    total : int, optional
        Total number of iterations (if known).
    desc : str, optional
        Description text to show beside the progress bar.
    disable : bool
        If True, disables the progress bar.

    Returns
    -------
    iterable
        The tqdm-wrapped iterable.
    """
    from tqdm import tqdm
    return tqdm(iterable, total=total, desc=desc, disable=disable)


def is_file_empty(path):
    """
    Check whether a file is empty.

    Parameters
    ----------
    path : str or Path
        Path to the file.

    Returns
    -------
    bool
        True if file is empty or does not exist, False otherwise.
    """
    try:
        with open(path, 'r') as f:
            return f.readline() == ''
    except FileNotFoundError:
        return True


def save_restart_state(
    path,
    sample_id,
    hla_groups=None,
    hla_pairs=None,
    coverage=None,
    cumulative_cover=None,
    include_timestamp=True
):
    """
    Save solver restart information to a CSV file.

    Parameters
    ----------
    path : str or Path
        File path to write the state.
    sample_id : str
        Identifier of the selected sample.
    hla_groups : list of str, optional
        HLA groups such as ['HLA-A', 'HLA-B'].
    hla_pairs : dict, optional
        Dictionary of group -> alleles.
    coverage : float, optional
        Coverage percentage for the selected sample.
    cumulative_cover : float, optional
        Cumulative coverage achieved so far.
    include_timestamp : bool
        Whether to include a timestamp column.

    Notes
    -----
    Appends to an existing file or creates a new one if necessary.
    """
    path = str(path)
    file_needs_header = is_file_empty(path)

    fieldnames = ["Sample ID"]

    if hla_groups:
        fieldnames.append("HLA Groups")
        hla_columns = [group.replace("HLA-", "HLA-") for group in hla_groups]
        fieldnames.extend(hla_columns)

    if coverage is not None:
        fieldnames.append("Coverage (%)")
    if cumulative_cover is not None:
        fieldnames.append("Cumulative Cover (%)")
    if include_timestamp:
        fieldnames.append("Timestamp M-D Time")

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if file_needs_header:
            writer.writeheader()

        row = {"Sample ID": sample_id}

        if hla_groups:
            row["HLA Groups"] = ";".join(hla_groups)
            for group in hla_groups:
                key = group.replace("HLA-", "HLA-")
                alleles = hla_pairs.get(group) if hla_pairs else None
                if alleles:
                    row[key] = ",".join(map(str, alleles))

        if coverage is not None:
            row["Coverage (%)"] = f"{coverage:.2f}"

        if cumulative_cover is not None:
            row["Cumulative Cover (%)"] = f"{cumulative_cover:.2f}"

        if include_timestamp:
            now = datetime.datetime.now()
            row["Timestamp M-D Time"] = now.strftime("%m-%d %H:%M:%S.%f")[:-3]

        writer.writerow(row)


def load_restart_state(path):
    """
    Load saved restart state from a CSV file.

    Parameters
    ----------
    path : str or Path
        File path of the saved restart state.

    Returns
    -------
    selected_sample_ids : list of str
        Sample IDs from the restart log.
    filter_sequence : list of list of str
        Applied filter sequence, if available.
    """
    selected_sample_ids = []
    filter_sequence = []

    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return selected_sample_ids, filter_sequence

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            selected_sample_ids.append(row.get("Sample ID", ""))
            if "Filter" in row and row["Filter"]:
                filter_sequence.append(row["Filter"].split(";"))

    return selected_sample_ids, filter_sequence


class TimingLogger:
    """
    Utility class for logging timing information for code blocks.

    Attributes
    ----------
    verbose : bool
        Whether to print messages to stdout.
    log_file_path : str
        Path to the file where timing logs are saved.

    Methods
    -------
    logger_func(message)
        Log a message to file and/or stdout.
    timing(label, enabled=True)
        Context manager for measuring execution time.
    """

    def __init__(self, verbose=True, log_dir=None):
        """
        Initialize TimingLogger.

        Parameters
        ----------
        verbose : bool
            Whether to print timing messages to stdout.
        log_dir : str or None
            Directory to save log file. If None, uses current directory.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"timing_{timestamp}.log"
        self.log_file_path = filename if log_dir is None else f"{log_dir}/{filename}"
        self.verbose = verbose

        with open(self.log_file_path, "w") as f:
            f.write(f"Timing log started at {datetime.datetime.now().isoformat()}\n\n")

    def logger_func(self, message):
        """
        Log a message to file and optionally print it.

        Parameters
        ----------
        message : str
            Message to log.

        :noindex:
        """
        try:
            with open(self.log_file_path, "a") as f:
                f.write(message + "\n")
        except Exception as e:
            if self.verbose:
                print(f"[TimingLogger Error] Failed to write to log file: {e}")
        if self.verbose:
            print(message)

    @contextmanager
    def timing(self, label, enabled=True):
        """
        Context manager for timing a code block.

        Parameters
        ----------
        label : str
            Description of the code block.
        enabled : bool
            If False, disables timing entirely.

        Yields
        ------
        None
            Allows usage in a with-statement.
        """
        if not enabled:
            yield
            return
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            elapsed = end - start
            message = f"[Timing] {label}: {elapsed:.4f} seconds"
            self.logger_func(message)

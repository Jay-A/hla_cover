# hla_solver/solver.py

import yaml
from typing import Optional, Union
from pathlib import Path
from hla_solver.backend import BackendManager
from hla_solver.preprocessing import preprocess
from hla_solver.utils import verbose_logger, TimingLogger, load_restart_state, PROJECT_ROOT

from pathlib import Path
from typing import Optional

class HLASolver:
    """
    Main interface for running the HLA greedy coverage solver.

    This class handles configuration loading, data preprocessing,
    backend parallelism setup, and execution of the greedy solver.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    """
    def __init__(self, config_path: str):
        self.project_root = PROJECT_ROOT
        
        self.config: dict = self._load_config(config_path)
        self.backend = None

        self.name = self.config.get("meta", {}).get("simulation_name", "default_sim_name")

        # Restart file path and directory
        self.restart_dir = Path(self.project_root) / "restarts"
        self.restart_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        
        # Default restart file path, can still be overridden in .run()
        self.restart_file = self.restart_dir / f"{self.name}_restart.csv"

        # Logging config
        self.verbose: bool = self.config.get("logging", {}).get("verbose", False)
        self.enable_timing: bool = self.config.get("logging", {}).get("enable_timing", False)
        self.progress_mode: str = self.config.get("logging", {}).get("progress_mode", "progress_bar")

        # Process log_dir and make it absolute relative to project_root if needed
        raw_log_dir = self.config.get("logging", {}).get("log_dir", None)
        if raw_log_dir:
            log_path = Path(raw_log_dir)
            if not log_path.is_absolute():
                self.log_dir = Path(self.project_root) / log_path
            else:
                self.log_dir = log_path
        else:
            self.log_dir = None

        # Preprocessing placeholders
        self.df: Optional[pd.DataFrame] = None
        self.hla_groups: Optional[List[Tuple[str, str, str]]] = None
        self.allele_to_id: Optional[Dict[str, int]] = None
        self.id_to_allele: Optional[Dict[int, str]] = None
        self.allele_columns: Optional[List[str]] = None
        self.sample_allele_sets: Optional[Dict[str, Dict[str, FrozenSet[int]]]] = None
        self.total_samples: int = 0
        self.timing_logger: Optional[TimingLogger] = None


    def get_id_to_allele(self):
        """
        Get the mapping from internal allele IDs back to their original string names.

        Returns
        -------
        dict
            Mapping from internal allele IDs (int) to allele names (str).
        """        
        return self.id_to_allele

    def _load_config(self, path):
        """
        Load and parse the YAML configuration file.

        Parameters
        ----------
        path : str
            Path to the YAML config file.

        Returns
        -------
        dict
            Parsed configuration as a dictionary.

        Raises
        ------
        FileNotFoundError
            If the configuration file cannot be found.

        ValueError
            If the YAML file cannot be parsed.
        """        
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {e}")

    @verbose_logger(label="HLASolver")
    def preprocess(self):
        """
        Run preprocessing on the input dataset.

        Loads sample data, encodes alleles, builds allele sets,
        and prepares all inputs needed for the solver.

        Configuration Keys
        ------------------
        - ``data.dataset_path``: Path to the CSV dataset.
        - ``data.number_doubles``: Number of duplicated rows to add for testing (optional).

        Outputs
        -------
        Sets internal attributes:
        - ``self.df``: DataFrame of input samples.
        - ``self.hla_groups``: List of (A, B, C) tuples.
        - ``self.allele_to_id``, ``self.id_to_allele``: Mapping dicts.
        - ``self.allele_columns``, ``self.sample_allele_sets``, etc.

        Raises
        ------
        KeyError
            If required config keys are missing.
        """        
        dataset_path = self.config["data"].get("dataset_path")
        number_doubles = self.config["data"].get("number_doubles", 0)
    
        (
            self.df,
            self.hla_groups,
            self.allele_to_id,
            self.id_to_allele,
            _,
            _,
            self.allele_columns,
            self.sample_allele_sets
        ) = preprocess(
            dataset_path,
            number_doubles=number_doubles,
            verbose=self.verbose,
            timing_logger=self.timing_logger  # 👈 Pass logger here
        )
    
        self.total_samples = len(self.df)
    
        if self.verbose:
            print(f"[Preprocessing] Completed. Total samples: {self.total_samples}")
            print(f"[Preprocessing] HLA Groups: {self.hla_groups}")
            print(f"[Preprocessing] Allele columns: {self.allele_columns}")
            print(f"[Preprocessing] Sample allele sets keys: {list(self.sample_allele_sets.keys())[:3]}")

    @verbose_logger(label="HLASolver")
    def run(self, *, restart_file: Optional[Union[str, Path]] = None):
        """
        Run or resume the HLA greedy coverage solver.

        If a restart file is provided, the solver resumes from that checkpoint.
        Otherwise, it starts a fresh run and creates a new restart file.

        Parameters
        ----------
        restart_file : Optional[str or Path], default=None
            Path to a restart CSV file. If provided, the solver resumes from it.
            If None, a new run is started from scratch.

        Returns
        -------
        List
            A list of the final best global filters produced by the solver.

        Raises
        ------
        FileNotFoundError
            If a provided restart file does not exist.
        """
        # --- Logging & Timing setup ---
        logging_config = self.config.get('logging', {})
        log_dir = logging_config.get('log_dir', None)
        verbose = logging_config.get('verbose', False)
        enable_timing = logging_config.get('enable_timing', False)
        progress_mode = logging_config.get('progress_mode', 'progress_bar')
    
        self.verbose = verbose
        self.enable_timing = enable_timing
        self.progress_mode = progress_mode
    
        self.timing_logger = TimingLogger(verbose=verbose, log_dir=log_dir) if enable_timing else None
        timing = self.timing_logger.timing if self.timing_logger else (lambda label, enabled=True: (yield))
    
        with timing("HLASolver::run()", enabled=enable_timing):
    
            # --- Preprocessing if needed ---
            if self.df is None or self.sample_allele_sets is None:
                with timing("Preprocessing", enabled=enable_timing):
                    self.preprocess()
    
            # --- Configuration ---
            parallel_config = self.config.get('parallel', {})
            solver_config = self.config.get('solver', {})
    
            parallel_mode = parallel_config.get('mode', 'serial')
            num_workers = parallel_config.get('num_workers', 1)
            max_iter = solver_config.get('max_iter', 10)
            top_n = solver_config.get('top_n', 1)
            top_k_global = solver_config.get('top_k_global', 5)
    
            # --- Restart logic ---
            if restart_file is not None:
                restart_file = Path(restart_file)
                if not restart_file.exists():
                    raise FileNotFoundError(f"[Restart] File not found: {restart_file}")
                self.restart_file = restart_file  # Resume from this file
            else:
                self.restart_file = None  # Let save_restart_state handle path generation
    
            # --- Backend setup ---
            with timing("BackendManager setup", enabled=enable_timing):
                self.backend = BackendManager(
                    mode=parallel_mode,
                    num_workers=num_workers,
                    verbose=verbose,
                    progress_mode=progress_mode,
                    restart_file=restart_file
                )
    
            # --- Solver Execution ---
            with timing("Greedy solver execution", enabled=enable_timing):
                best_filters = self.backend.run_local_global_greedy_cover(
                    df=self.df,
                    hla_groups=self.hla_groups,
                    allele_sets_by_sample=self.sample_allele_sets,
                    allele_to_id=self.allele_to_id,
                    id_to_allele=self.id_to_allele,
                    max_iter=max_iter,
                    top_n=top_n,
                    top_k_global=top_k_global,
                    verbose=verbose,
                    enable_timing=enable_timing,
                    timing=timing,
                )
    
            if self.timing_logger:
                self._log_run_summary(parallel_mode, num_workers, solver_config, parallel_config)
    
            if verbose:
                print("\n[HLASolver] Final best global filters:")
                for f in best_filters:
                    print(f)
    
            return best_filters


    def _log_run_summary(self, parallel_mode, num_workers, solver_config, parallel_config):
        """
        Logs configuration and timing summary after solver completes.

        Parameters
        ----------
        parallel_mode : str
            Parallelization mode used.
        num_workers : int
            Number of workers used for parallel execution.
        solver_config : dict
            Dictionary containing solver configuration parameters.
        parallel_config : dict
            Dictionary containing parallel execution configuration.
        """
        log = self.timing_logger.logger_func
        log("\n=== Solver Configuration ===")
        log(f"Parallel mode: {parallel_mode}")
        log(f"Num workers: {num_workers}")
        log(f"Total samples: {self.total_samples}")
        log(f"\n[solver config]: {solver_config}")
        log(f"[parallel config]: {parallel_config}")
        log(f"[data config]: {self.config.get('data', {})}")
        log("\n=== Results ===")

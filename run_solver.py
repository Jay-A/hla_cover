import sys
import os
import argparse

# -------------------------------
# Add project root to sys.path
# -------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hla_solver import HLASolver

# -------------------------------
# Argument parser setup
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HLASolver with a specified config file."
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        type=str,
        help="Path to the config YAML file."
    )
    parser.add_argument(
        "-r", "--restart",
        type=str,
        default=None,
        help="Optional path to a restart file."
    )
    return parser.parse_args()

# -------------------------------
# Main function
# -------------------------------
def main():
    args = parse_args()
    config_path = os.path.abspath(args.config)

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    solver = HLASolver(config_path)

    if args.restart:
        best_filters = solver.run(restart_file=args.restart)
    else:
        best_filters = solver.run()
    
    # Print or save results
    print("Best filters:", best_filters)

# -------------------------------
# Script entry point
# -------------------------------
if __name__ == "__main__":
    main()

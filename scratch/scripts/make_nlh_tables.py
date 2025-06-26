import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple

def parse_nlh_dat(file_path: Path) -> Dict[Tuple[int, int], Tuple[float, ...]]:
    """Parses the NLH data file and returns a dictionary of coefficients.

    Args:
        file_path (Path): The path to the NLH data file.

    Returns:
        Dict[Tuple[int, int], Tuple[float, ...]]: A dictionary mapping (Z1, Z2) to a tuple of six coefficients.
    """
    coeffs = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip().startswith('#') or not line.strip():
                continue
            parts = line.split()
            z1 = int(parts[0])
            z2 = int(parts[1])
            p = [float(x) for x in parts[2:8]]
            # The file format is a1, b1, a2, b2, a3, b3. We need (a1,a2,a3), (b1,b2,b3).
            six_coeffs = (p[0], p[2], p[4], p[1], p[3], p[5])
            coeffs[(z1, z2)] = six_coeffs
            coeffs[(z2, z1)] = six_coeffs
    return coeffs

def main():
    """Main function to download, parse, and pickle the NLH coefficients."""
    parser = argparse.ArgumentParser(description="Create NLH coefficient tables for Allegro.")
    parser.add_argument("--dat-file", type=str, default="nlh_coeffs.dat",
                        help="Path to the NLH potential parameters .dat file.")
    parser.add_argument("--pickle-file", type=str, default="../../forge/workflows/allegro_utils/nlh_coeffs.pkl",
                        help="Path to save the output pickle file.")
    args = parser.parse_args()

    dat_file_path = Path(args.dat_file)
    pickle_file_path = Path(args.pickle_file)

    if not dat_file_path.exists():
        raise FileNotFoundError(
            f"NLH data file not found at {dat_file_path}. "
            "Please download it from https://zenodo.org/records/15097303/files/NLH-potential-parameters.dat"
        )

    coeff_dict = parse_nlh_dat(dat_file_path)

    with open(pickle_file_path, 'wb') as f:
        pickle.dump(coeff_dict, f)

    print(f"Successfully created NLH coefficient pickle file at {pickle_file_path}")
    
if __name__ == "__main__":
    main() 
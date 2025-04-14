import os
import re
import numpy as np
from ase.io.vasp import read_vasp_out
from typing import Dict, Optional, Tuple, Any, List

class VaspParser:
    """
    Parses VASP OUTCAR file for energy, forces, stress, and metadata.

    Focuses solely on OUTCAR parsing. Initializes upon creation with
    results from a specific VASP calculation directory.
    """
    def __init__(self, output_dir: str, calculation_type: str = "static"):
        """
        Initializes the parser and attempts to parse the OUTCAR file
        in the given directory.

        Args:
            output_dir: Path to the VASP calculation directory.
            calculation_type: String identifier for the calculation type (e.g., 'static', 'relax').
        """
        self.output_dir = output_dir
        self.outcar_path = os.path.join(output_dir, 'OUTCAR')
        self.calculation_type = calculation_type

        # Initialize attributes
        self._atoms = None
        self._energy: Optional[float] = None
        self._forces: Optional[np.ndarray] = None
        self._stress: Optional[np.ndarray] = None
        self._metadata: Dict[str, Any] = self._initialize_metadata()
        self._success: bool = False
        self._error_message: Optional[str] = None

        # Attempt parsing upon initialization
        self._parse()

    def _initialize_metadata(self) -> Dict[str, Any]:
        """Returns the default structure for metadata."""
        return {
            "vasp_version": None,
            "date_run": None,
            "resources_used": {
                "mpi_ranks": None,
                "threads_per_rank": None,
                "nodes": None,
                "gpus_detected": False
            },
            "potcar": None,
            "kpoints": None,
            "nbands": None,
            "parser": "ase_outcar", # Indicate parser used
            "calculation_type": self.calculation_type
        }

    def _parse_metadata_from_lines(self, lines: List[str]) -> None:
        """Helper to parse metadata fields from OUTCAR lines."""
        header_limit = min(100, len(lines)) # Still useful for header info

        # --- Parse Header Info ---
        if lines:
            match = re.match(r'^\s*(vasp\.\S+\s+\S+)', lines[0])
            if match:
                self._metadata["vasp_version"] = match.group(1).strip()

        for i in range(1, header_limit):
            line = lines[i]
            if "executed on" in line and "date" in line:
                match = re.search(r'date\s+(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2})', line)
                if match:
                    self._metadata["date_run"] = match.group(1)
            elif "running" in line and "mpi-ranks" in line:
                match = re.search(r'running\s+(\d+)\s+mpi-ranks,\s+with\s+(\d+)\s+threads/rank,\s+on\s+(\d+)\s+nodes', line)
                if match:
                    self._metadata["resources_used"]["mpi_ranks"] = int(match.group(1))
                    self._metadata["resources_used"]["threads_per_rank"] = int(match.group(2))
                    self._metadata["resources_used"]["nodes"] = int(match.group(3))
            elif "OpenACC runtime initialized" in line and "GPUs detected" in line:
                self._metadata["resources_used"]["gpus_detected"] = True
            elif self._metadata["potcar"] is None and line.strip().startswith("POTCAR:"):
                potcar_text = line.split("POTCAR:", 1)[-1].strip()
                if not potcar_text.startswith("SHA256") and not potcar_text.startswith("COPYR"):
                    self._metadata["potcar"] = potcar_text

        # --- Parse Body Info (Search whole file) ---
        found_kgen_line = False
        for line in lines:
            # Kpoints (handle line after trigger)
            if found_kgen_line:
                match = re.search(r'generate k-points for:\s+(\d+)\s+(\d+)\s+(\d+)', line)
                if match:
                    self._metadata["kpoints"] = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                    found_kgen_line = False # Reset after finding

            # Kpoint trigger line
            if "Grid dimensions read from file:" in line:
                 found_kgen_line = True

            # NBANDS (search anywhere)
            if "NBANDS=" in line:
                match = re.search(r'NBANDS=\s*(\d+)', line)
                if match:
                    self._metadata["nbands"] = int(match.group(1))

    def _parse(self) -> None:
        """Performs the actual parsing of the OUTCAR file."""
        if not os.path.exists(self.outcar_path):
            self._success = False
            self._error_message = f"OUTCAR not found at {self.outcar_path}"
            print(f"[WARN] {self._error_message}")
            return

        try:
            # Parse core results with ASE
            self._atoms = read_vasp_out(self.outcar_path)
            self._energy = self._atoms.get_potential_energy()
            self._forces = self._atoms.get_forces()
            # ASE stress is 6-component Voigt by default, get full 3x3
            self._stress = self._atoms.get_stress(voigt=False)
            self._success = True # Core parsing succeeded

        except Exception as e:
            self._success = False
            self._error_message = f"Failed to parse core results from OUTCAR with ASE: {e}"
            print(f"[WARN] {self._error_message}")
            # Attempt to parse metadata even if core parsing fails
            pass

        # Always attempt metadata parsing if OUTCAR exists
        try:
            with open(self.outcar_path, 'r') as f:
                lines = f.readlines()
            self._parse_metadata_from_lines(lines)
        except Exception as e:
            # Don't overwrite core parsing error message if that failed
            if self._success:
                self._error_message = f"Failed during metadata parsing: {e}"
            else:
                 self._error_message += f" | Also failed during metadata parsing: {e}"
            # Metadata failure doesn't necessarily mean overall failure if core results ok
            print(f"[WARN] Failed during metadata parsing: {e}")


    @property
    def is_successful(self) -> bool:
        """Returns True if core results (energy, forces, stress) were parsed successfully."""
        return self._success

    @property
    def error_message(self) -> Optional[str]:
        """Returns the error message if parsing failed."""
        return self._error_message

    @property
    def energy(self) -> Optional[float]:
        """Returns the parsed potential energy."""
        return self._energy

    @property
    def forces(self) -> Optional[np.ndarray]:
        """Returns the parsed forces."""
        return self._forces

    @property
    def stress(self) -> Optional[np.ndarray]:
        """Returns the parsed stress tensor (3x3)."""
        return self._stress

    @property
    def metadata(self) -> Dict[str, Any]:
        """Returns the parsed metadata dictionary."""
        return self._metadata

    @property
    def atoms(self):
        """Returns the ASE Atoms object representing the final structure."""
        return self._atoms

    def get_calculation_data(self) -> Optional[Dict[str, Any]]:
        """
        Returns a dictionary formatted for DatabaseManager.add_calculation,
        including parsed results and metadata. Returns None if core parsing failed.
        """
        if not self.is_successful:
            return None

        # Prepare data, converting numpy arrays to lists for JSON compatibility
        forces_list = self._forces.tolist() if self._forces is not None else None
        stress_list = self._stress.tolist() if self._stress is not None else None

        # Combine results and metadata for the database
        # The metadata field in the DB will store our self._metadata dictionary
        calc_data = {
            "model_type": "vasp",
            "energy": self._energy,
            "forces": forces_list,
            "stress": stress_list,
            "metadata": self.metadata, # Embed the parsed metadata here
            "status": "completed" if self.is_successful else "parse_error"
        }
        return calc_data 
# TODO need to make this code add profiles from forge/workflows/hpc_profiles to create a slurm script for the hpc cluster 
# TODO need to also add vasp profiles from forge/workflows/vasp_settings to create a specific job

# profiles.py

# Import necessary modules
import json
from pathlib import Path

# Define a class to manage profiles
class ProfileManager:
    def __init__(self, profile_directory: str):
        # Initialize with the directory where profiles are stored
        self.profile_directory = Path(profile_directory)
        self.profiles = {}  # Dictionary to hold loaded profiles

    def load_profile(self, profile_name: str):
        # Load a profile from a JSON file
        profile_path = self.profile_directory / f"{profile_name}.json"
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                self.profiles[profile_name] = json.load(f)
        else:
            raise FileNotFoundError(f"Profile {profile_name} not found.")

    def add_vasp_profile(self, profile_name: str, settings: dict):
        # Add or update a VASP profile
        self.profiles[profile_name] = {
            "type": "vasp",
            "settings": settings
        }
        self.save_profile(profile_name)

    def add_slurm_profile(self, profile_name: str, directives: dict):
        # Add or update a SLURM profile
        self.profiles[profile_name] = {
            "type": "slurm",
            "slurm_directives": directives
        }
        self.save_profile(profile_name)

    def save_profile(self, profile_name: str):
        # Save the current profile to a JSON file
        profile_path = self.profile_directory / f"{profile_name}.json"
        with open(profile_path, 'w') as f:
            json.dump(self.profiles[profile_name], f, indent=4)

    def get_profile(self, profile_name: str):
        # Retrieve a profile by name
        return self.profiles.get(profile_name, None)

    def list_profiles(self):
        # List all available profiles
        return list(self.profiles.keys())

# Example usage
if __name__ == "__main__":
    profile_manager = ProfileManager("path/to/profiles")

    # Load existing profiles
    profile_manager.load_profile("Perlmutter-CPU")

    # Add a new VASP profile
    vasp_settings = {
        "incar": {
            "ENCUT": 520,
            "EDIFF": 1E-6,
            "IBRION": 2,
            "NSW": 50
        },
        "potcars": {
            "V": "V_pv",
            "Cr": "Cr_sv",
            "Ti": "Ti_sv",
            "W": "W_sv",
            "Zr": "Zr_sv"
        }
    }
    profile_manager.add_vasp_profile("New_VASP_Profile", vasp_settings)

    # Add a new SLURM profile
    slurm_directives = {
        "job-name": "vasp_job",
        "nodes": 1,
        "time": "01:00:00",
        "partition": "standard"
    }
    profile_manager.add_slurm_profile("New_SLURM_Profile", slurm_directives)

    # List all profiles
    print(profile_manager.list_profiles())
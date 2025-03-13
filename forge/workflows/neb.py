"""
Module for running Nudged Elastic Band (NEB) calculations, particularly focused on
vacancy diffusion in metallic systems.
"""
from typing import Optional, List, Dict, Union, Tuple, NamedTuple
from enum import Enum
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.mep import NEB, DyNEB
from ase.optimize import FIRE
import matplotlib.pyplot as plt
from dataclasses import dataclass
from ase.neighborlist import NeighborList
import json
from collections import defaultdict
import torch
from mace.calculators.mace import MACECalculator
from monty.json import MontyEncoder, MontyDecoder

class NEBMethod(Enum):
    """NEB calculation method."""
    REGULAR = "neb"
    DYNAMIC = "dyneb"

@dataclass
class NEBResult:
    """Container for NEB calculation results."""
    barrier: float
    energies: List[float]
    converged: bool
    n_steps: int

class NEBCalculation:
    """Handles running of individual NEB calculations."""
    
    def __init__(
        self,
        start_atoms: Atoms,
        end_atoms: Atoms,
        model_path: Union[str, List[str]],
        start_energy: float,
        end_energy: float,
        n_images: int = 7,
        method: NEBMethod = NEBMethod.DYNAMIC,
        climbing: bool = True,
        fmax: float = 0.01,
        steps: int = 200,
        seed: int = 42,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_cueq: bool = False,
    ):
        """
        Initialize NEB calculation.
        
        Args:
            start_atoms: Initial structure 
            end_atoms: Final structure
            model_path: Path(s) to MACE model(s)
            start_energy: Energy of start configuration
            end_energy: Energy of end configuration
            n_images: Number of images for NEB
            method: NEB method to use
            climbing: Enable climbing image
            fmax: Maximum force for convergence
            steps: Maximum optimization steps
            seed: Random seed for reproducibility
            device: Device to run calculations on
            use_cueq: Whether to use CUEQ
        """
        self.start_atoms = start_atoms
        self.end_atoms = end_atoms
        self.model_path = model_path if isinstance(model_path, list) else [model_path]
        self.start_energy = start_energy
        self.end_energy = end_energy
        self.n_images = n_images
        self.method = method
        self.climbing = climbing
        self.fmax = fmax
        self.steps = steps
        self.seed = seed
        self.device = device
        self.use_cueq = use_cueq
        np.random.seed(self.seed)  # Set seed for reproducibility

    def _create_calculator(self) -> MACECalculator:
        """Create a new MACE calculator instance."""
        return MACECalculator(
            model_paths=self.model_path,
            device=self.device,
            default_dtype="float32",
            use_cueq=self.use_cueq
        )

    def run(self) -> NEBResult:
        """
        Run the NEB calculation.
        
        Returns:
            NEBResult containing calculation results
        """
        # Create images (excluding endpoints which we already have energies for)
        images = [self.start_atoms]
        images += [self.start_atoms.copy() for _ in range(self.n_images)]
        images.append(self.end_atoms)

        # Set up NEB
        if self.method == NEBMethod.DYNAMIC:
            neb = DyNEB(images, climb=self.climbing)
        else:
            neb = NEB(images, climb=self.climbing)

        # Interpolate positions
        neb.interpolate()
        
        # Attach calculators only to intermediate images
        for image in images[1:-1]:
            image.calc = self._create_calculator()

        # Run optimization
        opt = FIRE(neb)
        opt.run(fmax=self.fmax, steps=self.steps)

        # Get energies for intermediate images
        intermediate_energies = [image.get_potential_energy() for image in images[1:-1]]
        
        # Combine with endpoint energies
        energies = [self.start_energy] + intermediate_energies + [self.end_energy]
        barrier = max(energies) - energies[0]
        
        return NEBResult(
            barrier=barrier,
            energies=energies,
            converged=opt.converged(),
            n_steps=opt.get_number_of_steps()
        )

class NEBAnalyzer:
    """Analyze and visualize results from multiple NEB calculations."""
    
    def __init__(self):
        self.calculations = []
        
    def add_calculation(self, calc_result: Dict):
        """Add results from a NEB calculation."""
        if calc_result.get("success", False):
            self.calculations.append(calc_result)
    
    def _group_barriers(self) -> Tuple[Dict, Dict]:
        """
        Group barriers by vacancy element and neighbor type.
        
        Returns:
            Tuple of (nn_barriers, nnn_barriers) dictionaries mapping
            'vacancy_element-target_element' to list of barriers
        """
        nn_barriers = defaultdict(list)
        nnn_barriers = defaultdict(list)
        
        for calc in self.calculations:
            if not calc.get("barrier"):
                continue
                
            key = f"{calc['vacancy_element']}-{calc['target_element']}"
            # TODO: Add logic to determine if NN or NNN based on distance
            # For now, assuming this is stored in the calculation results
            if calc.get("is_nearest_neighbor", True):
                nn_barriers[key].append(calc["barrier"])
            else:
                nnn_barriers[key].append(calc["barrier"])
                
        return dict(nn_barriers), dict(nnn_barriers)
    
    def calculate_statistics(self) -> Dict:
        """
        Calculate statistics for each type of transition.
        
        Returns:
            Dictionary containing mean, std, min, max for each transition type
            and neighbor category
        """
        nn_barriers, nnn_barriers = self._group_barriers()
        stats = {
            "nearest_neighbor": {},
            "next_nearest_neighbor": {},
            "overall": {}
        }
        
        # Calculate stats for each transition type
        for key, barriers in nn_barriers.items():
            stats["nearest_neighbor"][key] = {
                "mean": np.mean(barriers),
                "std": np.std(barriers),
                "min": np.min(barriers),
                "max": np.max(barriers),
                "count": len(barriers)
            }
            
        for key, barriers in nnn_barriers.items():
            stats["next_nearest_neighbor"][key] = {
                "mean": np.mean(barriers),
                "std": np.std(barriers),
                "min": np.min(barriers),
                "max": np.max(barriers),
                "count": len(barriers)
            }
        
        # Calculate overall statistics, handling empty arrays
        all_nn = np.concatenate([np.array(b) for b in nn_barriers.values()]) if nn_barriers else np.array([])
        all_nnn = np.concatenate([np.array(b) for b in nnn_barriers.values()]) if nnn_barriers else np.array([])
        
        stats["overall"] = {
            "nn_mean": float(np.mean(all_nn)) if len(all_nn) > 0 else None,
            "nn_std": float(np.std(all_nn)) if len(all_nn) > 0 else None,
            "nnn_mean": float(np.mean(all_nnn)) if len(all_nnn) > 0 else None,
            "nnn_std": float(np.std(all_nnn)) if len(all_nnn) > 0 else None,
            "nn_count": len(all_nn),
            "nnn_count": len(all_nnn)
        }
        
        return stats
    
    def filter_calculations(
        self,
        vacancy_element: Optional[str] = None,
        target_element: Optional[str] = None,
        neighbor_type: Optional[str] = None,  # 'nn' or 'nnn'
        min_barrier: float = 0.0,  # Skip negative or very small barriers
    ) -> List[Dict]:
        """
        Filter calculations based on specified criteria.
        
        Args:
            vacancy_element: Filter by vacancy element
            target_element: Filter by target element
            neighbor_type: Filter by neighbor type ('nn' or 'nnn')
            min_barrier: Minimum barrier height to include
            
        Returns:
            List of filtered calculation results
        """
        filtered = []
        for calc in self.calculations:
            if not calc.get("success") or calc.get("barrier", 0.0) < min_barrier:
                continue
                
            if vacancy_element and calc["vacancy_element"] != vacancy_element:
                continue
                
            if target_element and calc["target_element"] != target_element:
                continue
                
            if neighbor_type:
                is_nn = calc.get("is_nearest_neighbor", True)
                if neighbor_type == "nn" and not is_nn:
                    continue
                if neighbor_type == "nnn" and is_nn:
                    continue
                    
            filtered.append(calc)
            
        return filtered

    def plot_barrier_distributions(
        self,
        save_path: Optional[Path] = None,
        bins: int = 30,
        figsize: Tuple[int, int] = (12, 8),
        colors: Optional[Dict[str, str]] = None,  # Map transition to color
        labels: Optional[Dict[str, str]] = None,  # Map transition to custom label
        title: Optional[str] = None,
        show_stats: bool = True,  # Add mean/std to legend
        alpha: float = 0.5,
        min_barrier: float = 0.0
    ):
        """
        Plot histograms of migration barriers for NN and NNN transitions.
        
        Args:
            save_path: Path to save the plot
            bins: Number of histogram bins
            figsize: Figure size (width, height)
            colors: Dictionary mapping transition keys to colors
            labels: Dictionary mapping transition keys to custom labels
            title: Overall plot title
            show_stats: Include mean/std in legend
            alpha: Transparency of histogram bars
            min_barrier: Minimum barrier height to include
        """
        nn_barriers, nnn_barriers = self._group_barriers()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        def plot_barriers(ax, barriers, title):
            for key, values in barriers.items():
                # Filter out small/negative barriers
                values = [v for v in values if v >= min_barrier]
                if not values:
                    continue
                
                # Get plot settings
                color = colors.get(key) if colors else None
                label = labels.get(key, key) if labels else key
                if show_stats:
                    mean = np.mean(values)
                    std = np.std(values)
                    label = f"{label} (μ={mean:.2f}, σ={std:.2f})"
                
                ax.hist(values, bins=bins, alpha=alpha, label=label, color=color)
                
            ax.set_title(title)
            ax.set_xlabel("Barrier Height (eV)")
            ax.set_ylabel("Count")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plot_barriers(ax1, nn_barriers, "Nearest Neighbor Transitions")
        plot_barriers(ax2, nnn_barriers, "Next-Nearest Neighbor Transitions")
        
        if title:
            fig.suptitle(title)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()
            
    def save_results(self, filepath: Path, composition: Optional[Dict[str, float]] = None):
        """
        Save calculation results and statistics to JSON using Monty serialization.
        
        Args:
            filepath: Path to save the JSON file
            composition: Optional dictionary of composition fractions
        """
        stats = self.calculate_statistics()
        
        output = {
            "composition": composition,
            "statistics": stats,
            "calculations": self.calculations
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, cls=MontyEncoder)
    
    @classmethod
    def load_results(cls, filepath: Path) -> 'NEBAnalyzer':
        """
        Load results from a JSON file using Monty deserialization.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            NEBAnalyzer instance with loaded results
        """
        analyzer = cls()
        
        with open(filepath, 'r') as f:
            data = json.load(f, cls=MontyDecoder)
            
        for calc in data["calculations"]:
            analyzer.add_calculation(calc)
            
        return analyzer

@dataclass
class NeighborInfo:
    """Container for neighbor information."""
    nn_indices: np.ndarray
    nn_distances: np.ndarray
    nnn_indices: np.ndarray
    nnn_distances: np.ndarray
    center_element: str
    neighbor_elements: Dict[str, List[int]]  # Maps element to indices for both NN and NNN

class VacancyDiffusion:
    """High-level workflow for running multiple NEB calculations."""
    
    def __init__(
        self,
        atoms: Atoms,
        model_path: List[str],
        nn_cutoff: float = 2.8,
        nnn_cutoff: float = 3.2,
        seed: int = 42,
    ):
        """
        Initialize vacancy diffusion workflow.
        
        Args:
            atoms: Relaxed perfect structure to study
            model_path: Path(s) to MACE model(s)
            nn_cutoff: Cutoff radius for nearest neighbors
            nnn_cutoff: Cutoff radius for next-nearest neighbors
            seed: Random seed for reproducibility
        """
        self.atoms = atoms.copy()
        self.model_path = model_path
        self.nn_cutoff = nn_cutoff
        self.nnn_cutoff = nnn_cutoff
        self.seed = seed
        self.analyzer = NEBAnalyzer()
        
        # Set random seed
        np.random.seed(self.seed)
        
        # Cache for neighbor calculations
        self._neighbor_cache: Dict[int, NeighborInfo] = {}

    def get_neighbors(self, index: int) -> NeighborInfo:
        """
        Get nearest and next-nearest neighbors for an atom.
        
        Args:
            index: Index of the atom to find neighbors for
            
        Returns:
            NeighborInfo containing neighbor indices, distances, and element information
        """
        if index in self._neighbor_cache:
            return self._neighbor_cache[index]
            
        # Create neighbor list with larger cutoff
        cutoff = self.nnn_cutoff
        nl = NeighborList([cutoff/2] * len(self.atoms), 
                         skin=0.0, 
                         self_interaction=False, 
                         bothways=True)
        nl.update(self.atoms)
        
        # Get all neighbors and distances
        indices, offsets = nl.get_neighbors(index)
        positions = self.atoms.positions
        cell = self.atoms.get_cell()
        distances = []
        
        for i, offset in zip(indices, offsets):
            pos_i = positions[i] + np.dot(offset, cell)
            dist = np.linalg.norm(pos_i - positions[index])
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Separate into NN and NNN
        nn_mask = distances <= self.nn_cutoff
        nnn_mask = (distances > self.nn_cutoff) & (distances <= self.nnn_cutoff)
        
        # Sort both sets by distance
        nn_indices = indices[nn_mask]
        nn_distances = distances[nn_mask]
        nn_sort = np.argsort(nn_distances)
        
        nnn_indices = indices[nnn_mask]
        nnn_distances = distances[nnn_mask]
        nnn_sort = np.argsort(nnn_distances)
        
        # Group by elements
        center_element = self.atoms[index].symbol
        neighbor_elements = {}
        for elem in set(self.atoms.get_chemical_symbols()):
            elem_indices = []
            for idx in np.concatenate([nn_indices[nn_sort], nnn_indices[nnn_sort]]):
                if self.atoms[idx].symbol == elem:
                    elem_indices.append(idx)
            if elem_indices:
                neighbor_elements[elem] = elem_indices
        
        info = NeighborInfo(
            nn_indices=nn_indices[nn_sort],
            nn_distances=nn_distances[nn_sort],
            nnn_indices=nnn_indices[nnn_sort],
            nnn_distances=nnn_distances[nnn_sort],
            center_element=center_element,
            neighbor_elements=neighbor_elements
        )
        
        self._neighbor_cache[index] = info
        return info

    def create_endpoints(
        self,
        vacancy_index: int,
        target_index: int,
        relax_fmax: float = 0.01,
        relax_steps: int = 100
    ) -> Tuple[Atoms, Atoms, Dict[str, str]]:
        """
        Create and relax start and end configurations.
        
        Args:
            vacancy_index: Index of atom to remove (create vacancy)
            target_index: Index of atom to move to vacancy site
            relax_fmax: Force tolerance for endpoint relaxation
            relax_steps: Maximum steps for endpoint relaxation
            
        Returns:
            Tuple of (start_atoms, end_atoms, metadata)
        """
        # Get element information before modifications
        vacancy_element = self.atoms[vacancy_index].symbol
        target_element = self.atoms[target_index].symbol
        
        # Create start and end configurations preserving atom IDs
        start_atoms = self.atoms.copy()
        end_atoms = self.atoms.copy()
        
        # Get vacancy position
        vacancy_position = start_atoms.positions[vacancy_index].copy()

        # Move target atom to vacancy position in end configuration
        end_atoms.positions[target_index] = vacancy_position
        
        # Remove vacancy atom from both configurations
        start_atoms.pop(vacancy_index)
        end_atoms.pop(vacancy_index)
        
        # Relax configurations
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_cueq = device == "cuda"
        
        for atoms in (start_atoms, end_atoms):
            atoms.calc = MACECalculator(model_paths=[self.model_path], device=device, default_dtype="float32", use_cueq=use_cueq)
            opt = FIRE(atoms)
            opt.run(fmax=relax_fmax, steps=relax_steps)
        
        metadata = {
            "vacancy_element": vacancy_element,
            "target_element": target_element,
            "vacancy_index": str(vacancy_index),
            "target_index": str(target_index)
        }
        
        return start_atoms, end_atoms, metadata

    def sample_neighbors(
        self,
        vacancy_indices: List[int],
        n_nearest: int,
        n_next_nearest: int,
        rng_seed: Optional[int] = None
    ) -> List[Dict[str, Union[int, List[int]]]]:
        """
        Sample neighbor pairs for NEB calculations.
        
        Args:
            vacancy_indices: List of vacancy site indices
            n_nearest: Number of nearest neighbors to sample per vacancy
            n_next_nearest: Number of next-nearest neighbors to sample per vacancy
            rng_seed: Random seed for reproducibility
            
        Returns:
            List of dictionaries with keys:
                - 'vacancy_index': Index of the vacancy site
                - 'nn': List of sampled nearest neighbor indices
                - 'nnn': List of sampled next-nearest neighbor indices
        """
        # Use class seed if no specific seed provided
        seed_to_use = rng_seed if rng_seed is not None else self.seed
        rng = np.random.default_rng(seed_to_use)
        results = []
        
        for vac_idx in vacancy_indices:
            neighbors = self.get_neighbors(vac_idx)
            
            # Sample from NN
            if len(neighbors.nn_indices) >= n_nearest:
                nn_samples = rng.choice(neighbors.nn_indices, size=n_nearest, replace=False).tolist()
            else:
                nn_samples = neighbors.nn_indices.tolist()
                
            # Sample from NNN
            if len(neighbors.nnn_indices) >= n_next_nearest:
                nnn_samples = rng.choice(neighbors.nnn_indices, size=n_next_nearest, replace=False).tolist()
            else:
                nnn_samples = neighbors.nnn_indices.tolist()
            
            # Create structured result
            results.append({
                'vacancy_index': vac_idx,
                'nn': nn_samples,
                'nnn': nnn_samples
            })
                
        return results

    def run_single(
        self,
        vacancy_index: int,
        target_index: int,
        num_images: int = 5,
        neb_method: str = "dyneb",
        climb: bool = True,
        relax_fmax: float = 0.01,
        relax_steps: int = 100,
        neb_fmax: float = 0.01,
        neb_steps: int = 200,
        save_xyz: bool = False,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Run single NEB calculation between specified sites.
        
        Args:
            vacancy_index: Index of atom to remove
            target_index: Index of atom to move to vacancy site
            num_images: Number of interpolated images for NEB
            neb_method: Method to use for NEB calculation (dyneb or neb)
            climb: Whether to use climbing image for NEB (default: True)
            relax_fmax: Force tolerance for endpoint relaxation before NEB
            relax_steps: Maximum steps for endpoint relaxation before NEB
            neb_fmax: Force tolerance for NEB calculation
            neb_steps: Maximum steps for NEB calculation
            save_xyz: Save initial and final xyz files
            output_dir: Directory to save xyz files
            
        Returns:
            Dictionary containing calculation results and metadata
        """
        # Initialize metadata outside try block
        metadata = {
            "vacancy_element": self.atoms[vacancy_index].symbol,
            "target_element": self.atoms[target_index].symbol,
            "vacancy_index": str(vacancy_index),
            "target_index": str(target_index)
        }
        
        # Initialize variables for xyz saving
        base_name = None
        neb_calc = None
        
        if save_xyz and output_dir:
            formula = self.atoms.get_chemical_formula()
            vac_elem = metadata['vacancy_element']
            target_elem = metadata['target_element']
            base_name = f"{formula}_{vac_elem}_to_{target_elem}_site{vacancy_index}_to_{target_index}"
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create and relax endpoints
            start_atoms, end_atoms, endpoint_metadata = self.create_endpoints(
                vacancy_index, target_index, 
                relax_fmax=relax_fmax, 
                relax_steps=relax_steps
            )
            
            # Update metadata with any additional info from endpoints
            metadata.update(endpoint_metadata)
            
            # Calculate endpoint energies and remove calculators
            device = "cuda" if torch.cuda.is_available() else "cpu"
            use_cueq = device == "cuda"

            start_atoms.calc = MACECalculator(model_paths=[self.model_path], device=device, default_dtype="float32", use_cueq=use_cueq)
            end_atoms.calc = MACECalculator(model_paths=[self.model_path], device=device, default_dtype="float32", use_cueq=use_cueq)
            start_energy = start_atoms.get_potential_energy()
            end_energy = end_atoms.get_potential_energy()
            start_atoms.calc = None
            end_atoms.calc = None
            
            # Save initial configuration if requested
            if save_xyz and output_dir:
                # Create initial NEB images for visualization
                images = [start_atoms]
                images += [start_atoms.copy() for _ in range(num_images)]
                images.append(end_atoms)
                if neb_method == "dyneb" and climb:
                    neb = DyNEB(images, climb=True)
                elif neb_method == "dyneb" and not climb:
                    neb = DyNEB(images)
                elif neb_method == "neb" and climb:
                    neb = NEB(images, climb=True)
                elif neb_method == "neb" and not climb:
                    neb = NEB(images)
                else:
                    raise ValueError(f"Invalid NEB method or climb option: {neb_method} {climb}")
                
                neb.interpolate(mic=True)
                write(output_dir / f"{base_name}_initial.xyz", images)
            
            # Run NEB with seed
            neb_calc = NEBCalculation(
                start_atoms=start_atoms,
                end_atoms=end_atoms,
                model_path=self.model_path,
                start_energy=start_energy,
                end_energy=end_energy,
                n_images=num_images,
                method=NEBMethod.DYNAMIC if neb_method == "dyneb" else NEBMethod.REGULAR,
                climbing=climb,
                seed=self.seed,
                fmax=neb_fmax,
                steps=neb_steps
            )
            result = neb_calc.run()
            
            # Combine results and metadata
            output = {
                **metadata,
                "barrier": result.barrier,
                "energies": result.energies,
                "converged": result.converged,
                "n_steps": result.n_steps,
                "success": True,
                "error": None,
                "is_nearest_neighbor": True  # Add this flag for the analyzer
            }
            
            return output
            
        except Exception as e:
            output = {
                **metadata,
                "success": False,
                "error": str(e),
                "barrier": None,
                "energies": None,
                "converged": False,
                "n_steps": None
            }
            return output
            
        finally:
            # Save final configuration if requested, regardless of success/failure
            if save_xyz and output_dir and neb_calc is not None and hasattr(neb_calc, 'neb'):
                try:
                    write(output_dir / f"{base_name}_final.xyz", neb_calc.neb.images)
                except Exception as e:
                    print(f"Warning: Could not save final xyz: {str(e)}")

    def run_multiple(
        self,
        vacancy_indices: List[int],
        n_nearest: int = 8,
        n_next_nearest: int = 6,
        num_images: int = 5,
        neb_method: str = "dyneb",
        climb: bool = True,
        relax_fmax: float = 0.01,
        relax_steps: int = 100,
        neb_fmax: float = 0.01,
        neb_steps: int = 200,
        save_xyz: bool = False,
        output_dir: Optional[Path] = None,
        rng_seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Run multiple NEB calculations.
        
        Args:
            vacancy_indices: List of vacancy sites to test
            n_nearest: Number of nearest neighbors to sample per vacancy
            n_next_nearest: Number of next-nearest neighbors to sample per vacancy
            num_images: Number of interpolated images for NEB
            neb_method: Method to use for NEB calculation (dyneb or neb)
            climb: Whether to use climbing image for NEB (default: True)
            relax_fmax: Force tolerance for endpoint relaxation before NEB
            relax_steps: Maximum steps for endpoint relaxation before NEB
            neb_fmax: Force tolerance for NEB calculation
            neb_steps: Maximum steps for NEB calculation
            save_xyz: Save xyz files for each calculation
            output_dir: Directory to save xyz files
            rng_seed: Random seed for neighbor sampling
            
        Returns:
            List of calculation results
        """
        # Use class seed if no specific seed provided
        seed_to_use = rng_seed if rng_seed is not None else self.seed
        
        # Generate vacancy-target pairs with structured format
        neighbor_samples = self.sample_neighbors(
            vacancy_indices=vacancy_indices,
            n_nearest=n_nearest,
            n_next_nearest=n_next_nearest,
            rng_seed=seed_to_use
        )
        
        # Count total calculations
        total_calcs = sum(len(sample['nn']) + len(sample['nnn']) for sample in neighbor_samples)
        results = []
        progress_step = max(1, total_calcs // 10)  # Report every 10%
        
        print(f"Starting {total_calcs} NEB calculations...")
        calc_count = 0
        
        # Process each vacancy and its neighbors
        for sample in neighbor_samples:
            vac_idx = sample['vacancy_index']
            
            # Process nearest neighbors
            for target_idx in sample['nn']:
                result = self.run_single(
                    vacancy_index=vac_idx,
                    target_index=target_idx,
                    num_images=num_images,
                    neb_method=neb_method,
                    climb=climb,
                    relax_fmax=relax_fmax,
                    relax_steps=relax_steps,
                    neb_fmax=neb_fmax,
                    neb_steps=neb_steps,
                    save_xyz=save_xyz,
                    output_dir=output_dir
                )
                # Mark as nearest neighbor
                result["is_nearest_neighbor"] = True
                results.append(result)
                
                calc_count += 1
                # Report progress every 10%
                if calc_count % progress_step == 0:
                    print(f"Progress: {calc_count}/{total_calcs} calculations completed")
                
                # Add to analyzer if successful
                if result["success"]:
                    self.analyzer.add_calculation(result)
                else:
                    print(f"Calculation failed for vacancy {vac_idx} to NN {target_idx}: {result['error']}")
            
            # Process next-nearest neighbors
            for target_idx in sample['nnn']:
                result = self.run_single(
                    vacancy_index=vac_idx,
                    target_index=target_idx,
                    num_images=num_images,
                    neb_method=neb_method,
                    climb=climb,
                    relax_fmax=relax_fmax,
                    relax_steps=relax_steps,
                    neb_fmax=neb_fmax,
                    neb_steps=neb_steps,
                    save_xyz=save_xyz,
                    output_dir=output_dir
                )
                # Mark as next-nearest neighbor
                result["is_nearest_neighbor"] = False
                results.append(result)
                
                calc_count += 1
                # Report progress every 10%
                if calc_count % progress_step == 0:
                    print(f"Progress: {calc_count}/{total_calcs} calculations completed")
                
                # Add to analyzer if successful
                if result["success"]:
                    self.analyzer.add_calculation(result)
                else:
                    print(f"Calculation failed for vacancy {vac_idx} to NNN {target_idx}: {result['error']}")
        
        print(f"Completed {total_calcs} calculations")
        return results

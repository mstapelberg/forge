#!/usr/bin/env python3
"""
Comprehensive workflow combining composition generation, hybrid MD/MC optimization, 
and NEB calculations for vacancy diffusion studies.

This script:
1. Generates new alloy compositions using the CompositionAnalyzer
2. Creates initial structures with random atom positions
3. Optimizes structures using hybrid MD/MC simulation
4. Runs NEB calculations for vacancy diffusion barriers
5. Analyzes and visualizes results

Confidence: 8/10
"""

import numpy as np
import torch
import random
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import warnings

# ASE imports
from ase import Atoms
from ase.io import read, write
from ase.build import bulk

# Forge imports
from forge.analysis.composition import CompositionAnalyzer
from forge.workflows.mcmc import MonteCarloAlloySampler
from forge.workflows.neb import VacancyDiffusion, NEBAnalyzer
from forge.workflows.relax import relax

# Allegro/NequIP imports
try:
    from nequip.ase import NequIPCalculator
    ALLEGRO_AVAILABLE = True
except ImportError:
    print("Warning: NequIP not available. Please install nequip to use Allegro models.")
    ALLEGRO_AVAILABLE = False

# Suppress torch warnings
warnings.filterwarnings("ignore", category=FutureWarning, 
                       message=".*You are using `torch.load` with `weights_only=False`.*")


class AllegroVacancyDiffusion:
    """
    Vacancy diffusion workflow using Allegro calculator.
    This is a simplified version that adapts the original VacancyDiffusion
    to work with Allegro instead of MACE.
    """
    
    def __init__(
        self,
        atoms: Atoms,
        allegro_calculator,
        nn_cutoff: float = 2.8,
        nnn_cutoff: float = 3.2,
        seed: int = 42,
    ):
        """
        Initialize Allegro vacancy diffusion workflow.
        
        Args:
            atoms: Relaxed perfect structure to study
            allegro_calculator: Allegro calculator instance
            nn_cutoff: Cutoff radius for nearest neighbors
            nnn_cutoff: Cutoff radius for next-nearest neighbors
            seed: Random seed for reproducibility
        """
        self.atoms = atoms.copy()
        self.allegro_calculator = allegro_calculator
        self.nn_cutoff = nn_cutoff
        self.nnn_cutoff = nnn_cutoff
        self.seed = seed
        self.analyzer = NEBAnalyzer()
        
        # Set random seed
        np.random.seed(self.seed)
        
        # Cache for neighbor calculations
        self._neighbor_cache: Dict[int, 'NeighborInfo'] = {}

    def get_neighbors(self, index: int) -> 'NeighborInfo':
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
            List of dictionaries with neighbor information
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
        output_dir: Optional[Path] = None,
        verbose: int = 1
    ) -> Dict:
        """
        Run single NEB calculation between specified sites using Allegro calculator.
        
        Args:
            vacancy_index: Index of atom to remove
            target_index: Index of atom to move to vacancy site
            num_images: Number of interpolated images for NEB
            neb_method: Method to use for NEB calculation (dyneb or neb)
            climb: Whether to use climbing image for NEB
            relax_fmax: Force tolerance for endpoint relaxation before NEB
            relax_steps: Maximum steps for endpoint relaxation before NEB
            neb_fmax: Force tolerance for NEB calculation
            neb_steps: Maximum steps for NEB calculation
            save_xyz: Save initial and final xyz files
            output_dir: Directory to save xyz files
            verbose: Verbosity level
            
        Returns:
            Dictionary containing calculation results and metadata
        """
        # Convert indices to integers if they're not already
        try:
            vacancy_index = int(vacancy_index)
            target_index = int(target_index)
        except (TypeError, ValueError):
            if isinstance(target_index, list) and len(target_index) == 1:
                target_index = int(target_index[0])
            else:
                error_msg = f"Invalid indices: vacancy_index={vacancy_index}, target_index={target_index}"
                if verbose > 0:
                    print(f"Error: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "vacancy_index": str(vacancy_index),
                    "target_index": str(target_index)
                }
        
        # Initialize metadata
        metadata = {
            "vacancy_element": self.atoms[vacancy_index].symbol,
            "target_element": self.atoms[target_index].symbol,
            "vacancy_index": str(vacancy_index),
            "target_index": str(target_index)
        }
        
        try:
            # Create start and end configurations
            start_atoms = self.atoms.copy()
            end_atoms = self.atoms.copy()
            
            # Get vacancy position
            vacancy_position = start_atoms.positions[vacancy_index].copy()
            
            # Move target atom to vacancy position in end configuration
            end_atoms.positions[target_index] = vacancy_position
            
            # Remove vacancy atom from both configurations
            start_atoms.pop(vacancy_index)
            end_atoms.pop(vacancy_index)
            
            # Configure logfile based on verbosity
            logfile = None if verbose == 0 else '-'
            
            # Relax configurations using Allegro calculator
            start_calculator = self.allegro_calculator
            end_calculator = self.allegro_calculator

            rel_start_atoms = relax(
                atoms=start_atoms,
                calculator=start_calculator,
                relax_cell=False,
                fmax=relax_fmax,
                steps=relax_steps,
                optimizer="FIRE",
                logfile=logfile,
                verbose=verbose
            )

            rel_end_atoms = relax(
                atoms=end_atoms,
                calculator=end_calculator,
                relax_cell=False,
                fmax=relax_fmax,
                steps=relax_steps,
                optimizer="FIRE",
                logfile=logfile,
                verbose=verbose
            )
            
            start_energy = rel_start_atoms.get_potential_energy()
            end_energy = rel_end_atoms.get_potential_energy()
            
            # Create NEB calculation
            neb_calc = NEBCalculation(
                start_atoms=rel_start_atoms,
                end_atoms=rel_end_atoms,
                model_path=None,  # Not used for Allegro
                start_energy=start_energy,
                end_energy=end_energy,
                n_images=num_images,
                method=NEBMethod(neb_method),
                climbing=climb,
                fmax=neb_fmax,
                steps=neb_steps,
                seed=self.seed,
                device="cpu",  # Allegro handles device internally
                use_cueq=False,  # Not applicable for Allegro
                logfile=logfile
            )
            
            # Override the calculator creation to use Allegro
            def create_allegro_calculator():
                return self.allegro_calculator
            
            neb_calc._create_calculator = create_allegro_calculator
            
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
                "is_nearest_neighbor": True
            }
            
            return output
            
        except Exception as e:
            if verbose > 0:
                print(f"Error in NEB calculation: {e}")
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

    def run_multiple(
        self,
        vacancy_indices: Optional[List[int]] = None,
        num_images: int = 5,
        neb_method: str = "dyneb",
        climb: bool = True,
        relax_fmax: float = 0.01,
        relax_steps: int = 100,
        neb_fmax: float = 0.01,
        neb_steps: int = 200,
        save_xyz: bool = False,
        output_dir: Optional[Path] = None,
        n_nearest: int = 3,
        n_next_nearest: int = 3,
        rng_seed: Optional[int] = None,
        verbose: int = 1
    ) -> List[Dict]:
        """
        Run multiple NEB calculations for vacancy diffusion using Allegro.
        
        Args:
            vacancy_indices: List of vacancy sites to test
            num_images: Number of interpolated images for NEB
            neb_method: Method to use for NEB calculation (dyneb or neb)
            climb: Whether to use climbing image for NEB
            relax_fmax: Force tolerance for endpoint relaxation before NEB
            relax_steps: Maximum steps for endpoint relaxation before NEB
            neb_fmax: Force tolerance for NEB calculation
            neb_steps: Maximum steps for NEB calculation
            save_xyz: Save xyz files for each calculation
            output_dir: Directory to save xyz files
            n_nearest: Number of nearest neighbors to sample per vacancy
            n_next_nearest: Number of next-nearest neighbors to sample per vacancy
            rng_seed: Random seed for neighbor sampling
            verbose: Verbosity level
            
        Returns:
            List of dictionaries containing calculation results
        """
        # Use class seed if no specific seed provided
        seed_to_use = rng_seed if rng_seed is not None else self.seed
        
        # Generate vacancy-target pairs with structured format
        neighbor_samples = self.sample_neighbors(
            vacancy_indices=vacancy_indices if vacancy_indices else [i for i in range(len(self.atoms))],
            n_nearest=n_nearest,
            n_next_nearest=n_next_nearest,
            rng_seed=seed_to_use
        )
        
        # Count total calculations
        total_calcs = sum(len(sample['nn']) + len(sample['nnn']) for sample in neighbor_samples)
        results = []
        progress_step = max(1, total_calcs // 10)  # Report every 10%
        
        print(f"Starting {total_calcs} NEB calculations with Allegro...")
        calc_count = 0
        
        # Process each vacancy and its neighbors
        for i, sample in enumerate(neighbor_samples):
            vac_idx = sample['vacancy_index']
            
            # Process nearest neighbors
            for nn_idx in sample['nn']:
                if verbose > 0:
                    print(f"Running NEB {i+1}/{len(neighbor_samples)} - NN: vacancy at {vac_idx}, target at {nn_idx}")
                
                result = self.run_single(
                    vacancy_index=vac_idx,
                    target_index=nn_idx,
                    num_images=num_images,
                    neb_method=neb_method,
                    climb=climb,
                    relax_fmax=relax_fmax,
                    relax_steps=relax_steps,
                    neb_fmax=neb_fmax,
                    neb_steps=neb_steps,
                    save_xyz=save_xyz,
                    output_dir=output_dir,
                    verbose=verbose
                )
                result['is_nearest_neighbor'] = True
                results.append(result)
                
                calc_count += 1
                if calc_count % progress_step == 0:
                    print(f"Progress: {calc_count}/{total_calcs} calculations completed")
                
                if result["success"]:
                    self.analyzer.add_calculation(result)
                else:
                    print(f"Calculation failed for vacancy {vac_idx} to NN {nn_idx}: {result['error']}")
            
            # Process next-nearest neighbors
            for nnn_idx in sample['nnn']:
                if verbose > 0:
                    print(f"Running NEB {i+1}/{len(neighbor_samples)} - NNN: vacancy at {vac_idx}, target at {nnn_idx}")
                
                result = self.run_single(
                    vacancy_index=vac_idx,
                    target_index=nnn_idx,
                    num_images=num_images,
                    neb_method=neb_method,
                    climb=climb,
                    relax_fmax=relax_fmax,
                    relax_steps=relax_steps,
                    neb_fmax=neb_fmax,
                    neb_steps=neb_steps,
                    save_xyz=save_xyz,
                    output_dir=output_dir,
                    verbose=verbose
                )
                result['is_nearest_neighbor'] = False
                results.append(result)
                
                calc_count += 1
                if calc_count % progress_step == 0:
                    print(f"Progress: {calc_count}/{total_calcs} calculations completed")
                
                if result["success"]:
                    self.analyzer.add_calculation(result)
                else:
                    print(f"Calculation failed for vacancy {vac_idx} to NNN {nnn_idx}: {result['error']}")
        
        print(f"Completed {total_calcs} calculations")
        return results


class HybridNEBWorkflow:
    """
    Comprehensive workflow combining composition generation, hybrid MD/MC optimization, 
    and NEB calculations for vacancy diffusion studies.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
        output_dir: str = "hybrid_neb_results"
    ):
        """
        Initialize the hybrid NEB workflow.
        
        Args:
            model_path: Path to MACE model
            device: Device to run calculations on
            seed: Random seed for reproducibility
            output_dir: Directory to save results
        """
        self.model_path = model_path
        self.device = device
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Initialize components
        self.composition_analyzer = CompositionAnalyzer()
        self.allegro_calculator = None
        
        # Initialize calculators
        self._initialize_calculators()
        
        # Results storage
        self.compositions = []
        self.optimized_structures = []
        self.neb_results = []
        
    def _initialize_calculators(self):
        """Initialize Allegro calculator if available."""
        print(f"Initializing Allegro calculator on device: {self.device}")
        
        if ALLEGRO_AVAILABLE:
            try:
                # Initialize Allegro calculator
                # Note: You may need to adjust the species_to_type_name mapping
                # based on your specific Allegro model
                self.allegro_calculator = NequIPCalculator.from_deployed_model(
                    model_path=self.model_path,
                    species_to_type_name={'V': 0, 'Cr': 1, 'Ti': 2, 'W': 3, 'Zr': 4},
                    device=self.device
                )
                print("Successfully initialized Allegro calculator")
            except Exception as e:
                print(f"Warning: Could not initialize Allegro calculator: {e}")
                self.allegro_calculator = None
        else:
            print("Allegro not available. Please install nequip to use Allegro models.")
            self.allegro_calculator = None
    
    def generate_compositions(
        self,
        existing_compositions: List[Dict[str, float]],
        n_new_compositions: int = 5,
        elements: List[str] = ['V', 'Cr', 'Ti', 'W', 'Zr'],
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        balance_element: str = 'V'
    ) -> List[Dict[str, float]]:
        """
        Generate new alloy compositions using the CompositionAnalyzer.
        
        Args:
            existing_compositions: List of existing composition dictionaries
            n_new_compositions: Number of new compositions to generate
            elements: List of elements to include
            constraints: Optional constraints on element fractions
            balance_element: Element to balance the composition
            
        Returns:
            List of new composition dictionaries
        """
        print(f"Generating {n_new_compositions} new compositions...")
        
        # Set default constraints if none provided
        if constraints is None:
            constraints = {
                'V': (0.5, 0.9),    # Balance element should be majority
                'Cr': (0.01, 0.1),
                'Ti': (0.01, 0.1),
                'W': (0.01, 0.1),
                'Zr': (0.001, 0.01)
            }
        
        # Generate new compositions
        new_compositions = self.composition_analyzer.suggest_new_compositions(
            compositions=existing_compositions,
            n_suggestions=n_new_compositions,
            constraints=constraints,
            seed=self.seed
        )
        
        # Ensure compositions sum to 1.0
        for comp in new_compositions:
            total = sum(comp.values())
            if not np.isclose(total, 1.0, rtol=1e-3):
                # Adjust balance element
                comp[balance_element] = 1.0 - sum(v for k, v in comp.items() if k != balance_element)
        
        self.compositions = new_compositions
        print(f"Generated compositions: {new_compositions}")
        
        return new_compositions
    
    def create_initial_structures(
        self,
        compositions: List[Dict[str, float]],
        crystal_type: str = 'bcc',
        dimensions: List[int] = [8, 8, 8],
        lattice_constant: float = 3.01,
        balance_element: str = 'V'
    ) -> List[Atoms]:
        """
        Create initial atomic structures for each composition.
        
        Args:
            compositions: List of composition dictionaries
            crystal_type: Crystal structure type
            dimensions: Supercell dimensions
            lattice_constant: Lattice parameter in Angstrom
            balance_element: Element to use as base for structure
            
        Returns:
            List of ASE Atoms objects
        """
        print("Creating initial structures...")
        
        structures = []
        for i, composition in enumerate(compositions):
            print(f"Creating structure {i+1}/{len(compositions)}: {composition}")
            
            atoms = self.composition_analyzer.create_random_alloy(
                composition=composition,
                crystal_type=crystal_type,
                dimensions=dimensions,
                lattice_constant=lattice_constant,
                balance_element=balance_element,
                cubic=True
            )
            
            # Save initial structure
            formula = atoms.get_chemical_formula()
            write(self.output_dir / f"initial_structure_{i}_{formula}.xyz", atoms)
            
            structures.append(atoms)
        
        return structures
    
    def optimize_with_mcmc(
        self,
        structures: List[Atoms],
        temperature: float = 873.15,  # 600°C
        n_steps: int = 10000,
        convergence_window: int = 1000,
        energy_threshold: float = 0.0002
    ) -> List[Atoms]:
        """
        Optimize structures using MCMC simulation with Allegro calculator.
        
        Args:
            structures: List of ASE Atoms objects to optimize
            temperature: Simulation temperature in Kelvin
            n_steps: Number of simulation steps
            convergence_window: Steps to check for convergence
            energy_threshold: Energy change threshold for convergence
            
        Returns:
            List of optimized ASE Atoms objects
        """
        print("Optimizing structures with MCMC using Allegro calculator...")
        
        if self.allegro_calculator is None:
            raise ValueError("Allegro calculator not initialized. Please check model path and installation.")
        
        optimized_structures = []
        
        for i, atoms in enumerate(structures):
            print(f"Optimizing structure {i+1}/{len(structures)}")
            
            # Use standard MCMC with Allegro calculator
            optimized_atoms = self._optimize_with_mcmc(
                atoms, temperature, n_steps, convergence_window, energy_threshold
            )
            
            # Save optimized structure
            formula = optimized_atoms.get_chemical_formula()
            write(self.output_dir / f"optimized_structure_{i}_{formula}.xyz", optimized_atoms)
            
            optimized_structures.append(optimized_atoms)
        
        self.optimized_structures = optimized_structures
        return optimized_structures
    

    
    def _optimize_with_mcmc(
        self,
        atoms: Atoms,
        temperature: float,
        n_steps: int,
        convergence_window: int,
        energy_threshold: float
    ) -> Atoms:
        """Optimize structure using standard MCMC with Allegro calculator."""
        print("Using standard MCMC optimization with Allegro calculator")
        
        # Create MCMC sampler with Allegro calculator
        mc_sampler = MonteCarloAlloySampler(
            atoms=atoms,
            calculator=self.allegro_calculator,
            temperature=temperature,
            steps=n_steps,
            rng_seed=self.seed
        )
        
        # Run MCMC
        optimized_atoms = mc_sampler.run_mcmc(
            convergence_window=convergence_window,
            energy_threshold=energy_threshold
        )
        
        return optimized_atoms

    def _create_allegro_vacancy_diffusion(
        self,
        atoms: Atoms,
        nn_cutoff: float = 2.8,
        nnn_cutoff: float = 3.2,
        seed: int = 42
    ) -> 'AllegroVacancyDiffusion':
        """
        Create a VacancyDiffusion instance that uses Allegro calculator.
        
        Args:
            atoms: ASE Atoms object
            nn_cutoff: Cutoff radius for nearest neighbors
            nnn_cutoff: Cutoff radius for next-nearest neighbors
            seed: Random seed
            
        Returns:
            AllegroVacancyDiffusion instance
        """
        return AllegroVacancyDiffusion(
            atoms=atoms,
            allegro_calculator=self.allegro_calculator,
            nn_cutoff=nn_cutoff,
            nnn_cutoff=nnn_cutoff,
            seed=seed
        )
    
    def run_neb_calculations(
        self,
        structures: List[Atoms],
        vacancy_indices: Optional[List[int]] = None,
        n_nearest: int = 3,
        n_next_nearest: int = 3,
        num_images: int = 5,
        neb_method: str = "dyneb",
        climb: bool = True,
        relax_fmax: float = 0.01,
        relax_steps: int = 100,
        neb_fmax: float = 0.01,
        neb_steps: int = 200,
        save_xyz: bool = True,
        verbose: int = 1
    ) -> List[Dict]:
        """
        Run NEB calculations for vacancy diffusion on optimized structures.
        
        Args:
            structures: List of optimized ASE Atoms objects
            vacancy_indices: Specific vacancy sites to test (if None, sample randomly)
            n_nearest: Number of nearest neighbors to sample per vacancy
            n_next_nearest: Number of next-nearest neighbors to sample per vacancy
            num_images: Number of NEB images
            neb_method: NEB method ("dyneb" or "neb")
            climb: Whether to use climbing image
            relax_fmax: Force tolerance for endpoint relaxation
            relax_steps: Maximum steps for endpoint relaxation
            neb_fmax: Force tolerance for NEB calculation
            neb_steps: Maximum steps for NEB calculation
            save_xyz: Save xyz files for each calculation
            verbose: Verbosity level
            
        Returns:
            List of NEB calculation results
        """
        print("Running NEB calculations for vacancy diffusion...")
        
        all_neb_results = []
        
        for i, atoms in enumerate(structures):
            print(f"Running NEB calculations for structure {i+1}/{len(structures)}")
            
            # Create output directory for this structure
            structure_output_dir = self.output_dir / f"neb_structure_{i}"
            if save_xyz:
                structure_output_dir.mkdir(exist_ok=True)
            
            # Initialize vacancy diffusion workflow with Allegro calculator
            # We need to create a custom VacancyDiffusion that uses Allegro
            vacancy_diffusion = self._create_allegro_vacancy_diffusion(
                atoms=atoms,
                nn_cutoff=2.8,
                nnn_cutoff=3.2,
                seed=self.seed + i
            )
            
            # Run multiple NEB calculations
            neb_results = vacancy_diffusion.run_multiple(
                vacancy_indices=vacancy_indices,
                num_images=num_images,
                neb_method=neb_method,
                climb=climb,
                relax_fmax=relax_fmax,
                relax_steps=relax_steps,
                neb_fmax=neb_fmax,
                neb_steps=neb_steps,
                save_xyz=save_xyz,
                output_dir=structure_output_dir,
                n_nearest=n_nearest,
                n_next_nearest=n_next_nearest,
                rng_seed=self.seed + i,
                verbose=verbose
            )
            
            # Add structure information to results
            for result in neb_results:
                result['structure_index'] = i
                result['structure_formula'] = atoms.get_chemical_formula()
            
            all_neb_results.extend(neb_results)
            
            # Save results for this structure
            with open(structure_output_dir / "neb_results.json", 'w') as f:
                json.dump(neb_results, f, indent=2, default=str)
        
        self.neb_results = all_neb_results
        return all_neb_results
    
    def analyze_results(
        self,
        save_plots: bool = True,
        plot_barriers: bool = True,
        plot_compositions: bool = True
    ) -> Dict:
        """
        Analyze and visualize the results.
        
        Args:
            save_plots: Whether to save plots
            plot_barriers: Whether to plot barrier distributions
            plot_compositions: Whether to plot composition analysis
            
        Returns:
            Dictionary containing analysis results
        """
        print("Analyzing results...")
        
        analysis_results = {
            'compositions': self.compositions,
            'neb_results': self.neb_results,
            'summary': {}
        }
        
        # Create analyzer for NEB results
        neb_analyzer = NEBAnalyzer()
        for result in self.neb_results:
            if result.get('success', False):
                neb_analyzer.add_calculation(result)
        
        # Calculate statistics
        if neb_analyzer.calculations:
            stats = neb_analyzer.calculate_statistics()
            analysis_results['neb_statistics'] = stats
            
            # Print summary
            print("\nNEB Calculation Summary:")
            print(f"Total calculations: {len(self.neb_results)}")
            print(f"Successful calculations: {len(neb_analyzer.calculations)}")
            print(f"Overall NN mean barrier: {stats['overall'].get('nn_mean', 'N/A')} eV")
            print(f"Overall NNN mean barrier: {stats['overall'].get('nnn_mean', 'N/A')} eV")
            
            # Create plots if requested
            if save_plots and plot_barriers:
                print("Creating barrier distribution plots...")
                
                # Plot barrier distributions
                neb_analyzer.plot_barrier_distributions(
                    save_path=self.output_dir / "barrier_distributions.png",
                    title="Vacancy Diffusion Barriers",
                    min_barrier=0.0,
                    max_barrier=5.0
                )
                
                # Plot barriers by element
                neb_analyzer.plot_barriers_by_element(
                    save_path=self.output_dir / "barriers_by_element.png",
                    title="Energy Barriers by Element Type",
                    plot_type="box",
                    min_barrier=0.0,
                    max_barrier=5.0
                )
        
        # Composition analysis
        if plot_compositions and len(self.compositions) > 1:
            print("Creating composition analysis plots...")
            
            # Create composition visualization
            self.composition_analyzer.visualize_compositions(
                embeddings=np.array([[comp.get('V', 0), comp.get('Cr', 0), 
                                    comp.get('Ti', 0), comp.get('W', 0), 
                                    comp.get('Zr', 0)] for comp in self.compositions]),
                metadata=[{'composition': comp} for comp in self.compositions],
                save_path=self.output_dir / "composition_analysis.png"
            )
        
        # Save analysis results
        with open(self.output_dir / "analysis_results.json", 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        return analysis_results
    
    def run_full_workflow(
        self,
        existing_compositions: List[Dict[str, float]],
        n_new_compositions: int = 3,
        elements: List[str] = ['V', 'Cr', 'Ti', 'W', 'Zr'],
        crystal_type: str = 'bcc',
        dimensions: List[int] = [6, 6, 6],
        lattice_constant: float = 3.01,
        temperature: float = 873.15,
        n_steps: int = 5000,
        n_nearest: int = 2,
        n_next_nearest: int = 2,
        save_plots: bool = True
    ) -> Dict:
        """
        Run the complete workflow from composition generation to NEB analysis.
        
        Args:
            existing_compositions: List of existing composition dictionaries
            n_new_compositions: Number of new compositions to generate
            elements: List of elements to include
            crystal_type: Crystal structure type
            dimensions: Supercell dimensions
            temperature: Optimization temperature in Kelvin
            n_steps: Number of optimization steps
            mc_frequency: MC frequency for hybrid simulation
            n_nearest: Number of nearest neighbors for NEB
            n_next_nearest: Number of next-nearest neighbors for NEB
            save_plots: Whether to save plots
            
        Returns:
            Dictionary containing all workflow results
        """
        print("Starting full hybrid NEB workflow...")
        start_time = time.time()
        
        # Step 1: Generate new compositions
        compositions = self.generate_compositions(
            existing_compositions=existing_compositions,
            n_new_compositions=n_new_compositions,
            elements=elements
        )
        
        # Step 2: Create initial structures
        initial_structures = self.create_initial_structures(
            compositions=compositions,
            crystal_type=crystal_type,
            dimensions=dimensions,
            lattice_constant=lattice_constant
        )
        
        # Step 3: Optimize structures
        optimized_structures = self.optimize_with_mcmc(
            structures=initial_structures,
            temperature=temperature,
            n_steps=n_steps,
            convergence_window=1000,
            energy_threshold=0.0002
        )
        
        # Step 4: Run NEB calculations
        neb_results = self.run_neb_calculations(
            structures=optimized_structures,
            n_nearest=n_nearest,
            n_next_nearest=n_next_nearest
        )
        
        # Step 5: Analyze results
        analysis_results = self.analyze_results(save_plots=save_plots)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Create final summary
        workflow_summary = {
            'total_time_seconds': total_time,
            'n_compositions': len(compositions),
            'n_structures': len(optimized_structures),
            'n_neb_calculations': len(neb_results),
            'n_successful_neb': len([r for r in neb_results if r.get('success', False)]),
            'compositions': compositions,
            'analysis_results': analysis_results
        }
        
        # Save workflow summary
        with open(self.output_dir / "workflow_summary.json", 'w') as f:
            json.dump(workflow_summary, f, indent=2, default=str)
        
        print(f"\nWorkflow completed in {total_time:.2f} seconds")
        print(f"Results saved to: {self.output_dir}")
        
        return workflow_summary


def main():
    """Main function to run the hybrid NEB workflow."""
    
    # Configuration
    model_path = "../potentials/new_allegro/gen_7_2025-05-30_huberloss_thicc_model_0.nequip.zip"
    output_dir = "hybrid_neb_workflow_results"
    seed = 42
    
    # Example existing compositions (you can modify these)
    existing_compositions = [
        {'V': 0.85, 'Cr': 0.05, 'Ti': 0.05, 'W': 0.03, 'Zr': 0.02},
        {'V': 0.80, 'Cr': 0.08, 'Ti': 0.06, 'W': 0.04, 'Zr': 0.02},
        {'V': 0.75, 'Cr': 0.10, 'Ti': 0.08, 'W': 0.05, 'Zr': 0.02}
    ]
    
    # Initialize workflow
    workflow = HybridNEBWorkflow(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed,
        output_dir=output_dir
    )
    
    # Run full workflow
    results = workflow.run_full_workflow(
        existing_compositions=existing_compositions,
        n_new_compositions=2,
        elements=['V', 'Cr', 'Ti', 'W', 'Zr'],
        crystal_type='bcc',
        dimensions=[6, 6, 6],  # Smaller for faster testing
        lattice_constant=3.01,
        temperature=873.15,  # 600°C
        n_steps=2000,  # Reduced for faster testing
        mc_frequency=10,
        n_nearest=2,
        n_next_nearest=2,
        save_plots=True
    )
    
    print("\nWorkflow completed successfully!")
    print(f"Generated {results['n_compositions']} new compositions")
    print(f"Optimized {results['n_structures']} structures")
    print(f"Completed {results['n_neb_calculations']} NEB calculations")
    print(f"Successful NEB calculations: {results['n_successful_neb']}")


if __name__ == "__main__":
    main() 
"""
Hybrid MCMC-MD + NEB workflow for alloy structure optimization and vacancy diffusion studies.

This module provides a comprehensive workflow that combines:
1. Composition generation using the CompositionAnalyzer
2. Hybrid MCMC-MD optimization (NVT + Monte Carlo swaps)
3. NEB calculations for vacancy diffusion barriers
4. Analysis and visualization of results

The workflow follows the torch-sim pattern for efficient sampling and optimization.

Confidence: 9/10
"""

import numpy as np
import torch
import random
import time
import json
import os
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ASE imports
from ase import Atoms
from ase.io import read, write
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

# Forge imports
from forge.analysis.composition import CompositionAnalyzer
from forge.workflows.hybrid_mcmc import HybridMCMCSampler
from forge.workflows.neb import VacancyDiffusion, NEBAnalyzer
from forge.workflows.relax import relax
from forge.calculators.factory import create_ensemble_calculator, get_supported_backends

# Suppress torch warnings
warnings.filterwarnings("ignore", category=FutureWarning, 
                       message=".*You are using `torch.load` with `weights_only=False`.*")


class HybridNEBWorkflow:
    """
    Comprehensive workflow combining composition generation, hybrid MCMC-MD optimization, 
    and NEB calculations for vacancy diffusion studies.
    
    This workflow implements a complete pipeline for:
    1. Generating new alloy compositions using machine learning
    2. Creating and optimizing atomic structures using hybrid MCMC-MD
    3. Calculating vacancy diffusion barriers using NEB
    4. Analyzing and visualizing results
    
    The hybrid MCMC-MD approach combines NVT Langevin MD with Monte Carlo swaps,
    following the torch-sim pattern for efficient sampling.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
        output_dir: str = "hybrid_neb_results",
        backend: Optional[str] = None,
        species_to_type_name: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the hybrid NEB workflow.
        
        Args:
            model_path: Path to model file (Allegro, MACE, etc.)
            device: Device to run calculations on ('cuda' or 'cpu')
            seed: Random seed for reproducibility
            output_dir: Directory to save results
            backend: Calculator backend ('mace', 'allegro', or None for auto-detect)
            species_to_type_name: Species mapping for Allegro calculators
        """
        self.model_path = model_path
        self.device = device
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.species_to_type_name = species_to_type_name or {}
        
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Initialize components
        self.composition_analyzer = CompositionAnalyzer()
        self.ensemble_calculator = None
        
        # Initialize calculators
        self._initialize_calculators()
        
        # Results storage
        self.compositions = []
        self.optimized_structures = []
        self.neb_results = []
        
    def _initialize_calculators(self):
        """Initialize ensemble calculator for energy/force calculations."""
        print(f"Initializing ensemble calculator on device: {self.device}")
        
        # Check available backends
        available_backends = get_supported_backends()
        print(f"Available backends: {available_backends}")
        
        try:
            # Initialize ensemble calculator using the new factory
            self.ensemble_calculator = create_ensemble_calculator(
                model_paths=self.model_path,
                backend=self.backend,
                device=self.device,
                species_to_type_name=self.species_to_type_name
            )
            print(f"Successfully initialized {type(self.ensemble_calculator).__name__} calculator")
        except Exception as e:
            print(f"Error: Could not initialize calculator: {e}")
            print(f"Model path: {self.model_path}")
            print(f"Backend: {self.backend}")
            print(f"Device: {self.device}")
            print(f"Species mapping: {self.species_to_type_name}")
            print("\nTroubleshooting tips:")
            print("1. Check if the model file exists and is accessible")
            print("2. Verify the model file format (.zip, .nequip.zip, or .pt2)")
            print("3. Ensure the species_to_type_name mapping is correct")
            print("4. Check if the required packages (nequip, mace) are installed")
            self.ensemble_calculator = None
    
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
            constraints: Optional constraints on element fractions. Can be:
                - Range constraints: (min, max) tuples for element ranges
                - Specific values: (value, value) for exact amounts
                - Mixed constraints: Some ranges, some specific values
                Examples:
                    # Range constraints (recommended)
                    constraints = {
                        'V': (0.7, 0.95),    # V between 70-95%
                        'Cr': (0.01, 0.2),   # Cr between 1-20%
                        'Ti': (0.01, 0.2),   # Ti between 1-20%
                        'W': (0.01, 0.15),   # W between 1-15%
                        'Zr': (0.001, 0.03)  # Zr between 0.1-3%
                    }
                    
                    # Specific value constraints
                    constraints = {
                        'V': (0.85, 0.85),   # Exactly 85% V
                        'Cr': (0.10, 0.10),  # Exactly 10% Cr
                        'Ti': (0.03, 0.03),  # Exactly 3% Ti
                        'W': (0.01, 0.01),   # Exactly 1% W
                        'Zr': (0.01, 0.01)   # Exactly 1% Zr
                    }
                    
                    # Mixed constraints
                    constraints = {
                        'V': (0.8, 0.9),     # V between 80-90%
                        'Cr': (0.05, 0.15),  # Cr between 5-15%
                        'Ti': (0.025, 0.025), # Exactly 2.5% Ti
                        'W': (0.01, 0.05),   # W between 1-5%
                        'Zr': (0.005, 0.005) # Exactly 0.5% Zr
                    }
            balance_element: Element to balance the composition
            
        Returns:
            List of new composition dictionaries
        """
        print(f"Generating {n_new_compositions} new compositions...")
        
        # Set default constraints if none provided
        if constraints is None:
            constraints = {
                'V': (0.7, 0.95),   # Balance element should be majority
                'Cr': (0.01, 0.2),
                'Ti': (0.01, 0.2),
                'W': (0.01, 0.15),
                'Zr': (0.001, 0.03)
            }
        
        # Generate new compositions
        print(f"Generating compositions with constraints: {constraints}")
        new_compositions = self.composition_analyzer.suggest_new_compositions(
            compositions=existing_compositions,
            n_suggestions=n_new_compositions,
            constraints=constraints,
            seed=self.seed
        )
        
        print(f"Raw generated compositions: {new_compositions}")
        
        # Ensure compositions sum to 1.0
        for comp in new_compositions:
            total = sum(comp.values())
            if not np.isclose(total, 1.0, rtol=1e-3):
                # Adjust balance element
                comp[balance_element] = 1.0 - sum(v for k, v in comp.items() if k != balance_element)
        
        self.compositions = new_compositions
        print(f"Final compositions: {new_compositions}")
        
        # If no compositions were generated, create some simple interpolated ones
        if len(new_compositions) == 0:
            print("Warning: No compositions generated. Creating simple interpolated compositions.")
            new_compositions = self._create_simple_compositions(
                existing_compositions, n_new_compositions, balance_element
            )
            self.compositions = new_compositions
            print(f"Created simple compositions: {new_compositions}")
        
        return new_compositions
    
    def _create_simple_compositions(
        self,
        existing_compositions: List[Dict[str, float]],
        n_new_compositions: int,
        balance_element: str
    ) -> List[Dict[str, float]]:
        """
        Create simple interpolated compositions as a fallback.
        
        Args:
            existing_compositions: List of existing composition dictionaries
            n_new_compositions: Number of new compositions to create
            balance_element: Element to balance the composition
            
        Returns:
            List of new composition dictionaries
        """
        if len(existing_compositions) < 2:
            # If we only have one composition, create variations
            base_comp = existing_compositions[0]
            new_compositions = []
            
            for i in range(n_new_compositions):
                # Create small variations
                variation = 0.02 * (i + 1)  # Small variation
                new_comp = base_comp.copy()
                
                # Adjust non-balance elements slightly
                for element in new_comp:
                    if element != balance_element:
                        new_comp[element] = max(0.001, new_comp[element] + variation * (0.5 - np.random.random()))
                
                # Rebalance to sum to 1.0
                total = sum(v for k, v in new_comp.items() if k != balance_element)
                new_comp[balance_element] = max(0.4, 1.0 - total)
                
                # Normalize
                total = sum(new_comp.values())
                for element in new_comp:
                    new_comp[element] /= total
                
                new_compositions.append(new_comp)
        else:
            # Interpolate between existing compositions
            new_compositions = []
            
            for i in range(n_new_compositions):
                # Pick two random existing compositions
                idx1, idx2 = np.random.choice(len(existing_compositions), 2, replace=False)
                comp1 = existing_compositions[idx1]
                comp2 = existing_compositions[idx2]
                
                # Interpolate
                alpha = np.random.random()
                new_comp = {}
                
                all_elements = set(comp1.keys()) | set(comp2.keys())
                for element in all_elements:
                    val1 = comp1.get(element, 0.0)
                    val2 = comp2.get(element, 0.0)
                    new_comp[element] = alpha * val1 + (1 - alpha) * val2
                
                # Ensure balance element is present
                if balance_element not in new_comp:
                    new_comp[balance_element] = 0.5
                
                # Normalize to sum to 1.0
                total = sum(new_comp.values())
                for element in new_comp:
                    new_comp[element] /= total
                
                new_compositions.append(new_comp)
        
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
    
    def optimize_with_hybrid_mcmc(
        self,
        structures: List[Atoms],
        temperature: float = 873.15,  # 600°C
        md_temperature: Optional[float] = None,
        n_steps: int = 1000,
        md_steps_per_cycle: int = 100,
        mc_steps_per_cycle: int = 50,
        convergence_window: int = 1000,
        energy_threshold: float = 0.0002,
        final_cell_relax: bool = True,
        fmax: float = 0.05,
        steps: int = 1000,
        md_timestep: float = 2.0,
        md_thermostat: str = 'langevin',
        friction: float = 0.02
    ) -> List[Atoms]:
        """
        Optimize structures using hybrid MCMC-MD simulation with unified calculator.
        
        Args:
            structures: List of ASE Atoms objects to optimize
            temperature: Simulation temperature in Kelvin (for MC)
            md_temperature: MD temperature (defaults to temperature)
            n_steps: Number of hybrid cycles
            md_steps_per_cycle: MD steps per hybrid cycle
            mc_steps_per_cycle: MC steps per hybrid cycle
            convergence_window: Steps to check for convergence
            energy_threshold: Energy change threshold for convergence
            final_cell_relax: Whether to do final cell relaxation after convergence
            md_timestep: MD timestep in fs
            md_thermostat: 'langevin' or 'verlet'
            friction: Friction coefficient for Langevin
            
        Returns:
            List of optimized ASE Atoms objects
        """
        if self.ensemble_calculator is None:
            raise ValueError(
                "Ensemble calculator not initialized. Please check:\n"
                "1. Model file path and accessibility\n"
                "2. Model file format (.zip, .nequip.zip, or .pt2)\n"
                "3. Backend configuration\n"
                "4. Required packages installation (nequip, mace)\n"
                f"Model path: {self.model_path}"
            )
        
        print(f"Optimizing structures with hybrid MCMC-MD using {type(self.ensemble_calculator).__name__} calculator...")
        
        optimized_structures = []
        
        for i, atoms in enumerate(structures):
            print(f"Optimizing structure {i+1}/{len(structures)}")
            
            # Use hybrid MCMC with unified calculator
            optimized_atoms = self._optimize_with_hybrid_mcmc(
                atoms, temperature, md_temperature, n_steps, md_steps_per_cycle, 
                mc_steps_per_cycle, convergence_window, energy_threshold,
                final_cell_relax, fmax, steps, md_timestep, md_thermostat, friction
            )
            
            # Save optimized structure
            formula = optimized_atoms.get_chemical_formula()
            write(self.output_dir / f"optimized_structure_{i}_{formula}.xyz", optimized_atoms)
            
            optimized_structures.append(optimized_atoms)
        
        self.optimized_structures = optimized_structures
        return optimized_structures
    
    def _optimize_with_hybrid_mcmc(
        self,
        atoms: Atoms,
        temperature: float,
        md_temperature: Optional[float],
        n_steps: int,
        md_steps_per_cycle: int,
        mc_steps_per_cycle: int,
        convergence_window: int,
        energy_threshold: float,
        final_cell_relax: bool,
        fmax: float,
        steps: int,
        md_timestep: float,
        md_thermostat: str,
        friction: float
    ) -> Atoms:
        """Optimize structure using hybrid MCMC-MD with unified calculator."""
        print(f"Using hybrid MCMC-MD optimization with {type(self.ensemble_calculator).__name__} calculator")
        
        # Create hybrid MCMC sampler with unified calculator
        hybrid_sampler = HybridMCMCSampler(
            atoms=atoms,
            calculator=self.ensemble_calculator,
            temperature=temperature,
            md_temperature=md_temperature,
            steps=n_steps,
            md_steps_per_cycle=md_steps_per_cycle,
            mc_steps_per_cycle=mc_steps_per_cycle,
            final_cell_relax=final_cell_relax,
            md_timestep=md_timestep,
            md_thermostat=md_thermostat,
            friction=friction,
            rng_seed=self.seed
        )
        
        # Run hybrid MCMC
        optimized_atoms = hybrid_sampler.run_hybrid_mcmc(
            convergence_window=convergence_window,
            energy_threshold=energy_threshold,
            fmax=fmax,
            steps=steps
        )
        
        return optimized_atoms
    
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
        neb_fmax: float = 0.05,
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
        
        if self.ensemble_calculator is None:
            raise ValueError(
                "Ensemble calculator not initialized. Please check:\n"
                "1. Model file path and accessibility\n"
                "2. Model file format (.zip, .nequip.zip, or .pt2)\n"
                "3. Backend configuration\n"
                "4. Required packages installation (nequip, mace)\n"
                f"Model path: {self.model_path}"
            )
        
        all_neb_results = []
        
        for i, atoms in enumerate(structures):
            print(f"Running NEB calculations for structure {i+1}/{len(structures)}")
            
            # Create output directory for this structure
            structure_output_dir = self.output_dir / f"neb_structure_{i}"
            if save_xyz:
                structure_output_dir.mkdir(exist_ok=True)
            
            # Initialize vacancy diffusion workflow with unified calculator
            vacancy_diffusion = VacancyDiffusion(
                atoms=atoms,
                model_path=[self.model_path],
                nn_cutoff=2.8,
                nnn_cutoff=3.2,
                seed=self.seed + i,
                backend=self.backend,
                species_to_type_name=self.species_to_type_name
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
            
            # Use the proper workflow: reduce and cluster, then visualize
            embeddings, clusters, metadata = self.composition_analyzer.reduce_and_cluster(
                compositions=self.compositions,
                n_clusters=min(5, len(self.compositions)),
                comp_type='new',
                seed=self.seed
            )
            
            # Create composition visualization
            self.composition_analyzer.visualize_compositions(
                embeddings=embeddings,
                metadata=metadata,
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
            n_nearest: Number of nearest neighbors for NEB
            n_next_nearest: Number of next-nearest neighbors for NEB
            save_plots: Whether to save plots
            
        Returns:
            Dictionary containing all workflow results
        """
        print("Starting full hybrid MCMC-MD NEB workflow...")
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
        
        # Step 3: Optimize structures with hybrid MCMC-MD
        optimized_structures = self.optimize_with_hybrid_mcmc(
            structures=initial_structures,
            temperature=temperature,
            md_temperature=temperature,  # Use same temperature for MD
            n_steps=n_steps,
            md_steps_per_cycle=min(20*len(initial_structures[0]), 1000),  # Scale with system size
            mc_steps_per_cycle=min(50*len(initial_structures[0]), 5000),  # Scale with system size
            convergence_window=1000,
            energy_threshold=0.002,  # 2 meV/atom
            final_cell_relax=True,  # Do final cell relaxation
            md_timestep=1.0,  # 1 fs timestep
            md_thermostat='langevin',
            friction=0.02
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
    """Example usage of the HybridNEBWorkflow."""
    
    # Configuration
    model_path = "../data/potentials/allegro/gen-8-exploit_rmax6.00_lmax2_layers2_mlp384.nequip.zip"
    output_dir = "hybrid_neb_workflow_results"
    seed = 42
    
    # Example existing compositions
    existing_compositions = [
        {'V': 0.85, 'Cr': 0.05, 'Ti': 0.05, 'W': 0.03, 'Zr': 0.02},
        {'V': 0.80, 'Cr': 0.08, 'Ti': 0.06, 'W': 0.04, 'Zr': 0.02},
        {'V': 0.75, 'Cr': 0.10, 'Ti': 0.08, 'W': 0.05, 'Zr': 0.02}
    ]
    
    # Example constraint configurations (uncomment to use):
    
    # Range constraints (recommended for exploration)
    # constraints = {
    #     'V': (0.7, 0.95),    # V between 70-95%
    #     'Cr': (0.01, 0.2),   # Cr between 1-20%
    #     'Ti': (0.01, 0.2),   # Ti between 1-20%
    #     'W': (0.01, 0.15),   # W between 1-15%
    #     'Zr': (0.001, 0.03)  # Zr between 0.1-3%
    # }
    
    # Specific value constraints (for targeted compositions)
    # constraints = {
    #     'V': (0.85, 0.85),   # Exactly 85% V
    #     'Cr': (0.10, 0.10),  # Exactly 10% Cr
    #     'Ti': (0.03, 0.03),  # Exactly 3% Ti
    #     'W': (0.01, 0.01),   # Exactly 1% W
    #     'Zr': (0.01, 0.01)   # Exactly 1% Zr
    # }
    
    # Mixed constraints (ranges + specific values)
    # constraints = {
    #     'V': (0.8, 0.9),     # V between 80-90%
    #     'Cr': (0.05, 0.15),  # Cr between 5-15%
    #     'Ti': (0.025, 0.025), # Exactly 2.5% Ti
    #     'W': (0.01, 0.05),   # W between 1-5%
    #     'Zr': (0.005, 0.005) # Exactly 0.5% Zr
    # }
    
    # Initialize workflow
    workflow = HybridNEBWorkflow(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed,
        output_dir=output_dir,
        backend=None,  # Auto-detect based on model file
        species_to_type_name={'V': 0, 'Cr': 1, 'Ti': 2, 'W': 3, 'Zr': 4}
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
        n_steps=500,  # Hybrid cycles (reduced for faster testing)
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
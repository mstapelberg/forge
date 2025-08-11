"""
Hybrid Monte Carlo + Molecular Dynamics sampler for alloy structure optimization.

This module implements a hybrid approach combining Monte Carlo swaps with
NVT Molecular Dynamics relaxation, following the torch-sim pattern.
After convergence, optional cell relaxation can be performed.

Confidence: 9/10
"""

import numpy as np
from typing import Optional, Dict, List
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import VelocityVerlet, Langevin
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from ase.units import kB, fs  # Boltzmann constant and femtosecond

from forge.workflows.mcmc import MCMCTracker  # Assuming this exists for tracking

from forge.calculators.factory import create_ensemble_calculator

class HybridMCMCSampler:
    """Hybrid Monte Carlo + Molecular Dynamics sampler for alloy structures."""

    def __init__(
        self,
        atoms: Atoms,
        calculator,
        temperature: float = 1000.0,
        md_temperature: Optional[float] = None,
        steps: int = 1000,
        md_steps_per_cycle: int = 100,
        mc_steps_per_cycle: int = 50,
        tracker_settings: Optional[dict] = None,
        rng_seed: int = 42,
        final_cell_relax: bool = True,  # Whether to do final cell relaxation
        md_timestep: float = 2.0,  # fs
        md_thermostat: str = 'langevin',  # 'langevin' or 'verlet'
        friction: float = 0.02,  # For Langevin
        verbose: bool = True  # Whether to show detailed progress bars
    ) -> None:
        """
        Initialize the hybrid MCMC-MD sampler.

        Args:
            atoms: Initial atomic structure
            calculator: ASE calculator for energy/forces
            temperature: Simulation temperature in Kelvin (for MC)
            md_temperature: MD temperature (defaults to temperature)
            steps: Total number of hybrid cycles
            md_steps_per_cycle: MD steps per hybrid cycle
            mc_steps_per_cycle: MC steps per hybrid cycle
            tracker_settings: Settings for MCMCTracker
            rng_seed: Random seed
            final_cell_relax: Whether to do final cell relaxation after convergence
            md_timestep: MD timestep in fs
            md_thermostat: 'langevin' or 'verlet'
            friction: Friction coefficient for Langevin
            verbose: Whether to show detailed progress bars for MD and MC phases
        """
        self.atoms = atoms.copy()
        self.calculator = calculator
        self.temperature = temperature
        self.md_temperature = md_temperature or temperature
        self.steps = steps
        self.md_steps_per_cycle = md_steps_per_cycle
        self.mc_steps_per_cycle = mc_steps_per_cycle
        self.rng = np.random.default_rng(rng_seed)
        self.k_b = kB  # eV/K
        self.final_cell_relax = final_cell_relax
        self.md_timestep = md_timestep * fs
        self.md_thermostat = md_thermostat
        self.friction = friction
        self.verbose = verbose
        
        # Statistics tracking
        self.mc_acceptance_count = 0
        self.mc_total_attempts = 0
        
        # Set up tracker if requested
        if tracker_settings:
            tracker_settings = tracker_settings.copy()
            tracker_settings['atoms'] = self.atoms
            self.tracker = MCMCTracker(**tracker_settings)
        else:
            self.tracker = None
        
        # Handle calculator - now supports both old UnifiedCalculator and new ensemble calculators
        if hasattr(self.calculator, 'calculator'):
            # Old UnifiedCalculator interface
            self.atoms.calc = self.calculator.calculator
        else:
            # New ensemble calculator or direct ASE calculator
            self.atoms.calc = self.calculator
        self.current_energy = self.atoms.get_potential_energy()

    def run_hybrid_mcmc(self, convergence_window: int = 1000, energy_threshold: float = 0.0002, fmax: float = 0.05, steps: int = 500) -> Atoms:
        """
        Run hybrid MCMC-MD simulation with convergence checking.

        Args:
            convergence_window: Steps to check for convergence
            energy_threshold: Energy change per atom threshold (eV)

        Returns:
            Final optimized atomic configuration
        """
        n_atoms = len(self.atoms)
        energy_history = []
        converged = False
        total_steps = 0
        
        # Progress bar for cycles
        pbar = tqdm(range(self.steps), desc="Hybrid MCMC-MD", unit="cycle")
        
        for cycle in pbar:
            if converged:
                break
            
            # MD phase
            self._run_md_phase(cycle)
            
            # MC phase
            self._run_mc_phase(cycle)
            
            # Update total steps
            total_steps += self.md_steps_per_cycle + self.mc_steps_per_cycle
            
            # Get current energy
            self.current_energy = self.atoms.get_potential_energy()
            energy_history.append(self.current_energy)
            
            # Track if needed
            if self.tracker:
                self.tracker.record_energy(total_steps, self.current_energy)
                self.tracker.record_wc_params(total_steps, self.atoms)
            
            # Convergence check
            if len(energy_history) > convergence_window:
                energy_history.pop(0)
                energy_range = max(energy_history) - min(energy_history)
                energy_range_per_atom = energy_range / n_atoms
                
                # Calculate MC acceptance rate
                mc_acceptance_rate = self.mc_acceptance_count / max(self.mc_total_attempts, 1) * 100
                
                pbar.set_postfix({
                    'energy': f"{self.current_energy:.3f} eV",
                    'Δe/atom': f"{energy_range_per_atom:.6f} eV",
                    'MC_acc%': f"{mc_acceptance_rate:.1f}%"
                })
                
                if energy_range_per_atom < energy_threshold:
                    print(f"\nConverged at cycle {cycle}! ΔE/atom = {energy_range_per_atom:.6f} eV < {energy_threshold:.6f} eV")
                    converged = True
            else:
                # Calculate MC acceptance rate
                mc_acceptance_rate = self.mc_acceptance_count / max(self.mc_total_attempts, 1) * 100
                pbar.set_postfix({
                    'energy': f"{self.current_energy:.3f} eV",
                    'MC_acc%': f"{mc_acceptance_rate:.1f}%"
                })
        
        pbar.close()
        
        # Final cell relaxation if requested (following torch-sim pattern)
        if self.final_cell_relax:
            print("\nPerforming final cell relaxation...")
            self._final_cell_relaxation(fmax=fmax, steps=steps)
        
        return self.atoms
    
    def _run_md_phase(self, cycle: int):
        """Run NVT MD relaxation phase (no cell optimization during sampling)."""
        # Initialize velocities if not already set
        if not hasattr(self.atoms, 'get_velocities') or self.atoms.get_velocities() is None:
            MaxwellBoltzmannDistribution(self.atoms, temperature_K=self.md_temperature)
        
        # Set up NVT MD (following torch-sim pattern)
        if self.md_thermostat == 'langevin':
            md = Langevin(self.atoms, self.md_timestep, temperature_K=self.md_temperature, friction=self.friction)
        else:
            md = VelocityVerlet(self.atoms, self.md_timestep)
        
        # Run MD steps with progress bar if verbose
        if self.verbose and self.md_steps_per_cycle > 10:  # Only show progress for longer MD runs
            md_pbar = tqdm(range(self.md_steps_per_cycle), 
                          desc=f"MD Phase (Cycle {cycle})", 
                          unit="step", 
                          leave=False)
            for _ in md_pbar:
                md.run(1)
            md_pbar.close()
        else:
            # Run MD steps without progress bar
            for _ in range(self.md_steps_per_cycle):
                md.run(1)
    
    def _run_mc_phase(self, cycle: int):
        """Run Monte Carlo swap phase."""
        n_atoms = len(self.atoms)
        accepted_this_cycle = 0
        attempts_this_cycle = 0
        
        # Run MC steps with progress bar if verbose
        if self.verbose and self.mc_steps_per_cycle > 10:  # Only show progress for longer MC runs
            mc_pbar = tqdm(range(self.mc_steps_per_cycle), 
                          desc=f"MC Phase (Cycle {cycle})", 
                          unit="step", 
                          leave=False)
            for _ in mc_pbar:
                # Pick two sites
                site1 = self.rng.integers(0, n_atoms)
                site2 = self.rng.integers(0, n_atoms)
                if site1 == site2 or self.atoms[site1].symbol == self.atoms[site2].symbol:
                    continue
                
                attempts_this_cycle += 1
                old_symbol1 = self.atoms[site1].symbol
                old_symbol2 = self.atoms[site2].symbol
                
                # Propose swap
                self.atoms[site1].symbol = old_symbol2
                self.atoms[site2].symbol = old_symbol1
                trial_energy = self.atoms.get_potential_energy()
                delta_e = trial_energy - self.current_energy
                
                if delta_e <= 0 or self.rng.random() < np.exp(-delta_e / (self.k_b * self.temperature)):
                    self.current_energy = trial_energy
                    accepted_this_cycle += 1
                else:
                    # Revert
                    self.atoms[site1].symbol = old_symbol1
                    self.atoms[site2].symbol = old_symbol2
                
                # Update progress bar with acceptance rate
                if attempts_this_cycle > 0:
                    cycle_acceptance_rate = accepted_this_cycle / attempts_this_cycle * 100
                    mc_pbar.set_postfix({'acc%': f"{cycle_acceptance_rate:.1f}%"})
            
            mc_pbar.close()
        else:
            # Run MC steps without progress bar
            for _ in range(self.mc_steps_per_cycle):
                # Pick two sites
                site1 = self.rng.integers(0, n_atoms)
                site2 = self.rng.integers(0, n_atoms)
                if site1 == site2 or self.atoms[site1].symbol == self.atoms[site2].symbol:
                    continue
                
                attempts_this_cycle += 1
                old_symbol1 = self.atoms[site1].symbol
                old_symbol2 = self.atoms[site2].symbol
                
                # Propose swap
                self.atoms[site1].symbol = old_symbol2
                self.atoms[site2].symbol = old_symbol1
                trial_energy = self.atoms.get_potential_energy()
                delta_e = trial_energy - self.current_energy
                
                if delta_e <= 0 or self.rng.random() < np.exp(-delta_e / (self.k_b * self.temperature)):
                    self.current_energy = trial_energy
                    accepted_this_cycle += 1
                else:
                    # Revert
                    self.atoms[site1].symbol = old_symbol1
                    self.atoms[site2].symbol = old_symbol2
        
        # Update global statistics
        self.mc_acceptance_count += accepted_this_cycle
        self.mc_total_attempts += attempts_this_cycle
    
    def _final_cell_relaxation(self, fmax: float = 0.05, steps: int = 500):
        """Perform final cell relaxation after MCMC convergence."""
        from ase.optimize import FIRE
        from ase.filters import FrechetCellFilter
        
        print(f"  Initial energy: {self.atoms.get_potential_energy():.3f} eV")
        
        # Create cell filter for cell optimization
        cell_filter = FrechetCellFilter(self.atoms)
        optimizer = FIRE(cell_filter)
        
        # Run cell optimization
        optimizer.run(fmax=fmax, steps=steps)
        
        # Update current energy
        self.current_energy = self.atoms.get_potential_energy()
        print(f"  Final energy after cell relaxation: {self.current_energy:.3f} eV")
    
    def get_statistics(self) -> dict:
        """Get statistics from the hybrid MCMC-MD simulation.
        
        Returns:
            Dictionary containing simulation statistics
        """
        total_acceptance_rate = self.mc_acceptance_count / max(self.mc_total_attempts, 1) * 100
        
        return {
            'mc_total_attempts': self.mc_total_attempts,
            'mc_accepted_swaps': self.mc_acceptance_count,
            'mc_acceptance_rate_percent': total_acceptance_rate,
            'md_steps_per_cycle': self.md_steps_per_cycle,
            'mc_steps_per_cycle': self.mc_steps_per_cycle,
            'temperature': self.temperature,
            'md_temperature': self.md_temperature,
            'final_energy': self.current_energy
        } 
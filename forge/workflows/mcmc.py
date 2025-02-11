"""Monte Carlo simulation tools for alloy structures."""
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

from ase import Atoms

from forge.analysis.wc_sro import WarrenCowleyCalculator


def get_default_bcc_shells(lattice_constant: float) -> List[float]:
    """Get default shell distances for BCC structure.
    
    Args:
        lattice_constant: Lattice parameter of the BCC structure

    Returns:
        List of shell distances corresponding to 1st through 4th nearest neighbors
    """
    a = lattice_constant
    return [
        0.0,                # start
        np.sqrt(3)/2 * a,   # 1st NN
        a,                  # 2nd NN
        np.sqrt(2) * a,     # 3rd NN
        np.sqrt(11)/2 * a,  # 4th NN
    ]


class MCMCTracker:
    """Track and visualize MCMC simulation progress."""

    def __init__(
        self,
        atoms: Atoms,
        energy_freq: int = 1,
        wc_freq: Optional[int] = None,
        shells: Optional[List[float]] = None,
        lattice_constant: Optional[float] = None,
    ) -> None:
        """Initialize the MCMC tracker.

        Args:
            atoms: Atomic structure being simulated
            energy_freq: Save energy every N steps
            wc_freq: Calculate WC parameters every N steps (if None, same as energy_freq)
            shells: Shell distances for WC calculation (if None, use BCC defaults)
            lattice_constant: Required if using default BCC shells
        """
        self.atoms = atoms
        self.energy_freq = energy_freq
        self.wc_freq = wc_freq if wc_freq is not None else energy_freq
        
        if shells is None:
            if lattice_constant is None:
                raise ValueError("lattice_constant required if using default BCC shells")
            self.shells = get_default_bcc_shells(lattice_constant)
        else:
            self.shells = shells

        # Initialize storage
        self.energies: List[float] = []
        self.steps: List[int] = []
        self.wc_params: List[Dict[Tuple[str, str], List[float]]] = []
        self.wc_steps: List[int] = []

    def record_energy(self, step: int, energy: float) -> None:
        """Record energy at a given step."""
        self.steps.append(step)
        self.energies.append(energy)

    def record_wc_params(self, step: int, atoms: Atoms) -> None:
        """Calculate and record Warren-Cowley parameters."""
        calculator = WarrenCowleyCalculator(atoms, self.shells)
        params = calculator.calculate_parameters()
        self.wc_params.append(params)
        self.wc_steps.append(step)

    def plot_results(self, save_dir: Optional[str] = None, colormap: str = 'tab20') -> None:
        """Create plots of energy evolution and WC parameters.
        
        Args:
            save_dir: Directory to save plots and data. If None, only display.
            colormap: Name of matplotlib colormap to use (e.g. 'tab20', 'Set3', 'tab20b').
                     Qualitative colormaps recommended for distinct colors.
        """
        # Create figure with shared x-axis subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot energy evolution
        ax1.plot(self.steps, self.energies, 'k-')
        ax1.set_ylabel('Energy (eV)')
        ax1.set_title('MCMC Energy Evolution')
        
        # Get unique elements for WC parameter plotting
        if self.wc_params:
            # Get unique elements from first shell of first timestep
            first_params = self.wc_params[0][0]  # First timestep, first shell
            elements = sorted(set(self.atoms.get_chemical_symbols()))
            
            # Plot X-X parameters (first shell only)
            cmap_xx = plt.get_cmap(colormap)
            n_xx = len(elements)  # Number of X-X pairs
            colors_xx = [cmap_xx(i/min(20, max(n_xx, 1))) for i in range(n_xx)]
            
            for idx, elem in enumerate(elements):
                i = elements.index(elem)
                values = [params[0][i, i] for params in self.wc_params]  # First shell
                ax2.plot(self.wc_steps, values, color=colors_xx[idx], label=f'{elem}-{elem}', linewidth=2)
            ax2.set_ylabel('WC Parameter (1st shell)')
            ax2.set_title('Self-Interaction Parameters')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Plot X-Y parameters (first shell only)
            plotted = set()
            # Count number of unique X-Y pairs
            n_xy = sum(1 for i, elem1 in enumerate(elements) 
                      for j, elem2 in enumerate(elements) 
                      if i < j)
            
            cmap_xy = plt.get_cmap(colormap)
            colors_xy = [cmap_xy(i/min(20, max(n_xy, 1))) for i in range(n_xy)]
            color_idx = 0
            
            for i, elem1 in enumerate(elements):
                for j, elem2 in enumerate(elements):
                    if i >= j or (elem2, elem1) in plotted:
                        continue
                    values = [params[0][i, j] for params in self.wc_params]  # First shell
                    ax3.plot(self.wc_steps, values, 
                            color=colors_xy[color_idx], 
                            label=f'{elem1}-{elem2}',
                            linewidth=2)
                    plotted.add((elem1, elem2))
                    color_idx += 1
                    
            ax3.set_ylabel('WC Parameter (1st shell)')
            ax3.set_title('Cross-Interaction Parameters')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax3.set_xlabel('MC Steps')
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save plots with extra space for legend
            plt.savefig(save_path / 'mcmc_evolution.png', dpi=300, bbox_inches='tight')
            
            # Save raw data
            np.savez(
                save_path / 'mcmc_data.npz',
                steps=self.steps,
                energies=self.energies,
                wc_steps=self.wc_steps,
                wc_params=self.wc_params
            )
        else:
            plt.show()


class MonteCarloAlloySampler:
    """Perform Monte Carlo swaps in a structure for SRO refinement."""

    def __init__(
        self,
        atoms: Atoms,
        calculator,
        temperature: float = 1000.0,
        steps: int = 1000,
        tracker_settings: Optional[dict] = None,
        rng_seed: int = 42,
    ) -> None:
        """Initialize the Monte Carlo sampler.

        Args:
            atoms: Initial atomic structure
            calculator: ASE calculator for energy evaluation
            temperature: Simulation temperature in Kelvin
            steps: Number of Monte Carlo steps
            tracker_settings: Settings for the MCMCTracker
            rng_seed: Random seed for reproducibility
        """
        self.atoms = atoms.copy()
        self.calculator = calculator
        self.temperature = temperature
        self.steps = steps
        self.rng = np.random.default_rng(rng_seed)
        self.k_b = 8.617333262e-5  # Boltzmann constant in eV/K
        
        # Set up tracker if requested
        if tracker_settings:
            tracker_settings = tracker_settings.copy()  # Don't modify original
            tracker_settings['atoms'] = self.atoms
            self.tracker = MCMCTracker(**tracker_settings)
        else:
            self.tracker = None
        
        # Attach calculator and get initial energy
        self.atoms.calc = self.calculator
        self.current_energy = self.atoms.get_potential_energy()

    def run_mcmc(self, convergence_window: int = 1000, energy_threshold: float = 0.0002) -> Atoms:
        """Perform Metropolis-Hastings Monte Carlo swaps.

        Args:
            convergence_window: Number of steps to check for convergence
            energy_threshold: Maximum energy change per atom (in eV) allowed for convergence

        Returns:
            Final atomic configuration
        """
        n_atoms = len(self.atoms)
        
        # Record initial state if tracking
        if self.tracker:
            self.tracker.record_energy(0, self.current_energy)
            self.tracker.record_wc_params(0, self.atoms)
        
        # Create progress bar
        pbar = tqdm(range(self.steps), desc="Running MC", unit="steps")
        
        # Initialize energy history for convergence checking
        energy_history = []
        converged = False
        
        for step in pbar:
            if converged:
                break
            
            # 1. Randomly pick two different site indices
            site1 = self.rng.integers(0, n_atoms)
            site2 = self.rng.integers(0, n_atoms)
            if site1 == site2:
                continue
                
            old_symbol_1 = self.atoms[site1].symbol
            old_symbol_2 = self.atoms[site2].symbol
            
            # 2. If they're the same species, swapping does nothing
            if old_symbol_1 == old_symbol_2:
                continue
                
            # 3. Propose a swap
            self.atoms[site1].symbol = old_symbol_2
            self.atoms[site2].symbol = old_symbol_1
            
            # 4. Compute the energy
            trial_energy = self.atoms.get_potential_energy()
            delta_e = trial_energy - self.current_energy
            
            accepted = False
            if delta_e <= 0:
                # Accept
                self.current_energy = trial_energy
                accepted = True
            else:
                # Accept with p = exp(-ΔE / (k_B T))
                p_accept = np.exp(-delta_e / (self.k_b * self.temperature))
                if self.rng.random() < p_accept:
                    self.current_energy = trial_energy
                    accepted = True
                else:
                    # Revert swap
                    self.atoms[site1].symbol = old_symbol_1
                    self.atoms[site2].symbol = old_symbol_2
            
            # Track if needed
            if self.tracker:
                if step % self.tracker.energy_freq == 0:
                    self.tracker.record_energy(step + 1, self.current_energy)
                if step % self.tracker.wc_freq == 0:
                    self.tracker.record_wc_params(step + 1, self.atoms)
            
            # Update energy history and check convergence
            energy_history.append(self.current_energy)
            if len(energy_history) > convergence_window:
                energy_history.pop(0)  # Remove oldest energy
                
                if len(energy_history) == convergence_window:
                    energy_range = max(energy_history) - min(energy_history)
                    energy_range_per_atom = energy_range / n_atoms
                    
                    # Update progress bar with convergence info
                    pbar.set_postfix({
                        'energy': f"{self.current_energy:.3f} eV",
                        'Δe/atom': f"{energy_range_per_atom:.6f} eV"
                    })
                    
                    # Check for convergence
                    if energy_range_per_atom < energy_threshold:
                        print(f"\nConverged! Energy change per atom ({energy_range_per_atom:.6f} eV) "
                              f"below threshold ({energy_threshold:.6f} eV) "
                              f"over {convergence_window} steps")
                        converged = True
            else:
                # Just update energy if we haven't filled the window yet
                pbar.set_postfix({'energy': f"{self.current_energy:.3f} eV"})
        
        pbar.close()
        return self.atoms
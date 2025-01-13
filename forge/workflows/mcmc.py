import numpy as np
from ase import Atoms
from copy import deepcopy
import random
from datetime import datetime

class MonteCarloAlloySampler:
    """
    Perform Monte Carlo (MC) swaps in a small BCC supercell for SRO refinement.

    Parameters
    ----------
    atoms : ase.Atoms
        Initial structure (e.g., 128-atom supercell).
    calculator : ase.calculators.calculator.Calculator
        An ASE-compatible calculator, e.g. a MACE or Allegro potential.
    temperature : float
        Temperature in Kelvin for MC sampling.
    steps : int
        Number of MC steps (swaps per atom or total swaps depending on usage).
    allowed_species : list of str
        List of elements that can appear in the structure (e.g., ["V", "Cr", "Ti", "W", "Zr"]).
    rng_seed : int
        Random seed for reproducibility.

    Example
    -------
    >>> from forge.workflows.mcmc import MonteCarloAlloySampler
    >>> from mace.calculators.mace import MACECalculator
    >>> atoms = some_128_atom_supercell  # e.g. prepared or read from DB
    >>> calc = MACECalculator(model_paths=["/path/to/mace.model"])
    >>> mc = MonteCarloAlloySampler(atoms, calc, temperature=1000, steps=640, allowed_species=["V","Cr","Ti","W","Zr"])
    >>> final_atoms = mc.run_mcmc()
    """

    def __init__(self, atoms, calculator, temperature=1000.0, steps=1000,
                 allowed_species=None, rng_seed=42):
        self.atoms = atoms
        self.calculator = calculator
        self.temperature = temperature
        self.steps = steps
        self.allowed_species = allowed_species or list(set(atoms.get_chemical_symbols()))
        self.rng = np.random.default_rng(rng_seed)
        self.k_b = 8.617333262e-5  # Boltzmann constant in eV/K

        # Attach calculator
        self.atoms.calc = self.calculator
        # Initial energy
        self.current_energy = self.atoms.get_potential_energy()

    def run_mcmc(self):
        """
        Perform Metropolis-Hastings Monte Carlo swaps in place.
        Returns the final Atoms object.
        """
        n_atoms = len(self.atoms)
        for step in range(self.steps):
            # Randomly pick an atom to swap
            site_index = self.rng.integers(0, n_atoms)
            old_symbol = self.atoms[site_index].symbol

            # Randomly choose a new species (that might be the same as old_symbol)
            new_symbol = random.choice(self.allowed_species)

            if new_symbol == old_symbol:
                # No change -> skip
                continue

            # Propose the swap
            self.atoms[site_index].symbol = new_symbol
            trial_energy = self.atoms.get_potential_energy()

            # Metropolis acceptance
            delta_e = trial_energy - self.current_energy
            if delta_e <= 0:
                # Accept immediately
                self.current_energy = trial_energy
            else:
                # Accept with p = exp(-ΔE / (k_B * T))
                p_accept = np.exp(-delta_e / (self.k_b * self.temperature))
                if self.rng.random() < p_accept:
                    self.current_energy = trial_energy
                else:
                    # Revert swap
                    self.atoms[site_index].symbol = old_symbol

        return self.atoms 
import numpy as np
from ase import Atoms
from copy import deepcopy
import random
from datetime import datetime


# TODO: make the MCMC not change the composition of the atoms object, right now it does
# so the composition just goes to W... 

class MonteCarloAlloySampler:
    """
    Perform Monte Carlo (MC) swaps in a BCC supercell for SRO refinement, but keep composition fixed.
    This is an 'off-lattice' approach where we only swap species labels, leaving positions unchanged.
    ...
    """
    def __init__(self, atoms, calculator, temperature=1000.0, steps=1000,
                 allowed_species=None, rng_seed=42):
        self.atoms = atoms.copy() # make a copy of the atoms object
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
        Perform Metropolis-Hastings Monte Carlo swaps, preserving composition.
        Returns the final Atoms object.
        """
        n_atoms = len(self.atoms)
        for step in range(self.steps):
            # 1. Randomly pick two different site indices
            site1 = self.rng.integers(0, n_atoms)
            site2 = self.rng.integers(0, n_atoms)
            if site1 == site2:
                # same atom, skip
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
            if delta_e <= 0:
                # Accept
                self.current_energy = trial_energy
            else:
                # Accept with p = exp(-ΔE / (k_B T))
                p_accept = np.exp(-delta_e / (self.k_b * self.temperature))
                if self.rng.random() < p_accept:
                    self.current_energy = trial_energy
                else:
                    # Revert swap
                    self.atoms[site1].symbol = old_symbol_1
                    self.atoms[site2].symbol = old_symbol_2
        return self.atoms
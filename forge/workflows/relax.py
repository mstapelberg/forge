"""
Structure optimization module with configurable parameters.

This module provides functionality for relaxing atomic structures using various
optimization algorithms with control over cell relaxation, convergence criteria,
and verbosity of output.

Inspired by the ASE relax function and the M3GNet relax function.
"""

import os
import logging
from typing import Optional, Union, Literal

from ase.filters import FrechetCellFilter
from ase.optimize import FIRE, LBFGS, QuasiNewton, MDMin
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator


# Set up logging
logger = logging.getLogger(__name__)


def relax(
    atoms: Atoms,
    calculator: Calculator,
    relax_cell: bool = False,
    fmax: float = 0.04,
    steps: int = 100,
    optimizer: Literal["FIRE", "LBFGS", "BFGS", "MDMin"] = "FIRE",
    trajectory: Optional[str] = None,
    logfile: Optional[Union[str, os.PathLike]] = None,
    verbose: int = 1,
    interval: int = 10,
    **optimizer_kwargs
) -> Atoms:
    """
    Perform structure optimization on an atomic system.
    
    Parameters
    ----------
    atoms : Atoms
        The atomic structure to optimize
    calculator : Calculator
        ASE calculator to use for energy and force calculations
    relax_cell : bool, default=False
        Whether to relax the unit cell along with atomic positions
    fmax : float, default=0.04
        Maximum force criterion for convergence (eV/Å)
    steps : int, default=100
        Maximum number of optimization steps
    optimizer : str, default="FIRE"
        Optimization algorithm to use. Options: "FIRE", "LBFGS", "BFGS", "MDMin"
    trajectory : str, optional
        Filename to write trajectory to during optimization
    logfile : str or file-like, optional
        File to write optimization log to (use '-' for stdout)
    verbose : int, default=1
        Verbosity level:
        0 - Silent operation
        1 - Basic information (iterations, energy)
        2 - Detailed information (forces, stress)
    interval : int, default=10
        Interval for printing status updates when verbose > 0
    **optimizer_kwargs
        Additional keyword arguments to pass to the optimizer
        
    Returns
    -------
    Atoms
        Optimized atomic structure
    
    Raises
    ------
    ValueError
        If an unsupported optimizer is specified
    """
    # Configure logging based on verbosity
    if verbose == 0:
        log_level = logging.ERROR
        logfile = None
        trajectory = None
    elif verbose == 1:
        log_level = logging.INFO
        if logfile is None and trajectory is None:
            logfile = '-'  # stdout
    else:  # verbose >= 2
        log_level = logging.DEBUG
        if logfile is None:
            logfile = '-'  # stdout
    
    logger.setLevel(log_level)
    
    # Create a copy of atoms to avoid modifying the input
    atoms_copy = atoms.copy()
    
    # Set calculator on the atoms object first
    atoms_copy.calc = calculator
    
    # Apply cell filter if requested
    if relax_cell:
        system = FrechetCellFilter(atoms_copy)
        logger.info("Relaxing atomic positions and unit cell")
    else:
        system = atoms_copy
        logger.info("Relaxing atomic positions with fixed unit cell")
    
    # Select optimizer
    optimizer_kwargs.update({
        'logfile': logfile,
        'trajectory': trajectory,
    })
    
    if optimizer == "FIRE":
        opt = FIRE(system, **optimizer_kwargs)
    elif optimizer == "LBFGS":
        opt = LBFGS(system, **optimizer_kwargs)
    elif optimizer == "BFGS":
        opt = QuasiNewton(system, **optimizer_kwargs)
    elif optimizer == "MDMin":
        opt = MDMin(system, **optimizer_kwargs)
    else:
        raise ValueError(f"Optimizer '{optimizer}' not supported. Choose from: FIRE, LBFGS, BFGS, MDMin")
    
    # Configure optimizer verbosity
    if verbose > 0:
        opt.attach(lambda: _log_status(system, verbose), interval=interval)
    
    # Run optimization
    logger.info(f"Starting optimization with {optimizer}, fmax={fmax}, max steps={steps}")
    opt.run(fmax=fmax, steps=steps)
    
    # Log final results
    if verbose > 0:
        logger.info(f"Optimization completed after {opt.get_number_of_steps()} steps")
        logger.info(f"Final energy: {system.get_potential_energy():.6f} eV")
        logger.info(f"Maximum force: {max(abs(system.get_forces().flatten())):.6f} eV/Å")
    
    # Return the optimized structure
    # If we used FrechetCellFilter, we need to return the atoms object
    if relax_cell:
        return system.atoms
    return system


def _log_status(system, verbose):
    """Log the current status of the optimization."""
    atoms = system.atoms if hasattr(system, 'atoms') else system
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    fmax = max(abs(forces.flatten()))
    
    logger.info(f"Energy: {energy:.6f} eV, Max force: {fmax:.6f} eV/Å")
    
    if verbose >= 2:
        # Log more detailed information
        if hasattr(system, 'atoms'):  # FrechetCellFilter case
            stress = atoms.get_stress()
            logger.debug(f"Stress tensor (GPa): {stress}")
            logger.debug(f"Cell parameters: {atoms.get_cell_lengths_and_angles()}")
        
        logger.debug("Forces (eV/Å):")
        for i, force in enumerate(forces):
            logger.debug(f"  Atom {i}: {force[0]:.6f} {force[1]:.6f} {force[2]:.6f}")
from mace.calculators.mace import MACECalculator
from forge.analysis.composition import CompositionAnalyzer
from forge.workflows.mcmc import MonteCarloAlloySampler

import random
import torch
import numpy as np
from math import ceil

from ase.io import read, write
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from ase.mep import DyNEB

def relax(atoms, calc, relax_cell=False, steps=100, fmax=0.01):
    new_atoms = atoms.copy()
    new_atoms.calc = calc
    if relax_cell:
        fcf = FrechetCellFilter(new_atoms)
        opt = FIRE(fcf)
    else:
        opt = FIRE(new_atoms)

    opt.run(steps=steps, fmax=fmax)
    return opt.atoms


from ase.build import bulk
import numpy as np
from ase.neighborlist import NeighborList
import numpy as np
from ase.neighborlist import NeighborList

def get_neighbors_cutoff(atoms, index, nn_cutoff=2.8, nnn_cutoff=3.2):
    """
    Get nearest and next-nearest neighbors using distance cutoffs.
    
    Parameters:
    -----------
    atoms : ase.Atoms
        The atomic structure
    index : int
        Index of the atom to find neighbors for
    nn_cutoff : float
        Cutoff radius for nearest neighbors (default: 2.8 Å for V)
    nnn_cutoff : float
        Cutoff radius for next-nearest neighbors (default: 3.2 Å for V)
    
    Returns:
    --------
    dict
        Dictionary containing nearest and next-nearest neighbor information
    """
    # Create a neighbor list with the larger cutoff
    cutoff = nnn_cutoff
    nl = NeighborList([cutoff/2] * len(atoms), skin=0.0, self_interaction=False, bothways=True)
    nl.update(atoms)
    
    # Get all neighbors and distances
    indices, offsets = nl.get_neighbors(index)
    positions = atoms.positions
    cell = atoms.get_cell()
    distances = []
    
    for i, offset in zip(indices, offsets):
        pos_i = positions[i] + np.dot(offset, cell)
        dist = np.linalg.norm(pos_i - positions[index])
        distances.append(dist)
    
    distances = np.array(distances)
    
    # Separate into NN and NNN based on distances
    nn_mask = distances <= nn_cutoff
    nnn_mask = (distances > nn_cutoff) & (distances <= nnn_cutoff)
    
    # Sort both sets by distance
    nn_indices = indices[nn_mask]
    nn_distances = distances[nn_mask]
    nn_sort = np.argsort(nn_distances)
    
    nnn_indices = indices[nnn_mask]
    nnn_distances = distances[nnn_mask]
    nnn_sort = np.argsort(nnn_distances)
    
    return {
        'nn_indices': nn_indices[nn_sort],
        'nn_distances': nn_distances[nn_sort],
        'nnn_indices': nnn_indices[nnn_sort],
        'nnn_distances': nnn_distances[nnn_sort]
    }


def create_start_and_end_points(atoms, start_index, end_index):
    start_atoms = atoms.copy()
    end_atoms = atoms.copy()
    start_position = start_atoms.positions[start_index]
    #end_position = end_atoms.positions[end_index]

    start_atoms.pop(start_index)
    end_atoms.pop(start_index)

    end_atoms.positions[end_index] = start_position

    return start_atoms, end_atoms

def neb_relax_atoms(start_atoms, end_atoms, num_images, model_path, fmax=0.01):
    """Relax the atoms with the MACE calculator"""
    images = [start_atoms.copy() for _ in range(num_images)+1]
    images.append(end_atoms.copy())

    neb = DyNEB(images, climb=True)
    neb.interpolate(mic=True)

    for image in images:
        image.calc = MACECalculator(model_path=[model_path], device='cuda', default_dtype='float32')

    opt = FIRE(neb)
    opt.run(steps=150, fmax=fmax)

    return neb

# check if cuda is available
if torch.cuda.is_available():
    device = 'cuda'
    use_cueq = True
else:
    device = 'cpu'
    use_cueq = False

atoms = read('final_Cr132Ti177V3853W221Zr11.xyz')

model_path = '../potentials/gen_6_model_0_L0_isolated-2026-01-16_stagetwo.model'

calc = MACECalculator(model_paths=[model_path],
                      device=device,
                      default_dtype='float32',
                      use_cueq=use_cueq)

relaxed_perf = relax(atoms, calc, relax_cell=True, steps=250, fmax=0.01)

# pick a random atom index and get its neighbors
random_atom_index = random.randint(0, len(relaxed_perf)-1)
results = get_neighbors_cutoff(relaxed_perf, random_atom_index)

print("Nearest neighbors:")
print(f"Found {len(results['nn_indices'])} neighbors")
for idx, dist in zip(results['nn_indices'], results['nn_distances']):
    print(f"Atom {idx}: {dist:.3f} Å")

print("\nNext-nearest neighbors:")
print(f"Found {len(results['nnn_indices'])} neighbors")
for idx, dist in zip(results['nnn_indices'], results['nnn_distances']):
    print(f"Atom {idx}: {dist:.3f} Å")


start_atoms, end_atoms = create_start_and_end_points(relaxed_perf, random_atom_index, results['nn_indices'][0])

rel_start = relax(start_atoms, calc, relax_cell=False, steps=150, fmax=0.01)
rel_end = relax(end_atoms, calc, relax_cell=False, steps=150, fmax=0.01)

neb = neb_relax_atoms(rel_start, rel_end, num_images=10, model_path=model_path, fmax=0.01)

energies = []
for image in neb.images:
    energies.append(image.get_potential_energy())

barrier = np.max(energies) - energies[0]

print(f"Removed atom was {atoms.symbols[random_atom_index]} at site {random_atom_index}")
print(f"Barrier for {random_atom_index} to {results['nn_indices'][0]} is {barrier:.3f} eV")








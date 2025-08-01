import torch
import ase
from pathlib import Path

from nequip.data import AtomicDataDict
from nequip.data.ase import from_ase
from nequip.data.transforms import (
    NeighborListTransform,
    ChemicalSpeciesToAtomTypeMapper,
)
from forge.workflows.allegro_utils.pair_potential import NLH

def main():
    """Debug the NLH potential calculation."""
    # 1. Set up a simple test case (a C-O dimer)
    # Distance is 1.1 Angstroms. Cell is large to avoid PBC issues.
    atoms = ase.Atoms(['V','V'], positions=[(0, 0, 0), (0, 0, 1.4)], cell=[20, 20, 20], pbc=False)

    # 2. Convert to nequip's data format and apply transforms
    chemical_symbols = ['V','V']
    type_names = chemical_symbols
    
    data = from_ase(atoms=atoms)

    # Apply standard nequip transforms
    type_mapper_transform = ChemicalSpeciesToAtomTypeMapper(
        chemical_symbols=chemical_symbols
    )
    data = type_mapper_transform(data)

    # 3. Add batch dimension and generate neighbor list (edges)
    data = AtomicDataDict.with_batch_(data)
    nbl_transform = NeighborListTransform(r_max=2.0)
    data = nbl_transform(data)

    print("--- Input Data ---")
    print(f"Edge index: {data[AtomicDataDict.EDGE_INDEX_KEY]}")
    print(f"Atom types: {data[AtomicDataDict.ATOM_TYPE_KEY]}")
    print("-" * 20)

    # 4. Instantiate the NLH model
    nlh_model = NLH(
        type_names=type_names,
        chemical_species=chemical_symbols,
        units="metal",
    )

    # 5. Run the forward pass to get the per-atom energy contribution
    output_data = nlh_model(data)
    per_atom_energy = output_data[AtomicDataDict.PER_ATOM_ENERGY_KEY]
    total_pair_energy = torch.sum(per_atom_energy)

    print("--- Output ---")
    print(f"Edge lengths (r): {output_data[AtomicDataDict.EDGE_LENGTH_KEY].flatten()} Angstroms")
    print(f"Per-atom energy contribution from NLH: {per_atom_energy.flatten()} eV")
    print(f"Total pair energy for the dimer: {total_pair_energy} eV")

if __name__ == "__main__":
    main() 
import torch
import numpy as np
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace import data
from mace.tools import AtomicNumberTable, torch_geometric
import ase.io

class AtomicDataLoader:
    """Module to load atomic data from XYZ files"""
    
    def __init__(self, cutoff=5.0):
        self.cutoff = cutoff
        
    def load_from_xyz(self, xyz_file, batch_size=1):
        """Load atomic data from XYZ file
        
        Args:
            xyz_file: Path to XYZ file
            batch_size: Batch size for data loader
            
        Returns:
            data_loader: PyTorch Geometric DataLoader
            z_table: AtomicNumberTable
        """
        # Load atoms from XYZ file
        atoms_list = ase.io.read(xyz_file, index=":")
        
        # Create atomic number table from atoms
        atomic_numbers = np.concatenate([atoms.numbers for atoms in atoms_list])
        z_table = AtomicNumberTable(list(set(atomic_numbers)))
        
        # Create atomic data from atoms
        atomic_data = [
            data.AtomicData.from_config(
                data.config_from_atoms(atoms),
                z_table=z_table,
                cutoff=self.cutoff
            ) for atoms in atoms_list
        ]
        
        # Create data loader
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=atomic_data,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        
        return data_loader, z_table
    
    def get_batch_and_system_data(self, xyz_file, device="cuda", batch_idx=0):
        """Get a batch and system data from XYZ file
        
        Args:
            xyz_file: Path to XYZ file
            device: Device to put tensors on
            batch_idx: Index of batch to return
            
        Returns:
            batch_dict: Dictionary of batch data
            system_data: Dictionary with additional system information
        """
        data_loader, z_table = self.load_from_xyz(xyz_file)
        
        # Get batch
        for i, batch in enumerate(data_loader):
            if i == batch_idx:
                batch = batch.to(device)
                batch_dict = batch.to_dict()
                
                # Store system info
                system_data = {
                    "z_table": z_table,
                    "num_atoms": batch.num_nodes,
                    "atomic_numbers": z_table.zs,
                }
                return batch_dict, system_data
        
        raise IndexError(f"Batch index {batch_idx} out of range")

class CUEQMACEInterface(torch.nn.Module):
    def __init__(self, model_path, device="cuda"):
        super().__init__()
        # Load your trained model
        self.original_model = torch.load(model_path, map_location=device)
        
        # Convert to CUEQ version if available
        try:
            import cuequivariance as cue
            self.model = run_e3nn_to_cueq(self.original_model)
            self.using_cueq = True
            print("Using CUDA-Equivariance acceleration")
        except ImportError:
            self.model = self.original_model
            self.using_cueq = False
            print("CUDA-Equivariance not available, using standard model")
        
        self.model.to(device)
    
    def forward(self, batch_dict, **kwargs):
        return self.model(batch_dict, **kwargs)
    
    def get_energy_forces_stress(self, batch_dict, **kwargs):
        """Get energy, forces and stress for a system
        
        Args:
            batch_dict: Dictionary with required model inputs 
            **kwargs: Additional parameters for the model
            
        Returns:
            energy: Total energy
            forces: Forces on each atom [n_atoms, 3]
            stress: Stress tensor [3, 3]
        """
        # Make a copy and require gradients for positions
        positions = batch_dict["positions"].clone().requires_grad_(True)
        batch_dict = {**batch_dict, "positions": positions}
        
        # Prepare cell and add displacement for stress calculation
        num_graphs = batch_dict["ptr"].numel() - 1 if "ptr" in batch_dict else 1
        
        # Default cell if not provided
        cell = batch_dict.get("cell", torch.eye(3, device=positions.device))
        cell = cell.requires_grad_(True)
        batch_dict["cell"] = cell
        
        # Create displacement for stress calculation
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=positions.dtype,
            device=positions.device,
            requires_grad=True
        )
        
        # Run model with compute_stress=True
        output = self.model(
            batch_dict, 
            training=False, 
            compute_force=True, 
            compute_stress=True,
            compute_virials=True,
            displacement=displacement,
            **kwargs
        )
        
        energy = output["energy"]
        forces = output["forces"]
        
        # Get stress from model output if available, otherwise compute it
        if "stress" in output and output["stress"] is not None:
            stress = output["stress"]
        else:
            # Calculate stress from virial 
            stress = output.get("virials", None)
            if stress is not None and "cell" in batch_dict:
                # Scale by volume
                cell_matrix = cell.view(-1, 3, 3)
                volume = torch.abs(torch.linalg.det(cell_matrix))
                stress = stress / volume.view(-1, 1, 1)
        
        return energy, forces, stress
    
    @classmethod
    def from_xyz_and_model(cls, model_path, xyz_file, device="cuda"):
        """Create interface from XYZ file and model path
        
        Args:
            model_path: Path to model file
            xyz_file: Path to XYZ file
            device: Device to use
            
        Returns:
            interface: CUEQMACEInterface instance
            batch_dict: Dictionary with batch data
            system_data: Dictionary with system info
        """
        # Create interface
        interface = cls(model_path, device)
        
        # Load atomic data
        data_loader = AtomicDataLoader()
        batch_dict, system_data = data_loader.get_batch_and_system_data(
            xyz_file, device=device
        )
        
        return interface, batch_dict, system_data
    



import torch
from ase.calculators.mace import MACECalculator
import ase.io
import time

# Path to your model and XYZ file
model_path = "../potentials/mace_gen_6_ensemble/gen_7_model_0-2025-02-12_stagetwo_compiled.model"
xyz_file = "final_Cr132Ti177V3853W221Zr11.xyz"
device = "cuda"

# 1. Using custom interface
interface, batch_dict, system_info = CUEQMACEInterface.from_xyz_and_model(
    model_path, xyz_file, device
)

# Time the computation
start = time.time()
energy, forces, stress = interface.get_energy_forces_stress(batch_dict)
end = time.time()
custom_time = end - start

print("Custom Interface Results:")
print(f"Energy: {energy.item()} eV")
print(f"Forces shape: {forces.shape}")
print(f"Stress shape: {stress.shape if stress is not None else None}")
print(f"Computation time: {custom_time:.4f} s")

# 2. Using ASE interface
atoms = ase.io.read(xyz_file)
calc = MACECalculator(model_path=model_path, device=device)
atoms.calc = calc

# Time the computation
start = time.time()
ase_energy = atoms.get_potential_energy()
ase_forces = atoms.get_forces()
ase_stress = atoms.get_stress()
end = time.time()
ase_time = end - start

print("\nASE Interface Results:")
print(f"Energy: {ase_energy} eV")
print(f"Forces shape: {ase_forces.shape}")
print(f"Stress shape: {ase_stress.shape if ase_stress is not None else None}")
print(f"Computation time: {ase_time:.4f} s")

print(f"\nSpeedup: {ase_time/custom_time:.2f}x")
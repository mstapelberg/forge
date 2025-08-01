import itertools, json, random, numpy as np
from ase.build import bulk, make_supercell
from ase.io import write
import os, sys
from pathlib import Path

# Add forge to path to import db_to_vasp
forge_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(forge_root))
from forge.workflows.db_to_vasp import prepare_vasp_job_from_ase

def elasticity_grid(elements, ratios,
                    strains=(-0.03, 0.03),
                    sizes=('primitive', '3x3x3'),
                    vasp_profile_name='static',  # Add VASP profile parameter
                    hpc_profile_name='PSFC-GPU',  # Add HPC profile parameter
                    base_output_dir='elastic_jobs'):  # Add base output directory
    """Generate strained cells for elasticity training and create VASP jobs."""
    assert abs(sum(ratios)-1)<1e-6
    
    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    job_count = 0
    for size in sizes:
        base = bulk('V', 'bcc', a=3.01, cubic=True)        # lattice parameter is overwritten later
        if size != 'primitive':
            n = tuple(map(int,size.split('x')))
            P = np.diag(n)
            base = make_supercell(base, P)
        n_atoms = len(base)
        species = random.choices(elements, ratios, k=n_atoms)
        base.set_chemical_symbols(species)

        for eps in np.linspace(*strains, num=7):
            for α,β in itertools.combinations_with_replacement(range(3),2):
                strain = np.eye(3)
                strain[α,β] += eps; strain[β,α] += eps*(α!=β)
                strained = base.copy()
                strained.set_cell(strained.cell @ strain, scale_atoms=True)
                
                # Create job directory name
                tag = f"{size}_e{α}{β}_{eps:+.3f}"
                job_dir = os.path.join(base_output_dir, tag)
                
                # Add metadata to atoms object
                strained.info['elasticity'] = {
                    'size': size,
                    'strain_component': f'e{α}{β}',
                    'strain_value': eps,
                    'elements': elements,
                    'ratios': ratios
                }
                
                # Create VASP job instead of just writing POSCAR
                prepare_vasp_job_from_ase(
                    atoms=strained,
                    vasp_profile_name=vasp_profile_name,
                    hpc_profile_name=hpc_profile_name,
                    output_dir=job_dir,
                    auto_kpoints=True,  # Auto-determine k-points for each structure
                    job_name=tag
                )
                
                job_count += 1
    
    print(f"Created {job_count} VASP jobs for elasticity calculations in '{base_output_dir}/'")
    print(f"Using VASP profile: {vasp_profile_name}")
    print(f"Using HPC profile: {hpc_profile_name}")

# Example usage
if __name__ == "__main__":
    # Pure elements: V, Cr, Ti, W, Zr
    pure_elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
    
    for element in pure_elements:
        print(f"\nCreating jobs for pure {element}...")
        elasticity_grid(
            elements=[element],
            ratios=[1.0],  # 100% pure element
            strains=(-0.03, 0.03),  # -3% to +3% strain
            sizes=('primitive', '3x3x3'),  # primitive and 3x3x3 supercells
            vasp_profile_name='static',  # Change this to your desired VASP profile
            hpc_profile_name='PSFC-GPU',
            base_output_dir=f'../data/elastic_jobs_total/elastic_jobs_pure_{element}'
        )
    
    # V-rich alloy: V=0.8, Cr=Ti=W=Zr=0.05 each
    print(f"\nCreating jobs for V-rich alloy (V=0.8, others=0.05)...")
    elasticity_grid(
        elements=['V', 'Cr', 'Ti', 'W', 'Zr'],
        ratios=[0.8, 0.05, 0.05, 0.05, 0.05],  # V-rich composition
        strains=(-0.03, 0.03),  # -3% to +3% strain
        sizes=('primitive', '3x3x3'),  # primitive and 3x3x3 supercells
        vasp_profile_name='static',  # Change this to your desired VASP profile
        hpc_profile_name='PSFC-GPU',
        base_output_dir='../data/elastic_jobs_total/elastic_jobs_V_rich_alloy'
    )

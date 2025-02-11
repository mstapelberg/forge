import numpy as np 

from ase.build import bulk 
from forge.workflows.mcmc import MonteCarloAlloySampler
from mace.calculators.mace import MACECalculator 

from forge.analysis.composition import CompositionAnalyzer
from math import ceil, floor

from ase.io import write

num_atoms = 2*13**3

x_zr = round(ceil(0.0025 * num_atoms)/num_atoms, 4)
x_cr = round(ceil(0.03 * num_atoms)/num_atoms, 4)
x_ti = round(ceil(0.04 * num_atoms)/num_atoms, 4)
x_w = round(ceil(0.05 * num_atoms)/num_atoms, 4)
x_v = round(1 - x_zr - x_cr - x_ti - x_w, 4)


composition = {
    'V' : x_v,
    'Cr' : x_cr,
    'Ti' : x_ti,
    'W' : x_w,
    'Zr' : x_zr
}

# if sum of composition is not 1, adjust x_v
if sum(composition.values()) != 1:
    x_v = 1 - sum(composition.values())


print(f"Composition: {composition}")

analyzer = CompositionAnalyzer()
atoms = analyzer.create_random_alloy(composition = composition, 
                                         crystal_type = 'bcc', 
                                         dimensions=[13,13,13], 
                                         lattice_constant = 3.01,
                                         balance_element = 'V', 
                                         cubic=True)

print(atoms)

model_path = '../potentials/gen_6_model_0_L0_isolated-2026-01-16_stagetwo.model'

calc = MACECalculator(model_paths=[model_path],
                      device="cuda",
                      default_dtype="float32",
                      enable_cueq=True)

atoms.calc = calc

temperature = 600+273.15
steps_per_atom = 100
total_swaps = steps_per_atom * len(atoms)

mc_sampler = MonteCarloAlloySampler(
    atoms=atoms,
    calculator=calc,
    temperature=temperature,
    steps=total_swaps,
)

final_atoms = mc_sampler.run_mcmc()

write(f'final_{final_atoms.get_chemical_formula()}.xyz', final_atoms)
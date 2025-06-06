import os
import random
import numpy
from ase.io import read as ase_read
from ase import Atoms


# Define default motif types (without atom counts)
DEFAULT_MOTIF_TYPES = [
    'fcc',
    'hcp',
    'A15',
    'C15',
    'dia',
    'vacancy',
    'di-vacancy',
    'tri-vacancy',
    'sia',
    'di-sia',
    'surface_100',
    'surface_110',
    'surface_111',
    'surface_112',
    'surf_liquid',
    'liquid',
    'short_range',
    'gamma_surface'
]

def _normalize_composition(composition_dict):
    """
    Normalizes a composition dictionary to ensure fractions sum to 1.0.
    Handles zero sums and ensures non-negative values.

    Args:
        composition_dict (dict): {element_symbol: fraction}

    Returns:
        dict: Normalized {element_symbol: fraction}
    """
    if not composition_dict:
        raise ValueError("Composition dictionary cannot be empty.")

    # Ensure non-negative
    norm_comp = {el: max(0.0, val) for el, val in composition_dict.items()}

    total_frac = sum(norm_comp.values())
    if abs(total_frac - 0.0) < 1e-9: # All zero input
        return {el: 0.0 for el in norm_comp}
    if abs(total_frac - 1.0) > 1e-6: # If not already sum to 1
        norm_comp = {el: val / total_frac for el, val in norm_comp.items()}

    return norm_comp


def _calculate_single_stoichiometry(composition_fractions, N_total, base_rounding_method='ceil'):
    """
    Calculates one set of integer atom counts for a composition and N_total.
    Adjusts counts to sum exactly to N_total.

    Args:
        composition_fractions (dict): {element: fraction}, must sum to 1.0.
        N_total (int): Total number of atoms.
        base_rounding_method (str): 'ceil' or 'floor' to determine initial counts.

    Returns:
        dict: {element: count}
    """
    if N_total == 0:
        return {el: 0 for el in composition_fractions}
        
    elements = sorted(list(composition_fractions.keys())) # Sort for deterministic behavior
    ideal_counts_float = {el: composition_fractions[el] * N_total for el in elements}
    
    current_counts = {}
    if base_rounding_method == 'ceil':
        current_counts = {el: int(numpy.ceil(ideal_counts_float[el])) for el in elements}
    elif base_rounding_method == 'floor':
        current_counts = {el: int(numpy.floor(ideal_counts_float[el])) for el in elements}
    else:
        raise ValueError("base_rounding_method must be 'ceil' or 'floor'")

    current_sum = sum(current_counts.values())

    # Adjust sum to N_total
    while current_sum != N_total:
        if current_sum > N_total: # Need to decrement
            # Find element to decrement: one with largest (current_count - ideal_float_count)
            # and current_count > 0
            best_el_to_mod = None
            max_diff = -float('inf')
            for el in elements:
                if current_counts[el] > 0:
                    diff = current_counts[el] - ideal_counts_float[el]
                    if diff > max_diff:
                        max_diff = diff
                        best_el_to_mod = el
                    # Tie-breaking: if diff is same, prefer element with larger ideal count (more abundant)
                    elif abs(diff - max_diff) < 1e-9 and ideal_counts_float[el] > ideal_counts_float.get(best_el_to_mod, -float('inf')):
                        best_el_to_mod = el

            if best_el_to_mod:
                current_counts[best_el_to_mod] -= 1
                current_sum -= 1
            else: # Should not happen if sum > N_total and N_total >= 0
                break 
        else: # current_sum < N_total, need to increment
            # Find element to increment: one with largest (ideal_float_count - current_count)
            best_el_to_mod = None
            max_diff = -float('inf')
            for el in elements:
                diff = ideal_counts_float[el] - current_counts[el]
                if diff > max_diff:
                    max_diff = diff
                    best_el_to_mod = el
                 # Tie-breaking: if diff is same, prefer element with larger ideal count
                elif abs(diff - max_diff) < 1e-9 and ideal_counts_float[el] > ideal_counts_float.get(best_el_to_mod, -float('inf')):
                    best_el_to_mod = el
            
            if best_el_to_mod:
                current_counts[best_el_to_mod] += 1
                current_sum += 1
            else: # Should not happen if sum < N_total
                break
    
    # Final check, fallback if primary logic failed to sum to N_total (very unlikely)
    if sum(current_counts.values()) != N_total:
        # Fallback: floor all, then distribute remainder based on largest fractional parts
        current_counts = {el: int(numpy.floor(ideal_counts_float[el])) for el in elements}
        remainder = N_total - sum(current_counts.values())
        if remainder > 0:
            frac_parts = {el: ideal_counts_float[el] - current_counts[el] for el in elements}
            sorted_by_frac = sorted(elements, key=lambda el_key: frac_parts[el_key], reverse=True)
            for i in range(remainder):
                current_counts[sorted_by_frac[i % len(elements)]] += 1
    
    if sum(current_counts.values()) != N_total : # If still not summing, something is very wrong
         raise RuntimeError(f"Failed to calculate consistent stoichiometry for {composition_fractions} with N_total={N_total}")

    return current_counts


def _calculate_atom_counts_for_composition(composition_fractions, N_total):
    """
    Calculates the number of atoms of each element for a target composition and total atom count.

    Args:
        composition_fractions (dict): {element_symbol: fraction}. Must sum to ~1.0.
        N_total (int): Total number of atoms in the structure.

    Returns:
        list: A list of dictionaries, where each dict is {element_symbol: count}.
              Contains one dict (ceil-based), or two if N_total < 20 and floor-based is different.
    """
    if N_total <= 0: return [{el:0 for el in composition_fractions}]

    # Normalize fractions (should already be done by _normalize_composition, but as a safeguard)
    norm_comp = _normalize_composition(composition_fractions)

    results = []

    # Method 1: Adjusted Ceiling (primary)
    counts_ceil_based = _calculate_single_stoichiometry(norm_comp, N_total, base_rounding_method='ceil')
    results.append(counts_ceil_based)

    # Method 2: Adjusted Flooring (if N_total < 20 and it yields a different result)
    # Condition: "if going from X to X+1 changes the composition by more than 5%" -> 1/N > 0.05 => N < 20
    if N_total < 20:
        counts_floor_based = _calculate_single_stoichiometry(norm_comp, N_total, base_rounding_method='floor')
        
        # Add if it's different from the ceil_based one
        is_different = True
        if results: 
            if len(results[0]) == len(counts_floor_based) and \
               all(results[0].get(k) == v for k, v in counts_floor_based.items()):
                is_different = False
        
        if is_different:
            results.append(counts_floor_based)
            
    return results


def _load_motif_template(motif_type, motif_path):
    """
    Loads a motif template structure (ASE Atoms object) from the specified motifs directory.
    
    Args:
        motif_type (str): The type of defect motif (e.g., 'vacancy', 'sia').
        motif_path (str): Path to the motifs directory.

    Returns:
        tuple: (ase.Atoms object, int) or (None, None) if not found/invalid
    """
    for ext in ['.cif', '.vasp', '.poscar', '.xyz', '.gen', '.traj', '.extxyz']: 
        filepath = os.path.join(motif_path, f"{motif_type}{ext}")
        if os.path.exists(filepath):
            try:
                atoms = ase_read(filepath)
                return atoms, len(atoms)
            except Exception as e:
                print(f"Warning: Error reading template file {filepath} for {motif_type}: {e}")
                
    print(f"Warning: No suitable template file found for {motif_type} in '{motif_path}'")
    return None, None


def generate_defect_structures(
    target_compositions,
    exclude_motifs=None,
    include_motifs=None,
    custom_motif_path=None,
    random_seed=None
):
    """
    Generates defect structures for given compositions and defect motifs.

    Args:
        target_compositions (list): List of composition dictionaries. Each dict is {element_symbol: fraction}.
        exclude_motifs (list, optional): List of motif types to exclude (e.g., ['liquid', 'surface_100']).
        include_motifs (list, optional): List of motif types to include. If provided, only these motifs will be used.
        custom_motif_path (str, optional): Path to custom motif templates. If None, uses default in forge/core/motifs/.
        random_seed (int, optional): Seed for random number generator to ensure reproducibility.

    Returns:
        list: A list of dictionaries, each containing:
              {
                  'structure': ase.Atoms object (the generated structure),
                  'target_composition_input': original target composition input,
                  'parsed_composition_fractional': dict of {el: frac},
                  'motif_type': str (e.g., 'vacancy'),
                  'N_total_in_structure': int (actual atoms in the structure),
                  'atom_counts_in_structure': dict of {el: count},
                  'actual_composition_fractional': dict of {el: frac},
                  'variant_index': int (if N<20 rule generated multiple stoichiometries)
              }
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    # Determine which motifs to use
    motif_types = include_motifs if include_motifs is not None else DEFAULT_MOTIF_TYPES
    if exclude_motifs is not None:
        motif_types = [m for m in motif_types if m not in exclude_motifs]

    if not motif_types:
        raise ValueError("No valid motifs remain after filtering. Check your include/exclude lists.")

    # Determine motif template path
    if custom_motif_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        motif_path = os.path.join(current_dir, 'motifs')
    else:
        motif_path = custom_motif_path

    if not os.path.exists(motif_path):
        raise ValueError(f"Motif template directory not found: {motif_path}")

    generated_structures_info = []

    for comp_input in target_compositions:
        try:
            parsed_comp_frac = _normalize_composition(comp_input)
        except ValueError as e:
            print(f"Warning: Skipping composition '{comp_input}' due to normalization error: {e}")
            continue

        for motif_type in motif_types:
            # Load template structure for this motif
            template_atoms, N_total = _load_motif_template(motif_type, motif_path)
            
            if template_atoms is None:
                continue

            # Calculate atom counts for each element based on N_total
            list_of_atom_counts_dicts = _calculate_atom_counts_for_composition(parsed_comp_frac, N_total)

            for i, counts_dict in enumerate(list_of_atom_counts_dicts):
                if sum(counts_dict.values()) != N_total:
                    print(f"Error: Mismatch in calculated atom counts sum ({sum(counts_dict.values())}) "
                          f"and structure size ({N_total}) for {motif_type}, comp {comp_input}. Skipping this variant.")
                    continue

                new_structure = template_atoms.copy()
                
                symbols_to_assign = []
                for el, num in sorted(counts_dict.items()): # Sort for deterministic assignment if multiple elements
                    symbols_to_assign.extend([el] * num)
                
                # Shuffle symbols for random assignment to the sites in the template
                random.shuffle(symbols_to_assign)
                
                try:
                    new_structure.set_chemical_symbols(symbols_to_assign)
                except Exception as e:
                    print(f"Error setting chemical symbols for {motif_type}, comp {comp_input}: {e}. Skipping this variant.")
                    continue

                actual_comp_dict = {el: count / N_total for el, count in counts_dict.items()}

                generated_structures_info.append({
                    'structure': new_structure,
                    'target_composition_input': comp_input,
                    'parsed_composition_fractional': parsed_comp_frac,
                    'motif_type': motif_type,
                    'N_total_in_structure': N_total,
                    'atom_counts_in_structure': counts_dict,
                    'actual_composition_fractional': actual_comp_dict,
                    'variant_index': i # If N<20 rule generated multiple stoichiometries
                })
                
    return generated_structures_info


if __name__ == '__main__':
    # Example Usage
    print("Running example usage of defect_generator...")

    # Example compositions
    target_compositions = [
        {'V': 0.76, 'Cr': 0.17, 'Ti': 0.07},
        {'Fe': 0.5, 'Ni': 0.5},
        {'W': 1.0}
    ]

    # Example 1: Generate all default motifs
    results1 = generate_defect_structures(
        target_compositions,
        random_seed=42
    )

    # Example 2: Exclude certain motifs
    results2 = generate_defect_structures(
        target_compositions,
        exclude_motifs=['liquid', 'surface_100'],
        random_seed=42
    )

    # Example 3: Only include specific motifs
    results3 = generate_defect_structures(
        target_compositions,
        include_motifs=['vacancy', 'sia'],
        random_seed=42
    )
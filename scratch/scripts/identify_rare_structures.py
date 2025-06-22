# scratch/scripts/identify_rare_structures.py
import json
import logging
from pathlib import Path
from typing import List, Set, Dict, Optional

import numpy as np
from tqdm import tqdm

from forge.core.database import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_rare_structures_by_category(db_manager: DatabaseManager, rare_tags: List[str]) -> Set[int]:
    """Finds structure IDs based on a list of 'rare' config_type tags.

    Args:
        db_manager (DatabaseManager): An active database manager instance.
        rare_tags (List[str]): A list of config_type strings to query for.

    Returns:
        Set[int]: A set of unique structure IDs matching the rare tags.
    """
    rare_ids: Set[int] = set()
    logger.info(f"Querying for structures with the following rare tags: {rare_tags}")
    for tag in rare_tags:
        try:
            ids = db_manager.find_structures_by_metadata({'config_type': tag})
            if ids:
                logger.info(f"  - Found {len(ids)} structures with tag '{tag}'")
                rare_ids.update(ids)
            else:
                logger.info(f"  - Found 0 structures with tag '{tag}'")
        except Exception as e:
            logger.error(f"An error occurred while querying for tag '{tag}': {e}")
    return rare_ids

def find_high_property_structures(
    db_manager: DatabaseManager,
    structure_ids: List[int],
    property_weights: Dict[str, float],
    quantile: float,
) -> Set[int]:
    """Finds structures with high values for a weighted combination of properties.

    Args:
        db_manager (DatabaseManager): An active database manager instance.
        structure_ids (List[int]): The pool of structure IDs to analyze.
        property_weights (Dict[str, float]): A dict mapping property names
            ('energy', 'forces', 'stress') to their weight in the extremity score.
        quantile (float): The quantile to use for selecting "high" property structures.

    Returns:
        Set[int]: A set of unique structure IDs identified as having high properties.
    """
    logger.info(f"Analyzing {len(structure_ids)} structures for high property values (quantile={quantile}).")
    property_data = []

    # Fetch data in batches to avoid overwhelming memory
    batch_size = 2500
    for i in tqdm(range(0, len(structure_ids), batch_size), desc="Fetching properties"):
        batch_ids = structure_ids[i:i+batch_size]
        try:
            atoms_list = db_manager.get_batch_atoms_with_calculation(batch_ids)
            for atoms in atoms_list:
                sid = atoms.info.get("structure_id")
                if not sid:
                    continue
                
                num_atoms = len(atoms)
                energy = atoms.info["energy"] / num_atoms if num_atoms > 0 else 0
                forces = atoms.arrays["forces"]
                max_force = np.max(np.linalg.norm(forces, axis=1)) if forces.shape[0] > 0 else 0
                stress = atoms.info["stress"]
                max_stress = np.max(np.abs(stress)) if np.any(stress) else 0
                
                property_data.append({
                    "id": sid,
                    "energy": energy,
                    "force": max_force,
                    "stress": max_stress,
                })
        except Exception as e:
            logger.error(f"Failed to get calculations for batch starting at index {i}: {e}")

    if not property_data:
        logger.warning("No property data was successfully fetched. Returning empty set.")
        return set()

    # Normalize properties using min-max scaling to bring them to a common scale (0-1)
    for prop in ["energy", "force", "stress"]:
        values = [p[prop] for p in property_data]
        min_val, max_val = min(values), max(values)
        range_val = max_val - min_val
        if range_val > 1e-9: # Avoid division by zero
            for p in property_data:
                p[f"norm_{prop}"] = (p[prop] - min_val) / range_val
        else:
            for p in property_data:
                p[f"norm_{prop}"] = 0.0

    # Calculate weighted extremity score
    scores = []
    for p in property_data:
        score = (
            p.get("norm_energy", 0.0) * property_weights.get("energy", 0.0) +
            p.get("norm_force", 0.0) * property_weights.get("forces", 0.0) +
            p.get("norm_stress", 0.0) * property_weights.get("stress", 0.0)
        )
        scores.append(score)
    
    # Find the threshold for the top quantile and select IDs
    if not scores:
        return set()
        
    threshold = np.quantile(scores, quantile)
    high_prop_ids = {p["id"] for p, s in zip(property_data, scores) if s >= threshold}
    
    logger.info(f"Identified {len(high_prop_ids)} structures based on high property scores.")
    return high_prop_ids

def main():
    """
    Main function to identify rare structures and save their IDs.
    """
    # --- 1. Initialize Database Connection ---
    logger.info("Initializing database connection...")
    db_manager = DatabaseManager()

    # --- 2. Define Strategies for Identifying Rare Structures ---
    
    # --- Strategy 1: Automatic Categorical Filtering by Rarity ---
    logger.info("--- Starting Strategy 1: Categorical Rarity ---")
    config_type_counts = db_manager.get_metadata_key_counts("config_type")
    
    if not config_type_counts:
        logger.warning("Could not find any config_types in the database. Skipping categorical analysis.")
        auto_rare_tags = set()
    else:
        logger.info(f"Found {len(config_type_counts)} unique config_types with counts.")
        logger.info(f"Config types: {config_type_counts}")
        counts = np.array(list(config_type_counts.values()))
        rarity_quantile = 0.25 
        threshold = np.quantile(counts, rarity_quantile)
        auto_rare_tags = {tag for tag, count in config_type_counts.items() if count <= threshold}
        logger.info(f"Automatically identified {len(auto_rare_tags)} rare categories (bottom {rarity_quantile*100}%%, <= {threshold:.0f} structures).")

    # --- Strategy 2: Manual Override & Heuristics ---
    logger.info("--- Starting Strategy 2: Manual Overrides ---")
    manual_override_tags = {
        tag for tag in config_type_counts.keys() if "short_range" in tag.lower()
    }
    logger.info(f"Identified {len(manual_override_tags)} tags to include via manual override (e.g., 'aa' heuristic).")

    # Combine automatic and manual tags
    final_rare_tags = sorted(list(auto_rare_tags.union(manual_override_tags)))
    logger.info(f"Combined list contains {len(final_rare_tags)} unique tags to query.")
    rare_structure_ids = find_rare_structures_by_category(db_manager, final_rare_tags)
    logger.info(f"Found {len(rare_structure_ids)} structures from categorical and manual selection.")


    # --- Strategy 3: Property-Based Filtering ---
    logger.info("--- Starting Strategy 3: Property-Based Outliers ---")
    property_weights = {"energy": 0.1, "forces": 0.8, "stress": 0.1}
    
    logger.info("Finding all structures with VASP calculations for property analysis...")
    all_ids = set(db_manager.get_all_structure_ids())
    ids_without_vasp = set(db_manager.find_structures_without_calculation(calculator='vasp'))
    candidate_ids = sorted(list(all_ids - ids_without_vasp))

    if candidate_ids:
        logger.info(f"Found {len(candidate_ids)} candidates for property analysis.")
        high_prop_ids = find_high_property_structures(db_manager, candidate_ids, property_weights, quantile=0.95)
        
        initial_count = len(rare_structure_ids)
        rare_structure_ids.update(high_prop_ids)
        newly_added = len(rare_structure_ids) - initial_count
        
        logger.info(f"Added {newly_added} new structures from property-based filtering.")
        if newly_added > 0:
            logger.info("These are structures that are physically extreme but not in a rare category.")
    else:
        logger.warning("No candidate structures found for property-based filtering.")

    # Strategy 4 & 5: Future Implementations (UMAP, Active Learning)
    # ...

    # --- 4. Save the Final List of Rare IDs ---
    output_path = Path("./rare_structure_ids.json")
    final_id_list = sorted(list(rare_structure_ids))

    logger.info(f"\nFound a total of {len(final_id_list)} unique rare structures.")
    
    with open(output_path, 'w') as f:
        json.dump(final_id_list, f, indent=4)
        
    logger.info(f"Successfully saved the list of rare structure IDs to: {output_path.resolve()}")
    logger.info("This file can now be used by the 'run_allegro_experiment.py' script.")


if __name__ == "__main__":
    main() 
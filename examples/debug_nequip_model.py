#!/usr/bin/env python
"""Debug script to inspect NequIP model structure."""

import sys
from pathlib import Path

# Add the forge package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def inspect_object(obj, name="obj", max_depth=3, current_depth=0):
    """Recursively inspect an object to find attributes."""
    if current_depth >= max_depth:
        return
    
    indent = "  " * current_depth
    print(f"{indent}{name} ({type(obj).__name__}):")
    
    # Get non-private attributes
    attrs = [attr for attr in dir(obj) if not attr.startswith('_')]
    
    for attr in attrs:
        try:
            value = getattr(obj, attr)
            if callable(value):
                continue  # Skip methods
                
            print(f"{indent}  {attr}: {type(value).__name__}", end="")
            
            # Look for r_max, cutoff, chemical_symbols, etc.
            if any(keyword in attr.lower() for keyword in ['r_max', 'cutoff', 'chemical', 'symbol', 'type_name']):
                print(f" = {value}")
            else:
                print()
                
            # Recurse for important objects
            if attr in ['model', 'config'] and current_depth < max_depth - 1:
                inspect_object(value, f"{name}.{attr}", max_depth, current_depth + 1)
                
        except Exception as e:
            print(f"{indent}  {attr}: <error accessing: {e}>")

def main():
    """Main debugging function."""
    print("=== NequIP Model Structure Debug ===")
    
    try:
        from nequip.ase import NequIPCalculator
    except ImportError:
        print("NequIP not available")
        return
    
    # Find Allegro model files
    model_dir = Path(__file__).parent.parent / "scratch" / "data" / "potentials" / "compiled_gen-8-exploit"
    allegro_models = list(model_dir.glob("*.pt2"))
    
    if not allegro_models:
        print("No Allegro models found")
        return
    
    print(f"Found {len(allegro_models)} Allegro models")
    model_path = allegro_models[0]
    print(f"Inspecting: {model_path}")
    
    try:
        # Create calculator
        calc = NequIPCalculator.from_compiled_model(str(model_path), device='cuda')
        print("\n=== Calculator Structure ===")
        inspect_object(calc, "calc", max_depth=2)
        
        # Get the model
        model = calc.model
        print("\n=== Model Structure ===")
        inspect_object(model, "model", max_depth=3)
        
        # Look specifically for r_max and cutoff
        print("\n=== Searching for r_max/cutoff ===")
        def find_attribute(obj, attr_names, prefix=""):
            found = []
            for attr_name in attr_names:
                if hasattr(obj, attr_name):
                    value = getattr(obj, attr_name)
                    found.append(f"{prefix}{attr_name}: {value}")
                    
            # Check inner objects
            for inner_attr in ['model', 'config']:
                if hasattr(obj, inner_attr):
                    inner_obj = getattr(obj, inner_attr)
                    if inner_obj is not None:
                        found.extend(find_attribute(inner_obj, attr_names, f"{prefix}{inner_attr}."))
            
            return found
        
        r_max_attrs = find_attribute(calc, ['r_max', 'cutoff', 'R_max', 'rcut'])
        for attr in r_max_attrs:
            print(f"  {attr}")
        
        print("\n=== Searching for chemical symbols ===")
        symbol_attrs = find_attribute(calc, ['chemical_symbols', 'type_names', 'atomic_numbers', 'species'])
        for attr in symbol_attrs:
            print(f"  {attr}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
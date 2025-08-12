#!/usr/bin/env python3
"""
Test script to debug MACE descriptor extraction.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add forge to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_mace_descriptors():
    """Test MACE descriptor extraction with a simple structure."""
    
    # Test structure (simple FCC unit cell)
    from ase import Atoms
    from ase.build import bulk
    
    # Create a simple test structure
    atoms = bulk('V', 'bcc', a=3.01)
    print(f"Test structure: {len(atoms)} atoms")
    print(f"Cell: {atoms.get_cell()}")
    print(f"Positions: {atoms.get_positions()}")
    
    # Load MACE model
    model_path = "../../potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_0_pr_stagetwo.model"
    
    try:
        from mace.calculators.mace import MACECalculator
        print(f"\nLoading MACE model from: {model_path}")
        
        calc = MACECalculator(
            model_paths=[model_path],
            device='cuda',  # Use CPU for testing
            default_dtype="float32"
        )
        
        print(f"Model loaded successfully")
        print(f"Calculator attributes: {dir(calc)}")
        print(f"Calculator type: {type(calc)}")
        print(f"Models list length: {len(calc.models)}")
        print(f"First model type: {type(calc.models[0])}")
        print(f"First model attributes: {dir(calc.models[0])}")
        
        # Check the node_embedding attribute
        print(f"\n=== Checking node_embedding ===")
        if hasattr(calc.models[0], 'node_embedding'):
            print(f"✅ node_embedding found!")
            print(f"   Type: {type(calc.models[0].node_embedding)}")
            print(f"   Attributes: {dir(calc.models[0].node_embedding)}")
        else:
            print(f"❌ node_embedding not found")
        
        # Test 1: Try to get descriptors using our new method
        print(f"\n=== Test 1: Using get_descriptors method ===")
        try:
            from forge.calculators.mace_backend import MACEBackend
            backend = MACEBackend([model_path], device='cuda')
            descriptors = backend.get_descriptors(atoms)
            print(f"✅ get_descriptors successful!")
            print(f"   Descriptors shape: {descriptors.shape}")
            print(f"   Descriptors dtype: {descriptors.dtype}")
            print(f"   First few values: {descriptors[0][:5]}")
        except Exception as e:
            print(f"❌ get_descriptors failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 2: Direct model access
        print(f"\n=== Test 2: Direct model access ===")
        try:
            from mace.data import AtomicData, config_from_atoms
            from mace.tools.torch_geometric import Batch
            
            # Get model parameters
            r_max = calc.r_max.item() if hasattr(calc.r_max, 'item') else calc.r_max
            z_table = calc.z_table
            
            print(f"r_max: {r_max}")
            print(f"z_table: {z_table}")
            
            # Create MACE data format
            config = config_from_atoms(atoms)
            data = AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
            # Move data to the same device as the model (CUDA)
            data = Batch.from_data_list([data]).to('cuda')
            
            print(f"Data object type: {type(data)}")
            print(f"Data attributes: {dir(data)}")
            
            # Forward pass - use the model directly from calc.models[0]
            with torch.no_grad():
                output = calc.models[0](data, training=False)
                
                print(f"Model output type: {type(output)}")
                print(f"Model output keys: {list(output.keys())}")
                
                for key, value in output.items():
                    if value is not None:
                        print(f"  {key}: {type(value)}, shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
                    else:
                        print(f"  {key}: None")
                
                # Try to extract node features
                if "node_feats" in output:
                    node_feats = output["node_feats"]
                    print(f"✅ Found node_feats!")
                    print(f"   Shape: {node_feats.shape}")
                    print(f"   Dtype: {node_feats.dtype}")
                    print(f"   First few values: {node_feats[0][:5]}")
                else:
                    print(f"❌ No 'node_feats' found in output")
                    
        except Exception as e:
            print(f"❌ Direct model access failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 3: Try different output keys
        print(f"\n=== Test 3: Try different output keys ===")
        try:
            # Try the same forward pass but look for different keys
            config = config_from_atoms(atoms)
            data = AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
            # Move data to the same device as the model (CUDA)
            data = Batch.from_data_list([data]).to('cuda')
            
            with torch.no_grad():
                output = calc.models[0](data, training=False)
                
                # Try different possible key names
                possible_keys = [
                    "node_feats", "node_features", "atomic_features", 
                    "features", "node_attrs", "node_embeddings",
                    "atomic_embeddings", "descriptors"
                ]
                
                for key in possible_keys:
                    if key in output:
                        value = output[key]
                        print(f"✅ Found '{key}': {type(value)}, shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
                        if hasattr(value, 'shape') and len(value.shape) >= 2:
                            print(f"   First few values: {value[0][:5]}")
                    else:
                        print(f"❌ '{key}' not found")
                        
        except Exception as e:
            print(f"❌ Key testing failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 4: Try using node_embedding directly
        print(f"\n=== Test 4: Using node_embedding directly ===")
        try:
            if hasattr(calc.models[0], 'node_embedding'):
                # Try to use the node_embedding to get descriptors
                config = config_from_atoms(atoms)
                data = AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
                # Move data to the same device as the model (CUDA)
                data = Batch.from_data_list([data]).to('cuda')
                
                # Fix: data is a Batch object, not a dict, so we can't call .keys()
                print(f"Data object type: {type(data)}")
                print(f"Data attributes: {dir(data)}")
                print(f"Data shapes:")
                # Check common attributes that might have shapes
                for attr in ['node_attrs', 'pos', 'edge_index', 'edge_attr', 'batch']:
                    if hasattr(data, attr):
                        value = getattr(data, attr)
                        if hasattr(value, 'shape'):
                            print(f"  {attr}: {value.shape}")
                
                with torch.no_grad():
                    # Try to get node embeddings directly using node_attrs
                    print(f"node_attrs shape: {data.node_attrs.shape}")
                    node_emb = calc.models[0].node_embedding(data.node_attrs)
                    print(f"✅ node_embedding call successful!")
                    print(f"   Type: {type(node_emb)}")
                    print(f"   Shape: {node_emb.shape if hasattr(node_emb, 'shape') else 'N/A'}")
                    if hasattr(node_emb, 'shape'):
                        print(f"   First few values: {node_emb[0][:5]}")
                        print(f"   This looks like our atomic descriptors!")
            else:
                print(f"❌ node_embedding not available")
        except Exception as e:
            print(f"❌ node_embedding test failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Failed to load MACE model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mace_descriptors() 
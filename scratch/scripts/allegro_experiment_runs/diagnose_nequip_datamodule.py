#!/usr/bin/env python
"""Diagnostic script to understand NequIP's ASEDataModule structure."""

import logging
from pathlib import Path
import yaml
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def diagnose_datamodule(config_path: str):
    """Load a config and diagnose the datamodule structure."""
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent
    config_name = config_path.name
    
    # Load the config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n=== Original Config Data Section ===")
    print(yaml.dump(config.get('data', {}), default_flow_style=False))
    
    # Try to understand how NequIP expects the datamodule to work
    print("\n=== Attempting to instantiate datamodule ===")
    
    # Convert to OmegaConf
    cfg = OmegaConf.create(config)
    
    # Try to instantiate just the data section
    try:
        data_cfg = cfg.data
        print(f"\nData config type: {type(data_cfg)}")
        print(f"Data config _target_: {data_cfg.get('_target_', 'NOT SET')}")
        
        # Check what attributes are in the config
        print("\nData config keys:")
        for key in data_cfg:
            value = data_cfg[key]
            print(f"  - {key}: {type(value).__name__}")
            if key in ['train_dataloader', 'val_dataloader', 'test_dataloader']:
                if isinstance(value, dict):
                    print(f"    Keys: {list(value.keys())}")
        
        # Try minimal instantiation
        print("\n=== Attempting minimal datamodule instantiation ===")
        # Create a minimal version without file paths to avoid loading data
        minimal_cfg = OmegaConf.create({
            '_target_': data_cfg['_target_'],
            'train_file_path': '/tmp/dummy_train.xyz',
            'val_file_path': '/tmp/dummy_val.xyz', 
            'test_file_path': '/tmp/dummy_test.xyz',
            'ase_args': data_cfg.get('ase_args', {}),
            'key_mapping': data_cfg.get('key_mapping', {}),
            'transforms': [],  # Skip transforms for now
            'seed': 42,
        })
        
        # Add dataloader configs if they exist
        for dl_key in ['train_dataloader', 'val_dataloader', 'test_dataloader']:
            if dl_key in data_cfg:
                minimal_cfg[dl_key] = data_cfg[dl_key]
        
        print(f"\nMinimal config for instantiation:")
        print(OmegaConf.to_yaml(minimal_cfg))
        
        # Try to instantiate
        datamodule = instantiate(minimal_cfg)
        print(f"\nDatamodule instantiated: {type(datamodule)}")
        
        # Check attributes
        print("\nDatamodule attributes:")
        important_attrs = [
            'train_dataset', 'val_dataset', 'test_dataset',
            'train_dataloader', 'val_dataloader', 'test_dataloader',
            'train_dataloader_config', 'val_dataloader_config', 'test_dataloader_config',
            'collate_fn', 'setup', 'prepare_data'
        ]
        for attr in important_attrs:
            if hasattr(datamodule, attr):
                value = getattr(datamodule, attr)
                print(f"  - {attr}: {type(value).__name__}")
                if callable(value):
                    print(f"    (callable)")
                elif hasattr(value, '__dict__'):
                    print(f"    Keys: {list(value.__dict__.keys())[:5]}...")
        
    except Exception as e:
        logger.error(f"Failed to diagnose datamodule: {e}", exc_info=True)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python diagnose_nequip_datamodule.py <path_to_config.yaml>")
        sys.exit(1)
    
    diagnose_datamodule(sys.argv[1]) 
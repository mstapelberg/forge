#!/usr/bin/env python
"""Debug script to understand why we're only getting 1 training batch."""

import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_training_setup(config_path: str):
    """Analyze the training setup from config."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get basic info
    batch_size = config['data']['train_dataloader']['batch_size']
    num_gpus = config['trainer']['devices']
    
    logger.info(f"Batch size per GPU: {batch_size}")
    logger.info(f"Number of GPUs: {num_gpus}")
    logger.info(f"Effective batch size: {batch_size * num_gpus}")
    
    # Check if sampler config exists
    if 'sampler_config' in config['data']:
        sampler_cfg = config['data']['sampler_config']
        logger.info(f"\nSampler configuration found:")
        logger.info(f"  Type: {sampler_cfg.get('_target_', 'Unknown')}")
        logger.info(f"  Replica: {sampler_cfg.get('replica', 'Not set')}")
        logger.info(f"  Alpha: {sampler_cfg.get('alpha', 'Not set')}")
        logger.info(f"  Number of rare indices: {len(sampler_cfg.get('rare_idx', []))}")
    
    # Check data paths
    train_path = Path(config_path).parent / config['data']['train_file_path']
    if train_path.exists():
        # Count structures in training file
        with open(train_path, 'r') as f:
            lines = f.readlines()
        
        # In XYZ format, first line of each structure is the atom count
        num_structures = sum(1 for line in lines if line.strip().isdigit())
        logger.info(f"\nTraining data file: {train_path.name}")
        logger.info(f"Number of structures: {num_structures}")
        
        # Calculate expected batches
        if 'sampler_config' in config['data'] and 'replica' in config['data']['sampler_config']:
            replica = config['data']['sampler_config']['replica']
            rare_count = len(config['data']['sampler_config'].get('rare_idx', []))
            # This is approximate - actual count depends on which structures are rare
            approx_total_samples = num_structures + (rare_count * (replica - 1))
            logger.info(f"\nWith replication:")
            logger.info(f"  Approximate total samples: {approx_total_samples}")
            logger.info(f"  Samples per GPU: {approx_total_samples // num_gpus}")
            logger.info(f"  Expected batches per GPU: {approx_total_samples // (num_gpus * batch_size)}")
        else:
            logger.info(f"\nWithout replication:")
            logger.info(f"  Samples per GPU: {num_structures // num_gpus}")
            logger.info(f"  Expected batches per GPU: {num_structures // (num_gpus * batch_size)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python debug_training_batches.py <path_to_config.yaml>")
        sys.exit(1)
    
    analyze_training_setup(sys.argv[1]) 
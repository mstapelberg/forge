#!/usr/bin/env python
"""Test script to verify Allegro config generation without running full training."""

import logging
import yaml
from pathlib import Path
import sys
from forge.core.database import DatabaseManager
from forge.workflows.db_to_allegro import prepare_allegro_job

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_config_generation():
    """Test config generation with a simple case."""
    test_dir = Path("scratch/test_allegro_config")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Test with minimal configuration
        with DatabaseManager() as db:
            structure_ids = db.find_structures_by_metadata({'generation': 0}, operator='>=')[:100]
            
            if not structure_ids:
                logger.error("No structures found in database")
                return False
            
            logger.info(f"Testing with {len(structure_ids)} structures")
            
            # Test 1: Basic configuration
            logger.info("\n=== Test 1: Basic configuration ===")
            basic_job_dir = test_dir / "test_basic"
            prepare_allegro_job(
                db_manager=db,
                job_name="test_basic",
                job_dir=basic_job_dir,
                structure_ids=structure_ids,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                seed=42
            )
            
            # Check if config was created
            config_path = basic_job_dir / "config.yaml"
            if not config_path.exists():
                logger.error("Config file was not created!")
                return False
            
            # Load and validate config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info("Config structure:")
            logger.info(f"  - Top-level keys: {list(config.keys())}")
            logger.info(f"  - Data keys: {list(config.get('data', {}).keys())}")
            
            # Check critical keys
            critical_checks = [
                ('data' in config, "Missing 'data' section"),
                ('train_dataloader' in config.get('data', {}), "Missing 'train_dataloader'"),
                ('val_dataloader' in config.get('data', {}), "Missing 'val_dataloader'"),
                ('test_dataloader' in config.get('data', {}), "Missing 'test_dataloader'"),
            ]
            
            for check, msg in critical_checks:
                if not check:
                    logger.error(f"FAIL: {msg}")
                else:
                    logger.info(f"PASS: {msg}")
            
            # Test 2: With custom sampler
            logger.info("\n=== Test 2: With custom sampler ===")
            sampler_job_dir = test_dir / "test_sampler"
            prepare_allegro_job(
                db_manager=db,
                job_name="test_sampler",
                job_dir=sampler_job_dir,
                structure_ids=structure_ids,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                seed=42,
                sampler="rare_weighted",
                sampler_params={"replica": 2, "rare_idx": [0, 1, 2]}
            )
            
            # Check sampler config
            sampler_config_path = sampler_job_dir / "config.yaml"
            with open(sampler_config_path, 'r') as f:
                sampler_config = yaml.safe_load(f)
            
            if sampler_config.get('data', {}).get('_target_') == "forge.workflows.allegro_utils.data.CustomSamplingASEDataModule":
                logger.info("PASS: Custom data module is set")
            else:
                logger.error("FAIL: Custom data module not set correctly")
            
            if 'sampler_config' in sampler_config.get('data', {}):
                logger.info("PASS: Sampler config is present")
                logger.info(f"  - Sampler config: {sampler_config['data']['sampler_config']}")
            else:
                logger.error("FAIL: Sampler config missing")
            
            logger.info("\n=== All tests completed ===")
            return True
            
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_config_generation()
    sys.exit(0 if success else 1) 
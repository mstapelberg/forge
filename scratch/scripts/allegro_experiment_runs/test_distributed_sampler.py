#!/usr/bin/env python
"""Test script to validate distributed sampler behavior."""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from forge.workflows.allegro_utils.samplers import RareWeightedSampler
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
    """Simple dataset for testing."""
    def __init__(self, size=100):
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {"index": idx, "force_norm": 0.1 * idx}


def test_sampler_sizes():
    """Test that sampler sizes are correct."""
    dataset = DummyDataset(100)
    
    # Test 1: Regular sampler
    logger.info("\n=== Test 1: Regular RareWeightedSampler ===")
    rare_idx = [0, 10, 20, 30, 40]  # 5 rare samples
    replica = 3
    sampler = RareWeightedSampler(
        data_source=dataset,
        rare_idx=rare_idx,
        replica=replica,
        alpha=0.0  # No force weighting for simplicity
    )
    
    expected_size = 100 + (len(rare_idx) * (replica - 1))  # 100 + 5*(3-1) = 110
    actual_size = len(sampler)
    logger.info(f"Expected sampler size: {expected_size}")
    logger.info(f"Actual sampler size: {actual_size}")
    assert actual_size == expected_size, f"Size mismatch: {actual_size} != {expected_size}"
    
    # Test 2: DataLoader with custom sampler
    logger.info("\n=== Test 2: DataLoader with custom sampler ===")
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)
    num_batches = len(dataloader)
    logger.info(f"Number of batches: {num_batches}")
    logger.info(f"Expected batches: {expected_size // 4} (with {expected_size % 4} samples dropped)")
    
    # Test 3: Distributed scenario (simulate 4 GPUs)
    logger.info("\n=== Test 3: Simulated distributed training (4 GPUs) ===")
    for rank in range(4):
        # Create distributed sampler that wraps indices
        class SimpleDistributedWrapper:
            def __init__(self, indices, num_replicas=4, rank=0):
                self.indices = list(indices)
                self.num_replicas = num_replicas
                self.rank = rank
                self.num_samples = len(self.indices) // num_replicas
                
            def __iter__(self):
                # Simple round-robin distribution
                for i in range(self.rank, len(self.indices), self.num_replicas):
                    if i < len(self.indices):
                        yield self.indices[i]
                        
            def __len__(self):
                return self.num_samples
        
        dist_sampler = SimpleDistributedWrapper(sampler.indices, num_replicas=4, rank=rank)
        logger.info(f"  Rank {rank}: {len(dist_sampler)} samples")
    
    logger.info("\n=== All tests passed! ===")


if __name__ == "__main__":
    test_sampler_sizes() 
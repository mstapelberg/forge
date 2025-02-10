import logging
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


from pathlib import Path
from mace import data, tools
from mace.tools.slurm_distributed import DistributedEnvironment
from forge.core.database import DatabaseManager
from forge.potentials.mace import MACECalculator
from typing import Dict, List, Optional, Literal, Union
from mace.data import MeanAbsoluteErrorLoss

# --- Helper functions ---

def load_data_from_database(db_manager, train_ids, z_table, r_max):
    # Implementation of load_data_from_database function
    pass

def save_model_to_database(db_manager, model_state_dict, metadata):
    # Implementation of save_model_to_database function
    pass

# --- Main training function ---

def train_mlip(config: Dict):
    """Train an MLIP (MACE model) with distributed training and database integration."""

    # Setup distributed environment
    distr_env = DistributedEnvironment()
    world_size = distr_env.world_size
    print(f"World size: {world_size}")
    local_rank = distr_env.local_rank
    rank = distr_env.rank
    if rank == 0:
        print("Running on master node")
    torch.cuda.set_device(local_rank)

    # Setup logging
    tools.setup_logger(level=config.get("log_level", "INFO"), tag=config.get("name", "mace_training"), directory=config.get("log_dir", "logs"), rank=rank)

    # Database connection
    db_manager = DatabaseManager(config_path=config.get("db_config", "config/database.yaml"))

    # Load data
    if config.get("train_ids"):
        train_data = load_data_from_database(db_manager, config["train_ids"], config.get("z_table", "configs/z_table.yaml"), config.get("r_max", 5.0))
    else:
        train_data = data.dataset_from_sharded_hdf5(config["train_file"], r_max=config.get("r_max", 5.0), z_table=config.get("z_table", "configs/z_table.yaml"))
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True, seed=config.get("seed", 42)
    )
    train_loader = DataLoader(
        train_data, batch_size=config.get("batch_size", 32), sampler=train_sampler, shuffle=False, drop_last=False, pin_memory=True
    )

    # Initialize model
    if config.get("foundation_model"):
        model = MACECalculator(model_paths=config["foundation_model"], device=config.get("device", "cuda"))
    else:
        model = MACECalculator(
            name = config.get("model_name", "MACE_model"),
            r_max=config.get("r_max", 5.0),
            num_bessel=config.get("num_bessel", 8),
            num_polynomial_cutoff=config.get("num_polynomial_cutoff", 6),
            max_L=config.get("max_L", 2),
            hidden_irreps=config.get("hidden_irreps", "32x0e+32x1o+32x2e"),
            MLP_hidden_layers=config.get("MLP_hidden_layers", 2),
            avg_num_neighbors=config.get("avg_num_neighbors", 6.0),
            atomic_energies_refs=None,
            device=config.get("device", "cuda"),
            dtype=config.get("dtype", "float32"),
        )
    model.to(config.get("device", "cuda"))
    model = DDP(model, device_ids=[local_rank])

    # Optimizer - AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 1e-5)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

    # Loss function - MAE
    loss_fn = MeanAbsoluteErrorLoss(output_properties=["energy", "forces"])

    # Scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=config.get("amp", False))

    # Training loop
    print("Starting training loop...")
    for epoch in range(config.get("epochs", 100)):
        train_sampler.set_epoch(epoch)
        total_loss = 0.0 # Initialize total loss for epoch
        for batch in train_loader:
            batch = batch.to(config.get("device", "cuda"))
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=config.get("amp", False)):
                pred = model(batch)
                loss, loss_dict = loss_fn(pred, batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() # Accumulate batch loss

        avg_loss = total_loss / len(train_loader) # Calculate average epoch loss
        print(f"Epoch: {epoch+1}/{config.get('epochs', 100)}, Training Loss: {avg_loss:.4f}") # Basic logging

        # Validation (Placeholder - Add your validation logic here)
        # val_loss = validate(model, val_loader, loss_fn, device=config.get("device", "cuda"), amp=config.get("amp", False))
        # scheduler.step(val_loss) # Step scheduler based on validation loss
        scheduler.step(avg_loss) # Step scheduler based on training loss for now (replace with val_loss later)

    print("Training finished.")

    # Save model and metadata
    if rank == 0:
        print("Rank 0 process: Saving model...")
        metadata = {
            "train_ids": config.get("train_ids"),
            "training_config": config # Save the entire training config in metadata
        }
        save_model_to_database(db_manager, model.module.state_dict(), metadata)

    dist.destroy_process_group()

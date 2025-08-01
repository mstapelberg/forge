import argparse
import logging
import json
import time
from pathlib import Path

import torch
import ase.io
import numpy as np
import matplotlib.pyplot as plt

from mace import data, tools
from mace import data, modules, tools  # or wherever your MACE model is loaded
from mace.tools import torch_geometric
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.calculators.mace import MACECalculator

try:
    import cuequivariance as cue  # pylint: disable=unused-import
    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

CUET_AVAILABLE = False

def build_dataset(xyz_file: str, cutoff: float, batch_size: int):
    """
    Loads the xyz file and creates two dataloaders:
    one for warmup (batch size 1, first batch) and one for the full dataset.
    """
    atoms_list = ase.io.read(xyz_file, index=":")  # Read all structures
    # Use a fixed atomic number table as in the original code.
    table = tools.AtomicNumberTable([22, 23, 24, 40, 74])
    # Build dataset: one AtomicData per structure.
    dataset = [
        data.AtomicData.from_config(
            data.config_from_atoms(atoms),
            z_table=table,
            cutoff=cutoff,
        )
        for atoms in atoms_list
    ]
    # Warmup dataloader: batch size = 1 (we only need the first batch)
    warmup_loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    # Full dataset dataloader with batch size 1 to avoid inhomogeneous arrays
    full_loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=1,  # Always use batch size 1
        shuffle=False,
        drop_last=False,
    )
    return warmup_loader, full_loader

def get_node_features_from_mace(model, batch):
    """
    Extract node features from a MACE model without computing forces.
    This avoids the gradient computation that causes errors.
    """
    # Get the embeddings from the model
    if hasattr(model, "embed_network"):
        # For newer MACE versions
        node_feats = model.embed_network(batch)
    elif hasattr(model, "node_embedding"):
        # For older MACE versions
        node_feats = model.node_embedding(batch)
    else:
        # Try to access the features through the model's forward method
        # but modify it to avoid force computation
        original_compute_forces = model.compute_forces
        model.compute_forces = False
        try:
            out = model(batch)
            node_feats = out.get("node_feats", None)
        finally:
            # Restore the original compute_forces setting
            model.compute_forces = original_compute_forces
            
    if node_feats is None:
        raise ValueError("Could not extract node features from the MACE model")
        
    return node_feats

def extract_mace_descriptors(model, full_loader, device):
    """
    Extracts global structure descriptors from MACE node features for all structures in 'full_loader'.
    Returns descriptors and other available properties from the model.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        print("tqdm not installed. Install with 'pip install tqdm' for progress bars.")
        # Create a simple replacement if tqdm is not available
        def tqdm(iterable, **kwargs):
            return iterable
    
    model.eval()
    all_descriptors = []
    all_energies = []
    all_forces = []
    all_stresses = []
    all_compositions = []
    
    # Get total number of batches for progress bar
    total_batches = len(full_loader)
    
    # Loop over batches of structures with progress bar
    for batch_idx, batch in enumerate(tqdm(full_loader, desc="Extracting features", total=total_batches)):
        batch = batch.to(device)
        
        # Print batch attributes for debugging in the first batch
        if batch_idx == 0:
            print("Batch attributes that might contain atomic numbers:")
            for attr in ['x', 'node_attrs', 'charges']:
                if hasattr(batch, attr) and getattr(batch, attr) is not None:
                    print(f"  {attr}: {getattr(batch, attr).shape}")
        
        # Forward pass
        out = model(batch, training=False)
        
        # Debug: Print available keys in the output dictionary for the first batch
        if batch_idx == 0:
            print(f"Available keys in model output: {list(out.keys())}")
            for key, value in out.items():
                if value is not None:
                    print(f"  {key}: {type(value)}, shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
                else:
                    print(f"  {key}: None")
        
        # Extract node features
        if "node_feats" in out:
            node_feats = out["node_feats"]
        else:
            # Try to get node features from the model directly
            node_feats = get_node_features_from_mace(model, batch)
        
        # Extract energy if available - detach before converting to numpy
        if "energy" in out and out["energy"] is not None:
            energy = out["energy"]
            all_energies.append(energy.detach().cpu().numpy())
        
        # Extract forces if available - detach before converting to numpy
        if "forces" in out and out["forces"] is not None:
            forces = out["forces"]
            all_forces.append(forces.detach().cpu().numpy())
        
        # Extract stress if available - safely handle None values and detach
        if "stress" in out and out["stress"] is not None:
            stress = out["stress"]
            all_stresses.append(stress.detach().cpu().numpy())
        elif "virials" in out and out["virials"] is not None:
            # Some models output virials instead of stress
            virials = out["virials"]
            all_stresses.append(virials.detach().cpu().numpy())
        
        # 'batch.batch' indicates which graph (structure) each atom belongs to
        graph_indices = batch.batch
        for struct_id in range(batch.num_graphs):
            mask = (graph_indices == struct_id)
            struct_node_feats = node_feats[mask]  # shape = (n_atoms_in_struct, feat_dim)
            
            # Pool to get a single descriptor per structure - detach before converting to numpy
            struct_descriptor = struct_node_feats.mean(dim=0)
            all_descriptors.append(struct_descriptor.detach().cpu().numpy())
            
            # Get composition (count of each atom type)
            # Try different attributes that might contain atomic numbers
            if hasattr(batch, 'node_attrs') and batch.node_attrs is not None:
                atom_types = batch.node_attrs[mask, 0].detach().cpu().numpy()
            elif hasattr(batch, 'charges') and batch.charges is not None:
                atom_types = batch.charges[mask].detach().cpu().numpy()
            else:
                # If we can't find atomic numbers, use a placeholder
                print("Warning: Could not find atomic numbers in batch attributes")
                atom_types = np.ones(mask.sum().item())
                
            unique, counts = np.unique(atom_types, return_counts=True)
            composition = dict(zip(unique, counts))
            all_compositions.append(composition)
    
    return {
        "descriptors": all_descriptors,
        "compositions": all_compositions,
        "energies": all_energies if all_energies else None,
        "forces": all_forces if all_forces else None,
        "stresses": all_stresses if all_stresses else None
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xyz_file", type=str, help="Path to xyz file")
    parser.add_argument("model_file", type=str, help="Path to a single MACE model .pt file")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for full dataset")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Cutoff for neighbor building")
    parser.add_argument("--output_npy", type=str, default="descriptors.npy", help="Where to save the descriptors")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # 1) Build dataset + dataloader with batch_size=1
    warmup_loader, full_loader = build_dataset(args.xyz_file, args.cutoff, batch_size=1)
    
    # 2) Load MACE model
    logging.info(f"Loading MACE model from {args.model_file}")
    model = torch.load(args.model_file, map_location=device)
    
    # Modify the model to expose node features
    original_forward = model.forward
    
    def modified_forward(self, batch, training=False):
        # Call the original forward method
        result = original_forward(batch, training=training)
        
        # Add node features to the result if not already there
        if "node_feats" not in result and hasattr(self, "embed_network"):
            result["node_feats"] = self.embed_network(batch)
        elif "node_feats" not in result and hasattr(self, "node_embedding"):
            result["node_feats"] = self.node_embedding(batch)
        
        return result
    
    # Bind the modified forward method to the model
    import types
    model.forward = types.MethodType(modified_forward, model)
    
    # Create a calculator with compute_forces=False
    #calculator = MACECalculator(model, device=device, compute_forces=False)
    
    if CUET_AVAILABLE and args.device == "cuda":
        model_cueq = run_e3nn_to_cueq(model)
        model_cueq = model_cueq.to(device)
    else:
        model_cueq = None
    
    # 3) Extract descriptors
    logging.info("Extracting MACE descriptors...")
    results = extract_mace_descriptors(model, full_loader, device)
    
    descriptors = np.array(results["descriptors"])  # shape = (n_structures, feat_dim)
    np.save(args.output_npy, descriptors)
    logging.info(f"Saved descriptors to {args.output_npy}")
    
    # Save other properties if available - using pickle for inhomogeneous arrays
    output_base = args.output_npy.replace('.npy', '')
    
    # Use pickle for saving arrays with inhomogeneous shapes
    import pickle
    
    if results["energies"] is not None:
        with open(f"{output_base}_energies.pkl", 'wb') as f:
            pickle.dump(results["energies"], f)
        logging.info(f"Saved energies to {output_base}_energies.pkl")
    
    if results["forces"] is not None:
        with open(f"{output_base}_forces.pkl", 'wb') as f:
            pickle.dump(results["forces"], f)
        logging.info(f"Saved forces to {output_base}_forces.pkl")
    
    if results["stresses"] is not None:
        with open(f"{output_base}_stresses.pkl", 'wb') as f:
            pickle.dump(results["stresses"], f)
        logging.info(f"Saved stresses to {output_base}_stresses.pkl")
    
    # Save compositions as well
    with open(f"{output_base}_compositions.pkl", 'wb') as f:
        pickle.dump(results["compositions"], f)
    logging.info(f"Saved compositions to {output_base}_compositions.pkl")
    
    # 4) Dimensionality reduction and visualization
    from sklearn.decomposition import PCA
    import umap
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    
    logging.info("Performing PCA + UMAP on descriptors...")
    pca = PCA(n_components=min(50, descriptors.shape[1]))
    X_pca = pca.fit_transform(descriptors)
    
    # Plot PCA explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.savefig(f"{output_base}_pca_variance.png", dpi=150)
    
    # UMAP for visualization
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_2d = umap_model.fit_transform(X_pca)
    
    # Try different DBSCAN parameters to find good clustering
    best_score = -1
    best_eps = 0.5
    best_min_samples = 5
    best_labels = None
    
    for eps in [0.3, 0.5, 0.7, 1.0, 1.5]:
        for min_samples in [3, 5, 10, 15]:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_2d)
            
            # Skip if all points are noise or only one cluster
            if len(np.unique(labels)) <= 1 or -1 in labels and np.sum(labels == -1) > 0.5 * len(labels):
                continue
                
            try:
                score = silhouette_score(X_2d, labels)
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples
                    best_labels = labels
            except:
                # Silhouette score can fail if there's only one cluster
                pass
    
    # If no good clustering found, use default parameters
    if best_labels is None:
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        best_labels = dbscan.fit_predict(X_2d)
    
    # Visualization with cluster labels
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=best_labels, cmap="tab20", s=30, alpha=0.8)
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"UMAP of MACE Descriptors (eps={best_eps}, min_samples={best_min_samples})")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.savefig(f"{output_base}_umap_clusters.png", dpi=150)
    
    # Density plot to identify redundant regions
    plt.figure(figsize=(10, 8))
    from scipy.stats import gaussian_kde
    xy = np.vstack([X_2d[:,0], X_2d[:,1]])
    z = gaussian_kde(xy)(xy)
    
    idx = z.argsort()
    x, y, z = X_2d[idx,0], X_2d[idx,1], z[idx]
    
    plt.scatter(x, y, c=z, s=30, cmap='viridis')
    plt.colorbar(label='Density')
    plt.title('UMAP Density Plot - Identifying Redundant Regions')
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.savefig(f"{output_base}_umap_density.png", dpi=150)
    
    logging.info(f"Saved visualization plots to {output_base}_*.png")


if __name__ == "__main__":
    main()


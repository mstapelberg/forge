from forge.core.database import DatabaseManager
#from forge.workflows.adversarial_attack import run_adversarial_attacks
import random
import numpy as np 
from ase.io import read, write
import json
from glob import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

db_manager = DatabaseManager()


rare_ids = json.load(open('./analysis_output_full_gen8/rare_structure_ids.json'))

calcs = db_manager.get_batch_atoms_with_calculation(rare_ids)

# rank the atoms by force rmse
atoms_df = pd.read_csv('./analysis_output_full_gen8/atom_metrics.csv')
print(atoms_df.head())

# print the top 25 structures by force_error_mag
print(atoms_df.sort_values(by='force_error_mag', ascending=False).head(25))

structure_df = pd.read_csv('./analysis_output_full_gen8/structure_metrics.csv')
print(structure_df.head())

print(structure_df.columns)
print(structure_df.sort_values(by='force_rmse_metric', ascending=False).head(25))

#model_paths = ['../potentials/mace_gen_6_ensemble/gen_7_model_0-2025-02-12_stagetwo_compiled.model', '../potentials/mace_gen_6_ensemble/gen_7_model_1-2025-02-12_stagetwo_compiled.model', '../potentials/mace_gen_6_ensemble/gen_7_model_2-2025-02-12_stagetwo_compiled.model']
model_paths = glob('../data/potentials/compiled_gen-8-exploit/*.nequip.pt2')

random.seed(42)

# Set plot style
sns.set_style("whitegrid")

from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
import numpy as np
import json
import os

# --- This code should be placed after you load 'structure_df' ---

# Create a directory for outputs if it doesn't exist
output_dir = './analysis_output_full_gen8'
os.makedirs(output_dir, exist_ok=True)


### 1. Feature Selection and Scaling
# We select a rich set of metrics that describe the error's magnitude,
# distribution, spatial clustering, and the model's uncertainty.
print("Step 1: Selecting and scaling features...")
features_to_cluster = [
    'force_rmse_metric',
    'force_gini_metric',
    'force_kurtosis_metric',
    'morans_i_global_metric',
    'score_ensemble_metric',
    'difficulty_metric',
    'n_error_clusters_metric'
]

# Extract features and fill any potential NaNs (a robust practice)
structure_features = structure_df[features_to_cluster].copy().fillna(0)

# Scale features to have zero mean and unit variance. This is crucial
# for distance-based algorithms like UMAP.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(structure_features)
print("Features scaled successfully.")

### 2. Dimensionality Reduction with UMAP
# UMAP will project our 7D feature space into 2D for visualization and clustering.
# We are looking for the global structure, so we use a higher `n_neighbors`.
print("\nStep 2: Performing dimensionality reduction with UMAP...")
reducer = umap.UMAP(
    n_neighbors=40,
    min_dist=0.0,
    n_components=2,
    random_state=42
)
embedding = reducer.fit_transform(scaled_features)
structure_df['umap_x'] = embedding[:, 0]
structure_df['umap_y'] = embedding[:, 1]
print("UMAP embedding created.")

### 3. Density-Based Clustering with HDBSCAN
# HDBSCAN is ideal for UMAP results. It identifies clusters of varying densities
# and, importantly, can mark points as "noise" (outliers), which is very useful.
# You may need to install it: pip install hdbscan
print("\nStep 3: Clustering UMAP projections with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,  # Clusters must have at least 20 members
    gen_min_span_tree=True
)
cluster_labels = clusterer.fit_predict(embedding)
structure_df['cluster'] = cluster_labels
n_clusters = len(np.unique(cluster_labels)) - 1 # Subtract 1 for the noise cluster (-1)
print(f"HDBSCAN found {n_clusters} distinct clusters.")


### 4. Visualize the Clusters
print("\nStep 4: Generating cluster visualization...")
plt.figure(figsize=(14, 10))
# Plot non-noise points with cluster labels
clustered_points = structure_df[structure_df['cluster'] != -1]
sns.scatterplot(
    data=clustered_points,
    x='umap_x',
    y='umap_y',
    hue='cluster',
    palette=sns.color_palette("viridis", n_colors=n_clusters),
    s=30,
    alpha=0.9
)
# Plot noise points
noise_points = structure_df[structure_df['cluster'] == -1]
plt.scatter(
    noise_points['umap_x'],
    noise_points['umap_y'],
    c='lightgrey',
    s=10,
    alpha=0.5,
    label='Noise'
)
plt.title('UMAP Projection of Structure Errors with HDBSCAN Clusters', fontsize=16)
plt.xlabel('UMAP Dimension 1', fontsize=12)
plt.ylabel('UMAP Dimension 2', fontsize=12)
plt.legend(title='Cluster ID')
plt.savefig(os.path.join(output_dir, 'plots/umap_hdbscan_clusters.png'))
print("Cluster plot saved to 'analysis_output_full_gen8/plots/'.")


### 5. Stratified Sampling for Data Selection
print("\nStep 5: Selecting 250 diverse data points via stratified sampling...")
n_total_to_select = 250
selected_ids = []

# We will select points from the actual clusters found by HDBSCAN
selectable_clusters = structure_df[structure_df['cluster'] != -1]
total_in_clusters = len(selectable_clusters)
cluster_counts = selectable_clusters['cluster'].value_counts()

# Sample from each cluster, prioritizing the most difficult structures
for cluster_id, count in cluster_counts.items():
    # Number to select is proportional to cluster size
    n_to_select_from_cluster = int(np.round(n_total_to_select * (count / total_in_clusters)))
    
    # Get all structures in the current cluster
    cluster_subset = selectable_clusters[selectable_clusters['cluster'] == cluster_id]
    
    # Select the top N most "difficult" structures from this cluster
    selection = cluster_subset.nlargest(n_to_select_from_cluster, 'difficulty_metric')
    selected_ids.extend(selection['structure_id'].tolist())

# To ensure we get exactly 250, we can add top "noise" points if we are short,
# as these are often unique, high-error configurations.
if len(selected_ids) < n_total_to_select:
    n_needed = n_total_to_select - len(selected_ids)
    top_noise = noise_points.nlargest(n_needed, 'difficulty_metric')
    selected_ids.extend(top_noise['structure_id'].tolist())

# Ensure the final list is exactly 250 unique IDs
final_selection = list(np.unique(selected_ids))
# If over, trim it down by removing the 'least difficult' of the selected
if len(final_selection) > n_total_to_select:
    selected_df = structure_df[structure_df['structure_id'].isin(final_selection)]
    final_selection = selected_df.nlargest(n_total_to_select, 'difficulty_metric')['structure_id'].tolist()


print(f"\n✅ Successfully selected {len(final_selection)} diverse structures for augmentation.")

# Save the list of selected structure IDs to a JSON file
output_path = os.path.join(output_dir, 'selected_for_augmentation.json')
with open(output_path, 'w') as f:
    json.dump(final_selection, f, indent=4)
print(f"List of selected IDs saved to: {output_path}")

"""
trajectories = run_adversarial_attacks(
    db_manager = db_manager,
    model_paths = model_paths,
    structure_ids = rare_ids, # select 1000 at random from list
    generation = 9,
    n_iterations=200,
    learning_rate=0.01,
    temperature=1000,
    include_probability=False,
    min_distance=1.2,
    use_energy_per_atom=True,
    device='cuda',
    debug=False,
    top_n=318,
    save_output=True,
    output_dir='../data/adversarial_attacks/gen_9_rare',
    patience=25,
    shake=False,
    ranking_metric='force_rmse',
    reference_calculator='vasp',
    cache_rmse=True,
    plot_rmse_histogram=True,
    rmse_cutoff=0.1
)

# --- Handle None return value when saving --- 
if trajectories is not None:
    print(f"Number of trajectories returned: {len(trajectories)}")
else:
    print("Trajectories were saved to files.")

print(f"All done")

"""

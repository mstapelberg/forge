from typing import List, Dict, Optional
import numpy as np
from ase import Atoms
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
import json

class NNMDSimulator:
    def __init__(self, calculator, temp: float, timestep: float = 1.0,
                 friction: float = 0.002, trajectory_file: Optional[str] = None):
        self.calculator = calculator
        self.temp = temp
        self.timestep = timestep
        self.friction = friction
        self.trajectory_file = trajectory_file
        
    def run_md(self, atoms: Atoms, steps: int, sample_interval: int = 10):
        """Run MD simulation with neural network potential"""
        atoms.calc = self.calculator
        
        # Initialize velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=self.temp)
        
        # Setup dynamics
        dyn = Langevin(atoms, self.timestep, temperature_K=self.temp,
                      friction=self.friction)
                      
        # Setup trajectory writing
        if self.trajectory_file:
            def write_frame():
                atoms.write(self.trajectory_file, append=True)
            dyn.attach(write_frame, interval=sample_interval)
            
        # Run dynamics
        dyn.run(steps)
        
        return atoms

class CompositionAnalyzer:
    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.tsne = TSNE(n_components=n_components, random_state=random_state)
        self.kmeans = None
        
    def analyze_compositions(self, compositions: List[Dict[str, float]], 
                           n_clusters: int = 5):
        """Analyze composition space using t-SNE and clustering"""
        # Convert compositions to array
        comp_array = self._compositions_to_array(compositions)
        
        # Perform t-SNE
        embeddings = self.tsne.fit_transform(comp_array)
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.kmeans.fit_predict(comp_array)
        
        return embeddings, clusters
        
    def _compositions_to_array(self, compositions: List[Dict[str, float]]) -> np.ndarray:
        """Convert composition dictionaries to numpy array"""
        elements = sorted(set().union(*compositions))
        return np.array([[comp.get(elem, 0.0) for elem in elements]
                        for comp in compositions])
                        
    def plot_analysis(self, embeddings: np.ndarray, clusters: np.ndarray,
                     save_path: Optional[str] = None):
        """Create visualization of composition analysis"""
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1],
                            c=clusters, cmap='viridis')
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('Composition Space Analysis')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def suggest_new_compositions(self, compositions: List[Dict[str, float]],
                               n_suggestions: int = 10) -> List[Dict[str, float]]:
        """Suggest new compositions based on clustering analysis"""
        if self.kmeans is None:
            raise ValueError("Must run analyze_compositions first")
            
        # Find cluster centers
        centers = self.kmeans.cluster_centers_
        
        # Generate new compositions near cluster boundaries
        new_compositions = []
        for _ in range(n_suggestions):
            # Select random pair of clusters
            c1, c2 = np.random.choice(len(centers), 2, replace=False)
            
            # Generate composition between clusters
            mix = np.random.random()
            new_comp = centers[c1] * mix + centers[c2] * (1 - mix)
            
            # Convert to dictionary
            elements = sorted(set().union(*compositions))
            new_compositions.append(dict(zip(elements, new_comp)))
            
        return new_compositions
#!/usr/bin/env python
"""Test script for gradient-based adversarial attack."""

import argparse
import os
import ase.io
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from forge.core.adversarial_attack import (
    GradientAdversarialCalculator,
    GradientAscentOptimizer
)

def run_gradient_adversarial_attack(
    xyz_file: str,
    output_dir: str,
    model_paths: list[str],
    learning_rate: float = 0.01,
    n_iterations: int = 60,
    min_distance: float = 1.5,
    include_probability: bool = False,
    temperature: float = 0.86,
    device: str = "cuda",
    use_hessian: bool = True,
):
    """Run gradient-based adversarial attack on structures in XYZ file.
    
    Args:
        xyz_file: Path to input XYZ file containing structures
        output_dir: Directory to save optimization results
        model_paths: List of paths to model files
        learning_rate: Learning rate for gradient ascent
        n_iterations: Number of optimization iterations
        min_distance: Minimum allowed distance between atoms (Å)
        include_probability: Whether to include the probability term in the loss
        temperature: Temperature for probability weighting (eV)
        device: Device to run on (cpu/cuda)
        use_hessian: Whether to use the Hessian for more efficient gradient calculation
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize calculator and optimizer
    calculator = GradientAdversarialCalculator(
        model_paths=model_paths,
        device=device,
        temperature=temperature,
        use_autograd=use_hessian
    )
    optimizer = GradientAscentOptimizer(
        calculator=calculator,
        learning_rate=learning_rate,
        include_probability=include_probability
    )
    
    # Load structures
    atoms_list = ase.io.read(xyz_file, ':')
    print(f"[INFO] Loaded {len(atoms_list)} structures from {xyz_file}")
    
    # Run optimization for each structure
    results = []
    
    for i, atoms in enumerate(atoms_list):
        struct_name = atoms.info.get('structure_name', f'struct_{i}')
        print(f"\n[INFO] Optimizing structure: {struct_name}")
        
        # Calculate initial variance
        initial_variance = calculator.calculate_loss(atoms, include_probability)
        print(f"[INFO] Initial variance: {initial_variance:.6f}")
        
        # Run optimization
        best_atoms, best_variance, loss_history = optimizer.optimize(
            atoms=atoms,
            n_iterations=n_iterations,
            min_distance=min_distance,
            output_dir=output_dir
        )
        
        # Generate variance plot
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Force Variance')
        plt.title(f'Force Variance vs. Iteration - {struct_name}')
        plt.grid(True)
        plt.savefig(output_path / f"{struct_name}_variance_plot.png")
        plt.close()
        
        print(f"[INFO] Optimization complete for {struct_name}")
        print(f"[INFO] Initial variance: {initial_variance:.6f}")
        print(f"[INFO] Final variance: {best_variance:.6f}")
        print(f"[INFO] Improvement: {best_variance/initial_variance:.2f}x")
        print(f"[INFO] Results saved to {output_dir}")
        
        results.append({
            'structure_name': struct_name,
            'initial_variance': initial_variance,
            'final_variance': best_variance,
            'improvement_factor': best_variance/initial_variance,
            'trajectory_file': f"{struct_name}_adversarial.xyz",
            'variance_plot': f"{struct_name}_variance_plot.png"
        })
    
    # Save results summary
    import json
    with open(output_path / 'results_summary.json', 'w') as f:
        json.dump({
            'input_file': xyz_file,
            'parameters': {
                'learning_rate': learning_rate,
                'n_iterations': n_iterations,
                'min_distance': min_distance,
                'include_probability': include_probability,
                'temperature': temperature,
                'device': device,
                'use_hessian': use_hessian
            },
            'results': results
        }, f, indent=2)
    
    print("\n[INFO] All optimizations complete!")
    print(f"[INFO] Results saved to {output_dir}")
    print("[INFO] You can now visualize the trajectories in OVITO")

def main():
    parser = argparse.ArgumentParser(
        description="Test gradient-based adversarial attack optimization."
    )
    parser.add_argument(
        "xyz_file",
        help="Input XYZ file containing structures"
    )
    parser.add_argument(
        "output_dir",
        help="Directory to save optimization results"
    )
    parser.add_argument(
        "--model_paths",
        nargs='+',
        required=True,
        help="Paths to model files"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for gradient ascent"
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=60,
        help="Number of optimization iterations"
    )
    parser.add_argument(
        "--min_distance",
        type=float,
        default=1.5,
        help="Minimum allowed distance between atoms (Å)"
    )
    parser.add_argument(
        "--include_probability",
        action="store_true",
        help="Include probability term in adversarial loss"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.86,
        help="Temperature for probability weighting (eV)"
    )
    parser.add_argument(
        "--device",
        choices=['cpu', 'cuda'],
        default='cuda',
        help="Device to run on"
    )
    parser.add_argument(
        "--use_hessian",
        action="store_true",
        default=True,
        help="Use Hessian for more efficient gradient calculation"
    )
    
    args = parser.parse_args()
    
    run_gradient_adversarial_attack(
        xyz_file=args.xyz_file,
        output_dir=args.output_dir,
        model_paths=args.model_paths,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        min_distance=args.min_distance,
        include_probability=args.include_probability,
        temperature=args.temperature,
        device=args.device,
        use_hessian=args.use_hessian
    )

if __name__ == "__main__":
    main() 
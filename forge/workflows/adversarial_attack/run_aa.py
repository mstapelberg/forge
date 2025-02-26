#!/usr/bin/env python
"""Script to run adversarial attack optimization on a batch of structures."""

import argparse
from pathlib import Path
import ase.io
from forge.core.adversarial_attack import (
    AdversarialCalculator,
    DisplacementGenerator,
    AdversarialOptimizer
)

def run_gradient_aa_optimization(
    xyz_file: str,
    output_dir: str,
    model_paths: list[str],
    learning_rate: float = 0.01,
    n_iterations: int = 60,
    min_distance: float = 1.5,
    include_probability: bool = False,
    temperature: float = 0.86,
    device: str = "cuda",
    use_autograd: bool = False,
):
    """Run gradient-based adversarial attack optimization on structures in XYZ file.
    
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
        use_autograd: Whether to use the Hessian from MACECalculator for more efficient gradient calculation
    """
    from pathlib import Path
    import ase.io
    from forge.core.adversarial_attack import (
        GradientAdversarialCalculator,
        GradientAscentOptimizer
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize calculator and optimizer
    calculator = GradientAdversarialCalculator(
        model_paths=model_paths,
        device=device,
        temperature=temperature,
        use_autograd=use_autograd
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
    for atoms in atoms_list:
        struct_name = atoms.info.get('structure_name', 'unknown')
        initial_variance = atoms.info.get('initial_variance', None)
        print(f"\n[INFO] Optimizing structure: {struct_name}")
        if initial_variance:
            print(f"[INFO] Initial variance: {initial_variance:.6f}")
        
        # Run optimization
        best_atoms, best_variance, loss_history = optimizer.optimize(
            atoms=atoms,
            n_iterations=n_iterations,
            min_distance=min_distance,
            output_dir=output_dir
        )
        
        results.append({
            'structure_name': struct_name,
            'initial_variance': initial_variance or loss_history[0],
            'final_variance': best_variance,
            'loss_history': loss_history
        })
    
    # Save summary
    import json
    with open(output_path / 'optimization_summary.json', 'w') as f:
        json.dump({
            'input_file': xyz_file,
            'parameters': {
                'learning_rate': learning_rate,
                'n_iterations': n_iterations,
                'min_distance': min_distance,
                'include_probability': include_probability,
                'temperature': temperature,
                'device': device,
                'use_autograd': use_autograd
            },
            'results': results
        }, f, indent=2)

def run_aa_optimization(
    xyz_file: str,
    output_dir: str,
    model_paths: list[str],
    temperature: float = 1200.0,
    max_steps: int = 50,
    patience: int = 25,
    min_distance: float = 2.0,
    mode: str = "all",
    device: str = "cuda",
):
    """Run adversarial attack optimization on structures in XYZ file.
    
    Args:
        xyz_file: Path to input XYZ file containing structures
        output_dir: Directory to save optimization results
        model_paths: List of paths to model files
        temperature: Temperature for adversarial optimization (K)
        max_steps: Maximum optimization steps per structure
        patience: Stop if no improvement after this many steps
        min_distance: Minimum allowed distance between atoms (Å)
        mode: Optimization mode ('all' or 'single' atom)
        device: Device to run on (cpu/cuda)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize calculator and optimizer
    calculator = AdversarialCalculator(
        model_paths=model_paths,
        device=device
    )
    displacement_gen = DisplacementGenerator(min_distance=min_distance)
    optimizer = AdversarialOptimizer(
        adversarial_calc=calculator,
        displacement_gen=displacement_gen
    )
    
    # Load structures
    atoms_list = ase.io.read(xyz_file, ':')
    print(f"[INFO] Loaded {len(atoms_list)} structures from {xyz_file}")
    
    # Run optimization for each structure
    results = []
    for atoms in atoms_list:
        struct_name = atoms.info.get('structure_name', 'unknown')
        initial_variance = atoms.info.get('initial_variance', None)
        print(f"\n[INFO] Optimizing structure: {struct_name}")
        if initial_variance:
            print(f"[INFO] Initial variance: {initial_variance:.6f}")
        
        # Run optimization
        best_atoms, best_variance, accepted_moves = optimizer.optimize(
            atoms=atoms,
            temperature=temperature,
            max_iterations=max_steps,
            patience=patience,
            mode=mode,
            output_dir=output_dir
        )
        
        results.append({
            'structure_name': struct_name,
            'initial_variance': initial_variance,
            'final_variance': best_variance,
            'accepted_moves': accepted_moves
        })
    
    # Save summary
    import json
    with open(output_path / 'optimization_summary.json', 'w') as f:
        json.dump({
            'input_file': xyz_file,
            'parameters': {
                'temperature': temperature,
                'max_steps': max_steps,
                'patience': patience,
                'min_distance': min_distance,
                'mode': mode,
                'device': device
            },
            'results': results
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Run adversarial attack optimization on structures."
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
        "--gradient",
        action="store_true",
        help="Use gradient-based optimization instead of Metropolis"
    )
    
    # Parameters for Metropolis optimization
    parser.add_argument(
        "--temperature",
        type=float,
        default=1200.0,
        help="Temperature for adversarial optimization (K) or energy scaling (eV)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Maximum optimization steps per structure (Metropolis mode)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Stop if no improvement after this many steps (Metropolis mode)"
    )
    
    # Parameters for gradient-based optimization
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for gradient ascent (gradient mode)"
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=60,
        help="Number of optimization iterations (gradient mode)"
    )
    parser.add_argument(
        "--include_probability",
        action="store_true",
        help="Include probability term in adversarial loss (gradient mode)"
    )
    parser.add_argument(
        "--use_autograd",
        action="store_true",
        help="Use Hessian-based gradient calculation (faster but requires MACECalculator.get_hessian)"
    )
    
    # Common parameters
    parser.add_argument(
        "--min_distance",
        type=float,
        default=2.0,
        help="Minimum allowed distance between atoms (Å)"
    )
    parser.add_argument(
        "--mode",
        choices=['all', 'single'],
        default='all',
        help="Optimization mode ('all' or 'single' atom) (Metropolis mode)"
    )
    parser.add_argument(
        "--device",
        choices=['cpu', 'cuda'],
        default='cuda',
        help="Device to run on"
    )
    
    args = parser.parse_args()
    
    if args.gradient:
        # Use gradient-based optimization
        from forge.core.adversarial_attack import (
            GradientAdversarialCalculator,
            GradientAscentOptimizer
        )
        
        run_gradient_aa_optimization(
            xyz_file=args.xyz_file,
            output_dir=args.output_dir,
            model_paths=args.model_paths,
            learning_rate=args.learning_rate,
            n_iterations=args.n_iterations,
            min_distance=args.min_distance,
            include_probability=args.include_probability,
            temperature=args.temperature,
            device=args.device,
            use_autograd=args.use_autograd
        )
    else:
        # Use original Metropolis optimization
        run_aa_optimization(
            xyz_file=args.xyz_file,
            output_dir=args.output_dir,
            model_paths=args.model_paths,
            temperature=args.temperature,
            max_steps=args.max_steps,
            patience=args.patience,
            min_distance=args.min_distance,
            mode=args.mode,
            device=args.device
        )

def _main():
    parser = argparse.ArgumentParser(
        description="Run adversarial attack optimization on structures."
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
        "--temperature",
        type=float,
        default=1200.0,
        help="Temperature for adversarial optimization (K)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Maximum optimization steps per structure"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Stop if no improvement after this many steps"
    )
    parser.add_argument(
        "--min_distance",
        type=float,
        default=2.0,
        help="Minimum allowed distance between atoms (Å)"
    )
    parser.add_argument(
        "--mode",
        choices=['all', 'single'],
        default='all',
        help="Optimization mode ('all' or 'single' atom)"
    )
    parser.add_argument(
        "--device",
        choices=['cpu', 'cuda'],
        default='cuda',
        help="Device to run on"
    )
    
    args = parser.parse_args()
    run_aa_optimization(
        xyz_file=args.xyz_file,
        output_dir=args.output_dir,
        model_paths=args.model_paths,
        temperature=args.temperature,
        max_steps=args.max_steps,
        patience=args.patience,
        min_distance=args.min_distance,
        mode=args.mode,
        device=args.device
    )


if __name__ == "__main__":
    main() 
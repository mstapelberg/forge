# Adversarial Attack Workflow

This guide explains how to run the adversarial attack workflow, from splitting initial structures to running attacks.

## Prerequisites
- Python environment with ASE, numpy, monty
- SLURM cluster access
- Required model files

## Step 1: Split XYZ File
Split large xyz file into smaller parts for parallel processing:
```bash
python split_xyz.py input.xyz 24
```
This creates files named `input_part0.xyz`, `input_part1.xyz`, etc.

## Step 2: Calculate Variances
1. Edit `variance_calculation.slurm` to set correct paths:
   - BASE_XYZ: path to your split xyz files
   - OUTPUT_DIR: where variance results will be saved
   - MODEL_PATHS: paths to your model files

2. Submit the job:
```bash
sbatch variance_calculation.slurm
```
This runs 24 parallel jobs, calculating force variances for each structure.

## Step 3: Combine Variance Results
After variance calculations complete, combine results and select structures:
```bash
python combine_variances.py \
    path/to/variance_results \
    path/to/original.xyz \
    --output_dir path/to/output \
    --n_structures 120
```
This:
- Combines all variance results
- Selects top 120 structures
- Creates 24 batch files (5 structures each)
- Saves metadata and combined results

## Step 4: Run Adversarial Attacks
1. Edit `adversarial_attack.slurm` to set correct paths:
   - BATCH_DIR: where combined variance results are
   - OUTPUT_DIR: where attack results will be saved
   - MODEL_PATHS: paths to your model files

2. Submit the job:
```bash
sbatch adversarial_attack.slurm
```
This runs 24 parallel jobs, each processing 5 structures.

## Directory Structure
```
project/
├── data/
│   └── system_name/
│       ├── original.xyz
│       ├── variance_results/
│       └── aa_results/
└── scripts/
    ├── split_xyz.py
    ├── calculate_variance.py
    ├── combine_variances.py
    ├── run_aa.py
    ├── variance_calculation.slurm
    └── adversarial_attack.slurm
```

## Additional Tools
Combine all results into single xyz:
```bash
python combine_xyz.py path/to/aa_results combined_results.xyz
```

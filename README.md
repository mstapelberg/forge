# Forge

A HPC workflow management and materials analysis toolkit built with Python, ASE, and MACE.

# Features
- Adversarial Attack workflows for materials potential uncertainty
- HPC job templates for VASP, MACE training, NEB, etc.
- Composition analysis (short-range order, clustering, etc.)

# Installation

## Step 1: Install PyTorch and mace‑torch Manually

Forge does **not** include PyTorch or mace‑torch in its dependency list. Please install them manually based on your system's requirements. Additionally, you will need a CUDA-enabled GPU to run FORGE (and preferably slurm for spawning multiple jobs at once).

```bash
### Setting up forge conda/mamba environment
conda create -n forge python=3.11

# for cuda 12.6
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip3 install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12

# for cuda 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu11

# Optional for WANDB logging during training
pip3 install wandb

## Installing MACE 
git clone https://github.com/ACEsuit/mace.git
pip install ./mace[wandb] # ./mace without wandb
```

## Step 2: Install Forge and dependencies
All of the needed dependencies are specified in the `pyproject.toml` file and can be installed with `pip install -e .` from the repository root.
```bash
## Basic installation of Forge
git clone https://github.com/mstapelberg/forge.git
cd forge

pip install -e .
```

# Quick Usage
Refer to the scripts in `forge/scratch/scripts` or notebooks in `forge/scratch/notebooks` for examples. 

# Repository Layout
- `analysis/` - analysis scripts, e.g. Warren-Cowley
- `core/` - database manager, calculators
- `workflows/` - HPC job generation scripts, templates
- `tests/` - test suite
- `config/` - example database config
- `scratch/` - current scratch work repository 

## HPC Profiles
You can create profiles for clusters that you use. An example is put together for Perlmutter-CPU and Perlmutter-GPU. 
It is advised to have different types of profiles for jobs on different clusters.
In the future we will implement folders for each cluster that you can save your job templates to. 

## Adversarial Attack Workflow
1. **Calculate variance**: ...
2. **Run optimization**: ...
3. **Generate Structures**: ...
4. **Pick Structures for VASP**: ...

## Configuration
The central tenant of FORGE is using AWS relational databases. Configuration of your database is done in `config/database.yaml`. This allows you to just call DatabaseManager() without passing in a config file.

## Contributing
We use ruff for linting and black for formatting. Contact myless@mit.edu if you'd like to contribute or drop an issue/discussion.

## License
[MIT License](./LICENSE)

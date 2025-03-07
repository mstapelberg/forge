# Forge

A HPC workflow management and materials analysis toolkit built with Python, ASE, and MACE.

# Features
- Adversarial Attack workflows for materials potential uncertainty
- HPC job templates for VASP, MACE training, NEB, etc.
- Composition analysis (short-range order, clustering, etc.)

# Installation
```bash
### Setting up forge conda/mamba environment
conda create -n forge python=3.11

# for cuda 12.6
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip3 install cuequivariance==0.1.0 cuequivariance-torch==0.1.0 cuequivariance-ops-torch-cu12==0.1.0

# for cuda 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install cuequivariance==0.1.0 cuequivariance-torch==0.1.0 cuequivariance-ops-torch-cu11==0.1.0

# Optional for WANDB logging during training
pip3 install wandb

## Installing MACE 
git clone https://github.com/ACEsuit/mace.git
pip install ./mace[wandb] # ./mace without wandb

## Basic installation of Forge
git clone https://github.com/mstapelberg/forge.git
cd forge

pip install -e .
```

# Quick Usage
Examples of CLI usage or minimal code:
```bash
forge calculate-variance --xyz_file data/structures.xyz --output_dir results ...
forge create-aa-jobs ...
forge run-aa-jobs ...
forge create-aa-vasp-jobs ...
```

# Repository Layout
- `analysis/` - analysis scripts, e.g. Warren-Cowley
- `core/` - database manager, calculators
- `workflows/` - HPC job generation scripts, templates
- `tests/` - test suite
- `config/` - example database config, HPC profiles

## HPC Profiles
Describe usage of HPC JSON templates.

## Adversarial Attack Workflow
1. **Calculate variance**: ...
2. **Run optimization**: ...
3. **Generate VASP jobs**: ...

## Configuration
Points about \`config/database.yaml\`, environment variables, etc.

## Contributing
Guidelines for dev environment setup, testing, linting, etc.

## License
[MIT License](./LICENSE) or whichever is relevant
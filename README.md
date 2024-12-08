# forge
FORGE (Flexible Optimizer for Rapid Generation and Exploration) to guide machine learning interatomic potential development. 



## Structure of the Package
forge/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── calculator.py     # Base calculator interface
│   ├── database.py      # PostgreSQL integration
│   └── config.py        # Configuration management
├── potentials/
│   ├── __init__.py
│   └── mace.py          # MACE calculator implementation (priority)
├── analysis/
│   ├── __init__.py
│   ├── clustering.py     # Composition space analysis
│   ├── ensemble.py       # Ensemble predictions analysis
│   └── visualization.py  # Plotting utilities
├── workflows/
│   ├── __init__.py
│   ├── md.py            # MD simulation workflow
│   ├── adversarial.py   # Adversarial attack workflow
│   └── slurm.py         # SLURM job management
├── utils/
│   ├── __init__.py
│   └── io.py            # File I/O utilities
├── tests/
│   ├── __init__.py
│   └── test_core.py
├── setup.py
├── pyproject.toml
├── config/
│   ├── database.yaml    # Database configuration
│   └── slurm.yaml       # SLURM templates and configs
└── README.md
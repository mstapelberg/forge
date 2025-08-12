# Creating Defect Motifs Scripts

This folder contains scripts for generating defect structures, running adversarial attacks, and creating VASP jobs. Each script serves a specific purpose in the defect motif workflow.

## Script Overview

### 🎯 **Two-Step Workflow (RECOMMENDED)**

**Step 1: `full_defect_motif_workflow_step1.py`**
- Generate defect structures for target compositions
- Run adversarial attacks using **Allegro** models
- Save trajectory XYZ files for later processing

**Step 2: `full_defect_motif_workflow_step2.py`**
- Load trajectory files from Step 1
- Use **MACE** models and `aa_selection` to choose diverse structures
- Create VASP jobs for the selected structures

**Usage:**
```bash
# Step 1: Generate and attack with Allegro
python full_defect_motif_workflow_step1.py --compositions compositions.json \
  --model-paths model1.nequip.zip model2.nequip.zip --output-dir step1_output

# Step 2: Select and create VASP jobs with MACE
python full_defect_motif_workflow_step2.py --input-dir step1_output \
  --mace-model-paths model1.model model2.model --output-dir step2_output
```

**This is the recommended approach!** It allows you to use the best tool for each step.

---

### `full_defect_motif_workflow.py` (Single-Step Alternative)
**Complete end-to-end workflow that combines all steps in one script:**
1. Generate defect structures for target compositions
2. Run adversarial attacks on the structures  
3. Use `aa_selection` to choose diverse structures from trajectories
4. Create VASP jobs for the selected structures

**Usage:**
```bash
python full_defect_motif_workflow.py --compositions compositions.json \
  --model-paths model1.model model2.model --output-dir workflow_output
```

**Note:** This requires both Allegro and MACE to be available in the same environment.

---

### `generate_defect_motifs.py`
**Purpose:** Generate defect structures and create VASP jobs directly
- Takes compositions → generates defect structures → creates VASP job directories
- **No adversarial attacks involved**
- Creates static VASP jobs for the initial defect structures

**Usage:**
```bash
python generate_defect_motifs.py --compositions compositions.json --output-dir vasp_jobs
```

---

### `defect_adversarial_attack.py`
**Purpose:** Run adversarial attacks on defect structures and generate trajectories
- Takes defect structures → runs adversarial optimization → produces trajectories
- **No VASP job creation**
- Outputs trajectory files (.xyz) with optimized structures
- Uses factory pattern with MACE/Allegro backends

**Usage:**
```bash
python defect_adversarial_attack.py --compositions compositions.json \
  --model-paths model1.model model2.model --top-n 10 --generation 1
```

---

### `example_defect_aa.py`
**Purpose:** Example usage of the adversarial attack workflow
- Demonstrates how to use `defect_adversarial_attack.py`
- Shows both MACE and Allegro model usage
- Good starting point for understanding the workflow

### `example_full_workflow.py`
**Purpose:** Example usage of the complete workflow
- Demonstrates how to use `full_defect_motif_workflow.py`
- Shows MACE, Allegro, and auto-detection examples
- Good starting point for the complete workflow

---

## Test Scripts

### `test_allegro_integration.py`
Tests Allegro backend integration with adversarial attacks.

### `test_mace_integration.py` 
Tests MACE backend integration with adversarial attacks.

### `test_basic_refactor.py`
Tests basic refactoring functionality.

### `test_refactor_integration.py`
Tests integration of refactored components.

### `test_structure_id_handling.py`
Tests structure ID handling in the workflow.

### `test_core_functionality.py`
**Purpose:** Test core functionality before running the full workflow
- Tests composition loading from JSON
- Tests SIA and di-SIA defect generation
- Tests interstitial detection in both defect types
- Tests calculator factory functionality
- **Run this first to verify everything works!**

---

## Workflow Comparison

| Script | Defect Generation | Adversarial Attacks | Structure Selection | VASP Jobs |
|--------|------------------|-------------------|-------------------|-----------|
| `generate_defect_motifs.py` | ✅ | ❌ | ❌ | ✅ |
| `defect_adversarial_attack.py` | ✅ | ✅ | ❌ | ❌ |
| `full_defect_motif_workflow.py` | ✅ | ✅ | ✅ | ✅ |
| `example_full_workflow.py` | ✅ | ✅ | ✅ | ✅ |

## Quick Start

For your goal of generating defect motifs with adversarial attacks and creating VASP jobs:

1. **Test core functionality first:**
```bash
python test_core_functionality.py
```

2. **Create a compositions file:**
```bash
python generate_defect_motifs.py --create-example compositions.json
```

3. **Run the two-step workflow (RECOMMENDED):**

**Step 1: Generate and attack with Allegro**
```bash
python full_defect_motif_workflow_step1.py \
  --compositions compositions.json \
  --model-paths model1.nequip.zip model2.nequip.zip \
  --output-dir step1_output
```

**Step 2: Select and create VASP jobs with MACE**
```bash
python full_defect_motif_workflow_step2.py \
  --input-dir step1_output \
  --mace-model-paths model1.model model2.model \
  --output-dir step2_output
```

This will:
- Generate defect structures for your compositions
- Run adversarial attacks using Allegro models
- Save trajectory files for later processing
- Use MACE and `aa_selection` to pick diverse structures from trajectories
- Create VASP jobs for the selected structures

**Alternative: Single-step workflow (requires both Allegro and MACE in same environment)**
```bash
python full_defect_motif_workflow.py \
  --compositions compositions.json \
  --model-paths model1.model model2.model \
  --output-dir my_workflow_output
```

## Backend Support

All scripts support both MACE and Allegro backends through the factory pattern:

- **MACE models:** `.model` files
- **Allegro models:** `.nequip.zip` files (packaged models)

For Allegro models, you'll need to provide a species mapping:
```bash
--species-mapping '{"Ti": "Ti", "V": "V", "Cr": "Cr", "Zr": "Zr", "W": "W"}'
```

## Output Structure

The `full_defect_motif_workflow.py` creates:
```
workflow_output/
├── adversarial_attacks/     # Adversarial attack results
├── vasp_jobs/              # VASP job directories
├── structure_selection_plot.png  # Selection visualization
└── workflow_stats.json     # Overall statistics
```

## Detailed Documentation

- `README_defect_motifs.md` - Detailed documentation for defect motif generation
- `README_defect_adversarial_attacks.md` - Detailed documentation for adversarial attacks  
- `README_allegro_integration.md` - Detailed documentation for Allegro integration 
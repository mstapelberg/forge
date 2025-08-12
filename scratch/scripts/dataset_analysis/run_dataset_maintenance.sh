#!/bin/bash
#SBATCH --job-name=analyze_db
#SBATCH --output=analyze_db.out
#SBATCH --error=analyze_db.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH --time=5-06:00:00
#SBATCH -p regular

# Load Conda environment
source /home/myless/.mambaforge/etc/profile.d/conda.sh
conda activate forge-allegro

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=8

# Run Allegro training with the provided YAML
python dataset_maintenance.py analyze-db --model /home/myless/Packages/forge/scratch/data/potentials/allegro/gen-8-aa/*.nequip.zip --backend allegro --device cuda --rare-quantile 0.95 --output-dir /home/myless/Packages/forge/scratch/data/dataset_analysis_output_gen_9 --generation 9

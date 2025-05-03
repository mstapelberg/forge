#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${JOB_NAME}.out
#SBATCH --error=${JOB_NAME}.err
#SBATCH --ntasks-per-node=${GPU_COUNT}
#SBATCH --cpus-per-task=8
#SBATCH --nodes=${NUM_NODES}
#SBATCH --gres=gpu:${GPU_COUNT}
#SBATCH --constraint=${GPU_TYPE}
#SBATCH --time=5-06:00:00
#SBATCH -p regular

# Load Conda environment
source /home/myless/.mambaforge/etc/profile.d/conda.sh
conda activate allegro-new

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=8

# Run Allegro training with the provided YAML
srun nequip-train -cn ${YAML_FILE}
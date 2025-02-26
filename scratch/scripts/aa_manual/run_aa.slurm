#!/bin/bash
#SBATCH --job-name=adv_attack
#SBATCH --output=aa_gen_%A_%a.out
#SBATCH --error=aa_gen_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=5-06:00:00
#SBATCH --array=0-23
#SBATCH -p regular

# Load Conda environment
source /home/myless/.mambaforge/etc/profile.d/conda.sh
conda activate mace-11.7

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=10

# Define directories
BATCH_DIR="../../data/zr-w-v-ti-cr/gen_4_2024-11-13/variance_results"
OUTPUT_DIR="../../data/zr-w-v-ti-cr/gen_4_2024-11-13/aa_results"

# Get the batch file for this array job
INPUT_FILE="${BATCH_DIR}/batch_${SLURM_ARRAY_TASK_ID}.xyz"
BATCH_OUTPUT_DIR="${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}"


# Model paths - adjust these as needed
MODEL_PATHS="../../Models/zr-w-v-ti-cr/gen_2_2024-11-11/gen_2_model_0-11-11_stagetwo_compiled.model \
             ../../Models/zr-w-v-ti-cr/gen_2_2024-11-11/gen_2_model_1-11-11_stagetwo_compiled.model \
             ../../Models/zr-w-v-ti-cr/gen_2_2024-11-11/gen_2_model_2-11-11_stagetwo_compiled.model \
             ../../Models/zr-w-v-ti-cr/gen_3_2024-11-13/gen_3_model_0-11-13_stagetwo_compiled.model \
             ../../Models/zr-w-v-ti-cr/gen_3_2024-11-13/gen_3_model_1-11-13_stagetwo_compiled.model \
             ../../Models/zr-w-v-ti-cr/gen_3_2024-11-13/gen_3_model_2-11-13_stagetwo_compiled.model"

# Run the adversarial attack script
python run_aa.py "$INPUT_FILE" "$BATCH_OUTPUT_DIR" --model_paths $MODEL_PATHS --n_structures 5 --temperature 1200 --device cuda

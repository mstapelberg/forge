#!/bin/bash
#SBATCH --job-name=calc_var
#SBATCH --output=var_calc_%A_%a.out
#SBATCH --error=var_calc_%A_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-15 #change this number based on how man u split 
#SBATCH -p regular

# Load Conda environment
source /home/myless/.mambaforge/etc/profile.d/conda.sh
conda activate mace-11.7

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=10

# Define input and output directories
BASE_XYZ="../../data/zr-w-v-ti-cr/gen_4_2024-11-14/gen_4_2024-11-14"
OUTPUT_DIR="../../data/zr-w-v-ti-cr/gen_4_2024-11-14/variance_results"

# Get the input file for this array job
INPUT_FILE="${BASE_XYZ}_part${SLURM_ARRAY_TASK_ID}.xyz"

# Model paths - adjust these as needed
MODEL_PATHS="../../Models/zr-w-v-ti-cr/gen_2_2024-11-11/gen_2_model_0-11-11_stagetwo_compiled.model \
             ../../Models/zr-w-v-ti-cr/gen_2_2024-11-11/gen_2_model_1-11-11_stagetwo_compiled.model \
             ../../Models/zr-w-v-ti-cr/gen_2_2024-11-11/gen_2_model_2-11-11_stagetwo_compiled.model \
             ../../Models/zr-w-v-ti-cr/gen_3_2024-11-13/gen_3_model_0-11-13_stagetwo_compiled.model \
             ../../Models/zr-w-v-ti-cr/gen_3_2024-11-13/gen_3_model_1-11-13_stagetwo_compiled.model \
             ../../Models/zr-w-v-ti-cr/gen_3_2024-11-13/gen_3_model_2-11-13_stagetwo_compiled.model "

# Run the variance calculation script
python calculate_variance.py "$INPUT_FILE" "$OUTPUT_DIR" --model_paths $MODEL_PATHS --device cuda

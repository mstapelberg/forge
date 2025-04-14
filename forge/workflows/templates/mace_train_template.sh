#!/bin/bash
#
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${JOB_NAME}.out
#SBATCH --error=${JOB_NAME}.err
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gres=gpu:${GPUS_PER_NODE}
#SBATCH --constraint=${GPU_TYPE}
#SBATCH --time=5-06:00:00
#SBATCH -p regular

# Load Conda environment
source /home/myless/.mambaforge/etc/profile.d/conda.sh

# Activate the 'allegro' environment
conda activate mace-cueq

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=10

srun python /home/myless/Packages/mace_all/mace/scripts/run_train.py \
    --name='${RUN_NAME}' \
    --model='MACE' \
    --num_interactions=${NUM_INTERACTIONS} \
    --num_channels=${NUM_CHANNELS} \
    --max_L=${MAX_L} \
    --correlation=3 \
    --E0s='${E0S_STR}' \
    --loss='stress' \
    --forces_weight=${FORCES_WEIGHT} \
    --energy_weight=${ENERGY_WEIGHT} \
    --stress_weight=${STRESS_WEIGHT} \
    --compute_stress=True \
    --compute_forces=True \
    --energy_key='REF_energy' \
    --forces_key='REF_force' \
    --stress_key='REF_stress' \
    --r_max=${R_MAX} \
    --lr=${LR} \
    --train_file="data/${BASE_NAME}_train.xyz" \
    --valid_file="data/${BASE_NAME}_val.xyz" \
    --test_file="data/${BASE_NAME}_test.xyz" \
    --swa \
    --swa_lr=0.0005 \
    --start_swa=160 \
    --swa_energy_weight=1000 \
    --num_workers=1 \
    --batch_size=4 \
    --valid_batch_size=4 \
    --max_num_epochs=200 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --scheduler='ReduceLROnPlateau' \
    --scheduler_patience=25 \
    --lr_scheduler_gamma=0.99 \
    --error_table='PerAtomRMSEstressvirials' \
    --default_dtype='float32' \
    --device='cuda' \
    --enable_cueq="True" \
    --distributed \
    --restart_latest \
    --seed=${SEED} \
    --wandb \
    --wandb_name='${WANDB_NAME}' \
    --wandb_project='mace_vcrtiwzr_fixed_al' \
    --wandb_entity='mnm-shortlab-mit' \
    --wandb_dir='.'
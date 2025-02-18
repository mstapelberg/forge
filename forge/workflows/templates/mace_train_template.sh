#!/bin/bash
#
#SBATCH --job-name=gen_6_model_0_L0_isolated_energy
#SBATCH --output=gen_6_model_0_L0_isolated_energy.out
#SBATCH --error=gen_6_model_0_L0_isolated_energy.err
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=rtx6000
#SBATCH --time=5-06:00:00
#SBATCH -p regular

# Load Conda environment
source /home/myless/.mambaforge/etc/profile.d/conda.sh

# Activate the 'allegro' environment
conda activate mace-cueq

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=10

srun python /home/myless/Packages/mace_all/mace/scripts/run_train.py \
    --name='gen_6_model_0_L0_isolated-2026-01-16' \
    --model='MACE' \
    --num_interactions=2 \
    --num_channels=128 \
    --max_L=0 \
    --correlation=3 \
    --E0s='{22: -2.15203187, 23 : -3.55411419, 24 : -5.42767241, 40 : -2.3361286, 74 : -4.55186158}' \
    --loss='stress' \
    --forces_weight=50.0 \
    --energy_weight=1.0 \
    --stress_weight=25.0 \
    --compute_stress=True \
    --compute_forces=True \
    --energy_key='REF_energy' \
    --forces_key='REF_force' \
    --stress_key='REF_stress' \
    --r_max=5.0 \
    --lr=0.001 \
    --train_file="data/gen_6_2025-01-16_train.xyz" \
    --valid_file="data/gen_6_2025-01-16_val.xyz" \
    --test_file="data/gen_6_2025-01-16_test.xyz" \
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
    --seed=0 \
    --wandb \
    --wandb_name='gen_6_model_0_L0_isolated-2025-01-16' \
    --wandb_project='mace_vcrtiwzr_fixed_al' \
    --wandb_entity='mnm-shortlab-mit' \
    --wandb_dir='.'
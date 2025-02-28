#!/bin/bash
#SBATCH --job-name=aa_id22090_var0.0577
#SBATCH --output=vasp-gpu-%j.out
#SBATCH --error=vasp-gpu-%j.err
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --constraint=v100s
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8

# Module loading


# Environment setup
export OMP_NUM_THREADS=8
export VASP_EXE=/home/myless/VASP/vasp.6.4.2/bin/vasp_std
export MPI_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/comm_libs/mpi/bin/mpirun

# Job execution
cd $SLURM_SUBMIT_DIR
${MPI_PATH} -np 4 ${VASP_EXE} > "vasp_out.log" 2> "vasp_err.log"

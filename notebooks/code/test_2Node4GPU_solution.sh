#! /bin/bash
# SBATCH --job-name=dli_firstSlurmJob
# SBATCH --nodes 2
# SBATCH --ntasks-per-node 1
# SBATCH --time=48:00:00



set -x

echo "Master execution:"
hostname
nvidia-smi
echo "Distributed execution:"
srun hostname
srun nvidia-smi

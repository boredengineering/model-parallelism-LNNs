#!/bin/bash

#SBATCH -N 2
#SBATCH --job-name=dli_secondSlurmJob
#SBATCH -o /dli/megatron/logs/%j.out
#SBATCH -e /dli/megatron/logs/%j.err

srun -l /dli/code/test.sh   

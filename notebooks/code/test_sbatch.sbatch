#!/bin/bash

#SBATCH -N 2
#SBATCH --job-name=dli_firstSlurmJob
#SBATCH -o /dli/megatron/logs/%j.out
#SBATCH -e /dli/megatron/logs/%j.err

srun -l /dli/code/test.sh  

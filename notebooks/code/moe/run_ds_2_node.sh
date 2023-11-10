#!/bin/bash
#SBATCH --job-name=dli_moe2nodes
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=32 ### Number of threads per task (OMP threads)
#SBATCH -o /dli/megatron/logs/%j.out
#SBATCH -e /dli/megatron/logs/%j.err

# Number of nodes
NUM_NODES=2
# Number of GPUs per node
NUM_GPUS=2


deepspeed --num_nodes=${NUM_NODES} --hostfile /dli/code/moe/hostfile --num_gpus=${NUM_GPUS} cifar10_deepspeed.py \
	--log-interval 100 \
	--deepspeed \
	--deepspeed_config ./ds_config.json

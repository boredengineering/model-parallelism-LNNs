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
# Size of expert parallel world (should be less than total world size)
EP_SIZE=2
# Number of total experts per layer
EXPERTS=2
# Number of total expert layers
EXPERT_LAYERS=2

/home/admin/.local/bin/deepspeed --hostfile /dli/code/moe/hostfile --num_nodes=${NUM_NODES} --num_gpus=${NUM_GPUS} /dli/code/moe/cifar10_deepspeed.py \
	--log-interval 100 \
	--deepspeed \
	--deepspeed_config /dli/code/moe/ds_config.json \
	--moe \
	--ep-world-size ${EP_SIZE} \
	--num-experts-per-layer ${EXPERTS} \
    --num-expert-layers ${EXPERT_LAYERS} \
	--top-k 1 \
	--noisy-gate-policy 'RSample' \
	--moe-param-group

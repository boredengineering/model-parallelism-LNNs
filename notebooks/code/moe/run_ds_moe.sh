#!/bin/bash

# Number of nodes
NUM_NODES=1
# Number of GPUs per node
NUM_GPUS=2
# Size of expert parallel world (should be less than total world size)
EP_SIZE=2
# Number of total experts per layer
EXPERTS=2
# Number of total expert layers
EXPERT_LAYERS=2

deepspeed --num_nodes=${NUM_NODES} --num_gpus=${NUM_GPUS} cifar10_moe_deepspeed.py \
	--log-interval 100 \
	--deepspeed \
	--deepspeed_config ds_config.json \
	--moe \
	--ep-world-size ${EP_SIZE} \
	--num-experts-per-layer ${EXPERTS} \
    --num-expert-layers ${EXPERT_LAYERS} \
	--top-k 1 \
	--noisy-gate-policy 'RSample' \
	--moe-param-group

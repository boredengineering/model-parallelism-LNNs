#!/bin/bash
#SBATCH --job-name=dli_2nodes_hybrid
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=32 ### Number of threads per task (OMP threads)
#SBATCH -o /dli/megatron/logs/%j.out
#SBATCH -e /dli/megatron/logs/%j.err

set -x -e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Distributed training args
NNODES=2
GPUS_PER_NODE=2
TP_SIZE=2
PP_SIZE=2 

# SLURM args
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# Distributed training 
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=2    # <--- CHANGED HERE

# Model architecture 
NLAYERS=12
NHIDDEN=768
NHEADS=32
SEQ_LEN=1024
VOCAB_SIZE=50257

# Data Paths
DATA_OUTPUT_PATH=/dli/megatron/checkpoints/test
CHECKPOINT_PATH=/dli/megatron/checkpoints
TENSORBOARD_PATH=/dli/megatron/tensorboard
LOGS_PATH=/dli/megatron/logs
VOCAB_FILE=/dli/data/GPT-2_assets/gpt2-vocab.json
MERGE_FILE=/dli/data/GPT-2_assets/gpt2-merges.txt
DATA_PATH=/dli/data/GPT-2_assets/my-gpt2_text_document

NAME="log_2Nodes4GPUS_hybrid"       # <--- CHANGED HERE 





OPTIMIZER_ARGS=" \
            --optimizer adam \
            --adam-beta1 0.9 \
            --adam-beta2 0.95 \
            --adam-eps 1e-8 \
            --lr 6e-5 \
            --min-lr 6e-6 \
            --lr-decay-style cosine \
            --lr-decay-iters 800 \
            --lr-warmup-fraction .01 \
            --clip-grad 1.0 \
            --weight-decay 1e-1 \
            --exit-duration-in-mins 1190 \
              "

GPT_ARGS=" \
            --num-layers $NLAYERS \
            --hidden-size $NHIDDEN \
            --num-attention-heads $NHEADS \
            --seq-length $SEQ_LEN \
            --max-position-embeddings $SEQ_LEN \
            --micro-batch-size $MICRO_BATCH_SIZE \
            --global-batch-size $GLOBAL_BATCH_SIZE \
            --train-iters 100 \
            --vocab-file $VOCAB_FILE \
            --merge-file $MERGE_FILE \
            --init-method-std 0.006 \
            $OPTIMIZER_ARGS \
        "

OUTPUT_ARGS=" \
            --log-interval 10 \
            --save-interval 300 \
            --eval-interval 1000 \
            --eval-iters 10 \
            --tensorboard-dir $TENSORBOARD_PATH \
            --tensorboard-queue-size 5 \
            --log-timers-to-tensorboard \
            --log-batch-size-to-tensorboard \
            --log-validation-ppl-to-tensorboard \
            "

export LAUNCHER="python -u -m torch.distributed.launch \
             --nproc_per_node $GPUS_PER_NODE \
             --nnodes $NNODES \
             --master_addr $MASTER_ADDR \
             --master_port $MASTER_PORT \
             "

export CMD=" \
             /dli/megatron/Megatron-LM-main/pretrain_gpt.py \
             --tensor-model-parallel-size $TP_SIZE \
             --pipeline-model-parallel-size $PP_SIZE \
             $GPT_ARGS \
             $OUTPUT_ARGS \
             --save $CHECKPOINT_PATH \
             --data-path $DATA_PATH \
             --data-impl mmap \
             --split 949,50,1 \
             --distributed-backend nccl \
           "

clear; srun --jobid $SLURM_JOBID bash -c 'NCCL_DEBUG=INFO  $LAUNCHER --node_rank $SLURM_PROCID $CMD' 2>&1 | tee -a $LOGS_PATH/$NAME.txt


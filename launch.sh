#!/bin/bash
#
# Usage: ./launch.sh <mode> <model_size> [steps] [nodes] [time] [kernel fusion] [fp8] 
#
# Modes:     throughput  (50 steps, no logging)
#            train       (N steps, with W&B and Tensorboard)
#
# Sizes:     125m, 350m, 760m, 1.5b, 3b, 8b
#
# Steps:     required for train mode (e.g., 1000, 5000, 15000)
# Nodes:     optional, default 4 (max 8)
# Time:      optional, format HH:MM:SS (default: 00:30:00 for throughput, 02:30:00 for train)
#
# Examples:  ./launch.sh throughput 760m
#            ./launch.sh throughput 8b 50 1 00:45:00 true bf16
#            ./launch.sh train 760m 5000
#            ./launch.sh train 1.5b 3000 8 03:00:00 true bf16

set -euo pipefail

MODE=${1:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes] [time] [kernel fusion] [fp8]}
MODEL_SIZE=${2:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes] [time] [kernel fusion] [fp8]}

################ Mode config ################
case $MODE in
    throughput)
        TRAINING_STEPS=${3:-50}
        NODES=${4:-4}
        TIME=${5:-00:30:00}
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        LOGGING_EXTRA=""
        WANDB=false
        # `set -u` (line 19) crashes on any unset variable, so this must be defined in all branches.
        CHECKPOINT_ARGS_CONTENT=""
        ;;
    train)
        TRAINING_STEPS=${3:?Usage: ./launch.sh train <model_size> <steps> [nodes] [time] [kernel fusion] [fp8]}
        NODES=${4:-4}
        TIME=${5:-02:30:00}
        KERNEL_FUSION=${6:-false}
        FP8=${7:-false}
        EVAL_INTERVAL=1000
        EVAL_ITERS=10
        LR_WARMUP_ITERS=200
        LOGGING_EXTRA="
    --tensorboard-dir \$TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard"
        WANDB=true
        # Graceful exit: the job saves a checkpoint and stops cleanly instead of being killed mid-step.
        # Two flags complement each other:
        #   --exit-duration-in-mins: after each step, stop if elapsed time > threshold (primary path).
        #   --exit-signal-handler:   save on SLURM's kill warning (SIGTERM) as a fallback.
        # We use a 5-min buffer because saving across many GPUs takes 1-3 min; SLURM's warning
        # is only ~60 s (not enough time to save safely without the head start).
        # --load is safe on first run: Megatron-LM starts fresh if no checkpoint exists yet.
        #
        # Split HH:MM:SS into total minutes minus 5-min buffer. e.g. 02:30:00 → 145.
        EXIT_DURATION_MINS=$(awk -F: '{print $1*60+$2-5}' <<< "$TIME")
        CHECKPOINT_ARGS_CONTENT="
    --save \$LOG_DIR/checkpoints
    --load \$LOG_DIR/checkpoints
    --exit-signal-handler
    --exit-duration-in-mins ${EXIT_DURATION_MINS}"
        ;;
    *)
        echo "Unknown mode: $MODE. Choose: throughput, train"
        exit 1
        ;;
esac

################ Model config ################
case $MODEL_SIZE in
    125m)
        NUM_LAYERS=12;  HIDDEN=768;  FFN=2048;  HEADS=12; KV_HEADS=4
        MBS=16
        ;;
    350m)
        NUM_LAYERS=24; HIDDEN=1024; FFN=2816;  HEADS=16; KV_HEADS=4
        MBS=8
        ;;
    760m)
        NUM_LAYERS=24; HIDDEN=1536; FFN=4096;  HEADS=16; KV_HEADS=4
        MBS=4
        ;;
    1.5b)
        NUM_LAYERS=48; HIDDEN=1600; FFN=4352;  HEADS=20; KV_HEADS=4
        MBS=4
        ;;
    3b)
        NUM_LAYERS=32; HIDDEN=3072; FFN=8192;  HEADS=24; KV_HEADS=8
        MBS=4
        ;;
    8b)
        NUM_LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; KV_HEADS=8
        MBS=2
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Choose: 125m, 350m, 760m, 1.5b, 3b, 8b"
        exit 1
        ;;
esac

GBS=256
SEQ_LEN=4096
JOB_NAME="gipfel-${MODE}-${MODEL_SIZE}-${TRAINING_STEPS}s-${NODES}n-${TIME}-h-kernel_fusion-${KERNEL_FUSION}-fp8-${FP8}"

################ W&B block ################
if [ "$WANDB" = true ]; then
    WANDB_BLOCK='
# WANDB
if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB enabled."
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOG_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi'
else
    WANDB_BLOCK='export WANDB_MODE=disabled'
fi

################ Generate script ################
mkdir -p logs

SCRIPT="logs/${JOB_NAME}.sbatch"

cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
HEADER

cat >> "$SCRIPT" << SBATCH_DIRECTIVES
#SBATCH --account=lsaie-ss26
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=debug
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue
SBATCH_DIRECTIVES

cat >> "$SCRIPT" << 'BODY'

echo "START TIME: $(date)"

################ Configs ################
WORKDIR=/users/$USER/lsaie-ss26-gipfelsturm
MEGATRON_LM_DIR=$WORKDIR/Megatron-LM-repo
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/cache
BODY

cat >> "$SCRIPT" << CONFIGS

# Training config
MBS=${MBS}
GBS=${GBS}
SEQ_LEN=${SEQ_LEN}
TRAINING_STEPS=${TRAINING_STEPS}

# Logging
PROJECT_NAME=gipfelsturm
EXP_NAME=${MODE}-${MODEL_SIZE}-\${SLURM_NNODES}n-kernel_fusion-${KERNEL_FUSION}-fp8-${FP8}
LOG_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/\$PROJECT_NAME/\$EXP_NAME
TENSORBOARD_DIR=\$LOG_DIR/tensorboard
CONFIGS

cat >> "$SCRIPT" << 'SETUP'

#########################################

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)

SETUP

cat >> "$SCRIPT" << MODEL
NETWORK_SIZE_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN}
    --ffn-hidden-size ${FFN}
    --num-attention-heads ${HEADS}
    --group-query-attention
    --num-query-groups ${KV_HEADS}
    --max-position-embeddings \$SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length \$SEQ_LEN
)
MODEL

cat >> "$SCRIPT" << TRAINING

TRAINING_ARGS=(
    --micro-batch-size \$MBS
    --global-batch-size \$GBS
    --train-iters \$TRAINING_STEPS
    --log-interval 1
    --eval-interval ${EVAL_INTERVAL}
    --eval-iters ${EVAL_ITERS}
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer adam
    --dataloader-type single
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 50
    ${KERNEL_FUSION:+--enable-cuda-graph}
)



REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr 3e-4
    --lr-decay-style constant
    --lr-warmup-iters ${LR_WARMUP_ITERS}
)
TRAINING


if [ "$FP8" = "true" ]; then
    cat >> "$SCRIPT" << FP8
MIXED_PRECISION_ARGS=(
    --bf16 
    --fp8-format hybrid
)
FP8
else
cat >> "$SCRIPT" << BF16
MIXED_PRECISION_ARGS=(
    --bf16
)
BF16
fi

cat >> "$SCRIPT" << REST

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)
DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
REST

cat >> "$SCRIPT" << LOGGING_EXTRA
${LOGGING_EXTRA}
)
LOGGING_EXTRA

# launch.sh writes a .sbatch script that SLURM runs later on the cluster.
# This block writes the CHECKPOINT_ARGS array into that generated script.
# In throughput mode CHECKPOINT_ARGS_CONTENT is empty, so CHECKPOINT_ARGS=() -> expands to nothing.
cat >> "$SCRIPT" << CKPT_BLOCK

CHECKPOINT_ARGS=(${CHECKPOINT_ARGS_CONTENT}
)
CKPT_BLOCK

cat >> "$SCRIPT" << 'TOKENIZER'

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/data/gpt2-vocab.json
    --merge-file $WORKDIR/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $DATA_PREFIX
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 1
)

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

TRAINING_CMD="torchrun ${TORCHRUN_ARGS[@]} $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${CHECKPOINT_ARGS[@]}"

TOKENIZER

cat >> "$SCRIPT" << 'WANDB_PLACEHOLDER'
WANDB_PLACEHOLDER

# Replace placeholder with actual W&B block
sed -i '/^WANDB_PLACEHOLDER$/d' "$SCRIPT"
cat >> "$SCRIPT" << WANDB_INSERT
${WANDB_BLOCK}
WANDB_INSERT

cat >> "$SCRIPT" << 'FOOTER'

echo "CMD: $TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=/users/$USER/lsaie-ss26-gipfelsturm/alps3.toml --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "numactl --membind=0-3 $TRAINING_CMD"

echo "END TIME: $(date)"
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT"
sbatch "$SCRIPT"

export PYTHONWARNINGS="ignore"
export WANDB_MODE=offline
export NGPUS_PER_NODE=4
export CUDA_VISIBLE_DEVICES='4,5,6,7'


# export PROJ_NAME="SFT_math_CIR"
export EXP_NAME="Qwen3-4B-Thinking-2507_0316"

export MODEL_PATH="/home/pengxia3/dc/models/Qwen3-4B-Thinking-2507"
export TRAIN_FILE="['data/trainset/cir_thinking_0303.parquet', 'data/trainset/cir_w_cot_0304.parquet', 'data/trainset/cir_w_cot_0316.parquet']"
export TEST_FILE=${TRAIN_FILE}
export CKPTS_ROOT_DIR="/home/pengxia3/dc/ckpts"


# bash exp_scripts/train_dapo.sh
# bash exp_scripts/train_grpo.sh
bash exp_scripts/base_scripts/train_sft.sh

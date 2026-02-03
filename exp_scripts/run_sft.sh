export PYTHONWARNINGS="ignore"
export WANDB_MODE=offline
export NGPUS_PER_NODE=4
export CUDA_VISIBLE_DEVICES='3,4,6,7'


# export PROJ_NAME="SFT_math_CIR"
export EXP_NAME="Qwen2.5-3B-Instruct"

export MODEL_PATH="/home/pengxia3/dc/models/Qwen2.5-3B-Instruct"
export TRAIN_FILE="['data/milestone/2nd_iter_infer/Qwen2p5_3B_Instruct-math_lighteval_train-selected.parquet', 'data/milestone/2nd_iter_infer/Qwen2p5_7B_Instruct-math_lighteval_train-selected.parquet']"
export TEST_FILE=${TRAIN_FILE}
export CKPTS_ROOT_DIR="/home/pengxia3/dc/ckpts"


# bash exp_scripts/train_dapo.sh
# bash exp_scripts/train_grpo.sh
bash exp_scripts/train_sft.sh

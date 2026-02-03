export PYTHONWARNINGS="ignore"
export WANDB_MODE=offline
export NGPUS_PER_NODE=4
export CUDA_VISIBLE_DEVICES='3,4,6,7'


# export PROJ_NAME="GRPO_math_CIR"
export EXP_NAME="test"

export MODEL_PATH="/home/pengxia3/dc/ckpts/SFT_math_CIR/Qwen2.5-3B-Instruct/global_step_10_merged"
export TRAIN_FILE="data/code_integrated_reasoning/math_lighteval/train.parquet"
export TEST_FILE="['data/code_integrated_reasoning/math500/test.parquet']"
export CKPTS_ROOT_DIR="/home/pengxia3/dc/ckpts"


# bash exp_scripts/train_dapo.sh
# bash exp_scripts/train_grpo.sh
bash exp_scripts/train_grpo_cir_test.sh

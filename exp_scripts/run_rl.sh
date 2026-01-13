export PYTHONWARNINGS="ignore"
export CUDA_VISIBLE_DEVICES='7'
export WANDB_MODE=offline
export NGPUS_PER_NODE=1


export PROJ_NAME="DAPO_math"
export EXP_NAME="DAPO-Qwen2.5-1.5b-MATH-test"

export RAY_DATA_HOME=/home/pengxia3/dc
export MODEL_PATH=/home/pengxia3/dc/models/Qwen2.5-1.5B-Instruct
export TRAIN_FILE="/home/pengxia3/dc/data/normal_reasoning/dapo_math_17k/train.parquet"
export TEST_FILE="['/home/pengxia3/dc/data/normal_reasoning/aime/aime2023.parquet', '/home/pengxia3/dc/data/normal_reasoning/aime/aime2024.parquet', '/home/pengxia3/dc/data/normal_reasoning/aime/aime2025.parquet']"


# bash exp/dapo_basescript.sh
# bash exp/grpo_basescript.sh
bash exp/grpo_basescript_cir.sh

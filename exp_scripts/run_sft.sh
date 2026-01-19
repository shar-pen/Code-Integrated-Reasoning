export PYTHONWARNINGS="ignore"
export WANDB_MODE=offline
export NGPUS_PER_NODE=1
export CUDA_VISIBLE_DEVICES='7'


# export PROJ_NAME="SFT_math_CIR"
export EXP_NAME="Qwen2.5-1.5B-Instruct"

export MODEL_PATH="ckpts/cir_sft/qwen_2.5_1.5b/global_step_8_merged"
export TRAIN_FILE="data/code_integrated_reasoning/math_lighteval/train.parquet"
export TEST_FILE="['data/code_integrated_reasoning/math500/test.parquet']"


# bash exp_scripts/train_dapo.sh
# bash exp_scripts/train_grpo.sh
bash exp_scripts/train_grpo_cir.sh

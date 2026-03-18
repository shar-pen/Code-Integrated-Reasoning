export PYTHONWARNINGS="ignore"
# export WANDB_MODE=offline
export NGPUS_PER_NODE=4
export CUDA_VISIBLE_DEVICES='4,5,6,7'
# export HYDRA_FULL_ERROR=1
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export NCCL_TIMEOUT=1200
export NCCL_ASYNC_ERROR_HANDLING=1
# For PyTorch 1.11+ to increase the watchdog timeout
export TORCH_NCCL_WAIT_TIMEOUT=1800000 # 30 mins in ms


# export PROJ_NAME="GRPO_math_CIR"
export EXP_NAME="Qwen3-4B-Thinking-2507_SFT_0316_v2_step@40_v1"

export MODEL_PATH="/home/pengxia3/dc/ckpts/SFT_math_CIR/Qwen3-4B-Thinking-2507_0316/global_step_40_merged"
export TRAIN_FILE="data/CIR_thinking/math_lighteval/train.parquet"
export TEST_FILE="['data/CIR_thinking/math500/test.parquet']"
export CKPTS_ROOT_DIR="/home/pengxia3/dc/ckpts"


# bash exp_scripts/train_dapo.sh
# bash exp_scripts/train_grpo.sh
bash exp_scripts/base_scripts/train_grpo_cir.sh

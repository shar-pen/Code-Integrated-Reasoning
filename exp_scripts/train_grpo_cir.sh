#!/usr/bin/env bash
# set -xeuo pipefail

PROJ_NAME=${PROJ_NAME:-"GRPO_math_CIR"}
EXP_NAME=${EXP_NAME:-"default"}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Paths
MODEL_PATH=${MODEL_PATH:-"~/dc/models/Qwen2.5-1.5B-Instruct"}
CKPTS_ROOT_DIR=${CKPTS_ROOT_DIR:-"ckpts"}
CKPTS_DIR=${CKPTS_ROOT_DIR}/${PROJ_NAME}/${EXP_NAME}

FILE_DIR="data/code_integrated_reasoning"
TRAIN_FILE=${TRAIN_FILE:-"${FILE_DIR}/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"['${FILE_DIR}/gsm8k/test.parquet']"}

# training hyperparameters
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio=0.2
loss_agg_mode="token-mean"

train_prompt_bsz=128
train_prompt_mini_bsz=32
n_resp_per_prompt=16
n_resp_per_prompt_val=8

max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 4))

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7
val_do_sample=True

# Performance Related Parameter
sp_size=1
gen_tp=1
fsdp_size=-1
offload=True
use_dynamic_bsz=True
infer_micro_batch_size=null
train_micro_batch_size=null
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 8))


python3 -m verl.trainer.main_ppo \
	\
	data.train_files="${TRAIN_FILE}" \
	data.val_files="${TEST_FILE}" \
	data.prompt_key=prompt \
	data.truncation='left' \
	data.dataloader_num_workers=0 \
	data.max_prompt_length=${max_prompt_length} \
	data.max_response_length=${max_response_length} \
	data.train_batch_size=${train_prompt_bsz} \
	\
	actor_rollout_ref.actor.entropy_coeff=0 \
	actor_rollout_ref.actor.grad_clip=1.0 \
	actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
	actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
	actor_rollout_ref.actor.clip_ratio=${clip_ratio} \
	actor_rollout_ref.actor.clip_ratio_c=10.0 \
	actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
	\
	algorithm.adv_estimator=${adv_estimator} \
	algorithm.use_kl_in_reward=${use_kl_in_reward} \
	algorithm.kl_ctrl.kl_coef=${kl_coef} \
	\
	reward_model.reward_manager=naive \
	\
	actor_rollout_ref.rollout.code_integrated_generation.enable=True \
	actor_rollout_ref.rollout.code_integrated_generation.max_execution_count=3 \
	actor_rollout_ref.rollout.code_integrated_generation.max_try_execution_count=5 \
	actor_rollout_ref.rollout.code_integrated_generation.execution_parallel_num=32 \
	actor_rollout_ref.rollout.code_integrated_generation.execution_timeout=5 \
	actor_rollout_ref.rollout.code_integrated_generation.use_tqdm=True \
	\
	custom_reward_function.path=cir_utils/reward_score.py \
	custom_reward_function.name=compute_cir_score \
	\
	actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
	actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
	actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
	actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
	actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
	actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
	actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
	actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
	actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
	actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
	actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
	actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
	\
	actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
	actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
	actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
	actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
	\
	actor_rollout_ref.model.path="${MODEL_PATH}" \
	actor_rollout_ref.model.enable_gradient_checkpointing=True \
	actor_rollout_ref.model.use_remove_padding=True \
	\
	actor_rollout_ref.rollout.name=vllm \
	actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
	actor_rollout_ref.rollout.enable_chunked_prefill=True \
	actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
	actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
	\
	actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
	actor_rollout_ref.rollout.temperature=${temperature} \
	actor_rollout_ref.rollout.top_p=${top_p} \
	actor_rollout_ref.rollout.top_k=${top_k} \
	actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
	actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
	actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
	actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample} \
	actor_rollout_ref.rollout.val_kwargs.n=${n_resp_per_prompt_val} \
	\
	trainer.logger='["console","wandb"]' \
	trainer.project_name="${PROJ_NAME}" \
	trainer.experiment_name="${EXP_NAME}" \
	\
	actor_rollout_ref.actor.optim.lr=1e-6 \
	actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
	actor_rollout_ref.actor.optim.weight_decay=0.1 \
	\
	trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
	trainer.nnodes="${NNODES}" \
	trainer.val_before_train=True \
	trainer.test_freq=5 \
	trainer.save_freq=10 \
	trainer.total_epochs=2 \
	trainer.total_training_steps=50 \
	trainer.default_local_dir="${CKPTS_DIR}" \
	trainer.resume_mode=auto \
	trainer.log_val_generations=10

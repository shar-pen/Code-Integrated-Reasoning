#!/bin/bash

# command:
# NGPUS_PER_NODE=4 PROJ_NAME="sft" EXP_NAME="qwen_2.5_1.5b" CUDA_VISIBLE_DEVICES='0,1,2,3' bash exps/sft.sh
set -euo pipefail


PROJ_NAME=${PROJ_NAME:-"SFT_math_CIR"}
EXP_NAME=${EXP_NAME:-"default"}

NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}

MODEL_PATH=${MODEL_PATH:-"/home/pengxia3/dc/models/Qwen2.5-1.5B-Instruct"}
CKPTS_ROOT_DIR=${CKPTS_ROOT_DIR:-"ckpts"}
CKPTS_DIR=${CKPTS_ROOT_DIR}/${PROJ_NAME}/${EXP_NAME}

TRAIN_FILE=${TRAIN_FILE:-"data/infer/distillation_data.parquet"}
TEST_FILE=${TEST_FILE:-"data/infer/distillation_data.parquet"}

max_length=1024
train_batch_size=512
micro_batch_size=1

torchrun --nnodes=1 --nproc_per_node=${NGPUS_PER_NODE} \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
	data.code_integrated_generation.enable=true \
	data.max_length=${max_length} \
	data.truncation="left" \
	data.train_batch_size=${train_batch_size} \
    data.micro_batch_size=${micro_batch_size} \
	optim.lr=5e-6 \
    model.partial_pretrain="${MODEL_PATH}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.project_name=${PROJ_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.logger='["console","wandb"]' \
	trainer.total_epochs=1 \
	trainer.save_freq=10 \
    trainer.total_training_steps=10 \
	\
	# trainer.val_before_train=False \
	# trainer.test_freq=5 \
    # ulysses_sequence_parallel_size=2 \
    # use_remove_padding=true

#!/bin/bash

# command:
# NGPUS_PER_NODE=4 PROJ_NAME="sft" EXP_NAME="qwen_2.5_1.5b" CUDA_VISIBLE_DEVICES='0,1,2,3' bash exps/sft.sh
set -x

export PYTHONWARNINGS="ignore"
# export CUDA_VISIBLE_DEVICES='7'
export WANDB_MODE=offline


PROJ_NAME=${PROJ_NAME:-"cir_sft"}
EXP_NAME=${EXP_NAME:-"default"}

NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
MODEL_PATH=${MODEL_PATH:-"/home/pengxia3/dc/models/Qwen2.5-1.5B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-"ckpts/${PROJ_NAME}/${EXP_NAME}"}

TRAIN_FILE=${TRAIN_FILE:-"data/infer/distillation_data.parquet"}
TEST_FILE=${TEST_FILE:-"data/infer/distillation_data.parquet"}


torchrun --nnodes=1 --nproc_per_node=${NGPUS_PER_NODE} \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
	data.code_integrated_generation.enable=true \
	data.max_length=2048 \
	data.truncation="left" \
	data.train_batch_size=512 \
    data.micro_batch_size=1 \
    model.partial_pretrain=${MODEL_PATH} \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.project_name=${PROJ_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.logger=console \
	trainer.total_epochs=3 \
	trainer.save_freq=5 \
    # trainer.total_training_steps=10 \
	\
    # ulysses_sequence_parallel_size=2 \
    # use_remove_padding=true

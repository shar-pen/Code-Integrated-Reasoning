# export PYTHONWARNINGS="ignore"
# export WANDB_MODE=offline
# export NGPUS_PER_NODE=4
# export CUDA_VISIBLE_DEVICES='3,4,6,7'


# # export PROJ_NAME="GRPO_math_CIR"
# export EXP_NAME="test"

# export MODEL_PATH="/home/pengxia3/dc/ckpts/SFT_math_CIR/Qwen2.5-3B-Instruct/global_step_10_merged"
# export TRAIN_FILE="data/code_integrated_reasoning/math_lighteval/train.parquet"
# export TEST_FILE="['data/code_integrated_reasoning/math500/test.parquet']"
# export CKPTS_ROOT_DIR="/home/pengxia3/dc/ckpts"


# # bash exp_scripts/train_dapo.sh
# # bash exp_scripts/train_grpo.sh
# bash exp_scripts/base_scripts/train_grpo_cir_test.sh

export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=False
export CUDA_VISIBLE_DEVICES='4,5,6,7'

temperature=1.0
max_tokens=4096
n_responses=4
topp=0.7
topk=-1

model_path_array=(
	"/home/pengxia3/dc/ckpts/GRPO_math_CIR/Qwen3-4B-Thinking-2507_SFT_0316_v2_step@40_v1/global_step_50/actor_merged"
	"/home/pengxia3/dc/ckpts/GRPO_math_CIR/Qwen3-4B-Thinking-2507_SFT_0316_v2_step@40_v1/global_step_50/actor_merged"
	
	"/home/pengxia3/dc/ckpts/SFT_math_CIR/Qwen3-4B-Thinking-2507_0316/global_step_40_merged"
	"/home/pengxia3/dc/ckpts/SFT_math_CIR/Qwen3-4B-Thinking-2507_0316/global_step_40_merged"
)

data_file_array=(
	"data/NLR_thinking/math500/test.parquet"
	"data/CIR_thinking/math500/test.parquet"

	"data/NLR_thinking/math500/test.parquet"
	"data/CIR_thinking/math500/test.parquet"
)

output_path_array=(
	"data/infer/NLR_thinking/math500/test/Qwen3_4B_Thinking_0316_sft_step@40_rl_step@50-maxtoken@${max_tokens}_temp@${temperature}_topp@${topp}_topk@${topk}_n@${n_responses}.parquet"
	"data/infer/CIR_thinking/math500/test/Qwen3_4B_Thinking_0316_sft_step@40_rl_step@50-maxtoken@${max_tokens}_temp@${temperature}_topp@${topp}_topk@${topk}_n@${n_responses}.parquet"

	"data/infer/NLR_thinking/math500/test/Qwen3_4B_Thinking_0316_sft_step@40-maxtoken@${max_tokens}_temp@${temperature}_topp@${topp}_topk@${topk}_n@${n_responses}.parquet"
	"data/infer/CIR_thinking/math500/test/Qwen3_4B_Thinking_0316_sft_step@40-maxtoken@${max_tokens}_temp@${temperature}_topp@${topp}_topk@${topk}_n@${n_responses}.parquet"
)

index=0
while [ $index -lt ${#model_path_array[@]} ]; do
	model_path=${model_path_array[$index]}
	data_file=${data_file_array[$index]}
	output_path=${output_path_array[$index]}

	python3 -m exp_scripts.utils.infer \
		--model-name-or-path ${model_path} \
		--data-files ${data_file} \
		--custom-reward-function-path cir_utils/reward_score.py \
		--custom-reward-function-name compute_cir_score \
		--output-path ${output_path} \
		--temperature ${temperature} \
		--max-tokens ${max_tokens} \
		--batch-size 128 \
		--top-p ${topp} \
		--top-k ${topk} \
		--n-samples ${n_responses} \
		--tensor-parallel-size 4

	index=$((index + 1))
done

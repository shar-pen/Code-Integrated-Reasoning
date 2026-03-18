# set -xeuo pipefail

export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=False
export CUDA_VISIBLE_DEVICES='4,5,6,7'

temperature=1.0
max_tokens=2048
n_responses=4
topp=0.7
topk=-1
python3 -m exp_scripts.utils.infer \
	--model-name-or-path /home/pengxia3/dc/ckpts/SFT_math_CIR/Qwen3-4B-Thinking-2507_0303/global_step_20_merged \
	--data-files 'data/CIR_thinking/math_lighteval/test.parquet' \
	--custom-reward-function-path cir_utils/reward_score.py \
	--custom-reward-function-name compute_cir_score \
	--output-path data/infer/CIR_thinking/math_lighteval/test/Qwen3_4B_Thinking_2507_0303_sft_step@20-maxtoken@${max_tokens}_temp@${temperature}_topp@${topp}_topk@${topk}_n@${n_responses}.parquet \
	--temperature ${temperature} \
	--max-tokens ${max_tokens} \
	--batch-size 128 \
	--top-p ${topp} \
	--top-k ${topk} \
	--n-samples ${n_responses} \
	--tensor-parallel-size 4 


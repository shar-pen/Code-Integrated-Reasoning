set -xeuo pipefail

export PYTHONWARNINGS="ignore"
export CUDA_VISIBLE_DEVICES='7'
export TOKENIZERS_PARALLELISM=False

python3 -m exp_scripts.utils.infer \
	--model_name_or_path /home/pengxia3/xp/gits/Code-Integrated-Reasoning/ckpts/sft/qwen_2.5_1.5b/global_step_20_merged \
	--data_files "['data/code_integrated_reasoning/math_lighteval/train.parquet']" \
	--output_path 'data/infer/infer_1.5b_instruct_sft_20_step.parquet' \
	--temperature 1.0 \
	--max_tokens 2048 \
	--top_p 0.7 \
	--n_samples 4 \
	--tensor_parallel_size 1 \

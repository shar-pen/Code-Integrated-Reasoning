# set -xeuo pipefail

export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=False

python3 -m exp_scripts.utils.infer \
	--model-name-or-path '/home/pengxia3/dc/models/Qwen2.5-1.5B-Instruct' \
	--data-files 'data/code_integrated_reasoning/math_lighteval/train.parquet' \
	--output-path 'data/infer/Qwen2.5-1.5B-Instruct_math_lighteval_train.parquet' \
	--temperature 1.0 \
	--max-tokens 2048 \
	--batch-size 256 \
	--top-p 0.7 \
	--n-samples 4 \
	--tensor-parallel-size 1 \


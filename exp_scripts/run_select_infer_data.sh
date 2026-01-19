#!/usr/bin/env bash
# Wrapper to run the select_data.py script
# set -euo pipefail

INPUT_PATH="data/infer/Qwen2.5-3B-Instruct_math_lighteval_train.parquet"
OUTPUT_PATH="data/infer/Qwen2.5-3B-Instruct_math_lighteval_train_selected.parquet"

python3 -m exp_scripts.utils.select_data \
	--input-data-path ${INPUT_PATH} \
	--output-data-path ${OUTPUT_PATH} \
	--max-num-response-per-question 1 \

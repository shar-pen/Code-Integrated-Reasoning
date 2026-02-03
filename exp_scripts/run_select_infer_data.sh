#!/usr/bin/env bash
# Wrapper to run the select_data.py script
# set -euo pipefail

INPUT_PATH="data/infer/math_lighteval/test/Qwen2p5_3B_Instruct_sft_step@10-maxtoken@2048_temp@1.0_topp@0.7_topk@-1_n@4.parquet"
OUTPUT_PATH="data/infer/math_lighteval/test/Qwen2p5_3B_Instruct_sft_step@10-maxtoken@2048_temp@1.0_topp@0.7_topk@-1_n@4-selected.parquet"

python3 -m exp_scripts.utils.select_data \
	--input-data-path ${INPUT_PATH} \
	--output-data-path ${OUTPUT_PATH} \
	--max-num-response-per-question 1 \

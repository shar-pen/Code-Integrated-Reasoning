#!/usr/bin/env bash
# Wrapper to run the select_data.py script
set -euo pipefail

INPUT_PATH="data/infer/infer_1.5b_instruct_base.parquet"
OUTPUT_PATH="data/infer/infer_1.5b_instruct_base_selected.parquet"

python3 -m exp_scripts.utils.select_data \
	--input-data-path ${INPUT_PATH} \
	--output-data-path ${OUTPUT_PATH} \
	--max-num-response-per-question 2 \
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
from functools import partial
from datasets import load_dataset, Dataset
from .format import format_prompt, example_map_fn



def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def build_gsm8k(format_prompt_kwargs, enable_map=True):
    
	def process_gsm8k(example, **process_fn_kwargs):
		question = example["question"]
		ground_truth = example["answer"]
		ground_truth = extract_solution(ground_truth)
		return question, ground_truth

	data_source = "openai/gsm8k"
	dataset = load_dataset(data_source, "main")
	train_dataset = dataset["train"]
	test_dataset = dataset["test"]
      
	train_map_fn = partial(
		example_map_fn, process_fn=process_gsm8k, data_source=data_source, ability="math", split="train", format_prompt_kwargs=format_prompt_kwargs
	)
	test_map_fn = partial(
		example_map_fn, process_fn=process_gsm8k, data_source=data_source, ability="math", split="test", format_prompt_kwargs=format_prompt_kwargs
	)

	if enable_map:
		train_dataset = train_dataset.map(train_map_fn, with_indices=True, remove_columns=train_dataset.column_names)
		test_dataset = test_dataset.map(test_map_fn, with_indices=True, remove_columns=test_dataset.column_names)
	else:
		processed_train_data = []
		for idx, example in enumerate(train_dataset):
			processed_train_data.append(train_map_fn(example, idx))
		train_dataset = Dataset.from_list(processed_train_data)

		processed_test_data = []
		for idx, example in enumerate(test_dataset):
			processed_test_data.append(test_map_fn(example, idx))
		test_dataset = Dataset.from_list(processed_test_data)

	return train_dataset, test_dataset


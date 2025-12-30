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
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import json
import os
from datasets import load_dataset, Dataset
from functools import partial
from .format import example_map_fn



def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def extract_solution(solution_str):
	return remove_boxed(last_boxed_only_string(solution_str))



def build_math_lighteval(format_prompt_kwargs, enable_map=True):
	
	def process_math_lighteval(example, **process_fn_kwargs):
		problem = example["problem"]
		ground_truth = extract_solution(example["solution"])
		return problem, ground_truth

	# 'lighteval/MATH' is no longer available on huggingface.
	# Use mirror repo: DigitalLearningGmbH/MATH-lighteval
	data_source = "DigitalLearningGmbH/MATH-lighteval"
	dataset = load_dataset(data_source)
	
	train_dataset = dataset["train"]
	test_dataset = dataset["test"]

	map_fn = partial(
		example_map_fn, process_fn=process_math_lighteval, data_source=data_source, ability="math", split="test", format_prompt_kwargs=format_prompt_kwargs
	)
     
	if enable_map:
		train_dataset = train_dataset.map(map_fn, with_indices=True, remove_columns=train_dataset.column_names)
		test_dataset = test_dataset.map(map_fn, with_indices=True, remove_columns=test_dataset.column_names)
	else:
		processed_train_data = []
		for idx, example in enumerate(train_dataset):
			processed_train_data.append(map_fn(example, idx))
		train_dataset = Dataset.from_list(processed_train_data)

		processed_test_data = []
		for idx, example in enumerate(test_dataset):
			processed_test_data.append(map_fn(example, idx))
		test_dataset = Dataset.from_list(processed_test_data)

	return train_dataset, test_dataset


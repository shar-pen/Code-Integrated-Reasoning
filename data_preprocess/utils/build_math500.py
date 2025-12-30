import os
import json
import argparse
from functools import partial
from datasets import load_dataset, Dataset
from .format import example_map_fn


def build_math500(format_prompt_kwargs, enable_map=True):

	def process_math500(example, **process_fn_kwargs):
		problem = example["problem"]
		ground_truth = example["answer"]
		return problem, ground_truth
	
	data_source = "HuggingFaceH4/MATH-500"
	dataset = load_dataset(data_source, split="test")
	map_fn = partial(
		example_map_fn, process_fn=process_math500, data_source=data_source, ability="math", split="test", format_prompt_kwargs=format_prompt_kwargs
	)

	if enable_map:
		dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
	else:
		processed_data = []
		for idx, example in enumerate(dataset):
			processed_data.append(map_fn(example, idx))
		dataset = Dataset.from_list(processed_data)
	return dataset


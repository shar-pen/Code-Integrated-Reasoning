import argparse
from functools import partial
import os
from datasets import load_dataset, Dataset
from .format import format_prompt, example_map_fn


def remove_boxed(s):
	if "\\boxed " in s:
		left = "\\boxed "
		assert s[: len(left)] == left
		return s[len(left) :]

	left = "\\boxed{"

	assert s[: len(left)] == left
	assert s[-1] == "}"

	return s[len(left) : -1]


# this dataset is gone
def build_aime2023(format_prompt_kwargs, enable_map=True):
	
	def process_aime2023(example, **process_fn_kwargs):
		problem = example["question"]
		ground_truth = example["answer"]
		ground_truth = f"{ground_truth}"
		return problem, ground_truth

	data_source = "extraordinarylab/aime23"
	dataset = load_dataset(data_source, split="test")
	map_fn = partial(
		example_map_fn, process_fn=process_aime2023, data_source=data_source, ability="math", split="test", format_prompt_kwargs=format_prompt_kwargs
	)
	
	if enable_map:
		dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
	else:
		processed_data = []
		for idx, example in enumerate(dataset):
			processed_data.append(map_fn(example, idx))
		dataset = Dataset.from_list(processed_data)

	return dataset


def build_aime2024(format_prompt_kwargs, enable_map=True):
	
	def process_aime2024(example, **process_fn_kwargs):
		problem = example["problem"]
		ground_truth = example["solution"]
		ground_truth = remove_boxed(ground_truth)
		return problem, ground_truth

	data_source = "math-ai/aime24"
	dataset = load_dataset(data_source, split="test")
	map_fn = partial(
		example_map_fn, process_fn=process_aime2024, data_source=data_source, ability="math", split="test", format_prompt_kwargs=format_prompt_kwargs
	)
	
	if enable_map:
		dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
	else:
		processed_data = []
		for idx, example in enumerate(dataset):
			processed_data.append(map_fn(example, idx))
		dataset = Dataset.from_list(processed_data)
	return dataset


def build_aime2025(format_prompt_kwargs, enable_map=True):
	
	def process_aime2025(example, **process_fn_kwargs):
		problem = example["problem"]
		ground_truth = example["answer"]
		return problem, ground_truth

	data_source = "math-ai/aime25"
	dataset = load_dataset(data_source, split="test")
	map_fn = partial(
		example_map_fn, process_fn=process_aime2025, data_source=data_source, ability="math", split="test", format_prompt_kwargs=format_prompt_kwargs
	)

	if enable_map:
		dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
	else:
		processed_data = []
		for idx, example in enumerate(dataset):
			processed_data.append(map_fn(example, idx))
		dataset = Dataset.from_list(processed_data)
	return dataset


def build_aime2026(format_prompt_kwargs, enable_map=True):
	
	def process_aime2026(example, **process_fn_kwargs):
		problem = example["problem"]
		ground_truth = example["answer"]
		return problem, ground_truth

	data_source = "math-ai/aime26"
	dataset = load_dataset(data_source, split="test")
	map_fn = partial(
		example_map_fn, process_fn=process_aime2026, data_source=data_source, ability="math", split="test", format_prompt_kwargs=format_prompt_kwargs
	)

	if enable_map:
		dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
	else:
		processed_data = []
		for idx, example in enumerate(dataset):
			processed_data.append(map_fn(example, idx))
		dataset = Dataset.from_list(processed_data)
	return dataset

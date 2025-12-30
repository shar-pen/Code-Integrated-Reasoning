import os
import json
import argparse
from datasets import load_dataset, Dataset
from .format import format_prompt

def build_dapo_math_17k_official_dataset(format_prompt_kwargs, enable_map=True):
	# Login using e.g. `huggingface-cli login` to access this dataset

	data_source = "BytedTsinghua-SIA/DAPO-Math-17k" # origin dataset is 1.7m, i.e., 17k prompts replicated 100 times
	dataset = load_dataset(data_source)
	train_dataset = dataset["train"]
	print(f"Loaded DAPO-Math-17k train dataset with {len(train_dataset)} examples.")

	indexes = [item['extra_info']['index'] for item in train_dataset]

	seen_indexes = set()

	def filter_duplicates(example):
		index = example['extra_info']['index']
		if index not in seen_indexes:
			seen_indexes.add(index)
			return True
		return False

	train_dataset = train_dataset.filter(filter_duplicates)
	print(f"Deduplicated DAPO-Math-17k train dataset has {len(train_dataset)} examples.")

	def alter_prompt_style(example, index):
	
		prompt = example['prompt'][0]['content']
		problem = prompt.replace('Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.', '').replace('Remember to put your answer on its own line after "Answer:".', '').strip('\n')
		prompt = format_prompt(problem, **format_prompt_kwargs)
		example['prompt']= prompt
		example['data_source'] = f'{format_prompt_kwargs.get("output_style", "math_reward_style")}/{data_source}'
		example['extra_info']['index'] = index
		example['extra_info']['problem'] = problem
		example['extra_info']['ground_truth'] = example['reward_model']['ground_truth']
		example['ability']='math'

		return example


	if enable_map:
		train_dataset = train_dataset.map(alter_prompt_style, with_indices=True)
	else:
		processed_data = []
		for idx, example in enumerate(train_dataset):
			processed_data.append(alter_prompt_style(example, idx))
		train_dataset = Dataset.from_list(processed_data)
	return train_dataset


def build_dapo_math_17k_unofficial_dataset(format_prompt_kwargs, enable_map=True):
	# Login using e.g. `huggingface-cli login` to access this dataset

	data_source = 'zhuzilin/dapo-math-17k' # this is a deduplicated version, 17k
	dataset = load_dataset(data_source)
	train_dataset = dataset["train"]

	def alter_prompt_style(example, index):
	
		prompt = example['prompt'][0]['content']
		problem = prompt.replace('Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem.', '').replace('Remember to put your answer on its own line after "Answer:".', '').strip('\n')
		prompt = format_prompt(problem, **format_prompt_kwargs)
		
		ret = {
			'data_source': f'{format_prompt_kwargs.get("output_style", "math_reward_style")}/{data_source}',
			'prompt': prompt,
			'ability': 'math',
			"reward_model": {"style": "rule", "ground_truth": example['label']},
			"extra_info": {"split": 'train', "index": index, },
		}

		return ret

	
	if enable_map:
		train_dataset = train_dataset.map(alter_prompt_style, with_indices=True)
	else:
		processed_data = []
		for idx, example in enumerate(train_dataset):
			processed_data.append(alter_prompt_style(example, idx))
		train_dataset = Dataset.from_list(processed_data)

	return train_dataset


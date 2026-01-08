import os, json, fire
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset, concatenate_datasets
from cir_utils.python_interpreter import ExecutorManager, LocalPythonExecutor
from cir_utils.generate_utils import CodeIntegratedGenerationConfig, code_integrated_generate
from verl.utils.reward_score import default_compute_score


def main(
		model_name_or_path: str, 
		data_files: list[str], 
		output_path: str,
		prompt_key: str = 'prompt', 
		data_source_key: str = 'data_source', 
		reward_model_key: str = 'reward_model', 
		batch_size: int = 128,
		temperature: float = 1.0, 
		max_tokens: int = 2048, 
		top_p: float = 0.7, 
		n_samples: int = 4,
		tensor_parallel_size: int = 1,
	):

	sampling_params = SamplingParams(
		temperature=temperature,
		max_tokens=max_tokens,
		top_p=top_p,
	)

	excutor_manager = ExecutorManager(max_workers=8, executor_cls=LocalPythonExecutor)
	cir_config = CodeIntegratedGenerationConfig()

	inference_engine = LLM(
		model=model_name_or_path, 
		tensor_parallel_size=tensor_parallel_size, 
		gpu_memory_utilization=0.85,
		skip_tokenizer_init=False,
	)
	tokenizer = inference_engine.get_tokenizer()

	datasets = []
	for parquet_file in data_files:
		# read parquet files and cache
		dataset = pd.read_parquet(parquet_file)
		datasets.append(dataset)
	dataset: pd.DataFrame = pd.concat(datasets, ignore_index=True)
	
	chat_lst = list(dataset[prompt_key])
	chat_lst = [chat.tolist() if not isinstance(chat, list) else chat for chat in chat_lst]

	tokenizer.padding_side = "left"
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token


	total_samples = len(dataset)
	num_batch = -(-total_samples // batch_size)
	output_col_list = []
	score_list = []
	code_triggered_count_col_list = []
	code_execution_count_col_list = []

	for batch_idx in tqdm(range(num_batch), desc="Generating Batches"):

		batch_chat_lst = chat_lst[batch_idx * batch_size : (batch_idx + 1) * batch_size]
		inputs = tokenizer.apply_chat_template(
			batch_chat_lst,
			add_generation_prompt=True,
			truncation=True,
			max_length=1024,
			tokenize=True,
		)
		vllm_inputs = [
			{"prompt_token_ids": raw_prompt_ids} 
			for raw_prompt_ids in inputs
		]
		vllm_inputs = np.repeat(vllm_inputs, repeats=n_samples, axis=0).tolist()
		
		outputs, response_observation_mask, code_triggered_counts, code_execution_counts = code_integrated_generate(
			vllm_inference_engine=inference_engine,
			prompts=vllm_inputs,
			sampling_params=sampling_params,
			cir_config=cir_config,
			executor_manager=excutor_manager,
		)
		output_texts = [item.outputs[0].text for item in outputs]

		output_col_list.extend(output_texts)
		code_triggered_count_col_list.extend(code_triggered_counts)
		code_execution_count_col_list.extend(code_execution_counts)


	output_col_list = np.array(output_col_list).reshape(-1, n_samples).tolist()
	code_triggered_count_col_list = np.array(code_triggered_count_col_list).reshape(-1, n_samples).tolist()
	code_execution_count_col_list = np.array(code_execution_count_col_list).reshape(-1, n_samples).tolist()

	dataset["responses"] = output_col_list
	dataset["code_triggered_counts"] = code_triggered_count_col_list
	dataset["code_execution_counts"] = code_execution_count_col_list


	def process_item(data_source, response_lst, reward_data):
		reward_fn = default_compute_score
		ground_truth = reward_data["ground_truth"]
		score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
		return  score_lst
	
	responses = dataset['responses']
	data_sources = dataset[data_source_key]
	reward_model_data = dataset[reward_model_key]
	score_ls = []
	for i in range(len(dataset)):
		score_lst = process_item(data_sources[i], responses[i], reward_model_data[i])
		score_ls.append(score_lst)

	dataset['scores'] = score_ls

	output_dir = os.path.dirname(output_path)
	os.makedirs(output_dir, exist_ok=True)
	dataset.to_parquet(output_path)


if __name__ == "__main__":
	fire.Fire(main)		

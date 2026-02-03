import os
import argparse
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset, concatenate_datasets
from cir_utils.python_interpreter import ExecutorManager, LocalPythonExecutor
from cir_utils.generate_utils import CodeIntegratedGenerationConfig, code_integrated_generate
from verl.utils.reward_score import default_compute_score
from verl.trainer.ppo.reward import get_custom_reward_fn, load_reward_manager
from omegaconf import DictConfig

def main(
		model_name_or_path: str, 
		data_files: list[str], 
		output_path: str,
		prompt_key: str = 'prompt', 
		data_source_key: str = 'data_source', 
		reward_model_key: str = 'reward_model', 
		batch_size: int = -1,
		temperature: float = 1.0, 
		max_tokens: int = 2048, 
		top_p: float = 0.7, 
		top_k: int = -1,
		n_samples: int = 4,
		tensor_parallel_size: int = 1,
		custom_reward_function_path: str = None,
		custom_reward_function_name: str = None,
	):

	if custom_reward_function_path is not None and custom_reward_function_name is not None:
		config = DictConfig({
			"custom_reward_function": {
				"path": 'cir_utils/reward_score.py',
				"name": 'compute_cir_score',
			}
		})
		compute_score = get_custom_reward_fn(config)
		if compute_score is None:
			compute_score = default_compute_score
	else:
		compute_score = default_compute_score
	print(compute_score)

	# Load datasets
	datasets = []
	for parquet_file in tqdm(data_files, desc="Loading Parquet Files"):
		# read parquet files and cache
		dataset = pd.read_parquet(parquet_file)
		print(f"Loaded {len(dataset)} samples from {parquet_file}")
		datasets.append(dataset)
	dataset: pd.DataFrame = pd.concat(datasets, ignore_index=True)
	print(f"Total samples loaded: {len(dataset)}")
	
	chat_lst = list(dataset[prompt_key])
	chat_lst = [chat.tolist() if not isinstance(chat, list) else chat for chat in chat_lst]

	# Setup inference components
	sampling_params = SamplingParams(
		temperature=temperature,
		max_tokens=max_tokens,
		top_p=top_p,
		top_k=top_k,
	)
	excutor_manager = ExecutorManager(max_workers=8, executor_cls=LocalPythonExecutor)
	cir_config = CodeIntegratedGenerationConfig()

	print("Initializing LLM inference engine...")
	inference_engine = LLM(
		model=model_name_or_path, 
		tensor_parallel_size=tensor_parallel_size, 
		gpu_memory_utilization=0.9,
		skip_tokenizer_init=False,
	)
	tokenizer = inference_engine.get_tokenizer()
	tokenizer.padding_side = "left"
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	# Inference loop
	total_samples = len(dataset)
	if batch_size == -1:
		batch_size = total_samples
	num_batch = -(-total_samples // batch_size)
	output_col_list = []
	score_list = []
	code_triggered_count_col_list = []
	code_execution_count_col_list = []
	code_execution_error_count_col_list = []

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
		
		outputs, response_observation_mask, code_triggered_counts, code_execution_counts, code_execution_error_counts = code_integrated_generate(
			vllm_inference_engine=inference_engine,
			prompts=vllm_inputs,
			sampling_params=sampling_params,
			cir_config=cir_config,
			executor_manager=excutor_manager,
			use_tqdm=True,
		)
		output_texts = [item.outputs[0].text for item in outputs]

		output_col_list.extend(output_texts)
		code_triggered_count_col_list.extend(code_triggered_counts)
		code_execution_count_col_list.extend(code_execution_counts)
		code_execution_error_count_col_list.extend(code_execution_error_counts)


	output_col_list = np.array(output_col_list).reshape(-1, n_samples).tolist()
	code_triggered_count_col_list = np.array(code_triggered_count_col_list).reshape(-1, n_samples).tolist()
	code_execution_count_col_list = np.array(code_execution_count_col_list).reshape(-1, n_samples).tolist()
	code_execution_error_count_col_list = np.array(code_execution_error_count_col_list).reshape(-1, n_samples).tolist()

	dataset["responses"] = output_col_list
	dataset["code_triggered_counts"] = code_triggered_count_col_list
	dataset["code_execution_counts"] = code_execution_count_col_list
	dataset["code_execution_error_counts"] = code_execution_error_count_col_list


	def process_item(data_source, response_lst, reward_data):
		ground_truth = reward_data["ground_truth"]
		score_lst = np.array([compute_score(data_source, r, ground_truth) for r in response_lst])
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
	print(f"Saved inference results to {output_path}")


def parse_args():
	parser = argparse.ArgumentParser(description="Code-integrated inference runner")
	parser.add_argument("--model-name-or-path", required=True, help="Model identifier or local path")
	parser.add_argument("--data-files", nargs="+", required=True, help="One or more parquet files containing prompts",)
	parser.add_argument("--output-path", required=True, help="Destination parquet path for outputs")
	parser.add_argument("--prompt-key", default="prompt", help="Column storing chat prompts")
	parser.add_argument("--data-source-key", default="data_source", help="Column for data source metadata")
	parser.add_argument("--reward-model-key", default="reward_model", help="Column containing reward metadata")
	parser.add_argument("--batch-size", type=int, default=-1, help="Batch size for inference, -1 means full batch")
	parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
	parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum new tokens")
	parser.add_argument("--top-p", type=float, default=0.7, help="Top-p nucleus sampling")
	parser.add_argument("--top-k", type=int, default=-1, help="Top-k sampling")
	parser.add_argument("--n-samples", type=int, default=4, help="Number of samples per prompt")
	parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel world size")
	parser.add_argument("--custom-reward-function-path", type=str, default=None, help="Path to custom reward function file")
	parser.add_argument("--custom-reward-function-name", type=str, default=None, help="Name of the custom reward function")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	main(
		model_name_or_path=args.model_name_or_path,
		data_files=args.data_files,
		output_path=args.output_path,
		prompt_key=args.prompt_key,
		data_source_key=args.data_source_key,
		reward_model_key=args.reward_model_key,
		batch_size=args.batch_size,
		temperature=args.temperature,
		max_tokens=args.max_tokens,
		top_p=args.top_p,
		top_k=args.top_k,
		n_samples=args.n_samples,
		tensor_parallel_size=args.tensor_parallel_size,
		custom_reward_function_path=args.custom_reward_function_path,
		custom_reward_function_name=args.custom_reward_function_name,
	)

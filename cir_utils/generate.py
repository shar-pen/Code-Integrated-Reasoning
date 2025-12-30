import os
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Any
from vllm.outputs import RequestOutput, CompletionOutput

from .python_interpreter import JupyterExecutorManager

@dataclass
class CodeIntegratedGenerationConfig:
	enable: bool = False
	execution_tag_start: str = '<python_interpreter>'
	execution_tag_end: str = '</python_interpreter>'
	observation_tag_start: str = '<execution_result>'
	observation_tag_end: str = '</execution_result>'
	max_execution_count: int = 3
	execution_parallel_num: int = 8


def is_finished(vllm_output):
	# check if generation is finished normally
	return vllm_output.outputs[0].finish_reason == 'stop' and vllm_output.outputs[0].stop_reason == None

def is_needing_execution(vllm_output, end_tag: str):
	# check if generation is finished with needing execution triggered
	return vllm_output.outputs[0].finish_reason == 'stop' and vllm_output.outputs[0].stop_reason == end_tag

def extract_between_tags_regex(text: str, start_tag: str, end_tag: str) -> List[str]:
	pattern = re.compile(re.escape(start_tag) + r"(.*?)" + re.escape(end_tag), re.DOTALL)
	matches = pattern.findall(text)
	return [match.strip('\n') for match in matches]

def format_execution_result(result: str, start_tag: str, end_tag: str) -> str:
	return f'\n{start_tag}\n{result}\n{end_tag}\n'

def get_active_data(data_info):
	"""
	extract active data for next generation round, i.e., those not finished normally yet

	returns:
		active_vllm_inputs: List[Dict], vllm llm input
		active_idxes: List[int], indexes in the original data_info, for mapping back
	"""
	active_vllm_inputs = []
	active_idxes = []
	for info in data_info:
		if not info['is_finished']:
			active_vllm_inputs.append({"prompt_token_ids": info['prompt_token_ids'] + info['response_token_ids']})
			active_idxes.append(info['idx'])
	return active_vllm_inputs, active_idxes

def to_vllm_request_outputs(data_info: List[Dict[str, Any]], tokenizer) -> List[RequestOutput]:
	"""
	data_info: [{'idx', 'prompt', 'sequence', 'is_finished', ...}, ...]
	tokenizer: the tokenizer used to encode the prompts and completions

	try to imitate vLLM's RequestOutput format so that agentic generation can just use existing pipeline of normal generation.
	"""
	outs: List[RequestOutput] = []

	for r in data_info:
		prompt_text: str = r["prompt"]
		completion_text = r["response"]
		finished: bool = bool(r.get("is_finished", True))
		prompt_token_ids = r["prompt_token_ids"]
		completion_token_ids = r["response_token_ids"]

		outs.append(
			RequestOutput(
				request_id=str(r.get("idx", "")),
				prompt=prompt_text,
				prompt_token_ids=prompt_token_ids,
				prompt_logprobs=None,
				outputs=[
					CompletionOutput(
						index=0,
						text=completion_text,
						token_ids=completion_token_ids,
						cumulative_logprob=None,
						logprobs=None,
						finish_reason="stop" if finished else None,
						stop_reason=None,
					)
				],
				finished=finished,
				metrics=None,
				lora_request=None,
				encoder_prompt=None,
				encoder_prompt_token_ids=None,
				num_cached_tokens=None,
				multi_modal_placeholders={},
			)
		)

	return outs

def code_integrated_generate(
		vllm_inference_engine, executor_manager, prompts, sampling_params, cir_config: CodeIntegratedGenerationConfig, *, 
		lora_request=None, prompt_adapter_request=None, gided_options_request=None
	):
	"""
	agentic code-integrated generation
	inputs are basically like normal llm.generate inputs.
	"""
	tokenizer = vllm_inference_engine.get_tokenizer()
	data_info = [
		{
			'idx': idx,
			'prompt': tokenizer.decode(item['prompt_token_ids']),
			'prompt_token_ids': item['prompt_token_ids'],
			'response': "",
			'response_token_ids': [],
			'response_observation_mask': [],
			'is_finished': False,
			'code_triggered_count': 0,
			'code_execution_count': 0,
		}
		for idx, item in enumerate(prompts)
	]

	max_execution_count = cir_config.max_execution_count
	execution_parallel_num = cir_config.execution_parallel_num

	# initialize executor manager, maybe use existing ones and reset them, or create new ones
	ids = list(range(len(prompts)))
	executor_manager.add_executors(ids)

	# make a copy of sampling_params to modify
	sampling_params_override = deepcopy(sampling_params)
	sampling_params_override.stop = [cir_config.execution_tag_end]
	sampling_params_override.include_stop_str_in_output = True
	sampling_params_override.detokenize = True

	# main agentic generation loop
	while True:

		# check if there is active prompt needing generation
		active_vllm_inputs, active_idxes = get_active_data(data_info)
		if len(active_vllm_inputs) == 0:
			break

		# follow verl.workers.rollout.vllm_rollout.vlmm_rollout_spmd vLLMRollout.generate_sequences
		current_vllm_outputs = vllm_inference_engine.generate(
			prompts=active_vllm_inputs,
			sampling_params=sampling_params_override,
			lora_request=lora_request,
			use_tqdm=False,
		)
		
		# Collect execution requests to run them in parallel
		exec_requests = []  # list of (idx, code, gen_text)

		for a_vllm_output, idx in zip(current_vllm_outputs, active_idxes):

			# appending generated token ids to response buffer
			gen_text = a_vllm_output.outputs[0].text
			gen_token_ids = a_vllm_output.outputs[0].token_ids
			data_info[idx]['response'] += gen_text
			data_info[idx]['response_token_ids'] += gen_token_ids
			data_info[idx]['response_observation_mask'] += [1] * len(gen_token_ids)

			# stop or execute
			if is_finished(a_vllm_output):
				# stop generation
				data_info[idx]['is_finished'] = True
			elif is_needing_execution(a_vllm_output, cir_config.execution_tag_end):
				data_info[idx]['code_triggered_count'] += 1
				# extract code
				codeblocks = extract_between_tags_regex(gen_text, cir_config.execution_tag_start, cir_config.execution_tag_end)
				if len(codeblocks) > 0:
					# noramlly we only get one special code block per generation round
					code = codeblocks[0]
					if data_info[idx]['code_execution_count'] < max_execution_count:
						# waiting for batch execution
						exec_requests.append({'id': idx, 'code': code})
					else:
						# max execution count reached, appending plain text result
						execution_return = "Max execution count reached"
						tagged_execution_result = format_execution_result(execution_return, cir_config.observation_tag_start, cir_config.observation_tag_start)
						tagged_execution_result_token_ids = tokenizer.encode(tagged_execution_result, add_special_tokens=False)

						data_info[idx]['response'] += tagged_execution_result
						data_info[idx]['response_token_ids'] += tagged_execution_result_token_ids
						data_info[idx]['response_observation_mask'] += [0] * len(tagged_execution_result_token_ids)
				else:
					# abnormal case: no code block found
					execution_return = "Error: No valid code found for execution."
					tagged_execution_result = format_execution_result(execution_return, cir_config.observation_tag_start, cir_config.observation_tag_start)
					tagged_execution_result_token_ids = tokenizer.encode(tagged_execution_result, add_special_tokens=False)
					data_info[idx]['response'] += tagged_execution_result
					data_info[idx]['response_token_ids'] += tagged_execution_result_token_ids
					data_info[idx]['response_observation_mask'] += [0] * len(tagged_execution_result_token_ids)
			else:
				# unexpected case, not noraml stop or execution triggered
				warnings.warn(f"Unexpected generation result: {gen_text}")
				data_info[idx]['is_finished'] = True

		# Run collected execution requests via manager
		if len(exec_requests) > 0:
			# batch execution
			exec_results = executor_manager.execute_requests(exec_requests) 
			for item in exec_results:
				idx = item['id']
				execution_return = item['execution_return']
				# appending execution result to response buffer
				tagged_execution_result = format_execution_result(execution_return, cir_config.observation_tag_start, cir_config.observation_tag_end)
				tagged_execution_result_token_ids = tokenizer.encode(tagged_execution_result, add_special_tokens=False)
				data_info[idx]['response'] += tagged_execution_result
				data_info[idx]['response_token_ids'] += tagged_execution_result_token_ids
				data_info[idx]['response_observation_mask'] += [0] * len(tagged_execution_result_token_ids)
				data_info[idx]['code_execution_count'] += 1

	final_output_vllm_format = to_vllm_request_outputs(data_info, tokenizer)

	response_observation_mask = [item['response_observation_mask'] for item in data_info]
	code_triggered_count = [item['code_triggered_count'] for item in data_info]
	code_execution_count = [item['code_execution_count'] for item in data_info]
	
	executor_manager.reset_executors(ids)

	return final_output_vllm_format, response_observation_mask, code_triggered_count, code_execution_count
	
def extract_continuous_prompt_ids(prompt_id, mask, mask_value=0):
	"""
	extract continous id (satisfy mask==mask_value) from prompt_id 
	"""
	result = []
	current_group = []

	for pid, m in zip(prompt_id, mask):
		if m == mask_value:
			current_group.append(pid)
		else:
			if current_group:
				result.append(current_group)
				current_group = []

	if current_group:
		result.append(current_group)

	return result


if __name__ == "__main__":

	import numpy as np
	from vllm import LLM, SamplingParams


	excutor_manager = JupyterExecutorManager(max_workers=8)
	config = CodeIntegratedGenerationConfig()

	model_name_or_path= 'Qweb/Qwen2.5-7B-Instruct'
	inference_engine = LLM(
		model=model_name_or_path, 
		tensor_parallel_size=1, 
		gpu_memory_utilization=0.8,
		skip_tokenizer_init=False,
	)
	tokenizer = inference_engine.get_tokenizer()

	sampling_params = SamplingParams(
		temperature=1,
		top_p=0.95,
		max_tokens=2048,
		n=1,
	)

	system_prompt = """During your reasoning, if needed, you can choose to write python code between the tags <python_interpreter> and </python_interpreter> to help you with calculations or logic, such as <python_interpreter>\n# pure python code only (NO backticks, NO markdown, NO prose, NO extra tags)\n</python_interpreter>. 
	The code executor will run your code and return the output (stdout / plain text / error message) back to you between the tags <excution_result> and </excution_result>.
	You will continue your reasoning after receiving the execution result.
	Please reason step by step, and put your final answer within \\boxed{}. 
	"""

	batch_prompts = [
		[
			{'role': 'system', 'content': system_prompt}, 
			{'role': 'user', 'content': 'Calculate 1+2+...+100.'}, # one-shot example
			{'role': 'assistant', 'content': "To calculate the sum of the first 100 positive integers, we can use the formula for the sum of an arithmetic series. The formula for the sum \\( S \\) of the first \\( n \\) positive integers is given by:\n\n\\[ S = \\frac{n(n + 1)}{2} \\]\n\nIn this case, \\( n = 100 \\).\n\nLet's plug in \\( n = 100 \\) into the formula to calculate the sum.\n<python_interpreter>\nn = 100\nsum_of_integers = n * (n + 1) // 2\nsum_of_integers\n</python_interpreter>\n<execution_result>\n5050\n</execution_result>\nThe sum of the first 100 positive integers is \\(\\boxed{5050}\\)."},
			{'role': 'user', 'content': 'how many integers from 1 to 100 (inclusive) contain the digit 7 at least once in their decimal representation.'},
		],
	] * 8
	batch_prompts = [tokenizer.apply_chat_template(item, add_generation_prompt=True) for item in batch_prompts]
	vllm_inputs = [
		{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in batch_prompts
	]

	output, mask, code_triggered_count, code_execution_count = code_integrated_generate(
		inference_engine,
		excutor_manager,
		vllm_inputs,
		sampling_params,
		config,
	)

	print(output)

	print((np.array(code_triggered_count) > 0).astype(float).mean())
	print((np.array(code_execution_count) > 0).astype(float).mean())

	try:
		excutor_manager.shutdown()
	except:
		pass
	
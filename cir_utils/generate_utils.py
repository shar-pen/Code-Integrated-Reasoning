import os
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Any
from vllm.outputs import RequestOutput, CompletionOutput
from .encode_utils import encode_with_tag

from .python_interpreter import ExecutorManager

@dataclass
class CodeIntegratedGenerationConfig:
	enable: bool = False
	execution_tag_start: str = '<python_interpreter>'
	execution_tag_end: str = '</python_interpreter>'
	observation_tag_start: str = '<execution_result>'
	observation_tag_end: str = '</execution_result>'
	max_execution_count: int = 3
	max_try_execution_count: int = 5
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



class GenerationInfoManager:
	"""Encapsulates `data_info` and provides helper methods to update and query generation state."""

	def __init__(self, prompts: List[Dict[str, Any]], tokenizer, cir_config: CodeIntegratedGenerationConfig, max_tokens: int):
		self.tokenizer = tokenizer
		self.cir_config = cir_config
		self.max_tokens = max_tokens
		self.special_tags = [
			cir_config.execution_tag_start,
			cir_config.execution_tag_end,
			cir_config.observation_tag_start,
			cir_config.observation_tag_end,
		]
		self.data: List[Dict[str, Any]] = []
		for idx, item in enumerate(prompts):
			prompt_token_ids = item['prompt_token_ids']
			prompt = tokenizer.decode(prompt_token_ids)
			self.data.append({
				'idx': idx,
				'prompt': prompt,
				'prompt_token_ids': encode_with_tag(prompt, tokenizer, self.special_tags),
				'response': "",
				'response_token_ids': [],
				'response_observation_mask': [],
				'is_finished': False,
				'code_triggered_count': 0,
				'code_execution_count': 0,
				'code_execution_error_cnt': 0,
			})

	def append_generation(self, idx: int, gen_text: str):
		gen_token_ids = encode_with_tag(gen_text, self.tokenizer, self.special_tags)
		self.data[idx]['response'] += gen_text
		self.data[idx]['response_token_ids'] += gen_token_ids
		self.data[idx]['response_observation_mask'] += [1] * len(gen_token_ids)

	def mark_finished(self, idx: int):
		self.data[idx]['is_finished'] = True

	def increment_code_triggered(self, idx: int):
		self.data[idx]['code_triggered_count'] += 1

	def get_execution_count(self, idx: int) -> int:
		return int(self.data[idx]['code_execution_count'])
	
	def get_code_triggered_count(self, idx: int) -> int:
		return int(self.data[idx]['code_triggered_count'])

	def append_execution(self, idx: int, execution_result: str):
		"""Append the formatted execution result without mutating execution counters."""
		tagged_execution_result = format_execution_result(
			execution_result,
			self.cir_config.observation_tag_start,
			self.cir_config.observation_tag_end,
		)
		token_ids = encode_with_tag(tagged_execution_result, self.tokenizer, self.special_tags)
		self.data[idx]['response'] += tagged_execution_result
		self.data[idx]['response_token_ids'] += token_ids
		self.data[idx]['response_observation_mask'] += [0] * len(token_ids)
		if len(self.data[idx]['response_token_ids']) > self.max_tokens:
			self.data[idx]['is_finished'] = True
			self.data[idx]['response_token_ids'] = self.data[idx]['response_token_ids'][:self.max_tokens]
			self.data[idx]['response_observation_mask'] = self.data[idx]['response_observation_mask'][:self.max_tokens]

	def update_execution_counters(self, idx: int, *, increment_exec_count: bool = False, increment_exec_error: bool = False):
		"""Update bookkeeping counters after an execution result."""
		if increment_exec_count:
			self.data[idx]['code_execution_count'] += 1
		if increment_exec_error:
			self.data[idx]['code_execution_error_cnt'] += 1

	def get_active_data(self):
		active_vllm_inputs = []
		active_idxes = []
		for info in self.data:
			if not info['is_finished']:
				active_vllm_inputs.append({"prompt_token_ids": info['prompt_token_ids'] + info['response_token_ids']})
				active_idxes.append(info['idx'])
		return active_vllm_inputs, active_idxes

	def to_vllm_request_outputs(self) -> List[RequestOutput]:
		outs: List[RequestOutput] = []
		for r in self.data:
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

	def summaries(self):
		return (
			[item['response_observation_mask'] for item in self.data],
			[item['code_triggered_count'] for item in self.data],
			[item['code_execution_count'] for item in self.data],
		)


def code_integrated_generate(
		vllm_inference_engine, prompts, sampling_params, cir_config: CodeIntegratedGenerationConfig, *, executor_manager=None,
		lora_request=None, prompt_adapter_request=None, gided_options_request=None
	):
	"""
	agentic code-integrated generation
	inputs are basically like normal llm.generate inputs.
	"""
	tokenizer = vllm_inference_engine.get_tokenizer()
	max_execution_count = cir_config.max_execution_count
	max_try_execution_count = cir_config.max_try_execution_count
	assert max_try_execution_count >= max_execution_count, "max_try_execution_count should be >= max_execution_count"

	# initialize executor manager, maybe use existing ones and reset them, or create new ones
	ids = list(range(len(prompts)))
	if executor_manager is None:
		executor_manager = ExecutorManager(ids=ids, max_workers=cir_config.execution_parallel_num)
	else:
		executor_manager.add_executors(ids)

	# make a copy of sampling_params to modify
	sampling_params_override = deepcopy(sampling_params)
	sampling_params_override.stop = [cir_config.execution_tag_end]
	sampling_params_override.include_stop_str_in_output = True
	sampling_params_override.detokenize = True

	# initialize generation info manager
	generation_info_manager = GenerationInfoManager(prompts, tokenizer, cir_config, sampling_params_override.max_tokens)

	# main agentic generation loop
	while True:

		# check if there is active prompt needing generation
		active_vllm_inputs, active_idxes = generation_info_manager.get_active_data()
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
			# gen_token_ids = a_vllm_output.outputs[0].token_ids
			generation_info_manager.append_generation(idx, gen_text)

			# stop or execute
			if is_finished(a_vllm_output):
				# stop generation
				generation_info_manager.mark_finished(idx)
			elif is_needing_execution(a_vllm_output, cir_config.execution_tag_end):
				generation_info_manager.increment_code_triggered(idx)
				# extract code
				codeblocks = extract_between_tags_regex(gen_text, cir_config.execution_tag_start, cir_config.execution_tag_end)
				if len(codeblocks) > 0:
					# noramlly we only get one special code block per generation round
					code = codeblocks[0]
					if generation_info_manager.get_execution_count(idx) < max_execution_count:
						# waiting for batch execution
						exec_requests.append({'id': idx, 'code': code})
					else:
						# max execution count reached, appending plain text result
						execution_return = "Max execution count reached"
						generation_info_manager.append_execution(idx, execution_return)
						generation_info_manager.update_execution_counters(idx, increment_exec_count=False, increment_exec_error=False)
				else:
					# abnormal case: no code block found
					execution_return = "Error: No valid code found for execution."
					generation_info_manager.append_execution(idx, execution_return)
					generation_info_manager.update_execution_counters(idx, increment_exec_count=False, increment_exec_error=True)

				# too many code triggers, mark finished
				if generation_info_manager.get_code_triggered_count(idx) > max_try_execution_count:
					generation_info_manager.mark_finished(idx)
			else:
				# unexpected case, not noraml stop or execution triggered
				warnings.warn(f"Unexpected generation result: {gen_text}")
				generation_info_manager.mark_finished(idx)

		# Run collected execution requests via manager
		if len(exec_requests) > 0:
			# batch execution
			exec_results = executor_manager.execute_requests(exec_requests) 
			for item in exec_results:
				idx = item['id']
				execution_return = item['execution_return']
				execution_error = bool(item.get('execution_error', False))
				# appending execution result to response buffer and update counters
				generation_info_manager.append_execution(idx, execution_return)
				generation_info_manager.update_execution_counters(idx, increment_exec_count=True, increment_exec_error=execution_error)

	final_output_vllm_format = generation_info_manager.to_vllm_request_outputs()

	response_observation_mask, code_triggered_counts, code_execution_counts = generation_info_manager.summaries()
	
	try:
		executor_manager.shutdown_executor(ids)
	except:
		pass

	return final_output_vllm_format, response_observation_mask, code_triggered_counts, code_execution_counts
	


if __name__ == "__main__":

	import numpy as np
	from vllm import LLM, SamplingParams


	excutor_manager = ExecutorManager(max_workers=8)
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
	
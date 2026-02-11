r"""
Because verl use different reward function based on `data_source` value, I alter `verl.utils.reward_score.__init__.default_compute_score, to choose reward function also based on prefix of `data_source` value. 

Below is a collection of prompt templates for different math problem styles. Refer to these datasets (openai/gsm8k, DigitalLearningGmbH/MATH-lighteval, BytedTsinghua-SIA/DAPO-Math-17k) for more details.
"""


output_style_2_system_prompt = {
	'math_dapo_style': 'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem. Remember to put your answer on its own line after "Answer:".',
	'gsm8k_style': 'Let\'s think step by step and output the final answer after "####".',
	'math_reward_style': 'Let\'s think step by step and output the final answer within \\boxed{}.',
}

code_integrated_generation_prompt = (
	# old prompt
	# "During your reasoning, if needed, you can choose to write python code between the tags <python_interpreter> and </python_interpreter> to help you with calculations or logic, such as <python_interpreter>\n# pure python code only (NO backticks, NO markdown, NO prose, NO extra tags)\n</python_interpreter>. "
	# "The code executor will run your code and return the output (stdout / plain text / error message) back to you between the tags <execution_result> and </execution_result>. "
	# "You will continue your reasoning after receiving the execution result."
	
	# new prompt
	"During your reasoning, you can use python code to perform logic and calculations. And you should always trust the execution result returned by the code executor. After receiving the execution result, you can choose to write more code if needed. Remeber you should prefer reasoning with code to reasoning directly. "
	"If you decide to use python code, you must write python code between the tags <python_interpreter> and </python_interpreter>, such as <python_interpreter>\n# pure python code only (NO backticks, NO markdown, NO prose, NO extra tags)\n</python_interpreter>. "
	"The code executor will run your code and return the output (stdout / plain text / error message) back to you between the tags <execution_result> and </execution_result> to support your reasoning process. You will continue your reasoning after receiving the execution result. "
	"Remember that execution result is not formal enough as final answer and you need to follow the output format instructions to provide your final answer. "
)

format_kwargs = {}

def format_prompt(problem: str, **format_kwargs):
	
	system_prompt = []
	output_style = format_kwargs.get('output_style', 'math_reward_style')
	enable_code_integrated_generation = format_kwargs.get('enable_code_integrated_generation', False)
	# code integrated generation prompt
	if enable_code_integrated_generation:
		system_prompt.append(code_integrated_generation_prompt)
	# output style prompt
	if output_style in output_style_2_system_prompt:
		output_style_prompt = output_style_2_system_prompt[output_style]
		system_prompt.append(output_style_prompt)
	system_prompt = '\n'.join(system_prompt)

	msgs = [
		{'role': 'system', 'content': system_prompt},
		{'role': 'user', 'content': problem},
	]
	return msgs


def example_map_fn(example, idx, process_fn, data_source, ability, split, format_prompt_kwargs={}, process_fn_kwargs={}):
	output_style = format_prompt_kwargs.get('output_style', 'math_reward_style')
	problem, ground_truth = process_fn(example, **process_fn_kwargs)
	prompt = format_prompt(problem, **format_prompt_kwargs)
	data = {
		"data_source": f'{output_style}/{data_source}',
		"prompt": prompt,
		"ability": ability,
		"reward_model": {"style": "rule", "ground_truth": ground_truth},
		"extra_info": {"split": split, "index": idx, },
	}
	return data

def remove_boxed(s):
	if "\\boxed " in s:
		left = "\\boxed "
		assert s[: len(left)] == left
		return s[len(left) :]

	left = "\\boxed{"

	assert s[: len(left)] == left
	assert s[-1] == "}"

	return s[len(left) : -1]
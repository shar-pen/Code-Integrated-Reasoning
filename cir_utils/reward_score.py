import re
import numpy as np


from verl.utils.reward_score import default_compute_score


def compute_has_tag_score(solution_str, **kwargs):
	"""
	Compute the score based on the presence of specific tags in the solution string.
	Return the averaged score over the tags existing in the solution.

	Example:
	solution_str = "<python_interpreter>print('Hello World')</python_interpreter>"
	score = 1.0
	solution_str = "<python_interpreter>print('Hello World')<tag>"
	score = 1.0
	"""
	tags = ['<python_interpreter>', '</python_interpreter>'] # tags that model should generate
	is_tag_in_solution = [item in solution_str for item in tags]
	averaged_score = np.array(is_tag_in_solution, dtype=float).mean().item()
	return averaged_score


def compute_tag_format_score(solution_str, **kwargs):
	"""
	check whether the tags in the solution string are in the correct format, i.e., ...<python_interpreter>...</python_interpreter>...<execution_result>...</execution_result>...
	return 1.0 if the format is correct, else return 0.0
	"""
	
	pattern = r'<python_interpreter>.*?</python_interpreter>.*?<execution_result>.*?</execution_result>'
	cleaned_text, count = re.subn(pattern, '', solution_str, flags=re.DOTALL)

	# if there is no such pattern, return 0.0
	if count == 0:
		return 0.0
	# if there is any leftover tags, return 0.0, as the format is incorrect
	for tag in ['<python_interpreter>', '</python_interpreter>', '<execution_result>', '</execution_result>']:
		if tag in cleaned_text:
			return 0.0
	return 1.0

def compute_normal_execution_result_score(solution_str, **kwargs):
	"""
	check whether the execution results in the solution string are normal, i.e., do not contain error messages, or blank results.
	return the ratio of normal execution results among all execution results.

	good code-intergrated reasoning should produce codes that run successfully and produce normal execution results.
	"""
	pattern = r'<execution_result>(.*?)</execution_result>'
	substrs = re.findall(pattern, solution_str, flags=re.DOTALL)
	substrs = [s.strip('\n').strip(' ').lower() for s in substrs if s.strip() != '']

	is_substr_normal = [
		(
			not any([abnoraml_str in s for abnoraml_str in ['error', 'traceback', 'exception', 'timeout', 'failed']]) \
			and len(s) > 0
		)
		for s in substrs
	]

	score = np.array(is_substr_normal, dtype=float).mean().item() if len(is_substr_normal) > 0 else 0.0
	return score


def compute_cir_score(
	data_source,
	solution_str,
	ground_truth,
	extra_info=None,
	sandbox_fusion_url=None,
	concurrent_semaphore=None,
	memory_limit_mb=None,
):
	"""
	Compute the reward score for Code Integrated Reasoning tasks.
	"""

	tag_presence_score = compute_has_tag_score(solution_str)
	tag_format_score = compute_tag_format_score(solution_str)
	normal_execution_result_score = compute_normal_execution_result_score(solution_str)
	answer_score = default_compute_score(
		data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
	)

	final_score = sum([
		0.1 * tag_presence_score,
		0.2 * tag_format_score,
		0.2 * normal_execution_result_score,
		0.5 * answer_score,
	])

	return {
		"score": final_score,
		"sub_score/tag_presence_score": tag_presence_score,
		"sub_score/tag_format_score": tag_format_score,
		"sub_score/normal_execution_result_score": normal_execution_result_score,
		"sub_score/answer_score": answer_score,
	}


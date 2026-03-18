import re, json
import numpy as np


from verl.utils.reward_score import default_compute_score
from cir_utils.llm_as_a_judge import DEFAULT_CLIENT, get_completion, extract_codefence_by_type


# individual score functions

def compute_tag_presence_score(solution_str, **kwargs):
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

def compute_python_interpreter_within_think_score(solution_str, **kwargs):
	"""
	check whether </python_interpreter> is before </think>, return 1.0 if true, else return 0.0
	"""
	think_pos = solution_str.rfind('</think>')
	python_interpreter_pos = solution_str.rfind('</python_interpreter>')
	if think_pos == -1 or python_interpreter_pos == -1:
		return 0.0
	return 1.0 if python_interpreter_pos < think_pos else 0.0

# after experiment, i find that you shouldn't use this reward for RL, it is hard to learn. my suggestion is to ensure model output in good format during SFT phrase, and only use accuracy-based reward for RL, which is easier to learn and can also achieve good performance. 
def compute_answer_quality_score(solution_str, max_retries: int = 3, **kwargs):
	"""
	Rate answer quality based on the LASJ rubric, can be adjusted to personal likings.
	"""
	prompt_template = """You are an evaluator for mathematical answer presentation quality.

Your task is to evaluate ONLY the final answer written by the model.
Do NOT evaluate the reasoning process, hidden chain-of-thought, or whether the math is correct.
You are not given the original question, and you must judge the answer purely as a standalone piece of mathematical writing.

Evaluate how formal, detailed, clear, and mathematically well-presented the final answer is on its own.

[Model Final Answer To Evaluate]
```markdown
{final_answer_only}
```

Evaluate the final answer on the following dimensions:

1. Formality (0-2)
- 0: Informal, conversational, sloppy, or uses vague phrases.
- 1: Mostly formal but not consistently precise.
- 2: Fully formal, precise, and appropriate for a mathematical solution or explanation.

2. Detail and Self-Contained Completeness (0-4)
- 0: Extremely brief, almost no explanation, just states a result or fragment.
- 1: Minimal explanation, with little supporting detail or justification.
- 2: Some useful explanation, but still terse, underdeveloped, or missing important connecting statements.
- 3: Detailed and reasonably self-contained, with enough explanation for a reader to follow the main idea.
- 4: Very detailed, well-elaborated, and self-contained, with clear supporting statements and sufficient justification throughout.

3. Structure and Clarity (0-2)
- 0: Disorganized, hard to follow, poor flow or formatting.
- 1: Mostly understandable but could be better organized.
- 2: Well-structured, clear flow, readable formatting, and easy to follow.

4. Mathematical Style and Precision (0-2)
- 0: Poor notation, ambiguous references, imprecise wording.
- 1: Mostly precise but with some ambiguity or notation issues.
- 2: Uses precise mathematical language and notation appropriately.

Important instructions:
- Ignore whether the answer is mathematically correct.
- Ignore whether the answer actually solves the original problem, since the problem is not provided.
- Judge only the visible final answer as a piece of mathematical writing.
- Prefer answers that clearly state claims, use precise wording, and present explanation in an organized and rigorous style.
- Do NOT reward verbosity alone. Extra text should improve clarity, rigor, detail, or readability.
- Do NOT give a high score to filler, repetition, generic mathematical phrases, or empty elaboration.
- If the answer is only a short final result with little or no explanation, it should score low on detail/completeness even if it appears polished.

Return your evaluation in valid JSON with the following schema:
```json
{{
  "formality": <0-2>,
  "detail_self_contained_completeness": <0-4>,
  "structure_clarity": <0-2>,
  "math_style_precision": <0-2>,
}}
```
"""
	max_score = 10.0 # need to adjust according to your rubric score setting
	solution_str = solution_str.split('</think>')[-1].strip('\n')
	prompt = prompt_template.format(final_answer_only = solution_str)
	for attempt in range(1, max_retries + 1):
		try:
			ret = get_completion(DEFAULT_CLIENT, model='gpt-5-mini', prompt=prompt, return_type='content')
			if ret.startswith('{') and ret.endswith('}'):
				json_str_dict = json.loads(ret)
			else:
				json_str = extract_codefence_by_type(ret, 'json')[0]
				json_str_dict = json.loads(json_str)
			score = sum(json_str_dict.values())
			score_normalized = score / max_score
			return score_normalized
		except Exception as e:
			if attempt == max_retries:
				raise RuntimeError(f"Failed to evaluate answer quality score: {e}")


# aggregated score function

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

	tag_presence_score = compute_tag_presence_score(solution_str)
	tag_format_score = compute_tag_format_score(solution_str)
	normal_execution_result_score = compute_normal_execution_result_score(solution_str)
	python_interpreter_within_think_score = compute_python_interpreter_within_think_score(solution_str)
	# answer_quality_score = compute_answer_quality_score(solution_str)

	answer_score = default_compute_score(
		data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
	)

	final_score = sum([
		0.1 * tag_presence_score,
		0.1 * tag_format_score,
		0.1 * normal_execution_result_score,
		0.2 * python_interpreter_within_think_score,
		0.5 * tag_format_score * answer_score,
		# 0.5 * answer_score * answer_quality_score,
	])

	return {
		"score": final_score,
		"sub_score/tag_presence_score": tag_presence_score,
		"sub_score/tag_format_score": tag_format_score,
		"sub_score/normal_execution_result_score": normal_execution_result_score,
		"sub_score/python_interpreter_within_think_score": python_interpreter_within_think_score,
		"sub_score/answer_score": answer_score,
	}


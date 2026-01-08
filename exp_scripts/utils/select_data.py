import re, fire
import numpy as np
import pandas as pd


def has_paired_tag(text: str, tags: list[tuple[str, str]]) -> bool:
    start_tags = {tag[0] for tag in tags}
    end_map = {tag[1]: tag[0] for tag in tags}
    search_tags = sorted(list(start_tags) + list(end_map.keys()), key=len, reverse=True)
    
    if not search_tags:
        return True
        
    pattern = '|'.join(map(re.escape, search_tags))
    stack = []
    
    for match in re.finditer(pattern, text):
        tag = match.group()
        if tag in start_tags:
            stack.append(tag)
        elif tag in end_map:
            if not stack or stack[-1] != end_map[tag]:
                return False
            stack.pop()
            
    return len(stack) == 0


# input_data_path = 'data/infer/infer.parquet'
# output_data_path = 'data/infer/distillation_data.parquet'

def main(input_data_path: str, output_data_path: str):
	tags = [["<python_interpreter>", "</python_interpreter>"], ["<execution_result>", "</execution_result>"]]
	select_min_exec = True
	dataset = pd.read_parquet(input_data_path)

	data = []
	for idx, row in dataset.iterrows():

		data_source = row['data_source']
		if 'gsm8k' in data_source:
			continue

		prompt = row['prompt']
		scores = row['scores']
		responses = row['responses']
		code_triggered_counts = row['code_triggered_counts']
		code_execution_counts = row['code_execution_counts']

		correctness_mask = (scores > 0.5)
		has_execution_mask = (code_execution_counts > 0)
		good_execution_format_mask_mask = (code_execution_counts == code_triggered_counts)
		has_paired_tag_mask = np.array([has_paired_tag(resp, tags) for resp in responses], dtype=bool)

		qualified_mask = correctness_mask & has_execution_mask & good_execution_format_mask_mask & has_paired_tag_mask

		if (qualified_mask).any():
			responses = responses[qualified_mask]
			scores = scores[qualified_mask]
			code_execution_counts = code_execution_counts[qualified_mask]

			if select_min_exec:
				min_exec_cnt = np.min(code_execution_counts)
				for resp, exec_cnt in zip(responses, code_execution_counts):
					if exec_cnt == min_exec_cnt:
						data.append({
							'messages': [
								*prompt,
								{'role': 'assistant', 'content': resp}
							]
						})
			else:
				for resp, exec_cnt in zip(responses, code_execution_counts):
					data.append({
						'messages': [
							*prompt,
							{'role': 'assistant', 'content': resp}
						]
					})
		else:
			continue
		
	print(f"Qualified data size: {len(data)}")
		
	data = pd.DataFrame(data)
	data.to_parquet(output_data_path, index=False)

if __name__ == "__main__":
	fire.Fire(main)
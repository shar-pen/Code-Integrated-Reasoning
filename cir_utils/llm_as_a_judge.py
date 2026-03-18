import os
import re
from dotenv import load_dotenv
from openai import OpenAI
import time
load_dotenv()


DEFAULT_CLIENT = OpenAI(
	base_url=os.getenv("OPENAI_BASE_URL"),
	api_key=os.getenv("OPENAI_API_KEY"),
)

def get_completion(
		client: OpenAI, model: str, 
		messages: list[dict] = None, prompt: str = None, 
		return_type: str = None, 
		tools: list = None, max_retries: int = 3
	):
	"""
	Get a completion from the OpenAI API.
	
	- If messages is provided, use it as the conversation history. If prompt is provided, wrap it in a single user message. If both are provided, raise an error.
	- return_type can be one of [None, 'choice', 'message', 'content'], which determines the format of the returned result.
	- If tools are provided, include them in the API call to enable tool use in the model's response.
	"""
	
	if messages is None and prompt is None:
		raise ValueError("Either messages or prompt must be provided.")
	elif messages is None and prompt is not None:
		messages = [
			{'role':'user', 'content': prompt}
		]
	elif messages is not None and prompt is None:
		pass
	else:
		raise ValueError("Only one of messages or prompt should be provided.")
	
	for attempt in range(1, max_retries + 1):
		try:
			completion = client.chat.completions.create(
				model=model,
				messages=messages,
				tools=tools,
			)
			break
		except Exception as e:
			if attempt == max_retries:
				raise RuntimeError(f"Failed to get completion after {max_retries} attempts: {e}")
			time.sleep(1)

	if return_type is None:
		return completion
	elif return_type == 'choice':
		return completion.choices[0]
	elif return_type == 'message':
		return completion.choices[0].message
	elif return_type == 'content':
		return completion.choices[0].message.content
	else:
		raise ValueError(f"Unsupported return_type: {return_type}, must be one of [None, 'choice', 'message', 'content']. I suggest 'message' if you involve tool or need reasoning content, otherwise 'content' is enough.")
	

def extract_codefence_by_type(markdown_text, target_type=None):
	"""
	Extract code blocks from a markdown string.

	- If target_type is a string (e.g. "python"), returns a list of code contents, whose info-string language matches target_type (case-insensitive).
	  ["code content 1", "code content 2", ...]
	- If target_type is None, returns a list of dicts:
	  [{"type": <language_or_None>, "content": <code>} ...]
	Supports both ``` and ~~~ fences.
	"""
	# Match opening fence (``` or ~~~), capture info string, capture body, require same closing fence.
	pattern = r"(?s)(```|~~~)\s*([^\n]*)\n(.*?)\n\1"
	matches = re.findall(pattern, markdown_text)

	results = []
	for _, info, body in matches:
		info = info.strip()
		# Only use the first token as the language (handles e.g. "python hl_lines=1")
		lang = info.split()[0] if info else ""
		if target_type is None:
			results.append({"type": (lang if lang else None), "content": body})
		else:
			if lang.lower() == str(target_type).strip().lower():
				results.append(body)

	return results


if __name__ == "__main__":
	
	prompt = "Hello, how are you?"
	ret = get_completion(DEFAULT_CLIENT, model='gpt-4o-2024-11-20', prompt=prompt, return_type='content')
	print(ret)
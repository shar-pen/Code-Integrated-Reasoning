import re
from transformers import AutoTokenizer
from typing import Callable, List, Sequence, Tuple, Optional



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

def split_keep_tags(text: str, tags: List[str]) -> List[str]:
    """Split by tags and keep the tags as standalone segments."""
    if not tags:
        return [text]

    tags_sorted = sorted(tags, key=len, reverse=True)  # longer first
    pattern = "(" + "|".join(re.escape(t) for t in tags_sorted) + ")"
    parts = re.split(pattern, text)
    return [p for p in parts if p != ""]

def encode_with_tag(
    text: str,
    tokenizer,
    tags_to_keep: List[str],
) -> List[int]:
    """
    Split text by tag boundaries, encode each segment separately, and concatenate token IDs.
    IMPORTANT: keep add_special_tokens=False for consistency across segments.
    """
    parts = split_keep_tags(text, tags_to_keep)
    token_ids: List[int] = []
    for p in parts:
        token_ids.extend(tokenizer.encode(p, add_special_tokens=False))
    return token_ids

def _find_subseq(haystack: List[int], needle: List[int], start: int = 0) -> int:
    """Return the first index >= start where haystack[i:i+len(needle)] == needle, else -1."""
    if not needle:
        raise ValueError("needle must be non-empty")
    n, m = len(haystack), len(needle)
    for i in range(start, n - m + 1):
        if haystack[i : i + m] == needle:
            return i
    return -1

def encode_and_mask(
    text: str,
    tokenizer,
    tags_to_keep: List[str],
    mask_pairs: List[Tuple[str, str]],
) -> Tuple[List[int], List[int]]:
    """
    Encode `text` with `encode_with_tag`, and compute a per-token loss mask.

    Masking rule (span mode, no nesting expected):
      - For each (start_marker, end_marker) in `mask_pairs`, find occurrences of
        start_marker ... end_marker in the tokenized sequence and set loss_mask=0
        for the entire span INCLUDING the start/end markers themselves.
      - All other tokens have loss_mask=1.

    Notes:
      - Both the full `text` and each marker string are tokenized using the SAME
        `encode_with_tag` to ensure consistency with tag-splitting behavior.
      - If a start marker is found but the corresponding end marker is not found,
        this function raises ValueError (assuming data generation is correct and non-nested).

    Returns:
      (token_ids, loss_mask)
        - token_ids: List[int]
        - loss_mask: List[int] with the same length as token_ids (0=masked, 1=keep)
    """
    token_ids = encode_with_tag(text, tokenizer, tags_to_keep)
    loss_mask = [1] * len(token_ids)

    if not mask_pairs or not token_ids:
        return token_ids, loss_mask

    # Pre-encode markers (must match the same encode_with_tag behavior)
    encoded_pairs: List[Tuple[List[int], List[int]]] = []
    for start_str, end_str in mask_pairs:
        start_ids = encode_with_tag(start_str, tokenizer, tags_to_keep)
        end_ids = encode_with_tag(end_str, tokenizer, tags_to_keep)
        if not start_ids or not end_ids:
            raise ValueError(
                f"Marker encodes to empty token ids. start={start_str!r}, end={end_str!r}"
            )
        encoded_pairs.append((start_ids, end_ids))

    # Collect all spans to mask: [l, r)
    spans: List[Tuple[int, int]] = []
    for start_ids, end_ids in encoded_pairs:
        i = 0
        while i < len(token_ids):
            s = _find_subseq(token_ids, start_ids, i)
            if s < 0:
                break
            search_from = s + len(start_ids)
            e = _find_subseq(token_ids, end_ids, search_from)
            if e < 0:
                raise ValueError(
                    f"Unmatched start marker at token index {s} (end marker not found)."
                    f"full text: {text}"
                )
            l, r = s, e + len(end_ids)
            spans.append((l, r))
            i = r  # continue after this block (no nesting assumed)

    if not spans:
        return token_ids, loss_mask

    # Merge spans (robust even if overlaps exist; well-formed data should not overlap/nest)
    spans.sort()
    merged: List[List[int]] = []
    for l, r in spans:
        if not merged or l > merged[-1][1]:
            merged.append([l, r])
        else:
            merged[-1][1] = max(merged[-1][1], r)

    # Apply mask
    for l, r in merged:
        for k in range(l, r):
            loss_mask[k] = 0

    return token_ids, loss_mask


if __name__ == "__main__":
    
	model_name_or_path= '../../../dc/models/Qwen2.5-3B-Instruct'
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
     
	test_str = "To calculate the sum of the first 100 positive integers, we can use the formula for the sum of an arithmetic series. The formula for the sum \\( S \\) of the first \\( n \\) positive integers is given by:\n\n\\[ S = \\frac{n(n + 1)}{2} \\]\n\nIn this case, \\( n = 100 \\).\n\nLet's plug in \\( n = 100 \\) into the formula to calculate the sum.\n<python_interpreter>\nn = 100\nsum_of_integers = n * (n + 1) // 2\nsum_of_integers\n</python_interpreter>\n<execution_result>\n5050\n</execution_result>\nThe sum of the first 100 positive integers is \\(\\boxed{5050}\\)."

	token_ids, loss_mask = encode_and_mask(
		test_str,
		tokenizer,
		tags_to_keep = ['<python_interpreter>', '</python_interpreter>', '<execution_result>', '</execution_result>'],
		mask_pairs=[("\n<execution_result>", "</execution_result>\n"),],
	)

	for item in extract_continuous_prompt_ids(token_ids, loss_mask):
		print(tokenizer.decode(item))
		print('=')
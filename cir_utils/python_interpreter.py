import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from jupyter_client import KernelManager
import concurrent.futures


# ANSI escape sequences (colors, cursor moves, etc.) often appear in tracebacks.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(text: str) -> str:
	"""Remove ANSI escape sequences from a string."""
	return _ANSI_RE.sub("", text)


@dataclass
class ExecResult:
	# Captured standard streams
	stdout: str = ""
	stderr: str = ""

	# All raw Jupyter output messages we observed on IOPub (useful for rich rendering)
	outputs: List[Dict[str, Any]] = field(default_factory=list)

	# Convenience: extracted "text/plain" payloads (e.g., last expression result)
	text_plain: List[str] = field(default_factory=list)

	# Error info (None if no error)
	# Recommended fields:
	#   ename, evalue, traceback (multi-line string)
	error: Optional[Dict[str, Any]] = None


class JupyterKernelExecutor:
	"""
	Execute code in a real IPython/Jupyter kernel (like a notebook cell).

	Features:
	- Supports "last-line expression auto display" (execute_result)
	- Supports %magic and !shell (IPython kernel)
	- Captures stdout/stderr (stream)
	- Captures rich outputs (display_data / execute_result)
	- Captures errors with readable multiline traceback
	"""

	def __init__(self, kernel_name: str = "python3", nocolor: bool = True):
		self.km = KernelManager(kernel_name=kernel_name)
		self.km.start_kernel()

		self.kc = self.km.client()
		self.kc.start_channels()
		self.kc.wait_for_ready(timeout=10)

		# Optionally disable colored tracebacks to avoid ANSI sequences.
		# We still keep strip_ansi as a fallback.
		if nocolor:
			self._run_silent("%colors nocolor", timeout=10)

	def shutdown(self, now: bool = True):
		"""Stop channels and shutdown the kernel process."""
		try:
			self.kc.stop_channels()
		finally:
			self.km.shutdown_kernel(now=now)

	def interrupt(self):
		"""Interrupt the kernel (useful for runaway code)."""
		self.km.interrupt_kernel()

	def _drain_iopub(self, max_seconds: float = 0.2):
		"""
		Drain pending IOPub messages to reduce cross-talk between executions.
		This is a best-effort cleanup.
		"""
		end = time.time() + max_seconds
		while time.time() < end:
			try:
				_ = self.kc.get_iopub_msg(timeout=0.05)
			except Exception:
				break

	def _run_silent(self, code: str, timeout: float = 10.0):
		"""
		Execute code silently and wait for idle (discard all outputs).
		Useful for one-time kernel configuration.
		"""
		self._drain_iopub()
		msg_id = self.kc.execute(code, silent=True, store_history=False)

		deadline = time.time() + timeout
		while True:
			remaining = deadline - time.time()
			if remaining <= 0:
				raise TimeoutError("Silent kernel execution timed out")

			msg = self.kc.get_iopub_msg(timeout=remaining)
			if msg.get("parent_header", {}).get("msg_id") != msg_id:
				continue

			if msg.get("msg_type") == "status":
				if msg.get("content", {}).get("execution_state") == "idle":
					return

	def execute(self, code: str, timeout: float = 30.0) -> ExecResult:
		"""
		Execute a notebook-like cell and collect outputs until the kernel becomes idle.
		"""
		self._drain_iopub()

		res = ExecResult()
		msg_id = self.kc.execute(code)

		deadline = time.time() + timeout
		while True:
			remaining = deadline - time.time()
			if remaining <= 0:
				raise TimeoutError("Kernel execution timed out")

			msg = self.kc.get_iopub_msg(timeout=remaining)

			# Only accept messages belonging to this execution request.
			if msg.get("parent_header", {}).get("msg_id") != msg_id:
				continue

			msg_type = msg.get("msg_type")
			content = msg.get("content", {}) or {}

			# Done signal
			if msg_type == "status" and content.get("execution_state") == "idle":
				break

			if msg_type == "stream":
				# content: {name: 'stdout'/'stderr', text: '...'}
				name = content.get("name")
				text = content.get("text", "")
				if name == "stdout":
					res.stdout += text
				else:
					res.stderr += text
				res.outputs.append({"type": "stream", **content})

			elif msg_type in ("execute_result", "display_data"):
				# content: {data: {...}, metadata: {...}, execution_count?: ...}
				data = content.get("data", {}) or {}
				tp = data.get("text/plain")
				if tp is not None:
					if isinstance(tp, list):
						res.text_plain.extend([str(x) for x in tp])
					else:
						res.text_plain.append(str(tp))
				res.outputs.append({"type": msg_type, **content})

			elif msg_type == "error":
				# content: {ename, evalue, traceback: [...]}
				tb_lines = content.get("traceback", []) or []
				tb_text = "\n".join(tb_lines)
				tb_text_clean = strip_ansi(tb_text)

				res.error = {
					"ename": content.get("ename"),
					"evalue": content.get("evalue"),
					# A "normal-looking" multiline traceback string (like notebook output)
					"traceback": tb_text_clean,
					# Keep lines too (optional but sometimes handy)
					"traceback_lines": [strip_ansi(line) for line in tb_lines],
				}
				res.outputs.append({"type": "error", **content})

			else:
				# Other message types: clear_output, update_display_data, etc.
				res.outputs.append({"type": msg_type, **content})

		return res

	def reset(self):
		"""
		Clear the kernel's outputs and error information while retaining the kernel instance.
		"""
		self._drain_iopub()  # Clear the IOPub messages
		# Reset the ExecResult instance
		self.exec_result = ExecResult()

		# Optionally, run %reset to clear variables in the kernel's memory
		self._run_silent("%reset -f", timeout=10)


def default_format_execution_return(exec_ret: ExecResult) -> str:
	"""Default formatter for ExecResult: shows error or last text/plain output."""

	execution_result = []
	if not exec_ret.stdout == '':
		execution_result.append(exec_ret.stdout)
	if len(exec_ret.text_plain) > 0:
		execution_result.append('\n'.join(exec_ret.text_plain))
	if exec_ret.error is not None:
		execution_result.append(exec_ret.error['traceback'])
	execution_result = '\n'.join(execution_result).strip()
	
	return execution_result


class JupyterExecutorManager:
	"""Manager that creates and coordinates multiple JupyterKernelExecutor instances.

	Responsibilities:
	- Optionally create a map of `JupyterKernelExecutor` instances.
	- Extract code between configured interpreter tags.
	- Execute code blocks in parallel and format results using
	  `default_format_execution_return`.
	"""

	def __init__(
		self,
		ids: Optional[list] = [],
		max_workers: int = 4,
		default_timeout: Optional[float] = 10.0,
	):
		self.max_workers = max_workers
		self.default_timeout = default_timeout
		print(f"Initialized JupyterExecutorManager with {max_workers} parallel workers, default timeout {default_timeout}s.")
		
		# Initialize executors only from provided ids.
		self.executor_map = {}
		if ids:
			self.add_executors(ids)


	def add_executor(self, id):
		"""Add a single JupyterKernelExecutor for `id` if it doesn't already exist.

		Returns True if created, False if already existed.
		"""
		if id in self.executor_map:
			return False
		self.executor_map[id] = JupyterKernelExecutor(nocolor=True)
	
	def add_executors(self, ids: list):
		"""Add JupyterKernelExecutor instances for the given ids (ignore existing keys) in parallel."""
		if not ids:
			return

		missing_ids = [id for id in ids if id not in self.executor_map]
		if not missing_ids:
			return

		workers = min(len(missing_ids), self.max_workers)
		with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
			future_to_id = {
				pool.submit(JupyterKernelExecutor, nocolor=True): id for id in missing_ids
			}
			for fut in concurrent.futures.as_completed(future_to_id):
				executor_id = future_to_id[fut]
				executor = fut.result()
				self.executor_map[executor_id] = executor

	def shutdown(self, now: bool = True):
		"""Shutdown all managed JupyterKernelExecutor instances.

		Calls each executor's `shutdown()` and clears the internal map.
		"""
		for idx, executor in list(self.executor_map.items()):
			try:
				executor.shutdown(now=now)
			except Exception:
				# best-effort shutdown; ignore individual failures
				pass
		self.executor_map.clear()

	def shutdown_executor(self, id, now: bool = True):
		"""Shutdown and remove a single executor by id."""
		if id in self.executor_map:
			try:
				self.executor_map[id].shutdown(now=now)
			except Exception:
				pass
			del self.executor_map[id]

	def shutdown_executors(self, ids: list, now: bool = True):
		"""Shutdown and remove a group of executors by id list in parallel."""
		if not ids:
			return

		for id in ids:
			if id not in self.executor_map:
				raise KeyError(f"Unknown executor id: {id}")

		workers = min(len(ids), self.max_workers)

		def _safe_shutdown(executor):
			try:
				executor.shutdown(now=now)
			except Exception:
				# best-effort shutdown; ignore individual failures
				pass

		with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
			futures = [pool.submit(_safe_shutdown, self.executor_map[id]) for id in ids]
			for fut in futures:
				# ensure completion; exceptions already swallowed in _safe_shutdown
				fut.result()

		for id in ids:
			if id in self.executor_map:
				del self.executor_map[id]

	def reset_executor(self, id):
		"""Reset a single executor by id (clears kernel state and outputs)."""
		if id not in self.executor_map:
			raise KeyError(f"Unknown executor id: {id}")
		try:
			self.executor_map[id].reset()
		except Exception:
			# best-effort reset; ignore individual failures
			pass

	def reset_executors(self, ids: list):
		"""Reset a group of executors by id list in parallel."""
		if not ids:
			return

		for id in ids:
			if id not in self.executor_map:
				raise KeyError(f"Unknown executor id: {id}")

		workers = min(len(ids), self.max_workers)

		def _safe_reset(executor):
			try:
				executor.reset()
			except Exception:
				# best-effort reset; ignore individual failures
				pass

		with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
			futures = [pool.submit(_safe_reset, self.executor_map[id]) for id in ids]
			for fut in futures:
				# ensure completion; exceptions already swallowed in _safe_reset
				fut.result()

	def execute_request(self, id, code: str, timeout: float = None) -> str:
		"""Execute code on a single executor identified by `id` and return formatted result."""
		if id not in self.executor_map:
			raise KeyError(f"Unknown executor id: {id}")
		executor = self.executor_map[id]
		if timeout is None:
			use_timeout = self.default_timeout
		else:
			use_timeout = timeout
		exec_ret = executor.execute(code, timeout=use_timeout)
		return default_format_execution_return(exec_ret)

	def execute_requests(self, exec_requests: list):
		"""Execute a list of requests in parallel.

		Each request should be a dict: {'id': <id>, 'code': '<code>', 'timeout': <optional float>}.
		Returns a list of dict: {'id': <id>, 'code': '<code>', 'execution_return': str} in the same
		order as the requests were provided.
		"""
		results = []
		if not exec_requests:
			return results

		workers = min(len(exec_requests), self.max_workers)
		results = [None] * len(exec_requests)
		with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
			future_to_meta = {}
			for idx, req in enumerate(exec_requests):
				id = req.get('id')
				code = req.get('code')
				if id is None or code is None:
					raise ValueError("Each request must contain 'id' and 'code'")
				if id not in self.executor_map:
					raise KeyError(f"Unknown executor id: {id}")
				timeout = req.get('timeout', self.default_timeout)
				executor = self.executor_map[id]
				future = pool.submit(executor.execute, code, timeout)
				future_to_meta[future] = (idx, id, code)

			for fut in concurrent.futures.as_completed(future_to_meta):
				idx, id, code = future_to_meta[fut]
				try:
					exec_ret = fut.result()
					execution_result = default_format_execution_return(exec_ret)
				except Exception as e:
					execution_result = f"Execution error: {e}"
				results[idx] = {
					"id": id,
					"code": code,
					"execution_return": execution_result,
				}

		return results

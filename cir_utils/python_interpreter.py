import re
import time
import random
import warnings
import multiprocessing
import io
import dill
import ast
import traceback
import contextlib
import sys
import socket
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

	# All raw  output messages we observed on IOPub (useful for rich rendering)
	outputs: List[Dict[str, Any]] = field(default_factory=list)

	# Convenience: extracted "text/plain" payloads (e.g., last expression result)
	text_plain: List[str] = field(default_factory=list)

	# Error info (None if no error)
	# Recommended fields:
	#   ename, evalue, traceback (multi-line string)
	error: Optional[Dict[str, Any]] = None


class BasePythonExecutor:
	"""Base class for Python executors."""
	def __init__(self, kernel_name: str = "python3", **kwargs):
		pass

	def execute(self, code: str, timeout: float = 30.0) -> ExecResult:
		raise NotImplementedError

	def shutdown(self, now: bool = True):
		raise NotImplementedError

	def interrupt(self):
		raise NotImplementedError

	def reset(self):
		raise NotImplementedError


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


def _run_code_with_timeout(code, globals_dict):
	"""
	Helper function to run code in a separate process with timeout support.
	"""
	res = ExecResult()
	stdout_capture = io.StringIO()
	stderr_capture = io.StringIO()
	
	with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
		try:
			# Parse AST to handle last expression value
			tree = ast.parse(code)
			if not tree.body:
				pass
			else:
				last_node = tree.body[-1]
				to_exec = None
				to_eval = None
				
				if isinstance(last_node, ast.Expr):
					# Last statement is an expression
					if len(tree.body) > 1:
						exec_node = ast.Module(body=tree.body[:-1], type_ignores=[])
						to_exec = compile(exec_node, filename="<string>", mode="exec")
					
					to_eval = compile(ast.Expression(body=last_node.value), filename="<string>", mode="eval")
				else:
					to_exec = compile(tree, filename="<string>", mode="exec")
					
				if to_exec:
					exec(to_exec, globals_dict)
				
				if to_eval:
					val = eval(to_eval, globals_dict)
					if val is not None:
						res.text_plain.append(str(val))
						# Add to outputs for compatibility
						res.outputs.append({
							"type": "execute_result",
							"data": {"text/plain": str(val)},
							"metadata": {}
						})
						
		except Exception:
			# Capture traceback
			exc_type, exc_value, tb = sys.exc_info()
			tb_str = "".join(traceback.format_exception(exc_type, exc_value, tb))
			# Remove ANSI colors if any
			tb_clean = strip_ansi(tb_str)
			
			res.error = {
				"ename": exc_type.__name__,
				"evalue": str(exc_value),
				"traceback": tb_clean
			}
			res.outputs.append({
				"type": "error",
				"ename": exc_type.__name__,
				"evalue": str(exc_value),
				"traceback": tb_clean
			})
	
	res.stdout = stdout_capture.getvalue()
	res.stderr = stderr_capture.getvalue()
	
	return res, globals_dict


def _process_wrapper(code, globals_bytes, return_dict):
	"""Wrapper to run code and store result in return_dict."""
	try:
		globals_dict = dill.loads(globals_bytes)
		result, new_globals = _run_code_with_timeout(code, globals_dict)
		return_dict['result'] = result
		return_dict['new_globals'] = dill.dumps(new_globals)
	except Exception as e:
		return_dict['error'] = str(e)


class LocalPythonExecutor(BasePythonExecutor):
	"""
	A local python executor using exec to run code in the current process.
	Maintains a global namespace dictionary for state persistence.
	"""
	def __init__(self, kernel_name: str = "python3", **kwargs):
		self.globals = {}
		self.unique_id = kernel_name

	def execute(self, code: str, timeout: float = 30.0) -> ExecResult:
		# Use multiprocessing.Manager to handle return values
		with multiprocessing.Manager() as manager:
			return_dict = manager.dict()
			globals_bytes = dill.dumps(self.globals)
			
			p = multiprocessing.Process(target=_process_wrapper, args=(code, globals_bytes, return_dict))
			p.start()
			p.join(timeout)
			
			if p.is_alive():
				p.kill()
				p.join()
				error_msg = f"Execution timed out after {timeout} seconds"
				# Return an ExecResult with the same error structure as runtime errors
				tb = error_msg
				res = ExecResult(
					stderr=error_msg,
					outputs=[{
						"type": "error",
						"ename": "TimeoutError",
						"evalue": error_msg,
						"traceback": tb
					}],
					error={
						"ename": "TimeoutError",
						"evalue": error_msg,
						"traceback": tb
					}
				)
				return res
			
			if 'result' in return_dict:
				self.globals = dill.loads(return_dict['new_globals'])
				return return_dict['result']
			elif 'error' in return_dict:
				raise RuntimeError(f"Execution failed: {return_dict['error']}")
			else:
				raise RuntimeError("Execution failed to return a result")

	def shutdown(self, now: bool = True):
		pass
			
	def interrupt(self):
		pass
			
	def reset(self):
		self.globals = {}


class ExecutorManager:
	"""Manager that creates and coordinates multiple KernelExecutor instances.

	Responsibilities:
	- Optionally create a map of `KernelExecutor` instances.
	- Extract code between configured interpreter tags.
	- Execute code blocks in parallel and format results using
	  `default_format_execution_return`.
	"""

	def __init__(
		self,
		ids: Optional[list] = [],
		max_workers: int = 4,
		default_timeout: Optional[float] = 10.0,
		executor_cls: type = LocalPythonExecutor,
	):
		self.max_workers = max_workers
		self.default_timeout = default_timeout
		self.executor_cls = executor_cls
		print(f"Initialized ExecutorManager with {max_workers} parallel workers, default timeout {default_timeout}s, executor class {executor_cls.__name__}.")
		
		# Initialize executors only from provided ids.
		self.executor_map = {}
		if ids:
			self.add_executors(ids)

	def add_executor(self, id):
		"""Add a single executor for `id` if it doesn't already exist.

		Returns True if created, False if already existed.
		"""
		if id in self.executor_map:
			return False
		self.executor_map[id] = self.executor_cls(kernel_name=id)
	
	def add_executors(self, ids: list):
		"""Add executor instances for the given ids (ignore existing keys) in parallel."""
		if not ids:
			return

		missing_ids = [id for id in ids if id not in self.executor_map]
		if not missing_ids:
			return

		workers = min(len(missing_ids), self.max_workers)
		with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
			future_to_id = {
				pool.submit(self.executor_cls, kernel_name=id): id for id in missing_ids
			}
			for fut in concurrent.futures.as_completed(future_to_id):
				executor_id = future_to_id[fut]
				try:
					executor = fut.result()
					self.executor_map[executor_id] = executor
				except Exception as e:
					print(f"Error creating executor {executor_id}: {e}")

	def shutdown(self, now: bool = True):
		"""Shutdown all managed KernelExecutor instances.

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
					error_status = exec_ret.error is not None
				except Exception as e:
					execution_result = f"Execution error: {e}"
					error_status = True
				results[idx] = {
					"id": id,
					"code": code,
					"execution_return": execution_result,
					"execution_error": error_status,
				}

		return results

if __name__ == "__main__":

	executor = LocalPythonExecutor()

	code = "def add(a,b):\n    return a + b\nadd(1)"
	r = executor.execute(code)
	print(r)

	code = "add(1,2)"
	r = executor.execute(code)
	print(r)
	
	code = "import time\ntime.sleep(2)\nadd(3,4)"
	r = executor.execute(code, 1)
	print(r)

	del executor


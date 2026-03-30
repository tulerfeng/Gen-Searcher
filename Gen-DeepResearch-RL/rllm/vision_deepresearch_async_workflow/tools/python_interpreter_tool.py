import asyncio
import json

from vision_deepresearch_async_workflow.tools.shared import (
    DeepResearchTool,
    log_tool_event,
    shorten_for_log,
)


class PythonInterpreterTool(DeepResearchTool):
    """Safe Python code execution (from existing implementation)."""

    def __init__(self):
        super().__init__(
            name="PythonInterpreter",
            description='Execute Python code in a sandboxed environment. Use this to run Python code and get the execution results.\n**Make sure to use print() for any output you want to see in the results.**\nFor code parameters, use placeholders first, and then put the code within <code></code> XML tags, such as:\n<tool_call>\n{"purpose": <detailed-purpose-of-this-tool-call>, "name": <tool-name>, "arguments": {"code": ""}}\n<code>\nHere is the code.\n</code>\n</tool_call>\n',
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute. Must be provided within <code></code> XML tags. Remember to use print() statements for any output you want to see.",
                    }
                },
                "required": ["code"],
            },
        )
        self.timeout = 30

    async def call(self, code: str, timeout: int = None, **kwargs) -> str:
        """Execute Python code safely with timeout."""
        timeout = timeout or self.timeout

        code_len = len(code or "")

        def log_result(
            status: str,
            message: str,
            extra: str | None = None,
            *,
            level: str | None = None,
        ) -> None:
            preview = shorten_for_log(message)
            details = f"code_len={code_len} result_len={len(message)} preview={json.dumps(preview, ensure_ascii=False)}"
            if extra:
                details += f" {extra}"
            log_tool_event(
                source="PythonInterpreter",
                status=status,
                message=details,
                level=level or "INFO",
            )

        # Security checks - check for dangerous imports/operations
        dangerous_patterns = [
            "import os",
            "import subprocess",
            "import sys",
            "from os import",
            "from subprocess import",
            "from sys import",
            "exec(",
            "eval(",
            "compile(",
            "open(",
            "file(",
        ]

        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                result = f"[Security Error] '{pattern}' not allowed for safety reasons"
                log_result(
                    "SecurityBlocked",
                    result,
                    extra=f"pattern={json.dumps(pattern, ensure_ascii=False)}",
                    level="WARNING",
                )
                return result

        import io
        import sys

        # Setup safe environment
        allowed_modules = {
            "math": __import__("math"),
            "datetime": __import__("datetime"),
            "json": __import__("json"),
            "random": __import__("random"),
            "re": __import__("re"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
            "statistics": __import__("statistics"),
        }

        # Add numpy/pandas if available
        try:
            import numpy as np

            allowed_modules["numpy"] = np
            allowed_modules["np"] = np
        except ImportError:
            pass

        try:
            import pandas as pd

            allowed_modules["pandas"] = pd
            allowed_modules["pd"] = pd
        except ImportError:
            pass

        # Restricted builtins with safe import capability
        def safe_import(name, *args, **kwargs):
            """Allow importing only safe modules."""
            safe_modules = [
                "math",
                "datetime",
                "json",
                "random",
                "re",
                "collections",
                "itertools",
                "statistics",
                "numpy",
                "pandas",
                "scipy",
                "scipy.linalg",  # Add scipy submodules
                "scipy.optimize",
                "scipy.signal",
                "scipy.special",
                "matplotlib",
                "matplotlib.pyplot",
                "urllib.request",
                "requests",
                "sys",
            ]
            # Check if the module or its parent is allowed
            if name in safe_modules or any(
                name.startswith(m + ".") for m in safe_modules
            ):
                return __import__(name, *args, **kwargs)
            else:
                raise ImportError(f"Module '{name}' is not allowed for safety reasons")

        restricted_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bin": bin,
            "bool": bool,
            "chr": chr,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "hex": hex,
            "int": int,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "oct": oct,
            "ord": ord,
            "pow": pow,
            "print": print,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "zip": zip,
            "__import__": safe_import,  # Allow safe imports
            # Add exception classes for proper error handling
            "Exception": Exception,
            "ImportError": ImportError,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "AttributeError": AttributeError,
        }

        global_vars = {"__builtins__": restricted_builtins}
        global_vars.update(allowed_modules)
        local_vars = {}

        # Capture output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        def execute_with_timeout():
            try:
                sys.stdout = stdout_buffer
                sys.stderr = stderr_buffer
                exec(code, global_vars, local_vars)
                return True
            except Exception as e:
                stderr_buffer.write(f"Execution error: {e}")
                return False
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        # Execute with timeout
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self.executor, execute_with_timeout)
        try:
            await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            result = f"[Timeout] Execution exceeded {timeout}s"
            log_result("Timeout", result, level="WARNING")
            return result
        except Exception as exc:  # noqa: BLE001
            result = f"[Error] Unexpected execution error: {exc}"
            log_result("UnexpectedError", result, level="ERROR")
            return result

        stdout_content = stdout_buffer.getvalue()
        stderr_content = stderr_buffer.getvalue()

        if stderr_content:
            result = f"[Error]\n{stderr_content}"
            log_result("Error", result, level="ERROR")
            return result
        elif stdout_content:
            cleaned_output = stdout_content.rstrip()
            result = f"[Output]\n{cleaned_output}"
            return result
        else:
            meaningful_vars = {
                k: v
                for k, v in local_vars.items()
                if not k.startswith("_") and k not in allowed_modules
            }
            if meaningful_vars:
                result = f"[Variables]\n{meaningful_vars}"
                return result
            else:
                result = "[Success] Code executed (no output)"
                return result

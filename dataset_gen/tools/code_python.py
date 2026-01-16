import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class PythonSandboxLimits:
    timeout_s: float = 6.0
    max_code_chars: int = 6000
    max_stdout_chars: int = 4000
    max_stderr_chars: int = 4000
    max_result_chars: int = 1000
    max_memory_mb: Optional[int] = 256
    disable_file_io: bool = True
    traceback_limit: int = 8


DEFAULT_ALLOWED_IMPORTS = ("math", "decimal", "fractions", "statistics", "re")


_RUNNER_SCRIPT = r"""
import ast
import builtins
import contextlib
import io
import json
import platform
import sys
import time
import traceback

try:
    payload = json.load(sys.stdin)
except Exception as exc:
    sys.stdout.write(json.dumps({"exec_status": "failed", "error": {"type": "bad_payload", "message": str(exc)}}))
    raise SystemExit(0)

code = str(payload.get("code") or "")
inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
allowed_imports = payload.get("allowed_imports") if isinstance(payload.get("allowed_imports"), list) else []
allowed_imports = [str(item) for item in allowed_imports if str(item).strip()]
allowed_set = set(allowed_imports)
max_stdout = int(payload.get("max_stdout_chars") or 0)
max_stderr = int(payload.get("max_stderr_chars") or 0)
max_result = int(payload.get("max_result_chars") or 0)
max_memory_mb = payload.get("max_memory_mb")
disable_file_io = bool(payload.get("disable_file_io", True))
traceback_limit = payload.get("traceback_limit")

try:
    traceback_limit = int(traceback_limit) if traceback_limit is not None else 8
except Exception:
    traceback_limit = 8

def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if limit <= 0:
        return text, False
    if len(text) <= limit:
        return text, False
    return text[: max(0, limit - 1)] + "…", True

class _BoundedWriter(io.TextIOBase):
    def __init__(self, limit: int):
        self._limit = max(0, int(limit))
        self._buf: list[str] = []
        self._count = 0
        self.truncated = False

    def write(self, s: str) -> int:
        text = str(s)
        if self._limit <= 0:
            self._buf.append(text)
            self._count += len(text)
            return len(text)
        remaining = self._limit - self._count
        if remaining <= 0:
            self.truncated = True
            return len(text)
        if len(text) <= remaining:
            self._buf.append(text)
            self._count += len(text)
            return len(text)
        self._buf.append(text[:remaining])
        self._count += remaining
        self.truncated = True
        return len(text)

    def getvalue(self) -> str:
        return "".join(self._buf)

def _apply_resource_limits(max_memory_mb):
    if max_memory_mb is None:
        return
    try:
        mb = int(max_memory_mb)
    except Exception:
        return
    if mb <= 0:
        return
    try:
        import resource

        limit_bytes = mb * 1024 * 1024
        for key in ("RLIMIT_AS", "RLIMIT_DATA"):
            if hasattr(resource, key):
                res_key = getattr(resource, key)
                try:
                    resource.setrlimit(res_key, (limit_bytes, limit_bytes))
                except Exception:
                    continue
    except Exception:
        return

def _validate_imports(code: str, allowed: set[str]) -> list[str]:
    missing: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return missing
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = str(alias.name).split(".", 1)[0]
                if top and top not in allowed:
                    missing.append(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = str(node.module).split(".", 1)[0]
                if top and top not in allowed:
                    missing.append(top)
    return sorted(set(missing))

def _install_import_guard(allowed: set[str]):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        try:
            target = str(name)
        except Exception:
            target = name
        top = str(target).split(".", 1)[0]
        if top and top not in allowed:
            raise ImportError(f"Import blocked: '{top}' is not in allowed_imports")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import

def _disable_file_io():
    def blocked_open(*args, **kwargs):
        raise PermissionError("file I/O is disabled in code_python sandbox")

    builtins.open = blocked_open

def _json_safe(obj):
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)

started = time.perf_counter()
stdout = _BoundedWriter(max_stdout)
stderr = _BoundedWriter(max_stderr)
status = "ok"
error = None
result_obj = None

blocked_imports = _validate_imports(code, allowed_set)
if blocked_imports:
    status = "failed"
    error = {"type": "import_blocked", "message": f"Blocked imports: {blocked_imports}", "blocked": blocked_imports}

if status == "ok":
    _apply_resource_limits(max_memory_mb)
    _install_import_guard(allowed_set)
    if disable_file_io:
        _disable_file_io()

    globals_env = {"__name__": "__main__", "INPUTS": dict(inputs)}
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            compiled = compile(code, "<code_python>", "exec")
            exec(compiled, globals_env, globals_env)
        result_obj = globals_env.get("result", None)
    except Exception as exc:
        status = "failed"
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=traceback_limit)}

elapsed_ms = int((time.perf_counter() - started) * 1000)

stdout_text, stdout_trunc = _truncate(stdout.getvalue(), max_stdout)
stderr_text, stderr_trunc = _truncate(stderr.getvalue(), max_stderr)

result_safe = _json_safe(result_obj)
result_text = "" if result_safe is None else (result_safe if isinstance(result_safe, str) else json.dumps(result_safe, ensure_ascii=False, default=str))
result_text, result_trunc = _truncate(result_text, max_result)

sys.stdout.write(
    json.dumps(
        {
            "exec_status": status,
            "elapsed_ms": elapsed_ms,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "stdout_truncated": stdout_trunc or getattr(stdout, "truncated", False),
            "stderr_truncated": stderr_trunc or getattr(stderr, "truncated", False),
            "result": result_safe,
            "result_text": result_text,
            "result_truncated": result_trunc,
            "error": error,
            # Avoid platform.platform(): it may invoke subprocess and trigger extra imports under guard.
            "env": {"python_version": sys.version.split()[0], "sys_platform": sys.platform, "allowed_imports": allowed_imports},
        },
        ensure_ascii=False,
        default=str,
    )
)
""".strip()


def run_code_python(
    *,
    code: str,
    inputs: Optional[Dict[str, Any]] = None,
    allowed_imports: Iterable[str] = DEFAULT_ALLOWED_IMPORTS,
    limits: PythonSandboxLimits = PythonSandboxLimits(),
) -> Dict[str, Any]:
    code = str(code or "")
    if not code.strip():
        return {"exec_status": "failed", "error": {"type": "empty_code", "message": "code is empty"}}
    if len(code) > int(limits.max_code_chars):
        return {"exec_status": "failed", "error": {"type": "code_too_long", "message": "code too long"}}

    payload = {
        "code": code,
        "inputs": inputs or {},
        "allowed_imports": [str(x) for x in allowed_imports if str(x).strip()],
        "max_stdout_chars": int(limits.max_stdout_chars),
        "max_stderr_chars": int(limits.max_stderr_chars),
        "max_result_chars": int(limits.max_result_chars),
        "max_memory_mb": limits.max_memory_mb,
        "disable_file_io": bool(limits.disable_file_io),
        "traceback_limit": int(limits.traceback_limit),
    }
    timeout = max(1.0, float(limits.timeout_s))
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _RUNNER_SCRIPT],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "exec_status": "failed",
            "error": {"type": "timeout", "message": f"Execution exceeded timeout_s={timeout}"},
            "stdout": "",
            "stderr": "",
            "result": None,
            "result_text": "",
            "env": {"python_version": platform.python_version(), "platform": platform.platform()},
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "exec_status": "failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "stdout": "",
            "stderr": "",
            "result": None,
            "result_text": "",
            "env": {"python_version": platform.python_version(), "platform": platform.platform()},
        }

    raw = (completed.stdout or "").strip()
    if not raw:
        return {
            "exec_status": "failed",
            "error": {
                "type": "empty_runner_output",
                "message": "Runner returned empty stdout",
                "stderr_preview": (completed.stderr or "")[:1200],
                "returncode": completed.returncode,
            },
        }
    try:
        obj = json.loads(raw) if raw else {}
        if isinstance(obj, dict):
            return obj
        return {
            "exec_status": "failed",
            "error": {"type": "bad_runner_output", "message": "Runner output is not a JSON object"},
        }
    except Exception:
        return {
            "exec_status": "failed",
            "error": {
                "type": "bad_runner_output",
                "message": "Runner did not return JSON",
                "stdout_preview": raw[:1200],
                "stderr_preview": (completed.stderr or "")[:1200],
            },
        }

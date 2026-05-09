from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


class PythonToolError(ValueError):
    """Raised when code is rejected before execution."""


@dataclass(slots=True)
class PythonToolResult:
    code: str
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def output(self) -> str:
        return self.stdout.strip() if self.ok else self.stderr.strip()


class PythonTool:
    """Small Python calculation tool inspired by the AIMO notebook.

    This is a calculation sandbox, not a strong security boundary. It is meant
    for generated arithmetic/symbolic snippets, with a timeout and AST checks
    that reject common filesystem, process, network, and introspection access.
    """

    allowed_import_roots = {
        "collections",
        "decimal",
        "fractions",
        "functools",
        "itertools",
        "math",
        "mpmath",
        "numpy",
        "statistics",
        "sympy",
    }
    banned_names = {
        "__builtins__",
        "__import__",
        "breakpoint",
        "compile",
        "dir",
        "eval",
        "exec",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
    }
    banned_import_roots = {
        "builtins",
        "ctypes",
        "glob",
        "importlib",
        "io",
        "os",
        "pathlib",
        "pickle",
        "shlex",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "tempfile",
    }
    banned_nodes = (
        ast.AsyncFor,
        ast.AsyncFunctionDef,
        ast.AsyncWith,
        ast.Await,
        ast.ClassDef,
        ast.Delete,
        ast.Global,
        ast.Nonlocal,
        ast.With,
    )

    def __init__(self, timeout: float = 3.0) -> None:
        self.timeout = timeout

    def validate(self, code: str) -> ast.Module:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise PythonToolError(str(exc)) from exc

        for node in ast.walk(tree):
            if isinstance(node, self.banned_nodes):
                raise PythonToolError(f"Unsupported statement: {type(node).__name__}")

            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in self.banned_import_roots or root not in self.allowed_import_roots:
                        raise PythonToolError(f"Import is not allowed: {alias.name}")

            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".")[0]
                if root in self.banned_import_roots or root not in self.allowed_import_roots:
                    raise PythonToolError(f"Import is not allowed: {module}")

            if isinstance(node, ast.Name) and node.id in self.banned_names:
                raise PythonToolError(f"Name is not allowed: {node.id}")

            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise PythonToolError(f"Private attribute access is not allowed: {node.attr}")

            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in self.banned_names:
                    raise PythonToolError(f"Call is not allowed: {node.func.id}")

        return tree

    def ensure_print(self, tree: ast.Module) -> str:
        if not tree.body:
            return ""
        last = tree.body[-1]
        if isinstance(last, ast.Expr):
            tree.body[-1] = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="print", ctx=ast.Load()),
                    args=[last.value],
                    keywords=[],
                )
            )
            ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    def execute(self, code: str) -> PythonToolResult:
        tree = self.validate(code)
        final_code = self.ensure_print(tree)
        with tempfile.TemporaryDirectory(prefix="exact_python_tool_") as tmp:
            script_path = Path(tmp) / "tool_code.py"
            script_path.write_text(final_code + "\n", encoding="utf-8")
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", str(script_path)],
                    cwd=tmp,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return PythonToolResult(
                    code=final_code,
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or f"Execution timed out after {self.timeout} seconds",
                    returncode=124,
                    timed_out=True,
                )
        return PythonToolResult(
            code=final_code,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )


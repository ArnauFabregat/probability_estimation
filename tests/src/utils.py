# type: ignore

import ast
import textwrap

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def get_source_segment(file_path, node):
    """Return source code for an AST node by reading the file text and slicing."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            src = f.read()
    except OSError:
        return ""

    # Preferred: use ast.get_source_segment if available and positions are present
    try:
        seg = ast.get_source_segment(src, node)
        if seg is not None:
            return seg
    except Exception:
        pass

    # Fallback: use lineno/end_lineno (available in modern Python)
    if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
        lines = src.splitlines()
        # Note: AST line numbers are 1-based
        snippet = "\n".join(lines[node.lineno - 1: node.end_lineno])
        # Optional: dedent so nested defs look nice
        return textwrap.dedent(snippet)

    return ""


def canonical_id(*parts):
    return "::".join(parts)


def normalize_signature(node):
    """Return a Python-like signature including annotations and return type."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""

    def unparse(x):
        try:
            return ast.unparse(x)
        except Exception:
            return ""

    parts = []

    # Pos-only args (Python 3.8+)
    for a in getattr(node.args, "posonlyargs", []):
        s = a.arg
        if a.annotation:
            s += f": {unparse(a.annotation)}"
        parts.append(s)
    if getattr(node.args, "posonlyargs", []):
        parts.append("/")

    # Regular args
    for a in node.args.args:
        s = a.arg
        if a.annotation:
            s += f": {unparse(a.annotation)}"
        parts.append(s)

    # Vararg
    if node.args.vararg:
        s = "*" + node.args.vararg.arg
        if node.args.vararg.annotation:
            s += f": {unparse(node.args.vararg.annotation)}"
        parts.append(s)
    elif node.args.kwonlyargs:
        # bare * to mark start of kw-only if no vararg
        parts.append("*")

    # kw-only args
    for a in node.args.kwonlyargs:
        s = a.arg
        if a.annotation:
            s += f": {unparse(a.annotation)}"
        parts.append(s)

    # kwargs
    if node.args.kwarg:
        s = "**" + node.args.kwarg.arg
        if node.args.kwarg.annotation:
            s += f": {unparse(node.args.kwarg.annotation)}"
        parts.append(s)

    ret = ""
    if node.returns:
        ret = f" -> {unparse(node.returns)}"

    return f"def {node.name}({', '.join(parts)}){ret}"

# type: ignore

import ast
import textwrap


def get_node_context(g, node_id):
    context: str = ""

    # ---------------------------------------
    # 1. Node Attributes
    # ---------------------------------------
    node_data = g.nodes[node_id]

    context += "### NODE INFO ###\n"
    context += f"file::type::name -> {node_id}\n\n"

    src = node_data.get("source") or ""
    context += "source:\n" + src + "\n\n"

    # ---------------------------------------
    # 2. Outgoing edges
    # ---------------------------------------
    out_edges = []
    for _, dst, data in g.out_edges(node_id, data=True):
        rel = data.get("rel")
        out_edges.append(f"{node_id} -[{rel}]-> {dst}")

    if out_edges:
        context += "### OUTGOING EDGES ###\n"
        for line in out_edges:
            context += line + "\n"
        context += "\n"

    # ---------------------------------------
    # 3. Incoming edges
    # ---------------------------------------
    in_edges = []
    for src_id, _, data in g.in_edges(node_id, data=True):
        rel = data.get("rel")
        in_edges.append(f"{src_id} -[{rel}]-> {node_id}")

    if in_edges:
        context += "### INCOMING EDGES ###\n"
        for line in in_edges:
            context += line + "\n"
        context += "\n"

    # ---------------------------------------
    # 4. Neighbor context
    # ---------------------------------------
    neighbor_ids = set()

    # nodes referenced by outgoing edges
    for rel, dst in [(rel, dst) for rel, dst in
                     [(data.get("rel"), dst) for _, dst, data in g.out_edges(node_id, data=True)]]:
        neighbor_ids.add(dst)

    # nodes referencing this node
    for src_id, data_rel in [(src, data.get("rel")) for src, _, data in g.in_edges(node_id, data=True)]:
        neighbor_ids.add(src_id)

    neighbor_blocks = []
    for nid in neighbor_ids:
        nd = g.nodes[nid]
        if nd.get("type") == "file":
            continue

        blk = "--- Neighbor Node ---\n"
        blk += f"file::type::name -> {nid}\n"
        blk += f"signature: {nd.get('signature')}\n"
        blk += "docstring:\n" + (nd.get("docstring") or "") + "\n"
        neighbor_blocks.append(blk)

    if neighbor_blocks:
        context += "### NEIGHBOR NODE DETAILS ###\n"
        for blk in neighbor_blocks:
            context += blk

    return context

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

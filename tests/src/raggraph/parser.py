# type: ignore
import os
import ast
from src.raggraph.utils import get_source_segment, canonical_id, normalize_signature


class CodeGraphBuilder1(ast.NodeVisitor):
    """
    First pass: build nodes for files, classes, functions/methods with metadata.
    Edges for "defines" and "has_method".
    """
    def __init__(self, file_path, graph):
        self.file_path = file_path
        self.graph = graph
        self.current_file_id = canonical_id(file_path)
        self.current_class = None
        self.current_scope = None

        # create the file node
        self.graph.add_node(
            self.current_file_id,
            type="file",
            name=os.path.basename(file_path),
            file=file_path,
            signature="None",
            docstring="None",
            source=open(file_path).read()
        )

    # ---------------------
    # Classes
    # ---------------------
    def visit_ClassDef(self, node):
        class_id = canonical_id(self.file_path, "class", node.name)

        self.graph.add_node(
            class_id,
            type="class",
            name=node.name,
            file=self.file_path,
            signature=node.name or "None",
            docstring=ast.get_docstring(node) or "None",
            source=get_source_segment(self.file_path, node)
        )

        self.graph.add_edge(self.current_file_id, class_id, rel="defines")

        prev_class = self.current_class
        self.current_class = class_id
        self.generic_visit(node)
        self.current_class = prev_class

    # ---------------------
    # Functions / Methods
    # ---------------------
    def visit_FunctionDef(self, node):
        parent_type = "method" if self.current_class else "function"
        name = f"{self.current_class.split('::')[-1]}.{node.name}" if self.current_class else node.name

        fn_id = canonical_id(
            self.file_path,
            parent_type,
            name
        )

        self.graph.add_node(
            fn_id,
            type=parent_type,
            name=name,
            file=self.file_path,
            signature=normalize_signature(node) or "None",
            docstring=ast.get_docstring(node) or "None",
            source=get_source_segment(self.file_path, node)
        )

        if self.current_class:
            self.graph.add_edge(self.current_class, fn_id, rel="has_method")
        else:
            self.graph.add_edge(self.current_file_id, fn_id, rel="defines")

        prev_scope = self.current_scope
        self.current_scope = fn_id
        self.generic_visit(node)
        self.current_scope = prev_scope


class CodeGraphBuilder2(ast.NodeVisitor):
    """
    Second pass: detect BOTH calls and references using ONLY visit_Name.
    visit_Call is used only to traverse children.
    """

    def __init__(self, file_path, graph):
        self.file_path = file_path
        self.graph = graph
        self.current_file_id = canonical_id(file_path)
        self.current_class = None
        self.current_scope = None
        self.dst_canonical_id = None  # storage for symbol lookup

    # ---------------------------------------------------------
    # Parent assignment (must be called before visiting)
    # ---------------------------------------------------------
    @staticmethod
    def attach_parents(node: ast.AST):
        for child in ast.iter_child_nodes(node):
            setattr(child, "parent", node)
            CodeGraphBuilder2.attach_parents(child)

    # ---------------------------------------------------------
    # Class definitions
    # ---------------------------------------------------------
    def visit_ClassDef(self, node: ast.ClassDef):
        class_id = canonical_id(self.file_path, "class", node.name)
        prev = self.current_class
        self.current_class = class_id
        self.generic_visit(node)
        self.current_class = prev

    # ---------------------------------------------------------
    # Functions / Methods
    # ---------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef):
        parent_type = "method" if self.current_class else "function"
        name = f"{self.current_class.split('::')[-1]}.{node.name}" if self.current_class else node.name

        fn_id = canonical_id(self.file_path, parent_type, name)

        prev = self.current_scope
        self.current_scope = fn_id
        self.generic_visit(node)
        self.current_scope = prev

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return self.visit_FunctionDef(node)

    # ---------------------------------------------------------
    # Symbol lookup (name-based)
    # ---------------------------------------------------------
    def is_local_symbol(self, name: str) -> bool:
        """
        Basic name match against existing graph nodes.
        """
        for nid, data in self.graph.nodes(data=True):
            if data.get("name") == name:
                self.dst_canonical_id = nid
                return True
        return False

    # ---------------------------------------------------------
    # Name = call or reference depending on parent
    # ---------------------------------------------------------
    def visit_Name(self, node: ast.Name):
        if not self.current_scope:
            return

        name = node.id

        # CASE 1: Name is a CALL (it sits in the callee position)
        parent = getattr(node, "parent", None)
        if isinstance(parent, ast.Call) and parent.func is node:
            if self.is_local_symbol(name):
                # Add call edge
                self.graph.add_edge(self.current_scope, self.dst_canonical_id, rel="calls")
            return

        # CASE 2: Otherwise, it's a REFERENCE
        if self.is_local_symbol(name):
            self.graph.add_edge(self.current_scope, self.dst_canonical_id, rel="references")

    # ---------------------------------------------------------
    # visit_Call – only traversal; NO call detection here
    # ---------------------------------------------------------
    def visit_Call(self, node: ast.Call):
        # Let Name logic detect call via parent=Call
        self.visit(node.func)

        # Traverse args normally (names here become references)
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            if kw.value:
                self.visit(kw.value)

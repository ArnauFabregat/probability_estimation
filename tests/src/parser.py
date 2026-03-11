# type: ignore
import os
import ast
from src.utils import get_source_segment, canonical_id, normalize_signature


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
    Second pass: add "calls" edges between functions/methods based on Call nodes.
     - This is a simplified heuristic that looks for direct name matches in the graph.
    """
    def __init__(self, file_path, graph):
        self.file_path = file_path
        self.graph = graph
        self.current_file_id = canonical_id(file_path)
        self.current_class = None
        self.current_scope = None

    # ---------------------
    # Classes
    # ---------------------
    def visit_ClassDef(self, node):
        class_id = canonical_id(self.file_path, "class", node.name)
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
        prev_scope = self.current_scope
        self.current_scope = fn_id
        self.generic_visit(node)
        self.current_scope = prev_scope

    # ---------------------
    # Calls
    # ---------------------
    def is_local_symbol(self, name: str) -> bool:
        # Search in graph for any node whose 'name' attribute matches.
        # This assumes you stored class/function names as node["name"].
        for nid, data in self.graph.nodes(data=True):
            if data.get("name") == name:
                self.dst_canonical_id = nid  # store the canonical ID for later use
                return True
        return False

    def visit_Call(self, node):
        if self.current_scope:
            target_name = None

            if isinstance(node.func, ast.Name):
                target_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                target_name = node.func.attr

            if target_name and self.is_local_symbol(target_name):
                self.graph.add_edge(self.current_scope, self.dst_canonical_id, rel="calls")

        self.generic_visit(node)

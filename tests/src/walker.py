# type: ignore
import ast
import os
import networkx as nx

from src.parser import CodeGraphBuilder1, CodeGraphBuilder2

# ------------------------------------------------------------
# Directory walker
# ------------------------------------------------------------


def iter_python_files(path, skip_init=True):
    """
    Iterate over .py files inside `path`, optionally skipping __init__.py.
    """
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".py"):
                if skip_init and f == "__init__.py":
                    continue
                yield os.path.join(root, f)


def build_graph_from_directory(path: str, skip_init: bool = True):
    graph = nx.DiGraph()
    for file_path in iter_python_files(path, skip_init=skip_init):
        tree = ast.parse(open(file_path, "r", encoding="utf-8").read())
        builder = CodeGraphBuilder1(file_path, graph)
        builder.visit(tree)

    for file_path in iter_python_files(path, skip_init=skip_init):
        tree = ast.parse(open(file_path, "r", encoding="utf-8").read())
        builder = CodeGraphBuilder2(file_path, graph)
        builder.visit(tree)

    return graph

# type: ignore
import ast
import os
from pathlib import Path
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
                yield Path(root, f).as_posix()


def build_graph_from_directory(path: str, skip_init: bool = True, save_graph: bool = False):
    graph = nx.DiGraph()
    for file_path in iter_python_files(path, skip_init=skip_init):
        tree = ast.parse(open(file_path, "r", encoding="utf-8").read())
        builder = CodeGraphBuilder1(file_path, graph)
        builder.visit(tree)

    for file_path in iter_python_files(path, skip_init=skip_init):
        tree = ast.parse(open(file_path, "r", encoding="utf-8").read())
        builder = CodeGraphBuilder2(file_path, graph)
        builder.attach_parents(tree)
        builder.visit(tree)

    if save_graph:
        # To open with Gephi in browser
        nx.write_graphml(graph, "repo_graph.graphml")

    return graph

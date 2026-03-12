# 🧩 What You Need in the Graph
To extract relevant context, the graph needs:

## 1. Nodes for:
- Files
- Classes
- Functions
- Methods

Fields:
- id: canonical ID (e.g., file::...::class::Calibrator / ...::method::Calibrator.fit)
- type: "file" | "class" | "method" | "function"
- name: symbol name ("Calibrator", "fit", etc.)
- file: file path
- signature: normalized function/method signature "def fit(self, probs, y) -> None"
- docstring: (trimmed) docstring
- source: actual source code of the node

## 2. Edges for:
- "defines" (file → symbol)
- "has_method" (class → method)
- "calls" (function/method → called symbol)
- "references" (function/method → referenced symbol)

Fields:
- src: source node id
- dst: destination node id
- rel: "defines" | "has_method" | "references" | "calls"

# 🕸️ Visualization

Upload `ast_graph.graphml` to https://lite.gephi.org/. 

# 🧠 Prompt template
```
Here is the target function you will write tests for:

<code of Calibrator.fit>

Here are the related components it depends on:
- <signature of IsotonicCalibrator.fit>
- <shape handling code>
- <error raising code>
- <signature of TemperatureScaler.fit>
- <any type constraints>

Generate high-quality unit tests...
```

# TODO
- Optionally filter `__init__.py` files before graph creation
- Maybe helps adding usage examples to docstrings
- Extend prompt with node relevant information
- Call to LLM with crewai or langchain
- Run tests, auto-fix erros if not working

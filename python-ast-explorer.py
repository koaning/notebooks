# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import ast
    return ast, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Understanding Python's AST

    An **Abstract Syntax Tree (AST)** is a tree representation of your code's structure.
    When Python runs your code, it first parses it into an AST before executing it.

    Tools like **linters**, **formatters**, **IDEs**, and **marimo** all use ASTs to understand code.

    **The question we'll answer:** How does marimo know which variables a cell *defines* vs *uses*?
    This is how it figures out cell dependencies and execution order.

    Let's explore!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Parsing Code with `ast.parse()`

    The `ast` module is built into Python. Use `ast.parse()` to turn a string of code into an AST:
    """)
    return


@app.cell
def _(mo):
    code_editor = mo.ui.code_editor(
        value="x = 1\ny = x + 2\nprint(y)",
        language="python",
        min_height=100,
    )
    code_editor
    return (code_editor,)


@app.cell
def _(ast, code_editor):
    tree = ast.parse(code_editor.value)
    print(ast.dump(tree, indent=2))
    return (tree,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Understanding the AST Structure

    The output above shows the tree structure. Key node types:

    - **`Module`** - The root node containing all statements
    - **`Assign`** - An assignment statement like `x = 1`
    - **`Name`** - A variable name (like `x`, `y`)
    - **`Constant`** - A literal value (like `1`, `2`)
    - **`BinOp`** - A binary operation (like `x + 2`)
    - **`Call`** - A function call (like `print(y)`)

    The most important insight: **`Name` nodes have a `ctx` (context) attribute:**

    - **`Store`** = the variable is being *assigned to* (written)
    - **`Load`** = the variable is being *read from* (used)

    This is exactly what marimo needs to know!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Walking the Tree with `ast.walk()`

    `ast.walk()` yields every node in the tree (in no particular order).
    We can use it to find all nodes of a specific type:
    """)
    return


@app.cell
def _(ast, tree):
    # Find all Name nodes in the tree
    name_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Name)]

    for node in name_nodes:
        ctx_type = type(node.ctx).__name__
        print(f"Name: {node.id!r:10} Context: {ctx_type}")
    return (name_nodes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Extracting Defined vs Used Variables

    Now we can build the functions that marimo uses conceptually:

    - **Defined variables**: `Name` nodes with `Store` context
    - **Used variables**: `Name` nodes with `Load` context
    """)
    return


@app.cell
def _(ast):
    def get_defined_names(code: str) -> set[str]:
        """Find all variable names that are assigned to (Store context)."""
        tree = ast.parse(code)
        return {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }

    def get_used_names(code: str) -> set[str]:
        """Find all variable names that are read from (Load context)."""
        tree = ast.parse(code)
        return {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }

    return get_defined_names, get_used_names


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Try It Out

    Edit the code below and see which variables are defined vs used:
    """)
    return


@app.cell
def _(mo):
    demo_editor = mo.ui.code_editor(
        value="result = a + b * 2\nprint(result)",
        language="python",
        min_height=80,
    )
    demo_editor
    return (demo_editor,)


@app.cell
def _(demo_editor, get_defined_names, get_used_names, mo):
    _defined = get_defined_names(demo_editor.value)
    _used = get_used_names(demo_editor.value)

    mo.md(f"""
    **Defined (Store):** `{_defined}`

    **Used (Load):** `{_used}`

    **External dependencies:** `{_used - _defined}` *(variables used but not defined in this code)*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## How Marimo Uses This

    For each cell, marimo figures out:

    1. **What variables does this cell define?** (Store context)
    2. **What variables does this cell use?** (Load context)
    3. **Which of those used variables come from other cells?** (used - defined)

    If Cell B uses a variable that Cell A defines, then Cell B *depends on* Cell A.
    That's how marimo knows to re-run Cell B when Cell A changes!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Bonus: `ast.NodeVisitor`

    For more complex analysis, you can subclass `ast.NodeVisitor`.
    It provides a cleaner way to handle different node types:
    """)
    return


@app.cell
def _(ast):
    class NameCollector(ast.NodeVisitor):
        def __init__(self):
            self.defined = set()
            self.used = set()

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store):
                self.defined.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                self.used.add(node.id)
            self.generic_visit(node)

    # Example usage
    _code = "x = y + 1"
    _collector = NameCollector()
    _collector.visit(ast.parse(_code))
    print(f"Code: {_code!r}")
    print(f"Defined: {_collector.defined}")
    print(f"Used: {_collector.used}")
    return (NameCollector,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    - `ast.parse(code)` turns code into a tree
    - `ast.walk(tree)` iterates over all nodes
    - `Name` nodes represent variable names
    - `Name.ctx` tells you if it's being read (`Load`) or written (`Store`)
    - This is the foundation for how tools like marimo analyze code dependencies
    """)
    return


if __name__ == "__main__":
    app.run()

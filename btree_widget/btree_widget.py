# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["marimo", "anywidget", "traitlets"]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import bisect
    from pathlib import Path
    import marimo as mo

    import anywidget
    import traitlets
    return Path, anywidget, bisect, mo, traitlets


@app.cell
def _(bisect):
    def make_node(keys=None, children=None):
        return {
            "keys": list(keys or []),
            "children": list(children or []),
        }

    def split_child(parent, index):
        child = parent["children"][index]
        keys = child["keys"]
        mid = len(keys) // 2
        promoted = keys[mid]
        left_keys = keys[:mid]
        right_keys = keys[mid + 1 :]

        if child["children"]:
            left_children = child["children"][: mid + 1]
            right_children = child["children"][mid + 1 :]
        else:
            left_children = []
            right_children = []

        left = make_node(left_keys, left_children)
        right = make_node(right_keys, right_children)

        parent["keys"].insert(index, promoted)
        parent["children"][index] = left
        parent["children"].insert(index + 1, right)

    def insert_nonfull(node, key, max_keys):
        if not node["children"]:
            bisect.insort(node["keys"], key)
            return

        index = bisect.bisect_left(node["keys"], key)
        insert_nonfull(node["children"][index], key, max_keys)

        if len(node["children"][index]["keys"]) > max_keys:
            split_child(node, index)

    def insert_key(root, key, max_keys):
        if root is None:
            return make_node([key], [])

        insert_nonfull(root, key, max_keys)

        if len(root["keys"]) > max_keys:
            new_root = make_node([], [root])
            split_child(new_root, 0)
            return new_root

        return root

    def build_btree(keys, branching_factor):
        if branching_factor < 3:
            raise ValueError("branching_factor must be at least 3")

        max_keys = branching_factor - 1
        root = None
        for key in keys:
            root = insert_key(root, key, max_keys)

        return root or make_node([], [])
    return (build_btree,)


@app.cell
def _(Path, anywidget, build_btree, traitlets):
    _ASSET_DIR = Path(__file__).parent
    _JS_SOURCE = (_ASSET_DIR / "btree_widget.js").read_text()
    _CSS_SOURCE = (_ASSET_DIR / "btree_widget.css").read_text()

    class BTreeWidget(anywidget.AnyWidget):
        _esm = _JS_SOURCE
        _css = _CSS_SOURCE

        btree_keys = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
        branching_factor = traitlets.Int(default_value=5).tag(sync=True)
        tree = traitlets.Dict(default_value={}).tag(sync=True)

        def __init__(self, keys=None, branching_factor=5):
            super().__init__()
            if branching_factor < 3:
                raise ValueError("branching_factor must be at least 3")

            self._keys = sorted(set(keys or []))
            self.branching_factor = branching_factor
            self.btree_keys = list(self._keys)
            self._rebuild()

        @traitlets.observe("branching_factor")
        def _on_branching_factor_change(self, _change):
            self._rebuild()

        def _rebuild(self):
            self.tree = build_btree(self._keys, self.branching_factor)

        def insert(self, key):
            if key in self._keys:
                return
            self._keys.append(key)
            self._keys.sort()
            self.btree_keys = list(self._keys)
            self._rebuild()

        def remove(self, key):
            if key not in self._keys:
                return
            self._keys.remove(key)
            self.btree_keys = list(self._keys)
            self._rebuild()
    return (BTreeWidget,)


@app.cell
def _(BTreeWidget):
    btree = BTreeWidget(
        keys=[1, 2, 5, 6, 7, 9, 12, 16, 18],
        branching_factor=5,
    )
    return (btree,)


@app.cell
def _(btree):
    btree.remove(6)
    return


@app.cell
def _(btree, mo):
    mo.ui.anywidget(btree)
    return


if __name__ == "__main__":
    app.run()

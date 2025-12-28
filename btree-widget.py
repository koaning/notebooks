"""
# B-Tree Visualization Widget

This marimo notebook provides an interactive visualization of a B-tree data structure.
You can insert, delete, search, and click nodes to drill down into the tree structure.
"""

import anywidget
import traitlets
from typing import Optional, List, Dict, Any
import pathlib

# __cell__

# B-tree Node class
class BTreeNode:
    def __init__(self, is_leaf=True):
        self.keys: List[int] = []
        self.values: List[str] = []
        self.children: List['BTreeNode'] = []
        self.is_leaf = is_leaf
        self.id = id(self)  # Unique identifier for visualization
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for JSON serialization"""
        result = {
            'id': self.id,
            'keys': self.keys,
            'values': self.values,
            'is_leaf': self.is_leaf
        }
        if not self.is_leaf:
            result['children'] = [child.to_dict() for child in self.children]
        return result

# __cell__

# B-tree implementation
class BTree:
    def __init__(self, order: int = 3):
        """
        Initialize B-tree with given order.
        Order 3 means max 2 keys per node (order - 1)
        """
        self.root = BTreeNode(is_leaf=True)
        self.order = order
        self.max_keys = order - 1
    
    def search(self, key: int) -> Optional[str]:
        """Search for a key and return its value, or None if not found"""
        return self._search(self.root, key)
    
    def _search(self, node: BTreeNode, key: int) -> Optional[str]:
        """Recursive search helper"""
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        
        if i < len(node.keys) and key == node.keys[i]:
            return node.values[i]
        
        if node.is_leaf:
            return None
        
        return self._search(node.children[i], key)
    
    def _search_path(self, node: BTreeNode, key: int, path: List[int]) -> List[int]:
        """Get path of node IDs visited during search"""
        path.append(node.id)
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        
        if i < len(node.keys) and key == node.keys[i]:
            return path
        
        if node.is_leaf:
            return path
        
        return self._search_path(node.children[i], key, path)
    
    def insert(self, key: int, value: str) -> None:
        """Insert a key-value pair into the B-tree"""
        if len(value) > 10:
            raise ValueError("Value must be at most 10 characters")
        
        root = self.root
        
        # If root is full, split it
        if len(root.keys) == self.max_keys:
            new_root = BTreeNode(is_leaf=False)
            new_root.children.append(root)
            self._split_child(new_root, 0)
            self.root = new_root
        
        self._insert_non_full(self.root, key, value)
    
    def _insert_non_full(self, node: BTreeNode, key: int, value: str) -> None:
        """Insert into a non-full node"""
        i = len(node.keys) - 1
        
        if node.is_leaf:
            # Insert into leaf
            node.keys.append(0)
            node.values.append("")
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.values[i + 1] = node.values[i]
                i -= 1
            node.keys[i + 1] = key
            node.values[i + 1] = value
        else:
            # Find child to insert into
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            
            # If child is full, split it
            if len(node.children[i].keys) == self.max_keys:
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            
            self._insert_non_full(node.children[i], key, value)
    
    def _split_child(self, parent: BTreeNode, index: int) -> None:
        """Split a full child node"""
        full_child = parent.children[index]
        new_child = BTreeNode(is_leaf=full_child.is_leaf)
        
        # Move middle key to parent
        mid = self.max_keys // 2
        parent.keys.insert(index, full_child.keys[mid])
        parent.values.insert(index, full_child.values[mid])
        
        # Move right half of keys to new child
        new_child.keys = full_child.keys[mid + 1:]
        new_child.values = full_child.values[mid + 1:]
        full_child.keys = full_child.keys[:mid]
        full_child.values = full_child.values[:mid]
        
        # Move children if not leaf
        if not full_child.is_leaf:
            new_child.children = full_child.children[mid + 1:]
            full_child.children = full_child.children[:mid + 1]
        
        parent.children.insert(index + 1, new_child)
    
    def delete(self, key: int) -> bool:
        """Delete a key from the B-tree. Returns True if deleted, False if not found"""
        if not self._search(self.root, key):
            return False
        
        self._delete(self.root, key)
        
        # If root becomes empty and has a child, make child the new root
        if len(self.root.keys) == 0 and not self.root.is_leaf:
            self.root = self.root.children[0]
        
        return True
    
    def _delete(self, node: BTreeNode, key: int) -> None:
        """Recursive delete helper"""
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        
        if i < len(node.keys) and key == node.keys[i]:
            # Key found in this node
            if node.is_leaf:
                # Simple case: remove from leaf
                node.keys.pop(i)
                node.values.pop(i)
            else:
                # Replace with predecessor or successor
                if len(node.children[i].keys) >= (self.order // 2):
                    # Predecessor
                    pred_key, pred_value = self._get_predecessor(node.children[i])
                    node.keys[i] = pred_key
                    node.values[i] = pred_value
                    self._delete(node.children[i], pred_key)
                elif len(node.children[i + 1].keys) >= (self.order // 2):
                    # Successor
                    succ_key, succ_value = self._get_successor(node.children[i + 1])
                    node.keys[i] = succ_key
                    node.values[i] = succ_value
                    self._delete(node.children[i + 1], succ_key)
                else:
                    # Merge children
                    self._merge_children(node, i)
                    self._delete(node.children[i], key)
        else:
            # Key not in this node, go to child
            if node.is_leaf:
                return  # Key not found
            
            # Ensure child has enough keys
            if len(node.children[i].keys) < (self.order // 2):
                self._ensure_minimum(node, i)
            
            # Adjust i if keys were merged
            if i > len(node.keys):
                i -= 1
            
            self._delete(node.children[i], key)
    
    def _get_predecessor(self, node: BTreeNode) -> tuple:
        """Get the rightmost key from a node"""
        while not node.is_leaf:
            node = node.children[-1]
        return node.keys[-1], node.values[-1]
    
    def _get_successor(self, node: BTreeNode) -> tuple:
        """Get the leftmost key from a node"""
        while not node.is_leaf:
            node = node.children[0]
        return node.keys[0], node.values[0]
    
    def _merge_children(self, parent: BTreeNode, index: int) -> None:
        """Merge child[index] with child[index+1]"""
        left = parent.children[index]
        right = parent.children[index + 1]
        
        # Move key from parent to left child
        left.keys.append(parent.keys[index])
        left.values.append(parent.values[index])
        
        # Move keys from right to left
        left.keys.extend(right.keys)
        left.values.extend(right.values)
        
        # Move children if not leaf
        if not left.is_leaf:
            left.children.extend(right.children)
        
        # Remove key and right child from parent
        parent.keys.pop(index)
        parent.values.pop(index)
        parent.children.pop(index + 1)
    
    def _ensure_minimum(self, parent: BTreeNode, index: int) -> None:
        """Ensure child[index] has at least minimum keys"""
        if index > 0 and len(parent.children[index - 1].keys) >= (self.order // 2):
            # Borrow from left sibling
            self._borrow_from_left(parent, index)
        elif index < len(parent.children) - 1 and len(parent.children[index + 1].keys) >= (self.order // 2):
            # Borrow from right sibling
            self._borrow_from_right(parent, index)
        else:
            # Merge with sibling
            if index > 0:
                self._merge_children(parent, index - 1)
            else:
                self._merge_children(parent, index)
    
    def _borrow_from_left(self, parent: BTreeNode, index: int) -> None:
        """Borrow a key from left sibling"""
        child = parent.children[index]
        left_sibling = parent.children[index - 1]
        
        # Move key from parent to child
        child.keys.insert(0, parent.keys[index - 1])
        child.values.insert(0, parent.values[index - 1])
        
        # Move key from left sibling to parent
        parent.keys[index - 1] = left_sibling.keys[-1]
        parent.values[index - 1] = left_sibling.values[-1]
        
        # Remove from left sibling
        left_sibling.keys.pop()
        left_sibling.values.pop()
        
        # Move child if not leaf
        if not child.is_leaf:
            child.children.insert(0, left_sibling.children.pop())
    
    def _borrow_from_right(self, parent: BTreeNode, index: int) -> None:
        """Borrow a key from right sibling"""
        child = parent.children[index]
        right_sibling = parent.children[index + 1]
        
        # Move key from parent to child
        child.keys.append(parent.keys[index])
        child.values.append(parent.values[index])
        
        # Move key from right sibling to parent
        parent.keys[index] = right_sibling.keys[0]
        parent.values[index] = right_sibling.values[0]
        
        # Remove from right sibling
        right_sibling.keys.pop(0)
        right_sibling.values.pop(0)
        
        # Move child if not leaf
        if not child.is_leaf:
            child.children.append(right_sibling.children.pop(0))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary for serialization"""
        return {
            'root': self.root.to_dict() if self.root else None
        }

# __cell__

# B-tree Widget using anywidget
# Read the JS and CSS files
_js_file = pathlib.Path("btree.js")
_css_file = pathlib.Path("btree.css")

class BTreeWidget(anywidget.AnyWidget):
    _esm = _js_file.read_text() if _js_file.exists() else """
    export function render({ model, el }) {
        el.innerHTML = '<div>Error: btree.js not found</div>';
    }
    """
    _css = _css_file.read_text() if _css_file.exists() else ""
    
    tree_data = traitlets.Dict().tag(sync=True)
    highlighted_path = traitlets.List(traitlets.Int()).tag(sync=True)
    insert_key = traitlets.Int(default_value=None, allow_none=True).tag(sync=True)
    insert_value = traitlets.Unicode(default_value="").tag(sync=True)
    delete_key = traitlets.Int(default_value=None, allow_none=True).tag(sync=True)
    search_key = traitlets.Int(default_value=None, allow_none=True).tag(sync=True)
    
    def __init__(self, order=3, **kwargs):
        super().__init__(**kwargs)
        self.btree = BTree(order=order)
        self._update_tree_data()
        
        # Observe changes from frontend
        self.observe(self._handle_insert, names='insert_key')
        self.observe(self._handle_delete, names='delete_key')
        self.observe(self._handle_search, names='search_key')
    
    def _update_tree_data(self):
        """Update the tree_data traitlet with current tree state"""
        self.tree_data = self.btree.to_dict()
    
    def _handle_insert(self, change):
        """Handle insert operation from frontend"""
        if change['new'] is not None:
            key = change['new']
            value = self.insert_value
            try:
                self.btree.insert(key, value)
                self._update_tree_data()
                # Reset the trigger
                self.insert_key = None
                self.insert_value = ""
            except Exception as e:
                print(f"Insert error: {e}")
    
    def _handle_delete(self, change):
        """Handle delete operation from frontend"""
        if change['new'] is not None:
            key = change['new']
            deleted = self.btree.delete(key)
            if deleted:
                self._update_tree_data()
            else:
                print(f"Key {key} not found")
            # Reset the trigger
            self.delete_key = None
    
    def _handle_search(self, change):
        """Handle search operation from frontend"""
        if change['new'] is not None:
            key = change['new']
            path = []
            if self.btree.root:
                self.btree._search_path(self.btree.root, key, path)
            self.highlighted_path = path
            
            result = self.btree.search(key)
            if result:
                print(f"Found key {key}: {result}")
            else:
                print(f"Key {key} not found")
            
            # Reset the trigger after a short delay to allow visualization
            import threading
            def reset_search():
                import time
                time.sleep(2)  # Show path for 2 seconds
                self.search_key = None
                self.highlighted_path = []
            threading.Thread(target=reset_search, daemon=True).start()
    
    # Python API methods
    def insert(self, key: int, value: str):
        """Insert a key-value pair"""
        self.btree.insert(key, value)
        self._update_tree_data()
    
    def delete(self, key: int) -> bool:
        """Delete a key"""
        result = self.btree.delete(key)
        self._update_tree_data()
        return result
    
    def search(self, key: int) -> Optional[str]:
        """Search for a key and highlight path"""
        path = []
        if self.btree.root:
            self.btree._search_path(self.btree.root, key, path)
        self.highlighted_path = path
        
        result = self.btree.search(key)
        
        # Clear highlight after 2 seconds
        import threading
        def clear_highlight():
            import time
            time.sleep(2)
            self.highlighted_path = []
        threading.Thread(target=clear_highlight, daemon=True).start()
        
        return result

# __cell__

# Create the widget
widget = BTreeWidget(order=3)
widget

# __cell__

# Insert some data using Python API
widget.insert(10, "ten")
widget.insert(20, "twenty")
widget.insert(5, "five")
widget.insert(15, "fifteen")
widget.insert(25, "twentyfive")
widget.insert(30, "thirty")
widget.insert(35, "thirtyfive")
widget.insert(1, "one")
widget.insert(2, "two")
widget.insert(3, "three")

# __cell__

# Search for a key (will highlight the path)
result = widget.search(15)
print(f"Search result: {result}")

# __cell__

# Delete a key
widget.delete(20)

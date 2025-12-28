// B-tree visualization widget
export function render({ model, el }) {
  // Create container
  const container = document.createElement('div');
  container.className = 'btree-container';
  
  // Create controls
  const controls = document.createElement('div');
  controls.className = 'btree-controls';
  
  const insertDiv = document.createElement('div');
  insertDiv.className = 'control-group';
  const insertKeyInput = document.createElement('input');
  insertKeyInput.type = 'number';
  insertKeyInput.placeholder = 'Key';
  insertKeyInput.className = 'key-input';
  const insertValueInput = document.createElement('input');
  insertValueInput.type = 'text';
  insertValueInput.placeholder = 'Value (max 10 chars)';
  insertValueInput.maxLength = 10;
  insertValueInput.className = 'value-input';
  const insertBtn = document.createElement('button');
  insertBtn.textContent = 'Insert';
  insertBtn.className = 'action-btn insert-btn';
  insertDiv.appendChild(insertKeyInput);
  insertDiv.appendChild(insertValueInput);
  insertDiv.appendChild(insertBtn);
  
  const deleteDiv = document.createElement('div');
  deleteDiv.className = 'control-group';
  const deleteKeyInput = document.createElement('input');
  deleteKeyInput.type = 'number';
  deleteKeyInput.placeholder = 'Key';
  deleteKeyInput.className = 'key-input';
  const deleteBtn = document.createElement('button');
  deleteBtn.textContent = 'Delete';
  deleteBtn.className = 'action-btn delete-btn';
  deleteDiv.appendChild(deleteKeyInput);
  deleteDiv.appendChild(deleteBtn);
  
  const searchDiv = document.createElement('div');
  searchDiv.className = 'control-group';
  const searchKeyInput = document.createElement('input');
  searchKeyInput.type = 'number';
  searchKeyInput.placeholder = 'Key';
  searchKeyInput.className = 'key-input';
  const searchBtn = document.createElement('button');
  searchBtn.textContent = 'Search';
  searchBtn.className = 'action-btn search-btn';
  searchDiv.appendChild(searchKeyInput);
  searchDiv.appendChild(searchBtn);
  
  controls.appendChild(insertDiv);
  controls.appendChild(deleteDiv);
  controls.appendChild(searchDiv);
  
  // Create canvas/visualization area
  const canvas = document.createElement('div');
  canvas.className = 'btree-canvas';
  
  container.appendChild(controls);
  container.appendChild(canvas);
  el.appendChild(container);
  
  // Render function
  function renderTree() {
    const treeData = model.get('tree_data');
    const highlightedPath = model.get('highlighted_path') || [];
    
    canvas.innerHTML = '';
    
    if (!treeData || !treeData.root) {
      const emptyMsg = document.createElement('div');
      emptyMsg.className = 'empty-message';
      emptyMsg.textContent = 'Tree is empty';
      canvas.appendChild(emptyMsg);
      return;
    }
    
    // Build tree visualization
    const treeDiv = buildTreeVisualization(treeData.root, highlightedPath);
    canvas.appendChild(treeDiv);
  }
  
  // Build tree visualization recursively
  function buildTreeVisualization(node, highlightedPath, level = 0) {
    const nodeDiv = document.createElement('div');
    nodeDiv.className = 'btree-node';
    
    // Check if this node is in highlighted path
    const isHighlighted = highlightedPath.includes(node.id);
    if (isHighlighted) {
      nodeDiv.classList.add('highlighted');
    }
    
    // Create keys container
    const keysDiv = document.createElement('div');
    keysDiv.className = 'node-keys';
    
    // Add keys
    for (let i = 0; i < node.keys.length; i++) {
      const keyValueDiv = document.createElement('div');
      keyValueDiv.className = 'key-value-pair';
      
      const keySpan = document.createElement('span');
      keySpan.className = 'key';
      keySpan.textContent = node.keys[i];
      
      const valueSpan = document.createElement('span');
      valueSpan.className = 'value';
      valueSpan.textContent = node.values[i] || '';
      
      keyValueDiv.appendChild(keySpan);
      keyValueDiv.appendChild(valueSpan);
      keysDiv.appendChild(keyValueDiv);
    }
    
    nodeDiv.appendChild(keysDiv);
    
    // Add children if not leaf
    if (node.children && node.children.length > 0) {
      const childrenDiv = document.createElement('div');
      childrenDiv.className = 'node-children';
      
      for (const child of node.children) {
        const childDiv = buildTreeVisualization(child, highlightedPath, level + 1);
        childrenDiv.appendChild(childDiv);
      }
      
      nodeDiv.appendChild(childrenDiv);
    }
    
    return nodeDiv;
  }
  
  // Event handlers
  insertBtn.addEventListener('click', () => {
    const key = parseInt(insertKeyInput.value);
    const value = insertValueInput.value;
    if (!isNaN(key) && value) {
      model.set('insert_key', key);
      model.set('insert_value', value);
      model.save_changes();
      insertKeyInput.value = '';
      insertValueInput.value = '';
    }
  });
  
  deleteBtn.addEventListener('click', () => {
    const key = parseInt(deleteKeyInput.value);
    if (!isNaN(key)) {
      model.set('delete_key', key);
      model.save_changes();
      deleteKeyInput.value = '';
    }
  });
  
  searchBtn.addEventListener('click', () => {
    const key = parseInt(searchKeyInput.value);
    if (!isNaN(key)) {
      model.set('search_key', key);
      model.save_changes();
      searchKeyInput.value = '';
    }
  });
  
  // Listen for tree updates
  model.on('change:tree_data', renderTree);
  model.on('change:highlighted_path', renderTree);
  
  // Initial render
  renderTree();
}

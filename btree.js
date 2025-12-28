// B-tree visualization widget
export function render({ model, el }) {
  // State for navigation
  let selectedNodeId = null;
  let navigationPath = []; // Stack of node IDs for navigation
  
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
  
  // Create navigation bar
  const navBar = document.createElement('div');
  navBar.className = 'navigation-bar';
  const navBreadcrumb = document.createElement('div');
  navBreadcrumb.className = 'breadcrumb';
  const backBtn = document.createElement('button');
  backBtn.textContent = '← Back to Root';
  backBtn.className = 'nav-btn';
  backBtn.style.display = 'none';
  navBar.appendChild(backBtn);
  navBar.appendChild(navBreadcrumb);
  
  // Create detail panel
  const detailPanel = document.createElement('div');
  detailPanel.className = 'detail-panel';
  detailPanel.style.display = 'none';
  const detailTitle = document.createElement('h3');
  detailTitle.textContent = 'Node Details';
  detailTitle.className = 'detail-title';
  const detailContent = document.createElement('div');
  detailContent.className = 'detail-content';
  detailPanel.appendChild(detailTitle);
  detailPanel.appendChild(detailContent);
  
  // Create canvas/visualization area
  const canvas = document.createElement('div');
  canvas.className = 'btree-canvas';
  
  // Create main content area
  const mainContent = document.createElement('div');
  mainContent.className = 'main-content';
  mainContent.appendChild(canvas);
  mainContent.appendChild(detailPanel);
  
  container.appendChild(controls);
  container.appendChild(navBar);
  container.appendChild(mainContent);
  el.appendChild(container);
  
  // Find node by ID in tree
  function findNodeById(root, nodeId) {
    if (!root) return null;
    if (root.id === nodeId) return root;
    if (root.children) {
      for (const child of root.children) {
        const found = findNodeById(child, nodeId);
        if (found) return found;
      }
    }
    return null;
  }
  
  // Update breadcrumb
  function updateBreadcrumb() {
    navBreadcrumb.innerHTML = '';
    if (navigationPath.length === 0) {
      navBreadcrumb.textContent = 'Root';
      backBtn.style.display = 'none';
    } else {
      backBtn.style.display = 'inline-block';
      const pathText = navigationPath.map((id, idx) => {
        const node = findNodeById(model.get('tree_data')?.root, id);
        const label = node ? `Node ${node.keys.join(',')}` : `Node ${id}`;
        const span = document.createElement('span');
        span.className = 'breadcrumb-item';
        span.textContent = label;
        span.onclick = () => navigateToNode(id);
        return span;
      });
      pathText.forEach((span, idx) => {
        navBreadcrumb.appendChild(span);
        if (idx < pathText.length - 1) {
          const sep = document.createElement('span');
          sep.className = 'breadcrumb-sep';
          sep.textContent = ' → ';
          navBreadcrumb.appendChild(sep);
        }
      });
    }
  }
  
  // Navigate to a specific node
  function navigateToNode(nodeId) {
    const treeData = model.get('tree_data');
    if (!treeData || !treeData.root) return;
    
    const node = findNodeById(treeData.root, nodeId);
    if (!node) return;
    
    // Update navigation path
    const idx = navigationPath.indexOf(nodeId);
    if (idx >= 0) {
      navigationPath = navigationPath.slice(0, idx + 1);
    } else {
      navigationPath.push(nodeId);
    }
    
    selectedNodeId = nodeId;
    updateBreadcrumb();
    renderTree();
    showNodeDetails(node);
  }
  
  // Show node details
  function showNodeDetails(node) {
    if (!node) {
      detailPanel.style.display = 'none';
      return;
    }
    
    detailPanel.style.display = 'block';
    detailContent.innerHTML = `
      <div class="detail-row">
        <strong>Node ID:</strong> <span>${node.id}</span>
      </div>
      <div class="detail-row">
        <strong>Is Leaf:</strong> <span>${node.is_leaf ? 'Yes' : 'No'}</span>
      </div>
      <div class="detail-row">
        <strong>Number of Keys:</strong> <span>${node.keys.length}</span>
      </div>
      <div class="detail-row">
        <strong>Keys:</strong> <span>[${node.keys.join(', ')}]</span>
      </div>
      <div class="detail-row">
        <strong>Values:</strong> <span>[${node.values.map(v => `"${v}"`).join(', ')}]</span>
      </div>
      ${node.children ? `<div class="detail-row">
        <strong>Number of Children:</strong> <span>${node.children.length}</span>
      </div>` : ''}
      <div class="detail-section">
        <strong>Key-Value Pairs:</strong>
        <div class="kv-list">
          ${node.keys.map((key, i) => `
            <div class="kv-item">
              <span class="kv-key">${key}</span>
              <span class="kv-sep">→</span>
              <span class="kv-value">"${node.values[i] || ''}"</span>
            </div>
          `).join('')}
        </div>
      </div>
      ${node.children && node.children.length > 0 ? `
        <div class="detail-section">
          <strong>Children:</strong>
          <div class="children-list">
            ${node.children.map((child, i) => `
              <div class="child-item" onclick="window.navigateToChild(${child.id})">
                Child ${i + 1}: [${child.keys.join(', ')}]
              </div>
            `).join('')}
          </div>
        </div>
      ` : ''}
    `;
    
    // Make child items clickable
    const childItems = detailContent.querySelectorAll('.child-item');
    childItems.forEach((item, i) => {
      item.style.cursor = 'pointer';
      item.onclick = () => {
        const node = findNodeById(model.get('tree_data')?.root, selectedNodeId);
        if (node && node.children && node.children[i]) {
          navigateToNode(node.children[i].id);
        }
      };
    });
  }
  
  // Back button handler
  backBtn.addEventListener('click', () => {
    navigationPath = [];
    selectedNodeId = null;
    updateBreadcrumb();
    renderTree();
    detailPanel.style.display = 'none';
  });
  
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
    
    // Determine which node to show based on navigation
    let rootNode = treeData.root;
    if (selectedNodeId) {
      const node = findNodeById(treeData.root, selectedNodeId);
      if (node) {
        rootNode = node;
      }
    }
    
    // Build tree visualization
    const treeDiv = buildTreeVisualization(rootNode, highlightedPath, selectedNodeId);
    canvas.appendChild(treeDiv);
    
    // Update details if node is selected
    if (selectedNodeId) {
      const node = findNodeById(treeData.root, selectedNodeId);
      if (node) {
        showNodeDetails(node);
      }
    }
  }
  
  // Build tree visualization recursively
  function buildTreeVisualization(node, highlightedPath, selectedId, level = 0) {
    const nodeDiv = document.createElement('div');
    nodeDiv.className = 'btree-node';
    nodeDiv.dataset.nodeId = node.id;
    
    // Check if this node is in highlighted path
    const isHighlighted = highlightedPath.includes(node.id);
    if (isHighlighted) {
      nodeDiv.classList.add('highlighted');
    }
    
    // Check if this node is selected
    if (selectedId === node.id) {
      nodeDiv.classList.add('selected');
    }
    
    // Make node clickable
    nodeDiv.style.cursor = 'pointer';
    nodeDiv.addEventListener('click', (e) => {
      e.stopPropagation();
      navigateToNode(node.id);
    });
    
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
        const childDiv = buildTreeVisualization(child, highlightedPath, selectedId, level + 1);
        childrenDiv.appendChild(childDiv);
      }
      
      nodeDiv.appendChild(childrenDiv);
    }
    
    return nodeDiv;
  }
  
  // Expose navigate function globally for detail panel
  window.navigateToChild = (nodeId) => {
    navigateToNode(nodeId);
  };
  
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
  model.on('change:tree_data', () => {
    // Reset navigation when tree changes
    navigationPath = [];
    selectedNodeId = null;
    updateBreadcrumb();
    renderTree();
    detailPanel.style.display = 'none';
  });
  model.on('change:highlighted_path', renderTree);
  
  // Initial render
  updateBreadcrumb();
  renderTree();
}

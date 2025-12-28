const SVG_NS = "http://www.w3.org/2000/svg";

function buildLevels(tree) {
  const levels = [];
  const edges = [];
  if (!tree) {
    return { levels, edges };
  }
  let idCounter = 0;
  const queue = [{ node: tree, depth: 0, id: idCounter }];
  while (queue.length > 0) {
    const current = queue.shift();
    if (!levels[current.depth]) {
      levels[current.depth] = [];
    }
    levels[current.depth].push({ id: current.id, node: current.node });
    const children = current.node.children || [];
    for (const child of children) {
      idCounter += 1;
      const childId = idCounter;
      edges.push({ from: current.id, to: childId });
      queue.push({ node: child, depth: current.depth + 1, id: childId });
    }
  }
  return { levels, edges };
}

function createNodeEl(node) {
  const nodeEl = document.createElement("div");
  nodeEl.className = "btree-node";

  const keysEl = document.createElement("div");
  keysEl.className = "btree-keys";

  const keys = node && Array.isArray(node.keys) ? node.keys : [];
  if (keys.length === 0) {
    const emptyEl = document.createElement("span");
    emptyEl.className = "btree-key btree-key-empty";
    emptyEl.textContent = "empty";
    keysEl.appendChild(emptyEl);
  } else {
    for (const key of keys) {
      const keyEl = document.createElement("span");
      keyEl.className = "btree-key";
      keyEl.textContent = String(key);
      keysEl.appendChild(keyEl);
    }
  }

  nodeEl.appendChild(keysEl);

  const children = node && Array.isArray(node.children) ? node.children : [];
  if (children.length > 0) {
    const pointersEl = document.createElement("div");
    pointersEl.className = "btree-pointers";
    const pointerCount = Math.max(1, keys.length + 1);
    for (let i = 0; i < pointerCount; i += 1) {
      const dot = document.createElement("span");
      dot.className = "btree-pointer";
      pointersEl.appendChild(dot);
    }
    nodeEl.appendChild(pointersEl);
  }

  return nodeEl;
}

function drawEdges({ container, svg, nodeEls, edges }) {
  svg.innerHTML = "";
  if (edges.length === 0) {
    return;
  }

  const rect = container.getBoundingClientRect();
  svg.setAttribute("width", rect.width);
  svg.setAttribute("height", rect.height);
  svg.setAttribute("viewBox", `0 0 ${rect.width} ${rect.height}`);

  const defs = document.createElementNS(SVG_NS, "defs");
  const marker = document.createElementNS(SVG_NS, "marker");
  marker.setAttribute("id", "btree-arrow");
  marker.setAttribute("markerWidth", "12");
  marker.setAttribute("markerHeight", "12");
  marker.setAttribute("refX", "8");
  marker.setAttribute("refY", "6");
  marker.setAttribute("orient", "auto");

  const arrowPath = document.createElementNS(SVG_NS, "path");
  arrowPath.setAttribute("d", "M0,0 L12,6 L0,12 Z");
  arrowPath.setAttribute("fill", "var(--btree-edge)");
  marker.appendChild(arrowPath);
  defs.appendChild(marker);
  svg.appendChild(defs);

  for (const edge of edges) {
    const fromEl = nodeEls.get(edge.from);
    const toEl = nodeEls.get(edge.to);
    if (!fromEl || !toEl) {
      continue;
    }
    const fromRect = fromEl.getBoundingClientRect();
    const toRect = toEl.getBoundingClientRect();

    const x1 = fromRect.left + fromRect.width / 2 - rect.left;
    const y1 = fromRect.bottom - rect.top - 6;
    const x2 = toRect.left + toRect.width / 2 - rect.left;
    const y2 = toRect.top - rect.top + 6;

    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", x1);
    line.setAttribute("y1", y1);
    line.setAttribute("x2", x2);
    line.setAttribute("y2", y2);
    line.setAttribute("marker-end", "url(#btree-arrow)");
    line.setAttribute("stroke", "var(--btree-edge)");
    line.setAttribute("stroke-width", "2.5");
    line.setAttribute("stroke-linecap", "round");
    svg.appendChild(line);
  }
}

function render({ model, el }) {
  el.classList.add("btree-root");

  const container = document.createElement("div");
  container.className = "btree-container";
  el.appendChild(container);

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.classList.add("btree-edges");
  container.appendChild(svg);

  const treeWrap = document.createElement("div");
  treeWrap.className = "btree-tree";
  container.appendChild(treeWrap);

  const nodeEls = new Map();

  function draw() {
    treeWrap.innerHTML = "";
    nodeEls.clear();

    const tree = model.get("tree");
    const { levels, edges } = buildLevels(tree);

    if (levels.length === 0) {
      return;
    }

    for (const level of levels) {
      const row = document.createElement("div");
      row.className = "btree-level";
      for (const item of level) {
        const nodeEl = createNodeEl(item.node);
        nodeEl.dataset.nodeId = String(item.id);
        row.appendChild(nodeEl);
        nodeEls.set(item.id, nodeEl);
      }
      treeWrap.appendChild(row);
    }

    requestAnimationFrame(() => {
      drawEdges({ container, svg, nodeEls, edges });
    });
  }

  const resizeObserver = new ResizeObserver(() => {
    draw();
  });
  resizeObserver.observe(container);

  model.on("change:tree", draw);
  model.on("change:branching_factor", draw);

  draw();

  return () => {
    resizeObserver.disconnect();
  };
}

export default { render };

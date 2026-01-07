function render({ model, el }) {
    const container = document.createElement("div");
    container.classList.add("maze-container");

    const gridDiv = document.createElement("div");
    gridDiv.classList.add("maze-grid");
    
    // Create status text
    const statusDiv = document.createElement("div");
    statusDiv.style.marginBottom = "10px";
    statusDiv.textContent = "Ready";

    container.appendChild(statusDiv);
    container.appendChild(gridDiv);
    el.appendChild(container);

    let cells = [];
    let agentDiv = document.createElement("div");
    agentDiv.classList.add("agent");

    function initGrid() {
        const layout = model.get("maze_layout");
        const height = layout.length;
        const width = layout[0].length;

        gridDiv.style.gridTemplateColumns = `repeat(${width}, 30px)`;
        gridDiv.style.gridTemplateRows = `repeat(${height}, 30px)`;
        gridDiv.innerHTML = "";
        cells = [];

        layout.forEach((row, r) => {
            row.forEach((cellType, c) => {
                const cell = document.createElement("div");
                cell.classList.add("maze-cell");
                cell.dataset.r = r;
                cell.dataset.c = c;

                if (cellType === 1) cell.classList.add("cell-wall");
                else if (cellType === 2) cell.classList.add("cell-start");
                else if (cellType === 3) cell.classList.add("cell-goal");

                // Arrow for policy visualization (optional)
                const arrow = document.createElement("span");
                arrow.classList.add("arrow");
                cell.appendChild(arrow);

                gridDiv.appendChild(cell);
                cells.push(cell);
            });
        });

        // Add agent to the grid container, but position absolutely relative to cells? 
        // Actually, let's put agent inside the correct cell or translate it.
        // Translating over the grid is smoother.
        // Let's append agent to the gridDiv and use transforms.
        // gridDiv needs relative positioning for this to work well if we want smooth transitions across gaps.
        // But simply appending to the current cell is easier for DOM updates, though less smooth if animating.
        // Let's stick to appending to the target cell for simplicity and robustness first.
        // Or overlay.
    }

    function updateAgent() {
        const [r, c] = model.get("agent_position");
        const layout = model.get("maze_layout");
        const width = layout[0].length;
        const cellIndex = r * width + c;
        const targetCell = cells[cellIndex];
        
        if (targetCell) {
            if (!agentDiv.parentNode) {
                targetCell.appendChild(agentDiv);
            } else if (agentDiv.parentNode !== targetCell) {
                targetCell.appendChild(agentDiv);
            }
        }
    }

    function getColorForValue(val, minVal, maxVal) {
        // Normalize to 0-1
        if (maxVal === minVal) return 'rgba(0,0,0,0)';
        
        // Simple heatmap: Red (low) -> Yellow -> Green (high)
        // Or simpler: transparent -> Green (positive), transparent -> Red (negative)
        
        // Let's use HSL.
        // -1 (bad) -> Red (0)
        // 100 (good) -> Green (120)
        // 0 -> somewhat neutral?
        
        // Use a fixed range for stability: -1 to 100
        // Or dynamic range from data?
        // Dynamic is better for visibility.
        
        let normalized = (val - minVal) / (maxVal - minVal);
        normalized = Math.max(0, Math.min(1, normalized));
        
        // Hue from 0 (red) to 120 (green)
        const hue = normalized * 120;
        return `hsla(${hue}, 80%, 50%, 0.6)`;
    }

    function updateQValues() {
        const show = model.get("show_q_values");
        const qGrid = model.get("q_values"); // [rows][cols][4]
        
        if (!show || !qGrid || qGrid.length === 0) {
            cells.forEach(cell => {
                const overlay = cell.querySelector(".q-overlay");
                if (overlay) overlay.classList.remove("visible");
            });
            return;
        }

        // Find global min/max for coloring
        let minVal = Infinity;
        let maxVal = -Infinity;
        qGrid.forEach(row => {
            row.forEach(cellActions => {
                cellActions.forEach(val => {
                    if (val < minVal) minVal = val;
                    if (val > maxVal) maxVal = val;
                });
            });
        });
        
        // If all zeros, avoid div by zero
        if (minVal === 0 && maxVal === 0) {
            maxVal = 1; 
        }

        const layout = model.get("maze_layout");
        cells.forEach((cell, i) => {
            const r = Math.floor(i / layout[0].length);
            const c = i % layout[0].length;
            let overlay = cell.querySelector(".q-overlay");
            
            if (!overlay) {
                overlay = document.createElement("div");
                overlay.classList.add("q-overlay");
                cell.appendChild(overlay);
            }
            
            const vals = qGrid[r][c]; // [up, right, down, left]
            
            // Only show if not a wall?
            const isWall = layout[r][c] === 1;
            
            if (isWall) {
                overlay.classList.remove("visible");
            } else {
                overlay.classList.add("visible");
                overlay.style.borderTopColor = getColorForValue(vals[0], minVal, maxVal);
                overlay.style.borderRightColor = getColorForValue(vals[1], minVal, maxVal);
                overlay.style.borderBottomColor = getColorForValue(vals[2], minVal, maxVal);
                overlay.style.borderLeftColor = getColorForValue(vals[3], minVal, maxVal);
            }
        });
    }

    // Listeners
    model.on("change:maze_layout", initGrid);
    model.on("change:agent_position", updateAgent);
    model.on("change:policy_grid", updatePolicy);
    model.on("change:q_values", updateQValues);
    model.on("change:show_q_values", updateQValues);
}

export default { render };

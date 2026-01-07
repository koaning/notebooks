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

    function updatePolicy() {
        // q_values is a list of lists of 4 floats [up, right, down, left]
        // or simplified: policy_grid which is just directions.
        // Let's assume we get a "policy_arrows" grid of strings.
        const policy = model.get("policy_grid");
        if (!policy || policy.length === 0) return;

        const layout = model.get("maze_layout");
        
        cells.forEach((cell, i) => {
            const r = Math.floor(i / layout[0].length);
            const c = i % layout[0].length;
            const arrow = cell.querySelector(".arrow");
            
            if (policy[r] && policy[r][c]) {
                const direction = policy[r][c];
                arrow.textContent = getArrowChar(direction);
                arrow.classList.add("visible");
            } else {
                arrow.classList.remove("visible");
            }
        });
    }

    function getArrowChar(dir) {
        if (dir === 0) return "↑"; // Up
        if (dir === 1) return "→"; // Right
        if (dir === 2) return "↓"; // Down
        if (dir === 3) return "←"; // Left
        return "";
    }

    function updateStatus() {
         // Could sync status text like "Episode: 5, Reward: 10"
    }

    // Initial render
    initGrid();
    updateAgent();

    // Listeners
    model.on("change:maze_layout", initGrid);
    model.on("change:agent_position", updateAgent);
    model.on("change:policy_grid", updatePolicy);
}

export default { render };

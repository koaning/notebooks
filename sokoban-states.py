# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "anywidget>=0.11",
#     "traitlets",
#     "matplotlib==3.11.1",
#     "wigglystuff==0.5.32",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from collections import deque
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    import anywidget
    import traitlets

    from wigglystuff import GraphWidget

    # SokobanWidget is defined in the appendix cell at the bottom of this
    # notebook, so it stays self-contained (no local module files).
    return Circle, GraphWidget, Rectangle, anywidget, deque, mo, plt, traitlets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sokoban state graph

    A Sokoban **state** is `(player position, crate positions)`. From your
    drawn board we run a breadth-first search over every state reachable by
    legal moves — walking *and* pushing crates — and map out the state graph.
    A **solved** board (all crates on goals) is a dead-end: we don't push crates
    any further from it, but every other branch keeps exploring.

    The grid boundary is a wall, and you can carve interior **walls** too —
    impassable for the player and crates. Because we count the *full* state
    (player + crates), the graph blows up fast, so keep the board small. Draw a
    board below (📦 crate, 🎯 goal, 🙂 player, ⬛ wall), then **hover any node**
    in the graph to see its board. Node color runs light→dark with BFS depth;
    the start is orange, the solution green.
    """)
    return


@app.cell
def _(SokobanWidget, mo):
    sokoban = SokobanWidget(width=4, height=4)
    view = mo.ui.anywidget(sokoban)
    view
    return (view,)


@app.cell
def _(view):
    # read via the mo.ui wrapper's reactive .value so this cell re-fires on edits;
    # tuple-ize so it stays hashable for downstream cells
    board_raw = view.value.get("board") or []
    board = tuple(tuple(row) for row in board_raw)

    def parse(grid):
        h = len(grid)
        w = len(grid[0]) if grid else 0
        start = None
        crates = []
        goals_ = []
        walls_ = []
        for y, row in enumerate(grid):
            for x, tile in enumerate(row):
                if tile == 1:
                    crates.append((x, y))
                elif tile == 2:
                    goals_.append((x, y))
                elif tile == 3:
                    start = (x, y)
                elif tile == 4:
                    walls_.append((x, y))
        return w, h, start, frozenset(crates), frozenset(goals_), frozenset(walls_)

    width, height, player_start, crates0, goals, walls = parse(board)
    return crates0, goals, height, player_start, walls, width


@app.cell(hide_code=True)
def _(crates0, mo, player_start):
    valid = player_start is not None
    # non-halting: let `valid` flow downstream so the graph clears when the
    # board becomes invalid, instead of mo.stop freezing the last render
    status = (
        mo.md(f"Player start `{player_start}` · {len(crates0)} crate(s).")
        if valid
        else mo.md("⚠️ **Place exactly one player (🙂) on the board to run the search.**")
    )
    status
    return (valid,)


@app.cell
def _(crates0, deque, goals, height, mo, player_start, valid, walls, width):
    MAX_STATES = 3000

    @mo.cache
    def compute_bfs(start_pos, crates_start, walls_set, goals_set, w, h, cap):
        def neighbors(state):
            (px, py), crates = state
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx_, ny_ = px + dx, py + dy
                if not (0 <= nx_ < w and 0 <= ny_ < h):
                    continue
                if (nx_, ny_) in walls_set:
                    continue
                if (nx_, ny_) in crates:
                    bx, by = nx_ + dx, ny_ + dy
                    if not (0 <= bx < w and 0 <= by < h):
                        continue
                    if (bx, by) in crates or (bx, by) in walls_set:
                        continue
                    yield ((nx_, ny_), frozenset((crates - {(nx_, ny_)}) | {(bx, by)}))
                else:
                    yield ((nx_, ny_), crates)

        def is_solved(crates):
            return bool(goals_set) and crates == goals_set

        start = (start_pos, crates_start)
        disc = {start: 0}
        depth = [0]  # index-aligned with nodes
        nodes = [start]
        edges = set()
        capped = False
        solved = {0} if is_solved(crates_start) else set()
        queue = deque([start])
        while queue:
            state = queue.popleft()
            if is_solved(state[1]):
                continue  # solved board is terminal — don't push crates further
            for nb in neighbors(state):
                if nb not in disc:
                    if len(nodes) >= cap:
                        capped = True
                        continue
                    disc[nb] = len(nodes)
                    depth.append(depth[disc[state]] + 1)
                    nodes.append(nb)
                    queue.append(nb)
                    if is_solved(nb[1]):
                        solved.add(disc[nb])
                # directed transition state -> nb (a walk adds both directions
                # over time; an irreversible push adds only the one)
                edges.add((disc[state], disc[nb]))
        return nodes, depth, frozenset(edges), capped, frozenset(solved)

    if valid:
        nodes, depth, edges, capped, solved = compute_bfs(
            player_start, crates0, walls, goals, width, height, MAX_STATES
        )
    else:
        nodes, depth, edges, capped, solved = [], [], frozenset(), False, frozenset()
    return capped, depth, edges, nodes, solved


@app.cell(hide_code=True)
def _(mo, nodes):
    n = max(len(nodes), 1)
    # reveal states in BFS order — fewer nodes lay out more cleanly
    reveal = mo.ui.slider(
        start=1, stop=n, step=1, value=n,
        label="reveal states (BFS order)", full_width=True, show_value=True,
    )
    node_size = mo.ui.slider(
        start=0.4, stop=3.0, step=0.1, value=1.0,
        label="node size", show_value=True,
    )
    directed_cb = mo.ui.checkbox(value=False, label="show move direction (arrows)")
    mo.vstack(
        [
            reveal,
            mo.hstack([node_size, directed_cb], justify="start", gap=2, align="center"),
        ]
    )
    return directed_cb, node_size, reveal


@app.cell(hide_code=True)
def _(GraphWidget, depth, mo, nodes, plt, solved):
    # Build the graph widget ONCE per board. The reveal slider mutates its
    # nodes/edges traits downstream — GraphWidget preserves the positions of
    # nodes it already knows, so revealing more states never re-lays-out the
    # graph or drops the hover state. Colors are computed here over the full
    # depth so they stay fixed as you drag the slider.
    MAX_GRAPH = 1800

    def build_all_nodes():
        maxd = max(depth) if depth else 1
        cmap = plt.get_cmap("Blues")

        def hexc(d):
            # 0.35→0.95 so shallow nodes aren't near-white; blues keeps green free
            r, g, b, _ = cmap(0.35 + 0.6 * ((d / maxd) if maxd else 0.0))
            return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

        gnodes = []
        for i in range(len(nodes)):
            if i == 0:
                color, size = "#f58518", 22  # start
            elif i in solved:
                color, size = "#2ca02c", 20  # solved (terminal)
            else:
                color, size = hexc(depth[i]), 9
            gnodes.append({"id": i, "name": f"#{i}", "color": color, "size": size})
        return gnodes

    all_gnodes = build_all_nodes()

    if len(nodes) == 0:
        graph = None
        graph_ui = None
        out = mo.md("*Place a player (🙂) on the board to build the graph.*")
    elif len(nodes) > MAX_GRAPH:
        graph = None
        graph_ui = None
        out = mo.md(
            f"Showing **{len(nodes)}** nodes — too many to draw interactively. "
            "Shrink the board."
        )
    else:
        # bounded=False so the layout can exceed the frame — scroll to zoom out,
        # drag to pan (bounded defaults to True, cramming nodes to edges). Start
        # empty; the reveal cell fills it in.
        graph = GraphWidget(
            nodes=[], edges=[], directed=False, bounded=False, height=780,
        )
        graph_ui = mo.ui.anywidget(graph)
        # small right padding so the widget's last pixel isn't clipped
        out = mo.vstack([graph_ui]).style({"padding-right": "4px"})

    out
    return all_gnodes, graph, graph_ui


@app.cell(hide_code=True)
def _(
    all_gnodes,
    capped,
    directed_cb,
    edges,
    graph,
    mo,
    node_size,
    nodes,
    reveal,
    solved,
):
    # Reactive on the reveal slider / size slider / arrows checkbox: push the
    # revealed subset into the existing widget's traits instead of rebuilding it.
    # hold_sync so the frontend rebuilds once from both node and edge changes.
    show_n = min(int(reveal.value), len(nodes))

    if graph is not None:
        scale = float(node_size.value)
        sub_nodes = [
            {**gn, "size": max(1, round(gn["size"] * scale))}
            for gn in all_gnodes[:show_n]
        ]
        sub_edges = [
            {"source": u, "target": v}
            for (u, v) in edges
            if u < show_n and v < show_n
        ]
        with graph.hold_sync():
            graph.nodes = sub_nodes
            graph.edges = sub_edges
            graph.directed = bool(directed_cb.value)

    shown_solved = sum(1 for s in solved if s < show_n)
    caption = mo.md(
        f"**{show_n} / {len(nodes)}** states shown"
        + (f" · {shown_solved} solved (dead-ends)" if shown_solved else "")
        + (" · ⚠️ capped (shrink the board)" if capped else "")
        + "\n\n"
        + "🟠 start · 🟢 solved · **blue = BFS depth** (lighter = fewer moves from start)"
    )
    caption
    return


@app.cell(hide_code=True)
def _(
    Circle,
    Rectangle,
    depth,
    goals,
    graph_ui,
    height,
    mo,
    nodes,
    plt,
    walls,
    width,
):
    if graph_ui is None or len(nodes) == 0:
        board_view = mo.md("*Hover a node in the graph to see its board.*")
    else:
        # read the mo.ui wrapper's reactive .value so hovering a node re-fires;
        # hovered_node is None when nothing is hovered → fall back to the start
        hovered = graph_ui.value.get("hovered_node")
        if hovered is None:
            idx = 0
        else:
            idx = hovered.get("id") if isinstance(hovered, dict) else hovered
        idx = max(0, min(int(idx), len(nodes) - 1))
        (player_pos, crates) = nodes[idx]

        fig, ax = plt.subplots(figsize=(min(4.5, width + 0.5), min(4.5, height + 0.5)))
        for cy in range(height):
            for cx in range(width):
                is_wall = (cx, cy) in walls
                ax.add_patch(
                    Rectangle(
                        (cx, height - 1 - cy), 1, 1,
                        facecolor="#555555" if is_wall else "#e6e6e6",
                        edgecolor="white", linewidth=1.5,
                    )
                )
        for gx, gy in goals:
            ax.add_patch(
                Circle((gx + 0.5, height - 1 - gy + 0.5), 0.36,
                       fill=False, edgecolor="#2e7d32", linewidth=2.5)
            )
        for cx, cy in crates:
            on_goal = (cx, cy) in goals
            ax.add_patch(
                Rectangle((cx + 0.15, height - 1 - cy + 0.15), 0.7, 0.7,
                          facecolor="#8a5a2b" if on_goal else "#c8935f",
                          edgecolor="#5b3a1a", linewidth=1.5)
            )
        px, py = player_pos
        ax.add_patch(
            Circle((px + 0.5, height - 1 - py + 0.5), 0.3,
                   facecolor="#4c78a8", edgecolor="#1f3a5f", linewidth=1.5)
        )
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect("equal")
        ax.axis("off")

        is_solved_state = bool(goals) and crates == goals
        board_view = mo.vstack(
            [
                mo.md(
                    f"**State #{idx}** · depth {depth[idx]}"
                    + (" · ✅ solved" if is_solved_state else "")
                ),
                fig,
            ]
        )
    board_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Appendix — widget definition

    `SokobanWidget` is inlined here so the notebook is self-contained (no
    local module files). marimo runs by dependency graph, so defining it at
    the bottom is fine.
    """)
    return


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class SokobanWidget(anywidget.AnyWidget):
        """Draw a Sokoban board. Tile enum per cell: 0 floor, 1 crate, 2 goal,
        3 player, 4 wall. The player is a singleton; the grid boundary is a wall."""

        _esm = r"""
    const TILES = {
      0: { name: 'Erase',  color: '#e6e6e6', glyph: ''  },
      1: { name: 'Crate',  color: '#c8935f', glyph: '📦' },
      2: { name: 'Goal',   color: '#c5e1a5', glyph: '🎯' },
      3: { name: 'Player', color: '#90caf9', glyph: '🙂' },
      4: { name: 'Wall',   color: '#555555', glyph: ''  },
    };

    function render({ model, el }) {
      el.classList.add('sokoban-widget-root');

      let currentTool = 1; // start painting crates
      let isDrawing = false;
      let board = [];

      const controls = document.createElement('div');
      controls.className = 'sokoban-controls';

      const widthLabel = document.createElement('label');
      widthLabel.textContent = 'W: ';
      const widthInput = document.createElement('input');
      widthInput.type = 'number';
      widthInput.min = '2';
      widthInput.max = '8';
      widthInput.value = model.get('width') || 4;
      widthLabel.appendChild(widthInput);

      const heightLabel = document.createElement('label');
      heightLabel.textContent = 'H: ';
      const heightInput = document.createElement('input');
      heightInput.type = 'number';
      heightInput.min = '2';
      heightInput.max = '8';
      heightInput.value = model.get('height') || 4;
      heightLabel.appendChild(heightInput);

      const toolButtons = {};
      const toolbar = document.createElement('div');
      toolbar.className = 'sokoban-toolbar';
      [0, 1, 2, 3, 4].forEach((t) => {
        const btn = document.createElement('button');
        btn.className = 'sokoban-tool';
        btn.textContent = (TILES[t].glyph ? TILES[t].glyph + ' ' : '') + TILES[t].name;
        btn.style.background = TILES[t].color;
        if (t === 4) btn.style.color = '#fff';
        btn.addEventListener('click', () => selectTool(t));
        toolButtons[t] = btn;
        toolbar.appendChild(btn);
      });

      const clearButton = document.createElement('button');
      clearButton.textContent = 'Reset';
      clearButton.className = 'sokoban-tool sokoban-reset';

      controls.appendChild(widthLabel);
      controls.appendChild(heightLabel);
      controls.appendChild(toolbar);
      controls.appendChild(clearButton);
      el.appendChild(controls);

      const container = document.createElement('div');
      container.className = 'sokoban-container';
      const grid = document.createElement('div');
      grid.className = 'sokoban-grid';
      container.appendChild(grid);
      el.appendChild(container);

      function selectTool(t) {
        currentTool = t;
        Object.entries(toolButtons).forEach(([k, b]) => {
          b.classList.toggle('sokoban-tool-active', Number(k) === t);
        });
      }

      function emptyBoard(w, h) {
        const b = [];
        for (let y = 0; y < h; y++) {
          b[y] = [];
          for (let x = 0; x < w; x++) b[y][x] = 0;
        }
        return b;
      }

      function paintCell(x, y) {
        if (currentTool === 3) {
          for (let yy = 0; yy < board.length; yy++)
            for (let xx = 0; xx < board[yy].length; xx++)
              if (board[yy][xx] === 3) board[yy][xx] = 0;
        }
        board[y][x] = currentTool;
        drawCells();
        pushModel();
      }

      function drawCells() {
        const cells = grid.querySelectorAll('.sokoban-cell');
        cells.forEach((cell) => {
          const x = Number(cell.dataset.x);
          const y = Number(cell.dataset.y);
          const t = board[y][x];
          cell.style.backgroundColor = TILES[t].color;
          cell.textContent = TILES[t].glyph;
        });
      }

      function buildGrid() {
        const width = parseInt(widthInput.value) || 4;
        const height = parseInt(heightInput.value) || 4;
        if (board.length !== height || (board[0] && board[0].length !== width)) {
          board = emptyBoard(width, height);
        }
        grid.innerHTML = '';
        grid.style.gridTemplateColumns = `repeat(${width}, 1fr)`;
        grid.style.gridTemplateRows = `repeat(${height}, 1fr)`;

        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const cell = document.createElement('div');
            cell.className = 'sokoban-cell';
            cell.dataset.x = x;
            cell.dataset.y = y;
            cell.addEventListener('mousedown', (e) => {
              e.preventDefault();
              isDrawing = true;
              paintCell(x, y);
            });
            cell.addEventListener('mouseenter', () => {
              if (isDrawing) paintCell(x, y);
            });
            grid.appendChild(cell);
          }
        }
        drawCells();
        pushModel();
      }

      function pushModel() {
        model.set('board', board.map((row) => [...row]));
        model.save_changes();
      }

      widthInput.addEventListener('change', () => {
        model.set('width', parseInt(widthInput.value) || 4);
        model.save_changes();
        buildGrid();
      });
      heightInput.addEventListener('change', () => {
        model.set('height', parseInt(heightInput.value) || 4);
        model.save_changes();
        buildGrid();
      });
      clearButton.addEventListener('click', () => {
        board = emptyBoard(parseInt(widthInput.value) || 4, parseInt(heightInput.value) || 4);
        drawCells();
        pushModel();
      });

      document.addEventListener('mouseup', () => { isDrawing = false; });

      model.on('change:width', () => { widthInput.value = model.get('width') || 4; buildGrid(); });
      model.on('change:height', () => { heightInput.value = model.get('height') || 4; buildGrid(); });

      const existing = model.get('board');
      if (existing && Array.isArray(existing) && existing.length > 0) {
        board = existing.map((row) => [...row]);
        buildGrid();
      } else {
        buildGrid();
      }
      selectTool(currentTool);

      return () => {};
    }

    export default { render };
    """

        _css = r"""
    .sokoban-widget-root {
      display: flex;
      flex-direction: column;
      gap: 1rem;
      padding: 1rem;
      border: 1px solid #ddd;
      border-radius: 8px;
      background: #fff;
    }
    .sokoban-controls {
      display: flex;
      gap: 1rem;
      align-items: center;
      flex-wrap: wrap;
    }
    .sokoban-controls label {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      font-weight: 500;
    }
    .sokoban-controls input[type="number"] {
      width: 52px;
      padding: 4px 8px;
      border: 1px solid #ccc;
      border-radius: 4px;
      font-size: 14px;
    }
    .sokoban-toolbar { display: flex; gap: 0.4rem; }
    .sokoban-tool {
      padding: 6px 12px;
      font-size: 14px;
      font-weight: 500;
      color: #333;
      border: 2px solid transparent;
      border-radius: 4px;
      cursor: pointer;
      transition: border-color 0.15s, transform 0.1s;
    }
    .sokoban-tool:hover { transform: translateY(-1px); }
    .sokoban-tool-active {
      border-color: #333;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
    }
    .sokoban-reset { color: #fff; background: #888; }
    .sokoban-container {
      display: flex;
      justify-content: center;
      padding: 1rem;
      background: #f5f5f5;
      border-radius: 4px;
    }
    .sokoban-grid {
      display: grid;
      gap: 2px;
      background: #ddd;
      padding: 2px;
      border-radius: 4px;
    }
    .sokoban-cell {
      width: 44px;
      height: 44px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 24px;
      background: #e6e6e6;
      border: 1px solid #bbb;
      cursor: pointer;
      user-select: none;
      transition: transform 0.08s;
    }
    .sokoban-cell:hover { transform: scale(1.08); z-index: 1; }
    """

        width = traitlets.Int(default_value=4).tag(sync=True)
        height = traitlets.Int(default_value=4).tag(sync=True)
        board = traitlets.List(
            traitlets.List(traitlets.Int()), default_value=[]
        ).tag(sync=True)

    return (SokobanWidget,)


if __name__ == "__main__":
    app.run()

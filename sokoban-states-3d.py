# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "anywidget>=0.11",
#     "traitlets",
#     "matplotlib==3.11.1",
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

    # SokobanWidget and Graph3DWidget are defined in the appendix cell at the
    # bottom of this notebook, so it stays self-contained (no local modules).
    return Circle, Rectangle, anywidget, deque, mo, plt, traitlets


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
    directed_cb = mo.ui.checkbox(value=False, label="show move direction (arrows)")
    dark_cb = mo.ui.checkbox(value=False, label="dark mode")
    twod_cb = mo.ui.checkbox(value=False, label="2D (flatten depth)")
    mo.hstack(
        [reveal, directed_cb, dark_cb, twod_cb],
        justify="start", gap=2, align="center",
    )
    return dark_cb, directed_cb, reveal, twod_cb


@app.cell(hide_code=True)
def _(Graph3DWidget, depth, mo, nodes, plt, solved):
    # Build the graph widget ONCE per board. The reveal slider mutates its
    # nodes/edges traits downstream — the widget reuses node objects it already
    # knows, so revealing more states never re-lays-out the graph or drops the
    # hover state. Colors are computed here over the full depth so they stay
    # fixed as you drag the slider. `depth` rides along on each node and pins
    # the vertical axis in 3D.
    MAX_GRAPH = 20000

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
            gnodes.append(
                {"id": i, "name": f"#{i}", "color": color, "size": size, "depth": depth[i]}
            )
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
        # 3D graph: vertical axis pinned to BFS depth. Drag to orbit, scroll to
        # zoom. Start empty; the reveal cell fills it in as you drag the slider.
        graph = Graph3DWidget(nodes=[], edges=[], directed=False, height=820)
        graph_ui = mo.ui.anywidget(graph)
        out = graph_ui

    out
    return all_gnodes, graph, graph_ui


@app.cell(hide_code=True)
def _(
    all_gnodes,
    capped,
    dark_cb,
    directed_cb,
    edges,
    graph,
    mo,
    nodes,
    reveal,
    solved,
    twod_cb,
):
    # Reactive on the reveal slider / arrows / dark-mode / 2D toggles: push the
    # revealed subset into the existing widget's traits instead of rebuilding
    # it. hold_sync so the frontend rebuilds once from both node and edge changes.
    show_n = min(int(reveal.value), len(nodes))

    if graph is not None:
        sub_edges = [
            {"source": u, "target": v}
            for (u, v) in edges
            if u < show_n and v < show_n
        ]
        with graph.hold_sync():
            graph.nodes = all_gnodes[:show_n]
            graph.edges = sub_edges
            graph.directed = bool(directed_cb.value)
            graph.dark = bool(dark_cb.value)
            graph.two_d = bool(twod_cb.value)

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
    ## Appendix — widget definitions

    The two anywidgets used above are inlined here so the notebook is
    self-contained (no local module files). marimo runs by dependency
    graph, so defining them at the bottom is fine.
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

    class Graph3DWidget(anywidget.AnyWidget):
        """3D force-directed graph (three.js via 3d-force-graph). z is pinned to
        each node's BFS depth, so the graph stacks in layers; x/y are force-laid
        out. Node dicts are {"id","name","color","size","depth"}; edges are
        {"source","target"}. Toggle two_d to flatten to a 2D graph."""

        _esm = r"""
    import ForceGraph3D from "https://esm.sh/3d-force-graph@1.80.0";
    import { Raycaster, Vector2 } from "https://esm.sh/three@0.180";

    function render({ model, el }) {
      const container = document.createElement("div");
      container.className = "graph3d-widget-container";
      container.style.height = `${model.get("height") || 520}px`;
      el.appendChild(container);

      const nodeById = new Map();
      let didFit = false;

      // z position for a node: pinned to its depth layer in 3D, flat (0) in 2D
      function zFor(node) {
        if (model.get("two_d")) return 0;
        return node.depth * (model.get("layer_gap") || 60);
      }

      function buildData() {
        const inNodes = model.get("nodes") || [];
        const inEdges = model.get("edges") || [];
        const ids = new Set(inNodes.map((n) => String(n.id)));

        for (const id of [...nodeById.keys()]) {
          if (!ids.has(id)) nodeById.delete(id);
        }

        const nodes = inNodes.map((n) => {
          const id = String(n.id);
          let o = nodeById.get(id);
          if (!o) {
            o = { id };
            nodeById.set(id, o);
          }
          o.name = n.name != null ? String(n.name) : id;
          o.color = n.color || "#4c78a8";
          o.size = n.size != null ? n.size : 6;
          o.depth = n.depth != null ? n.depth : 0;
          // Pin only z (the true 3rd dimension) to depth; leave x AND y free so the
          // force lays out the full 2D shape and depth just extrudes it.
          o.fz = zFor(o);
          return o;
        });

        const links = inEdges
          .filter((e) => ids.has(String(e.source)) && ids.has(String(e.target)))
          .map((e) => ({ source: String(e.source), target: String(e.target) }));

        return { nodes, links };
      }

      const Graph = ForceGraph3D()(container)
        .backgroundColor("rgba(0,0,0,0)")
        .numDimensions(3)
        .nodeColor((n) => n.color)
        .nodeVal((n) => n.size)
        .nodeLabel((n) => n.name)
        .nodeOpacity(0.95)
        .nodeRelSize(5)
        .linkColor(() => "#b4b4b4")
        .linkOpacity(0.8)
        .linkWidth(2)
        .enablePointerInteraction(true)
        .width(container.clientWidth || 600)
        .height(model.get("height") || 520)
        .onNodeDrag((node) => {
          // lock dragging to the node's depth layer: slide in x/y only, never z
          node.fz = zFor(node);
        })
        .onNodeDragEnd((node) => {
          node.fz = zFor(node);
        });

      // Switch between the flat 2D graph and the depth-stacked 3D view.
      function applyMode() {
        const flat = model.get("two_d");
        Graph.numDimensions(flat ? 2 : 3);
        const controls = Graph.controls();
        if (controls) {
          if ("noRotate" in controls) controls.noRotate = flat;
          if ("enableRotate" in controls) controls.enableRotate = !flat;
        }
        const data = Graph.graphData();
        if (data && data.nodes) data.nodes.forEach((n) => (n.fz = zFor(n)));
        if (flat) Graph.cameraPosition({ x: 0, y: 0, z: 220 }, { x: 0, y: 0, z: 0 }, 500);
        else Graph.cameraPosition({ x: 120, y: 90, z: 180 }, { x: 0, y: 0, z: 0 }, 500);
        if (nodeById.size > 0) setTimeout(() => Graph.zoomToFit(500, 40), 650);
      }

      function applyDirected() {
        const directed = model.get("directed");
        Graph.linkDirectionalArrowLength(directed ? 3.5 : 0).linkDirectionalArrowRelPos(1);
      }

      function applyTheme() {
        const dark = model.get("dark");
        Graph.backgroundColor(dark ? "#0e1117" : "rgba(0,0,0,0)");
        Graph.linkColor(() => (dark ? "#8a8a8a" : "#b4b4b4"));
        container.style.background = dark ? "#0e1117" : "transparent";
      }

      function update() {
        Graph.graphData(buildData());
        applyDirected();
        if (!didFit && nodeById.size > 0) {
          didFit = true;
          setTimeout(() => Graph.zoomToFit(600, 40), 1200);
        }
      }

      update();
      applyDirected();
      applyTheme();
      applyMode();

      // keep the renderer sized to the measured container so pointer picking maps
      // cursor to scene correctly (a width-0 init makes raycasts miss)
      function sizeToContainer() {
        const w = container.clientWidth;
        const h = container.clientHeight || model.get("height") || 520;
        if (w > 0) Graph.width(w);
        if (h > 0) Graph.height(h);
      }
      requestAnimationFrame(sizeToContainer);
      const ro = new ResizeObserver(sizeToContainer);
      ro.observe(container);

      // Own raycaster for hover/click: the library's built-in picking doesn't fire
      // reliably in this embed, so we hit-test the scene and drive the traits.
      const raycaster = new Raycaster();
      const pointer = new Vector2();
      let hoverId = null;
      let pickScheduled = false;
      const canvas = Graph.renderer().domElement;

      function pickNodeAt(clientX, clientY) {
        const rect = canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return null;
        pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(pointer, Graph.camera());
        const hits = raycaster.intersectObjects(Graph.scene().children, true);
        for (const hit of hits) {
          let o = hit.object;
          while (o) {
            if (o.__graphObjType === "node" && o.__data) return o.__data;
            o = o.parent;
          }
        }
        return null;
      }

      function onPointerMove(e) {
        if (pickScheduled) return;
        pickScheduled = true;
        const { clientX, clientY } = e;
        requestAnimationFrame(() => {
          pickScheduled = false;
          const node = pickNodeAt(clientX, clientY);
          if (!node) return;
          const id = String(node.id);
          if (id === hoverId) return;
          hoverId = id;
          model.set("hovered_node", id);
          model.save_changes();
        });
      }

      function onClick(e) {
        const node = pickNodeAt(e.clientX, e.clientY);
        if (!node) return;
        const id = String(node.id);
        hoverId = id;
        model.set("hovered_node", id);
        model.set("selected_nodes", [id]);
        model.save_changes();
      }

      canvas.addEventListener("pointermove", onPointerMove);
      canvas.addEventListener("click", onClick);

      model.on("change:nodes", update);
      model.on("change:edges", update);
      model.on("change:layer_gap", update);
      model.on("change:directed", applyDirected);
      model.on("change:dark", applyTheme);
      model.on("change:two_d", applyMode);

      return () => {
        model.off("change:nodes", update);
        model.off("change:edges", update);
        model.off("change:layer_gap", update);
        model.off("change:directed", applyDirected);
        model.off("change:dark", applyTheme);
        model.off("change:two_d", applyMode);
        canvas.removeEventListener("pointermove", onPointerMove);
        canvas.removeEventListener("click", onClick);
        ro.disconnect();
        if (Graph._destructor) Graph._destructor();
      };
    }

    export default { render };
    """

        _css = r"""
    .graph3d-widget-container {
      width: 100%;
      min-height: 200px;
      position: relative;
      overflow: hidden;
      border-radius: 6px;
      background: var(--marimo-background, #ffffff);
    }
    """

        nodes = traitlets.List([]).tag(sync=True)
        edges = traitlets.List([]).tag(sync=True)
        directed = traitlets.Bool(False).tag(sync=True)
        dark = traitlets.Bool(False).tag(sync=True)
        two_d = traitlets.Bool(False).tag(sync=True)
        height = traitlets.Int(520).tag(sync=True)
        layer_gap = traitlets.Float(60.0).tag(sync=True)
        hovered_node = traitlets.Unicode(None, allow_none=True).tag(sync=True)
        selected_nodes = traitlets.List([]).tag(sync=True)

    return Graph3DWidget, SokobanWidget


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["marimo", "numpy", "matplotlib", "anywidget", "traitlets"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 🏖️ Earth Mover's Distance, made of sand

    The **Earth Mover's Distance** (1-Wasserstein) is the minimum *work* to
    turn one pile of sand into another — where work is *mass × distance moved*.

    Pour sand into the two boxes below to sculpt a **source** and a **target**
    pile (each box shows the other's outline as a line). Then read off the exact
    EMD and watch, in the coloured histograms, exactly where each grain moves.
    """)
    return


@app.cell
def _(anywidget, traitlets):
    class SandInput(anywidget.AnyWidget):
        """Two falling-sand boxes for sculpting a source and a target distribution.

        Pour sand into each box with the mouse; grains fall and pile up via a little
        cellular automaton. Each box overlays the *other* box's current outline as a
        line so you can shape them relative to each other. The per-column pile heights
        are synced back to Python as ``source`` and ``target``.
        """

        _esm = r"""
    // Two falling-sand boxes used as an input for the source / target distributions.
    //
    // Each box is a little cellular automaton: cells hold sand, sand falls straight
    // down when it can, otherwise slides diagonally, and piles up. Pour or vacuum
    // with the mouse. Each box holds at most `budget` grains, so both piles carry the
    // same total mass — a genuine distribution — once full. Per-column pile heights
    // (and the grain count) sync back to Python. Each box also draws the OTHER box's
    // current outline as a line.

    const CELL = 5;

    const BOXES = [
      { key: "source", label: "source", color: "#2b6cff" },
      { key: "target", label: "target", color: "#ff8a3d" },
    ];

    function render({ model, el }) {
      el.classList.add("sand-root");

      const cols = model.get("cols");
      const rows = model.get("rows");
      const budget = model.get("budget");
      const W = cols * CELL;
      const H = rows * CELL;

      let mode = "pour"; // "pour" | "vacuum"

      // --- top toolbar: mode toggle + balance readout ------------------------
      const toolbar = document.createElement("div");
      toolbar.className = "sand-toolbar";
      const pourBtn = document.createElement("button");
      pourBtn.className = "sand-mode active";
      pourBtn.textContent = "⛱ pour";
      const vacBtn = document.createElement("button");
      vacBtn.className = "sand-mode";
      vacBtn.textContent = "🧹 vacuum";
      const balance = document.createElement("span");
      balance.className = "sand-balance";
      toolbar.append(pourBtn, vacBtn, balance);
      el.appendChild(toolbar);

      function setMode(m) {
        mode = m;
        pourBtn.classList.toggle("active", m === "pour");
        vacBtn.classList.toggle("active", m === "vacuum");
      }
      pourBtn.addEventListener("click", () => setMode("pour"));
      vacBtn.addEventListener("click", () => setMode("vacuum"));

      const panels = document.createElement("div");
      panels.className = "sand-panels";
      el.appendChild(panels);

      const boxes = {};

      for (const def of BOXES) {
        const panel = document.createElement("div");
        panel.className = "sand-panel";

        const head = document.createElement("div");
        head.className = "sand-head";
        const dot = document.createElement("span");
        dot.className = "sand-dot";
        dot.style.background = def.color;
        const name = document.createElement("span");
        name.textContent = def.label;
        const count = document.createElement("span");
        count.className = "sand-count";
        const clear = document.createElement("button");
        clear.className = "sand-clear";
        clear.textContent = "clear";
        head.append(dot, name, count, clear);

        const canvas = document.createElement("canvas");
        canvas.className = "sand-canvas";
        canvas.width = W;
        canvas.height = H;

        panel.append(head, canvas);
        panels.appendChild(panel);

        const box = {
          def,
          canvas,
          ctx: canvas.getContext("2d"),
          countEl: count,
          grid: new Uint8Array(cols * rows), // 0 = empty, else shade (128..255)
          pouring: false,
          pourCol: 0,
          lastHeights: null,
          stable: 0,
          dirty: false,
        };
        boxes[def.key] = box;

        clear.addEventListener("click", () => {
          box.grid.fill(0);
          box.dirty = true;
          box.stable = 0;
        });

        const columnAt = (e) => {
          const rect = canvas.getBoundingClientRect();
          const x = (e.clientX - rect.left) * (canvas.width / rect.width);
          return Math.max(0, Math.min(cols - 1, Math.floor(x / CELL)));
        };
        canvas.addEventListener("pointerdown", (e) => {
          box.pouring = true;
          box.pourCol = columnAt(e);
          canvas.setPointerCapture(e.pointerId);
        });
        canvas.addEventListener("pointermove", (e) => {
          if (box.pouring) box.pourCol = columnAt(e);
        });
        const stop = () => { box.pouring = false; };
        canvas.addEventListener("pointerup", stop);
        canvas.addEventListener("pointercancel", stop);
      }

      // --- cellular automaton -------------------------------------------------
      function step(grid) {
        for (let r = rows - 2; r >= 0; r--) {
          const ltr = Math.random() < 0.5;
          for (let k = 0; k < cols; k++) {
            const c = ltr ? k : cols - 1 - k;
            const i = r * cols + c;
            const v = grid[i];
            if (!v) continue;
            const below = i + cols;
            if (!grid[below]) { grid[below] = v; grid[i] = 0; continue; }
            const dl = c > 0 && !grid[below - 1];
            const dr = c < cols - 1 && !grid[below + 1];
            if (dl && dr) {
              if (Math.random() < 0.5) grid[below - 1] = v;
              else grid[below + 1] = v;
              grid[i] = 0;
            } else if (dl) { grid[below - 1] = v; grid[i] = 0; }
            else if (dr) { grid[below + 1] = v; grid[i] = 0; }
          }
        }
      }

      function count(box) {
        let n = 0;
        for (let i = 0; i < box.grid.length; i++) if (box.grid[i]) n++;
        return n;
      }

      function pour(box) {
        if (count(box) >= budget) return; // cap: both boxes share the same budget
        const c = box.pourCol;
        for (const dc of [-1, 0, 0, 1]) {
          const cc = c + dc;
          if (cc < 0 || cc >= cols) continue;
          if (!box.grid[cc]) box.grid[cc] = 200 + ((Math.random() * 55) | 0);
        }
      }

      function vacuum(box) {
        // suck the topmost grain out of the poured column and its neighbours
        for (const dc of [-1, 0, 1]) {
          const cc = box.pourCol + dc;
          if (cc < 0 || cc >= cols) continue;
          for (let r = 0; r < rows; r++) {
            const i = r * cols + cc;
            if (box.grid[i]) { box.grid[i] = 0; break; }
          }
        }
      }

      function heights(box) {
        const h = new Array(cols).fill(0);
        for (let c = 0; c < cols; c++) {
          let n = 0;
          for (let r = 0; r < rows; r++) if (box.grid[r * cols + c]) n++;
          h[c] = n;
        }
        return h;
      }

      function draw(box, otherHeights) {
        const { ctx } = box;
        ctx.clearRect(0, 0, W, H);

        const [rC, gC, bC] = hexToRgb(box.def.color);
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const v = box.grid[r * cols + c];
            if (!v) continue;
            ctx.globalAlpha = v / 255;
            ctx.fillStyle = `rgb(${rC},${gC},${bC})`;
            ctx.fillRect(c * CELL, r * CELL, CELL, CELL);
          }
        }
        ctx.globalAlpha = 1;

        ctx.strokeStyle = otherColor(box.def.key);
        ctx.lineWidth = 2;
        ctx.lineJoin = "round";
        ctx.beginPath();
        for (let c = 0; c < cols; c++) {
          const y = (rows - otherHeights[c]) * CELL;
          const x = c * CELL + CELL / 2;
          if (c === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      function otherColor(key) {
        return key === "source" ? BOXES[1].color : BOXES[0].color;
      }

      function hexToRgb(hex) {
        const n = parseInt(hex.slice(1), 16);
        return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
      }

      function sync(box) {
        model.set(box.def.key, heights(box).map((v) => v));
        model.save_changes();
      }

      // --- main loop ----------------------------------------------------------
      let anim = null;

      function frame() {
        for (const key of ["source", "target"]) {
          const box = boxes[key];
          if (box.pouring) {
            if (mode === "pour") pour(box);
            else vacuum(box);
            box.dirty = true;
            box.stable = 0;
          }
          step(box.grid);
        }

        const cs = count(boxes.source);
        const ct = count(boxes.target);
        boxes.source.countEl.textContent = `${cs} / ${budget}`;
        boxes.target.countEl.textContent = `${ct} / ${budget}`;
        balance.textContent =
          cs === 0 || ct === 0 ? "" :
          cs === ct ? "⚖ balanced" :
          cs > ct ? `source heavier by ${cs - ct}` : `target heavier by ${ct - cs}`;

        const hs = heights(boxes.source);
        const ht = heights(boxes.target);
        draw(boxes.source, ht);
        draw(boxes.target, hs);

        for (const [key, h] of [["source", hs], ["target", ht]]) {
          const box = boxes[key];
          if (!box.dirty) continue;
          const same = box.lastHeights &&
            box.lastHeights.length === h.length &&
            box.lastHeights.every((v, idx) => v === h[idx]);
          box.stable = same ? box.stable + 1 : 0;
          box.lastHeights = h;
          if (!box.pouring && box.stable >= 6) {
            sync(box);
            box.dirty = false;
          }
        }

        anim = requestAnimationFrame(frame);
      }
      frame();

      return () => { if (anim) cancelAnimationFrame(anim); };
    }

    export default { render };
    """

        _css = r"""
    .sand-root {
      font-family: ui-sans-serif, system-ui, sans-serif;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .sand-toolbar {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
    }

    .sand-mode {
      border: 1px solid #d0d3db;
      border-radius: 8px;
      background: #fff;
      padding: 4px 12px;
      font-size: 13px;
      cursor: pointer;
    }
    .sand-mode.active {
      background: #1a1d24;
      border-color: #1a1d24;
      color: #fff;
    }

    .sand-balance {
      margin-left: 6px;
      color: #6b7280;
      font-variant-numeric: tabular-nums;
    }

    .sand-panels {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
    }

    .sand-count {
      color: #9096a1;
      font-variant-numeric: tabular-nums;
      font-size: 12px;
      margin-left: 2px;
    }

    .sand-panel {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .sand-head {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      color: #1a1d24;
    }

    .sand-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }

    .sand-clear {
      margin-left: auto;
      border: 1px solid #d0d3db;
      border-radius: 6px;
      background: #fff;
      font-size: 12px;
      padding: 2px 8px;
      cursor: pointer;
    }
    .sand-clear:hover {
      background: #f4f5f7;
    }

    .sand-canvas {
      border: 1px solid #e3e5ea;
      border-radius: 8px;
      background: #fbfbfc;
      touch-action: none;
      cursor: crosshair;
      display: block;
    }
    """

        cols = traitlets.Int(64).tag(sync=True)
        rows = traitlets.Int(48).tag(sync=True)

        # Shared per-box mass budget: you can pour at most this many grains into each
        # box, so both piles carry the same total mass (a real distribution) once full.
        budget = traitlets.Int(500).tag(sync=True)

        source = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
        target = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)

    return (SandInput,)


@app.cell
def _(SandInput, mo):
    sand = mo.ui.anywidget(SandInput())
    sand
    return (sand,)


@app.cell
def _(np, sand):
    source = np.asarray(sand.source, float)
    target = np.asarray(sand.target, float)
    return source, target


@app.cell
def _(np, source, target):
    def emd_1d(s, t):
        """Exact 1-Wasserstein: integrated gap between the two CDFs."""
        s = np.asarray(s, float)
        t = np.asarray(t, float)
        if s.sum() <= 0 or t.sum() <= 0:
            return float("nan")
        s = s / s.sum(); t = t / t.sum()
        return float(np.abs(np.cumsum(s) - np.cumsum(t)).sum() / len(s))

    emd = emd_1d(source, target)
    return (emd,)


@app.cell(hide_code=True)
def _(emd, mo, np):
    _txt = "pour sand into both boxes" if np.isnan(emd) else f"`{emd:.4f}`"
    mo.md(f"**EMD (exact, from the CDF gap):** {_txt}")
    return


@app.cell
def _():
    # Ordinal blue ramp (dataviz palette, light->dark, validated). Colour encodes
    # *rank in the sorted mass* (quantile), NOT x-position — so a split pile isn't
    # "red on the right" just for sitting there. Same shade = the same sand.
    RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
    return (RAMP,)


@app.cell(hide_code=True)
def _(RAMP, np, plt, source, target):
    def transport_histogram(s, t, n_pixels=480):
        s = np.asarray(s, float)
        t = np.asarray(t, float)
        n = len(s)
        if s.sum() <= 0 or t.sum() <= 0:
            fig, ax = plt.subplots(figsize=(7, 1.4))
            ax.text(0.5, 0.5, "give both piles some sand",
                    ha="center", va="center", fontsize=11, color="#888")
            ax.axis("off")
            return fig

        # Resample both histograms onto a fine pixel grid and normalize to unit mass.
        idx = np.minimum(np.arange(n_pixels) * n // n_pixels, n - 1)
        sg = s[idx]; sg = sg / sg.sum()
        tg = t[idx]; tg = tg / tg.sum()
        pos = (np.arange(n_pixels) + 0.5) / n_pixels     # spatial position of each pixel
        width = 1.0 / n_pixels

        # Colour by quantile block: split the total mass into equal-mass chunks and
        # give each a shade. Under monotone transport the k-th mass-chunk of the
        # source is the k-th mass-chunk of the target, so shades line up 1:1.
        K = len(RAMP)
        Fs, Ft = np.cumsum(sg), np.cumsum(tg)
        block_s = np.clip(((Fs - 0.5 * sg) * K).astype(int), 0, K - 1)
        block_t = np.clip(((Ft - 0.5 * tg) * K).astype(int), 0, K - 1)
        before_color = [RAMP[b] for b in block_s]
        after_color = [RAMP[b] for b in block_t]

        # Bar heights = density on a SHARED scale, so both plots hold the same total
        # area (mass is conserved): a thin spike is tall, a broad pile is short.
        hmax = max(sg.max(), tg.max())
        h_src = sg / hmax
        h_tgt = tg / hmax

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(7, 3.4), sharex=True, constrained_layout=True
        )
        ax_top.bar(pos, h_src, width=width, color=before_color, align="center")
        ax_bot.bar(pos, h_tgt, width=width, color=after_color, align="center")

        for ax, title in ((ax_top, "before  ·  source, coloured by mass-rank"),
                          (ax_bot, "after  ·  target, same shade = same sand")):
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            ax.set_yticks([]); ax.set_title(title, fontsize=9, loc="left")
            for spine in ax.spines.values():
                spine.set_visible(False)
        ax_bot.set_xticks([])
        return fig

    transport_histogram(source, target)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why leftmost → leftmost?

    It's all on **one axis**: each grain hops from its spot in the source to a spot
    in the target, and the **length of the segment is the distance it travels** —
    a real length on the x-axis (the line just thickens with the cost).

    Swap the two destinations below. With **$|\Delta x|$** the swap can come out
    *exactly tied* — one long hop + one short hop equals two medium ones, your
    instinct was right. But with **$(\Delta x)^2$** the square punishes the long
    hop (watch its line fatten), so swapping is *strictly* worse and the sorted,
    leftmost-to-leftmost plan is the unique winner.
    """)
    return


@app.cell
def _(mo):
    cost = mo.ui.radio(
        options=["absolute  |Δx|   (W₁)", "squared  (Δx)²   (W₂)"],
        value="absolute  |Δx|   (W₁)",
        inline=True,
    )
    cross = mo.ui.switch(label="swap the two destinations")
    mo.hstack([cost, cross], justify="start", gap=2)
    return cost, cross


@app.cell(hide_code=True)
def _(RAMP, cost, cross, np, plt):
    # Real coordinates (not 0–1) so the distances read as plain numbers: a hop
    # from 2 to 8 is just 6, right there on the axis.
    src = np.array([2.0, 4.0])
    tgt = np.array([8.0, 10.0])
    lo, hi = 0.0, 12.0
    squared = cost.value.startswith("squared")
    dist = (lambda a, b: (a - b) ** 2) if squared else (lambda a, b: abs(a - b))

    perm = [1, 0] if cross.value else [0, 1]
    total = sum(dist(src[i], tgt[perm[i]]) for i in range(2))
    sorted_total = sum(dist(src[i], tgt[i]) for i in range(2))
    colors = [RAMP[0], RAMP[-1]]
    max_hop = (8.0 ** 2) if squared else 8.0  # to scale line thickness by cost

    # Everything lives on ONE axis (the position line) so a LENGTH on it is a real
    # distance. Each grain's hop is a straight, HORIZONTAL segment whose width is
    # exactly |Δx| — the distance it travels; thin drop-lines tie each end to its
    # spot on the axis. The line thickens with the cost, so squaring fattens the
    # long hop without ever distorting the honest length.
    fig, ax = plt.subplots(figsize=(7, 2.6), constrained_layout=True)
    base = 0.14
    ax.plot([lo, hi], [base, base], color="#c9ccd4", lw=1.5, zorder=0)
    for i in range(2):
        j = perm[i]
        a, b = src[i], tgt[j]
        hop = dist(a, b)
        lane = base + 0.24 + 0.26 * i
        ax.plot([a, a], [base, lane], color=colors[i], lw=1, alpha=0.4, zorder=1)
        ax.plot([b, b], [base, lane], color=colors[i], lw=1, alpha=0.4, zorder=1)
        ax.plot([a, b], [lane, lane], color=colors[i], lw=2 + 9 * hop / max_hop,
                solid_capstyle="butt", zorder=2)
        ax.scatter([a, b], [base, base], s=200, marker="s", color=colors[i],
                   zorder=3, edgecolors="white", linewidths=1.5)
        ax.annotate(f"{hop:g}", ((a + b) / 2, lane + 0.05), fontsize=9,
                    ha="center", va="bottom", color=colors[i])
    ax.text(3.0, base - 0.10, "source", fontsize=9, ha="center", color="#555")
    ax.text(9.0, base - 0.10, "target", fontsize=9, ha="center", color="#555")

    if not cross.value:
        verdict = "— the sorted plan (leftmost → leftmost)"
    elif abs(total - sorted_total) < 1e-9:
        verdict = "= sorted plan: a genuine tie (only |Δx| allows this)"
    else:
        verdict = f"> sorted plan by {total - sorted_total:g} — strictly worse"
    metric = "(Δx)²" if squared else "|Δx|"
    ax.set_title(f"total cost  ({metric})  =  {total:g}   {verdict}",
                 fontsize=11, loc="left")
    ax.set_xlim(lo - 0.3, hi + 0.3); ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(range(0, 13, 2))
    ax.tick_params(axis="x", length=0, labelsize=9, colors="#888")
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Now, the formula

    EMD is the *minimum* total work over **every** way to move the sand — every
    transport plan $\gamma$, where $\gamma_{ij}\ge 0$ is how much mass goes from
    bin $i$ to bin $j$:

    $$
    \mathrm{EMD}(P,Q)\;=\;\min_{\gamma\in\Pi(P,Q)}\;\sum_{i,j}\gamma_{ij}\,\lvert x_i-y_j\rvert
    $$

    The plan has to actually turn $P$ into $Q$ — everything leaving bin $i$ sums to
    $p_i$, everything arriving at $j$ sums to $q_j$:

    $$
    \sum_j \gamma_{ij}=p_i,\qquad \sum_i \gamma_{ij}=q_j .
    $$

    The per-unit cost is the distance moved $\lvert x_i-y_j\rvert$ — this is the
    **1-Wasserstein** $W_1$; square it to $\lvert x_i-y_j\rvert^2$ and you get
    $W_2$. In 1D the minimum has a closed form: the area between the two CDFs
    (exactly the EMD number above),

    $$
    W_1(P,Q)\;=\;\int_0^1 \lvert F_P(x)-F_Q(x)\rvert \, dx .
    $$
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import anywidget
    import traitlets

    return anywidget, mo, np, plt, traitlets


if __name__ == "__main__":
    app.run()

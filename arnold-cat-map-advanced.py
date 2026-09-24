# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "marimo",
#     "numpy==2.5.3",
#     "anywidget==0.11.0",
#     "traitlets==5.16.1",
#     "matplotlib==3.11.1",
#     "wigglystuff==0.5.32",
#     "marimo-studio==0.1.1"
# ]
#
# [tool.marimo-studio]
# default = "walkthrough"
#
# [tool.marimo-studio.cells]
# cell-11 = {ref = "cell:v1:3a476a7ddd818b4cc7eb3b588a0111ef6f059609b9bfe397bf182e416a558784:94f1b5057f2de0525d4adf8c403147fe0b93b68d74eeb331184e247276d6ba79:0"}
# cell-13 = {ref = "cell:v1:211986abd3354fbdc942d2293e9b44468ef936c9c2a54ca31b5b70c15b329b61:33d7883b20117127941600f1398e9dd7f2ee86e5e5eec70fe80d742fc9dfd0be:0"}
# cell-14 = {ref = "cell:v1:044b3c394e53353f14deac97bc9921bf12dafc7e864b05f61b23d2102ccd0b6e:bb7d8bd794a432814c118155cb2568cc24496fd97e7a39780d5332760f43c629:0"}
# cell-15 = {ref = "cell:v1:4bc3d48c30ef25c5041df8e73854d21e8e6737690e2b3dc60dd30a7377f93423:a04588d98ed877e54a1f88b5cc368534be7f1a6bb937a96beb618a77962ba23c:0"}
# cell-16 = {ref = "cell:v1:cb9499c7a268355c9ed29f98586fa44a215d4131022a6432d4b20fe726e26cc8:cb9499c7a268355c9ed29f98586fa44a215d4131022a6432d4b20fe726e26cc8:0"}
# cell-17 = {ref = "cell:v1:ce93c8dff0a273c998a97dbf841354198b31245c6928b0af9562cedb15eaaab1:242b286861cceadffd5c10fb8b2d4de9ee863aa5d3c792d13c1ca0165283a310:0"}
# cell-19 = {ref = "cell:v1:bbae17c4c7ee72805daf4bf953a28e5de202881ade85b62e8c7a5bfdc21d4c8f:2746382c53ec721e1e7bbd13baf41866407a6f458ed4032a05d33f68129a67ef:0"}
# cell-20 = {ref = "cell:v1:989b4c065d4800554e2c9138f3253c02be0ecbc7bf1f6bd77939c77a66f53503:e6bd6f7385c0feaead2250a11625afae215d6369e46d20cb86f8a5cb2adcb8a7:0"}
# cell-23 = {ref = "cell:v1:31aca5968e424c3652d837aea50f1cb30e5c8ef5068dd61aaf18e402b38988fd:c8936aaab6b14363d0ba4b25a765436d2b21012e2fcb8c0c7354f55e12cd0a6e:0"}
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def intro_md(mo):
    mo.md(r"""
    # 🐱 Arnold's cat map

    Take an $N\times N$ grid and move the tile at $(x, y)$ to

    $$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} x \\ y \end{pmatrix} \pmod{N}.$$

    This operation has a fun twist: if you keep applying it the tiles must
    eventually return home.That recurrence **period depends only on $N$**,
    not on the picture.

    Fun aside: $\begin{pmatrix} 2 & 1 \\ 1 & 1
    \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}^2$ and there are many (!) matrices that you can come up with that have this recurrence pattern.

    Hit **play** and watch the rainbow scramble — then snap back. Feel free to play around with your own matrices too!
    """)
    return


@app.cell(hide_code=True)
def _():
    import colorsys
    import math

    import anywidget
    import matplotlib.pyplot as plt
    import numpy as np
    import traitlets
    from wigglystuff import ChartPuck

    def hsv_hex(h, s, v):
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    def lerp_hex(stops, t):
        t = min(max(t, 0.0), 1.0) * (len(stops) - 1)
        i = min(int(t), len(stops) - 2)
        f = t - i
        r, g, b = (stops[i][j] + f * (stops[i + 1][j] - stops[i][j]) for j in range(3))
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

    PALETTES = {
        "sunset": [(45, 12, 82), (130, 30, 120), (220, 70, 90), (250, 150, 60), (255, 214, 102)],
        "ocean": [(4, 30, 63), (15, 82, 122), (24, 138, 160), (86, 196, 177), (204, 240, 220)],
        "viridis": [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
    }

    def palette(name, t):
        if name == "rainbow":
            return hsv_hex(t, 0.6, 0.92)
        return lerp_hex(PALETTES[name], t)

    return ChartPuck, anywidget, math, np, palette, plt, traitlets


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class CatMapWidget(anywidget.AnyWidget):
        n = traitlets.Int(29).tag(sync=True)
        colors = traitlets.List(traitlets.Unicode()).tag(sync=True)
        period = traitlets.Int(1).tag(sync=True)
        bijective = traitlets.Bool(True).tag(sync=True)
        a = traitlets.Float(2.0).tag(sync=True)
        b = traitlets.Float(1.0).tag(sync=True)
        c = traitlets.Float(1.0).tag(sync=True)
        d = traitlets.Float(1.0).tag(sync=True)
        speed = traitlets.Float(3.0).tag(sync=True)
        playing = traitlets.Bool(True).tag(sync=True)
        show_tiling = traitlets.Bool(False).tag(sync=True)
        selected = traitlets.Int(-1).tag(sync=True)

        _esm = r"""
        function render({ model, el }) {
          el.style.textAlign = "center";
          const wrap = document.createElement("div");
          wrap.style.display = "inline-block";
          const canvas = document.createElement("canvas");
          canvas.style.cursor = "pointer";
          wrap.appendChild(canvas);

          // Controls that live inside the widget: play/pause, a scrub timeline,
          // a tiling toggle, and a button to drop the followed tile.
          function styleBtn(b) {
            b.style.font = "13px ui-monospace, monospace";
            b.style.padding = "4px 10px";
            b.style.borderRadius = "6px";
            b.style.border = "1px solid rgba(128,128,128,0.5)";
            b.style.background = "transparent";
            b.style.color = "inherit";
            b.style.cursor = "pointer";
            return b;
          }
          const controls = document.createElement("div");
          controls.style.display = "flex";
          controls.style.alignItems = "center";
          controls.style.gap = "10px";
          controls.style.margin = "12px auto 0";
          const playBtn = styleBtn(document.createElement("button"));
          const nextBtn = styleBtn(document.createElement("button"));
          nextBtn.textContent = "▶ to next";
          const timeline = document.createElement("input");
          timeline.type = "range";
          timeline.min = "0";
          timeline.step = "any";
          timeline.value = "0";
          timeline.style.flex = "1";
          timeline.style.cursor = "pointer";
          const tileBtn = styleBtn(document.createElement("button"));
          const clearBtn = styleBtn(document.createElement("button"));
          clearBtn.textContent = "clear tile";
          const stepBack = styleBtn(document.createElement("button"));
          stepBack.textContent = "◀ chapter";
          const stepFwd = styleBtn(document.createElement("button"));
          stepFwd.textContent = "chapter ▶";
          controls.appendChild(playBtn);
          controls.appendChild(nextBtn);
          controls.appendChild(stepBack);
          controls.appendChild(timeline);
          controls.appendChild(stepFwd);
          controls.appendChild(tileBtn);
          controls.appendChild(clearBtn);

          const label = document.createElement("div");
          label.style.fontFamily = "ui-monospace, monospace";
          label.style.marginTop = "14px";
          label.style.opacity = "0.85";
          el.appendChild(wrap);
          el.appendChild(controls);
          el.appendChild(label);
          const ctx = canvas.getContext("2d");

          const MIN_SIZE = 360;   // canvas never shrinks below this, even for small n
          let PX = 18;
          const SPLIT = 0.5;   // first half of a step = stretch, second half = fold
          const HOLD_MS = 1300; // pause on the reassembled image each full cycle
          // pos is a continuous timeline position in [0, period]: its integer part
          // is the iteration, its fractional part the stretch/fold phase.
          let n, colors, bx, by, gx, gy, px, py, period, pos, hold, loadedIter;
          let curIter, curPhase, sc, ox0, oy0;
          let rafId = null, lastT = 0, stopAt = null;

          const ease = (t) => t * t * (3 - 2 * t);
          const mod = (a, m) => ((a % m) + m) % m;

          function commitOne() {
            // Advance every tile one full step: (x, y) -> A·(x, y) mod n.
            const a = model.get("a"), b = model.get("b");
            const c = model.get("c"), d = model.get("d");
            for (let i = 0; i < n * n; i++) {
              const x = gx[i], y = gy[i];
              gx[i] = mod(a * x + b * y, n);
              gy[i] = mod(c * x + d * y, n);
            }
          }

          function gotoIter(target) {
            // Put gx/gy at the start-of-iteration state for `target`. Forward is
            // incremental; going back rewinds to the base grid then replays.
            if (target === loadedIter) return;
            if (target < loadedIter) {
              for (let i = 0; i < n * n; i++) { gx[i] = bx[i]; gy[i] = by[i]; }
              loadedIter = 0;
            }
            for (let s = loadedIter; s < target; s++) commitOne();
            loadedIter = target;
          }

          function syncState() {
            if (model.get("bijective")) {
              curIter = ((Math.floor(pos) % period) + period) % period;
            } else {
              curIter = Math.floor(pos);
            }
            curPhase = pos - Math.floor(pos);
            gotoIter(curIter);
          }

          function setup() {
            n = model.get("n");
            colors = model.get("colors");
            period = Math.max(model.get("period"), 1);
            PX = Math.max(18, Math.ceil(MIN_SIZE / n));
            const size = n * PX;
            canvas.width = size;
            canvas.height = size;
            bx = new Float64Array(n * n);
            by = new Float64Array(n * n);
            gx = new Float64Array(n * n);
            gy = new Float64Array(n * n);
            px = new Float64Array(n * n);
            py = new Float64Array(n * n);
            for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
              const i = y * n + x;
              bx[i] = x; by[i] = y;
              gx[i] = x; gy[i] = y;
            }
            loadedIter = 0;
            pos = 0;
            hold = 0;
            const biject = model.get("bijective");
            timeline.max = String(biject ? period : 1);
            timeline.disabled = !biject;
            syncState();
            draw();
          }

          function draw() {
            // Split each step in two: first the pure matmul (stretch, no wrap),
            // then the mod (fold the overhang back into the box by ±n).
            let stretch, fold;
            if (curPhase < SPLIT) { stretch = ease(curPhase / SPLIT); fold = 0; }
            else { stretch = 1; fold = ease((curPhase - SPLIT) / (1 - SPLIT)); }

            // Compute every tile's current position, tracking a bounding box so
            // we can auto-frame — this zooms out for the stretch and back in for
            // the fold, showing exactly where the folded-in pieces come from.
            const a = model.get("a"), b = model.get("b");
            const c = model.get("c"), d = model.get("d");
            let minx = 0, miny = 0, maxx = n, maxy = n;
            for (let i = 0; i < n * n; i++) {
              const x = gx[i], y = gy[i];
              const ux = a * x + b * y, uy = c * x + d * y;   // A·(x,y), un-wrapped
              let cx, cy;
              if (fold === 0) {
                cx = x + stretch * (ux - x);
                cy = y + stretch * (uy - y);
              } else {
                cx = ux - fold * (ux - mod(ux, n));      // slide overhang back by ±n
                cy = uy - fold * (uy - mod(uy, n));
              }
              px[i] = cx; py[i] = cy;
              if (cx < minx) minx = cx; else if (cx > maxx) maxx = cx;
              if (cy < miny) miny = cy; else if (cy > maxy) maxy = cy;
            }

            const V = Math.max(maxx - minx, maxy - miny) * 1.18;
            sc = (n * PX) / V;
            ox0 = (minx + maxx) / 2 - V / 2;
            oy0 = (miny + maxy) / 2 - V / 2;

            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.setTransform(sc, 0, 0, sc, -ox0 * sc, -oy0 * sc);

            const sel = model.get("selected");
            const size = 0.76, g = (1 - size) / 2;
            for (let i = 0; i < n * n; i++) {
              ctx.fillStyle = (sel >= 0 && i !== sel) ? "rgba(120,120,120,0.22)" : colors[i];
              ctx.fillRect(px[i] + g, py[i] + g, size, size);
            }
            // Redraw the followed tile on top, a touch larger and outlined.
            if (sel >= 0) {
              const s2 = 0.94, g2 = (1 - s2) / 2;
              ctx.fillStyle = colors[sel];
              ctx.fillRect(px[sel] + g2, py[sel] + g2, s2, s2);
              ctx.lineWidth = 2.5 / sc;
              ctx.strokeStyle = "rgba(0,0,0,0.85)";
              ctx.strokeRect(px[sel] + g2, py[sel] + g2, s2, s2);
            }

            // Optional: the full mod-N tiling — every periodic copy of the box
            // that the fold pulls overhang back from.
            if (model.get("show_tiling")) {
              ctx.lineWidth = 1 / sc;
              ctx.strokeStyle = "rgba(255,255,255,0.28)";
              // Draw the mod-N lattice across the whole visible frame, not just
              // where the tiles are, so every line reaches the canvas edge and no
              // leading box ever looks open. Frame spans [ox0, ox0+V] on each axis.
              ctx.beginPath();
              for (let gl = Math.ceil(ox0 / n) * n; gl <= ox0 + V; gl += n) {
                ctx.moveTo(gl, oy0); ctx.lineTo(gl, oy0 + V);
              }
              for (let gl = Math.ceil(oy0 / n) * n; gl <= oy0 + V; gl += n) {
                ctx.moveTo(ox0, gl); ctx.lineTo(ox0 + V, gl);
              }
              ctx.stroke();
              // Thin cell lines inside the home window, one per tile.
              ctx.strokeStyle = "rgba(255,255,255,0.14)";
              ctx.beginPath();
              for (let k = 1; k < n; k++) {
                ctx.moveTo(k, 0); ctx.lineTo(k, n);
                ctx.moveTo(0, k); ctx.lineTo(n, k);
              }
              ctx.stroke();
            }

            // The fixed [0,n)×[0,n) window — the actual image the map acts on.
            ctx.lineWidth = 1.5 / sc;
            ctx.strokeStyle = "rgba(255,255,255,0.6)";
            ctx.strokeRect(0, 0, n, n);

            ctx.setTransform(1, 0, 0, 1, 0, 0);

            // Reflect the animation on the controls (unless the user is scrubbing).
            if (document.activeElement !== timeline) timeline.value = String(pos);
            playBtn.textContent = !model.get("bijective")
              ? "↻ restart"
              : (model.get("playing") ? "⏸ pause" : "▶ play");
            tileBtn.textContent = model.get("show_tiling") ? "hide mod-N grid" : "show mod-N grid";

            let stage;
            if (Math.abs(curPhase - SPLIT) < 1e-9) stage = "fully stretched (before fold)";
            else if (curPhase === 0) stage = "folded home";
            else stage = curPhase < SPLIT ? "matmul (stretch)" : "mod n (fold)";
            if (model.get("bijective")) {
              const home = curIter === 0 && curPhase < SPLIT ? "   ← back to the original!" : "";
              label.textContent = `iteration ${curIter} / period ${period} · ${stage}${home}`;
            } else {
              label.textContent = `iteration ${curIter} · ${stage} · det shares a factor with ${n} — not invertible, won't reassemble`;
            }
          }

          function frame(t) {
            if (!lastT) lastT = t;
            const dt = t - lastT;
            lastT = t;
            if (model.get("playing")) {
              if (hold > 0) {
                hold -= dt;               // linger on the reassembled image
              } else {
                const interval = 1000 / Math.max(model.get("speed"), 0.1);
                pos += dt / interval;
                if (stopAt !== null && pos >= stopAt) {
                  pos = stopAt;         // reached the requested chapter: park here
                  stopAt = null;
                  model.set("playing", false);
                  model.save_changes();
                } else if (model.get("bijective") && pos >= period) {
                  pos = 0;                // wrapped a full cycle: back home
                  hold = HOLD_MS;
                }
              }
            }
            syncState();
            draw();
            rafId = requestAnimationFrame(frame);
          }

          function snap(dir) {
            // Jump to the next/prev chapter — a half-step keyframe: fully
            // stretched at .5, folded home at .0 — and pause there.
            const half = pos * 2;
            let t = dir > 0 ? (Math.floor(half) + 1) / 2 : (Math.ceil(half) - 1) / 2;
            if (model.get("bijective")) t = mod(t, period);
            else if (t < 0) t = 0;
            pos = t;
            hold = 0;
            stopAt = null;
            model.set("playing", false);
            model.save_changes();
            syncState();
            draw();
          }
          stepBack.addEventListener("click", () => snap(-1));
          stepFwd.addEventListener("click", () => snap(1));
          nextBtn.addEventListener("click", () => {
            // Play smoothly up to the next chapter (half-step keyframe), then pause.
            let target = (Math.floor(pos * 2 + 1e-9) + 1) / 2;
            if (model.get("bijective") && pos >= period - 1e-9) { pos = 0; target = 0.5; }
            stopAt = target;
            hold = 0;
            model.set("playing", true);
            model.save_changes();
          });
          playBtn.addEventListener("click", () => {
            stopAt = null;
            if (!model.get("bijective")) {
              // Broken loop: no home to wrap back to, so the button just
              // rewinds to the clean grid and plays the fold from the top.
              pos = 0;
              hold = 0;
              model.set("playing", true);
              model.save_changes();
              syncState();
              draw();
              return;
            }
            model.set("playing", !model.get("playing"));
            model.save_changes();
          });
          tileBtn.addEventListener("click", () => {
            model.set("show_tiling", !model.get("show_tiling"));
            model.save_changes();
          });
          clearBtn.addEventListener("click", () => {
            model.set("selected", -1);
            model.save_changes();
          });
          timeline.addEventListener("input", () => {
            stopAt = null;
            model.set("playing", false);
            model.save_changes();
            pos = parseFloat(timeline.value);
            hold = 0;
            syncState();
            draw();
          });
          canvas.addEventListener("click", (ev) => {
            // Map the click back to grid coordinates and select the nearest tile.
            const rect = canvas.getBoundingClientRect();
            const ux = (ev.clientX - rect.left) * (canvas.width / rect.width) / sc + ox0;
            const uy = (ev.clientY - rect.top) * (canvas.height / rect.height) / sc + oy0;
            let best = -1, bd = Infinity;
            for (let i = 0; i < n * n; i++) {
              const dx = px[i] + 0.5 - ux, dy = py[i] + 0.5 - uy;
              const dd = dx * dx + dy * dy;
              if (dd < bd) { bd = dd; best = i; }
            }
            const cur = model.get("selected");
            model.set("selected", cur === best ? -1 : best);
            model.save_changes();
          });

          setup();
          rafId = requestAnimationFrame(frame);

          model.on("change:n", setup);
          model.on("change:colors", setup);
          model.on("change:period", setup);

          return () => { if (rafId) cancelAnimationFrame(rafId); };
        }
        export default { render };
        """

    return (CatMapWidget,)


@app.cell(hide_code=True)
def _(mo):
    n_slider = mo.ui.slider(5, 45, value=29, label="grid size N (= modulus)")
    speed_slider = mo.ui.slider(0.05, 1, value=0.25, step=0.05, label="steps / sec")
    palette_dropdown = mo.ui.dropdown(
        ["rainbow", "viridis", "sunset", "ocean"], value="rainbow", label="palette"
    )
    matrix_ui = mo.ui.matrix(
        [[2, 1], [1, 1]], min_value=-6, max_value=6, step=1, label="matrix A"
    )
    return matrix_ui, n_slider, palette_dropdown, speed_slider


@app.cell(hide_code=True)
def catmap_controls(matrix_ui, mo, n_slider, palette_dropdown, speed_slider):
    (m_a, m_b), (m_c, m_d) = ((int(round(v)) for v in row) for row in matrix_ui.value)
    det_caption = mo.md(f"$\\det A = {m_a * m_d - m_b * m_c}$")
    mo.hstack(
        [
            mo.vstack([matrix_ui, det_caption]),
            mo.vstack([n_slider, speed_slider, palette_dropdown]),
        ],
        justify="start",
        align="center",
        gap=2,
    )
    return


@app.cell
def _(math, matrix_ui, n_slider, np, palette, palette_dropdown):
    n = n_slider.value

    # A smooth diagonal gradient so the scramble — and its return — is obvious.
    colors = [
        palette(palette_dropdown.value, ((x + y) % n) / n)
        for y in range(n)
        for x in range(n)
    ]

    # The chosen matrix A (integer entries from the UI).
    (a, b), (c, d) = ((int(round(v)) for v in row) for row in matrix_ui.value)
    A = np.array([[a, b], [c, d]])

    # The map (x,y) -> A(x,y) mod n is a bijection — and eventually returns to
    # the identity — only when det(A) is invertible mod n, i.e. coprime to it.
    det = a * d - b * c
    bijective = math.gcd(det % n, n) == 1

    if bijective:
        M = A % n
        period = 1
        while not np.array_equal(M, np.eye(2, dtype=int)):
            M = (M @ A) % n
            period += 1
    else:
        period = 1  # unused when not bijective
    return A, a, b, bijective, c, colors, d, n, period


@app.cell
def catmap_widget(CatMapWidget, a, b, bijective, c, colors, d, mo, n, period):
    cat = CatMapWidget(
        n=n, colors=colors, period=period, bijective=bijective,
        a=a, b=b, c=c, d=d,
    )
    widget = mo.ui.anywidget(cat)
    widget
    return (cat,)


@app.cell
def _(cat, speed_slider):
    # Live control that shouldn't rebuild (and reset) the animation. Play/pause,
    # scrubbing, the mod-N grid and tile-follow all live inside the widget now.
    cat.speed = speed_slider.value
    return


@app.cell(hide_code=True)
def why_returns_md(mo):
    mo.md(r"""
    ## Why does the picture come back?

    Each step of the map is a reshuffle of the grid. It sends every tile to a
    new spot, and no two tiles ever land on the same place. Repeat any reshuffle
    of a finite set and you must eventually return to the exact starting
    arrangement, because there are only so many arrangements and none of them
    merge. The number of steps this takes is the **period**.

    That return hands you something tidy. After a full period you are back to
    the identity,

    $$A^{\text{period}} = I \pmod{N},$$

    which rearranges into a formula for the inverse:

    $$A^{-1} = A^{\text{period} - 1} \pmod{N}.$$

    Going forward $(\text{period} - 1)$ steps leaves you one step behind where you
    started, which is the same as stepping back once.

    Put generally, there are values of $N$ and $period$ for which the above
    equality holds, so that means that there are situations where we can
    repeat this map indefinately.

    > For the cat map with $N = 29$ the period is $7$, so $A^{-1} = A^{6}$: applying the map six more
    times undoes a single step. Undoing one step and looping all the way home
    are the same idea.

    So everything rests on that reshuffle property: the map has to be reversible. What makes a matrix reversible?
    """)
    return


@app.cell(hide_code=True)
def follow_tile_md(mo):
    mo.md(r"""
    ## Follow one tile home

    Pick a tile by dragging the dot. The line traces every spot that tile visits
    as you apply the map again and again. It hops around the grid and then lands
    right back where it started.

    The number of hops is that tile's own period. It always divides the period of
    the whole map, so no single tile takes longer to come home than the picture
    does.
    """)
    return


@app.cell
def orbit_chart(A, ChartPuck, bijective, matrix_ui, mo, n, n_slider, period):
    # Match marimo's light/dark theme so the chart doesn't glare or vanish.
    dark = mo.app_meta().theme == "dark"
    fg = "#e5e7eb" if dark else "#111827"

    def orbit_of(x0, y0):
        # Follow one tile: p -> A·p mod n, until it returns home or we hit the cap.
        cap = period if bijective else n * n
        pts = [(x0, y0)]
        x, y = x0, y0
        for step in range(cap):
            x, y = (
                (A[0, 0] * x + A[0, 1] * y) % n,
                (A[1, 0] * x + A[1, 1] * y) % n,
            )
            if (x, y) == (x0, y0):
                break
            pts.append((x, y))
        returned = (x, y) == (x0, y0)
        return pts, returned

    def draw_orbit(ax, widget):
        x0 = int(min(max(round(widget.x[0]), 0), n - 1))
        y0 = int(min(max(round(widget.y[0]), 0), n - 1))
        pts, returned = orbit_of(x0, y0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # Close the loop back to the start when the tile returns.
        if returned:
            xs = xs + [x0]
            ys = ys + [y0]

        ax.figure.set_facecolor("none")
        ax.set_facecolor("none")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.grid(True, alpha=0.25, color=fg)
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
        for spine in ax.spines.values():
            spine.set_color(fg)
        ax.plot(xs, ys, "-", color="#457b9d", linewidth=1.5, alpha=0.8)
        ax.plot(xs, ys, "o", color="#457b9d", markersize=3)
        ax.plot([x0], [y0], "o", color="#e63946", markersize=6, zorder=5)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_aspect("equal")
        if returned:
            ax.set_title(f"tile ({x0}, {y0}) · orbit length {len(pts)} of period {period}", color=fg)
        else:
            ax.set_title(f"tile ({x0}, {y0}) · does not return (not invertible)", color=fg)

    orbit_puck = mo.ui.anywidget(
        ChartPuck.from_callback(
            draw_fn=draw_orbit,
            x_bounds=(-0.5, n - 0.5),
            y_bounds=(-0.5, n - 0.5),
            figsize=(6, 6),
            x=float(n // 2),
            y=float(n // 2),
            puck_radius=6,
            throttle=100,
        )
    )
    det_A = int(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
    mo.hstack(
        [orbit_puck, mo.vstack([matrix_ui, mo.md(f"$\\det A = {det_A}$"), n_slider])],
        justify="start",
        align="center",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def reduce_to_matrix_md(mo):
    mo.md(r"""
    ## From the whole grid to one small matrix

    The period argument was about the shuffle of every tile at once. That is a big
    object: it moves all $N^2$ tiles around.

    But one small rule drives the whole thing. The same $2\times2$ matrix moves
    every tile, so the whole shuffle is decided by that matrix alone. The shuffle
    never drops two tiles on the same spot exactly when the small matrix can be
    undone mod $N$.

    So from here on we can forget the giant grid and look at the $2\times2$ matrix
    and the one number it hands us: its determinant.
    """)
    return


@app.cell(hide_code=True)
def determinant_md(mo):
    mo.md(r"""
    ## What makes it reversible? The determinant

    One number decides it. Write the matrix as

    $$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}.$$

    The deciding number is the determinant

    $$\det A = ad - bc,$$

    the product down the main diagonal minus the product down the other one.
    You can picture it as the area of the parallelogram drawn by the two columns
    $(a, c)$ and $(b, d)$. When that area is $0$ the columns lie on a single
    line, the map flattens the whole plane onto that line, and there is no way
    back. So "can it be undone" is really "do the two columns point in different
    directions."

    The positions matter again when you build the inverse. You swap the two
    diagonal entries, flip the sign of the two off-diagonal entries, and scale
    by $1 / \det A$:

    $$A^{-1} = \frac{1}{ad - bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix} = \frac{1}{\det A}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}.$$

    So each slot in the matrix has its own job, both in deciding whether an
    inverse exists and in shaping what that inverse looks like.
    """)
    return


@app.cell(hide_code=True)
def dividing_modn_md(mo):
    mo.md(r"""
    ## The catch: dividing mod $N$

    Building $A^{-1}$ meant dividing by $\det A$. On the grid our only numbers are
    $0, 1, \dots, N-1$. They're all integers! So we can't have fractions here.

    If we use the definition of inverse, we usually write.

    $$A \cdot A^{-1} = I$$

    But in our case we are talking about $\det A$ and we have that modulo to worry about.

    $$\det A \cdot (\det A)^{-1} \equiv 1 \pmod{N}.$$

    That inverse plays the role of $1/\det A$. The catch is that for some values it
    simply does not exist because we cannot have fractions. So whether the matrix can be undone comes down to one
    question: does $\det A$ have an inverse mod $N$?

    ## When does the inverse exist? The gcd

    Here we get to introduce a new function: the **greatest common divisor** of $\det A$ and $N$ —
    "gcd" for short, written $g = \gcd(\det A, N)$. It is the largest
    number that divides both $\det A$ and $N$.

    Here's what makes it useful as a tool.

    - If $\det A$ and $N$ share a factor $g > 1$, then every
    multiple $\det A, 2\det A, 3\det A, \dots$ is also a multiple of $g$.
    - So mod $N$ they only ever land on multiples of $g$, and $1$ is never one of them.
    - No multiple reaches $1$, so there is no inverse.

    When $\gcd(\det A, N) = 1$ the multiples instead sweep through every residue
    $0, 1, \dots, N-1$, so one of them has to be $1$. That one is the inverse.

    The table below runs this over many pairs. Columns are $\det A$, rows are the
    modulus $N$. Each cell is $\gcd(\det A, N)$: green when it is $1$ (an inverse
    exists), red when it is bigger (no inverse).
    """)
    return


@app.cell(hide_code=True)
def gcd_table(math, mo):
    dets = range(1, 11)  # det A along the columns
    mods = range(2, 53)  # N down the rows

    def head(text):
        return (
            f'<th style="padding:5px 9px;font-family:ui-monospace,monospace;'
            f'font-weight:600;opacity:0.75">{text}</th>'
        )

    def body(k, modulus):
        g = math.gcd(k, modulus)
        coprime = g == 1
        bg = "#dcfce7" if coprime else "#fee2e2"
        fg = "#16a34a" if coprime else "#dc2626"
        return (
            f'<td style="padding:5px 9px;text-align:center;border-radius:4px;'
            f"background:{bg};color:{fg};font-family:ui-monospace,monospace"
            f'">{g}</td>'
        )

    def axis(text, span=""):
        return (
            f'<th {span} style="padding:5px 9px;font-family:ui-monospace,monospace;'
            f'font-weight:700;letter-spacing:0.04em">{text}</th>'
        )

    n_det = len(list(dets))
    n_mod = len(list(mods))

    # Two header rows: the "det A" title spanning the value columns, then the
    # determinant values. Two empty leading cells keep them over the value grid.
    title_row = "<tr><td></td><td></td>" + axis("det A", f'colspan="{n_det}"') + "</tr>"
    num_row = "<tr><td></td><td></td>" + "".join(head(k) for k in dets) + "</tr>"

    # Body rows: the first one carries a vertical "M" title spanning every row.
    body_rows = []
    for i, modulus in enumerate(mods):
        cells = "".join(body(k, modulus) for k in dets)
        if i == 0:
            m_axis = (
                f'<th rowspan="{n_mod}" style="padding:5px 9px;font-weight:700;'
                f"font-family:ui-monospace,monospace;writing-mode:vertical-rl;"
                f'transform:rotate(180deg)">N</th>'
            )
            body_rows.append("<tr>" + m_axis + head(modulus) + cells + "</tr>")
        else:
            body_rows.append("<tr>" + head(modulus) + cells + "</tr>")

    mo.Html(
        '<table style="border-collapse:separate;border-spacing:3px">'
        + title_row + num_row + "".join(body_rows) + "</table>"
    )
    return


@app.cell(hide_code=True)
def try_it_md(mo):
    mo.md(r"""
    ## Try it yourself

    The whole question is about $\det A$ and $N$. The sliders let you pick both and
    watch the multiples of $\det A$ march across the residues. For the cat map
    $\det A = 1$; try other values to see when an inverse appears.
    """)
    return


@app.cell(hide_code=True)
def try_it_controls(mo):
    inv_n_slider = mo.ui.slider(2, 40, value=6, label="N")
    det_slider = mo.ui.slider(1, 39, value=5, label="det A")
    return det_slider, inv_n_slider


@app.cell(hide_code=True)
def try_it_result(det_slider, inv_n_slider, math, mo):
    N = inv_n_slider.value
    det_a = min(det_slider.value, N - 1)

    shared = math.gcd(det_a, N)
    inverse = next((k for k in range(1, N) if (det_a * k) % N == 1), None)

    tiy_dark = mo.app_meta().theme == "dark"
    tiy_txt = "#e5e7eb" if tiy_dark else "#111827"

    def tiy_cell(text, bg="transparent", color=None, weight="400"):
        color = color or tiy_txt
        return (
            f'<td style="padding:3px 6px;text-align:center;border-radius:4px;'
            f"font-family:ui-monospace,monospace;font-size:12px;font-weight:{weight};"
            f'background:{bg};color:{color}">{text}</td>'
        )

    def tiy_label(text):
        return (
            f'<th style="padding:3px 10px 3px 0;text-align:right;white-space:nowrap;'
            f"font-family:ui-monospace,monospace;font-size:12px;font-weight:700;"
            f'color:{tiy_txt}">{text}</th>'
        )

    def tiy_row(cells, delay):
        return f'<tr class="tiy-row" style="animation-delay:{delay}s">' + cells + "</tr>"

    ks = range(N)
    row_k = tiy_row(
        tiy_label("k") + "".join(tiy_cell(k, color="#9ca3af") for k in ks), 0.0
    )
    row_scaled = tiy_row(
        tiy_label(f"× {det_a}") + "".join(tiy_cell(det_a * k) for k in ks), 0.6
    )

    def mod_cell(k):
        r = (det_a * k) % N
        if r == 1:
            return tiy_cell(r, bg="#dcfce7", color="#16a34a", weight="700")
        return tiy_cell(r, bg="rgba(59,130,246,0.14)")

    row_mod = tiy_row(tiy_label(f"mod {N}") + "".join(mod_cell(k) for k in ks), 1.2)

    table = (
        "<style>"
        "@keyframes tiyReveal{from{opacity:0}to{opacity:1}}"
        ".tiy-anim tr.tiy-row{opacity:0;animation:tiyReveal .45s ease both}"
        "</style>"
        '<div class="tiy-anim" style="overflow-x:auto;padding-bottom:4px">'
        '<table style="border-collapse:separate;border-spacing:2px">'
        + row_k + row_scaled + row_mod + "</table></div>"
    )

    if inverse is not None:
        verdict = (
            f'<b style="color:#16a34a">gcd({det_a}, {N}) = 1</b> → '
            f"det A⁻¹ = <b>{inverse}</b>, since {det_a}·{inverse} = "
            f"{det_a * inverse} ≡ 1 (mod {N})."
        )
    else:
        verdict = (
            f'<b style="color:#dc2626">gcd({det_a}, {N}) = {shared}</b> → no inverse. '
            f"After the mod the bottom row only lands on multiples of {shared}, so it never hits 1."
        )

    mo.vstack(
        [
            mo.hstack([inv_n_slider, det_slider], justify="start", gap=2),
            mo.md(
                f"Take $0 \\dots N-1$, scale every number by $\\det = {det_a}$, then fold "
                f"back with mod ${N}$. An inverse exists exactly when the bottom row hits **1**."
            ),
            mo.Html(table),
            mo.Html(f'<div style="margin-top:8px">{verdict}</div>'),
        ]
    )
    return


@app.cell(hide_code=True)
def forward_back_md(mo):
    mo.md(r"""
    ## Forward, then back: watch the determinant

    Take the unit square, apply $A$, and it becomes a parallelogram whose area is
    exactly $\det A$. To come back you apply the inverse — but the inverse is the
    swap-flip matrix scaled by $1/\det A$. Toggle that $1/\det A$ on and off to see
    what it does: without it, the swap-flip lands you back on a *square*, but one
    blown up by $\det A$. The $1/\det A$ is the shrink that undoes the stretch.
    """)
    return


@app.cell(hide_code=True)
def inv_det_toggle(mo):
    use_inv_det = mo.ui.checkbox(value=False, label="include 1/det(A) in the inverse")
    return (use_inv_det,)


@app.cell(hide_code=True)
def det_geometry(matrix_ui, mo, np, plt, use_inv_det):
    inv_dark = mo.app_meta().theme == "dark"
    inv_fg = "#e5e7eb" if inv_dark else "#111827"

    (fa, fb), (fc, fd) = ((float(v) for v in row) for row in matrix_ui.value)
    F = np.array([[fa, fb], [fc, fd]])
    f_det = fa * fd - fb * fc
    adj = np.array([[fd, -fb], [-fc, fa]])

    # Closed unit square as a 2x5 array of corners.
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
    forward = F @ unit
    # adj @ forward = det * unit. The 1/det knob turns that back into unit.
    back_scale = (1.0 / f_det) if (use_inv_det.value and f_det != 0) else 1.0
    backward = back_scale * (adj @ forward)

    # Shared fixed limits (include the det-scaled square) so toggling 1/det
    # visibly shrinks the square instead of the axes silently rescaling.
    bounds = np.hstack([unit, forward, adj @ forward])
    lo = float(bounds.min())
    hi = float(bounds.max())
    pad = 0.15 * (hi - lo) + 0.3
    lo, hi = lo - pad, hi + pad

    def style(ax, title):
        ax.set_facecolor("none")
        ax.axhline(0, color=inv_fg, alpha=0.25, lw=1)
        ax.axvline(0, color=inv_fg, alpha=0.25, lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(title, color=inv_fg, fontsize=11)
        for s in ax.spines.values():
            s.set_color(inv_fg)
        ax.tick_params(colors=inv_fg, labelsize=8)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 4.6), constrained_layout=True)
    fig.patch.set_facecolor("none")

    axL.fill(unit[0], unit[1], color="#94a3b8", alpha=0.35)
    axL.plot(unit[0], unit[1], color="#94a3b8", lw=1.5)
    axL.fill(forward[0], forward[1], color="#457b9d", alpha=0.4)
    axL.plot(forward[0], forward[1], color="#457b9d", lw=2)
    style(axL, "forward: A · square")

    # Parallelogram in gray (the starting shape), then where the inverse sends it.
    axR.fill(forward[0], forward[1], color="#94a3b8", alpha=0.3)
    axR.plot(forward[0], forward[1], color="#94a3b8", lw=1.5)
    axR.fill(backward[0], backward[1], color="#e07a5f", alpha=0.45)
    axR.plot(backward[0], backward[1], color="#e07a5f", lw=2)
    style(axR, "backward: inverse")

    # The matrix actually applied backward: swap-flip alone, or scaled by 1/det.
    inv_shown = (back_scale * adj).tolist()
    inv_label = "A⁻¹ (with 1/det)" if use_inv_det.value else "swap-flip (no 1/det)"
    inv_view = mo.ui.matrix(inv_shown, disabled=True, precision=2, label=inv_label)

    mo.vstack(
        [
            mo.hstack(
                [matrix_ui, inv_view, use_inv_det],
                justify="start", align="center", gap=2,
            ),
            mo.md(f"$\\det A = {f_det:.2f}$ — the parallelogram's area, and the factor $1/\\det A$ undoes."),
            mo.as_html(fig),
        ]
    )
    return


@app.cell(hide_code=True)
def animated_md(mo):
    mo.md(r"""
    ## The same thing, animated

    Watch the unit square move through the four stages: apply $A$ (stretch to the
    parallelogram), apply the swap-flip (a square, but blown up by $\det A$), then
    divide by $\det A$ to land back home. Hit play, or step chapter by chapter.
    """)
    return


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class MorphWidget(anywidget.AnyWidget):
        a = traitlets.Float(2.0).tag(sync=True)
        b = traitlets.Float(1.0).tag(sync=True)
        c = traitlets.Float(1.0).tag(sync=True)
        d = traitlets.Float(3.0).tag(sync=True)
        speed = traitlets.Float(0.8).tag(sync=True)
        playing = traitlets.Bool(True).tag(sync=True)
        fg = traitlets.Unicode("#111827").tag(sync=True)

        _esm = r"""
        function render({ model, el }) {
          el.style.textAlign = "center";
          const wrap = document.createElement("div");
          wrap.style.display = "inline-block";
          const canvas = document.createElement("canvas");
          wrap.appendChild(canvas);

          function styleBtn(b) {
            b.style.font = "13px ui-monospace, monospace";
            b.style.padding = "4px 10px";
            b.style.borderRadius = "6px";
            b.style.border = "1px solid rgba(128,128,128,0.5)";
            b.style.background = "transparent";
            b.style.color = "inherit";
            b.style.cursor = "pointer";
            return b;
          }
          const controls = document.createElement("div");
          controls.style.cssText = "display:flex;align-items:center;gap:10px;margin:12px auto 0";
          const playBtn = styleBtn(document.createElement("button"));
          const nextBtn = styleBtn(document.createElement("button"));
          nextBtn.textContent = "▶ to next";
          const backBtn = styleBtn(document.createElement("button"));
          backBtn.textContent = "◀ chapter";
          const fwdBtn = styleBtn(document.createElement("button"));
          fwdBtn.textContent = "chapter ▶";
          const timeline = document.createElement("input");
          timeline.type = "range";
          timeline.min = "0";
          timeline.max = "3";
          timeline.step = "any";
          timeline.value = "0";
          timeline.style.flex = "1";
          timeline.style.cursor = "pointer";
          controls.appendChild(playBtn);
          controls.appendChild(nextBtn);
          controls.appendChild(backBtn);
          controls.appendChild(timeline);
          controls.appendChild(fwdBtn);

          const label = document.createElement("div");
          label.style.cssText = "font-family:ui-monospace,monospace;margin-top:14px;opacity:0.85";
          el.appendChild(wrap);
          el.appendChild(controls);
          el.appendChild(label);
          const ctx = canvas.getContext("2d");

          const SIZE = 440;
          canvas.width = SIZE;
          canvas.height = SIZE;
          const PERIOD = 3;
          const HOLD_MS = 1000;
          const STAGES = [
            "chapter 0/3 · unit square",
            "chapter 1/3 · x A  (parallelogram, area = det)",
            "chapter 2/3 · x swap-flip  (square, blown up by det)",
            "chapter 3/3 · / det  (back to the unit square)",
          ];
          const UNIT = [[0, 0], [1, 0], [1, 1], [0, 1]];
          const ease = (t) => t * t * (3 - 2 * t);

          let pos = 0, hold = 0, stopAt = null, rafId = null, lastT = 0;
          let keyframes, frameMin, frameMax;

          function applyM(M, p) {
            return [M[0] * p[0] + M[1] * p[1], M[2] * p[0] + M[3] * p[1]];
          }

          function setup() {
            const a = model.get("a"), b = model.get("b");
            const c = model.get("c"), d = model.get("d");
            const det = a * d - b * c;
            // M slides identity -> A -> det*I -> identity.
            keyframes = [[1, 0, 0, 1], [a, b, c, d], [det, 0, 0, det], [1, 0, 0, 1]];
            // Fixed frame over every keyframe so the view never jumps.
            let lo = 0, hi = 1;
            for (const M of keyframes) for (const p of UNIT) {
              const q = applyM(M, p);
              lo = Math.min(lo, q[0], q[1]);
              hi = Math.max(hi, q[0], q[1]);
            }
            const pad = 0.2 * (hi - lo) + 0.3;
            frameMin = lo - pad;
            frameMax = hi + pad;
          }

          function curM() {
            const seg = Math.min(Math.floor(pos), keyframes.length - 2);
            const f = ease(pos - seg);
            const A = keyframes[seg], B = keyframes[seg + 1];
            return A.map((v, i) => v + f * (B[i] - v));
          }

          function draw() {
            const V = frameMax - frameMin;
            const sc = SIZE / V;
            const tx = (p) => [(p[0] - frameMin) * sc, SIZE - (p[1] - frameMin) * sc];
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.clearRect(0, 0, SIZE, SIZE);

            // axes through the origin
            const o = tx([0, 0]);
            ctx.strokeStyle = model.get("fg");
            ctx.globalAlpha = 0.25;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, o[1]); ctx.lineTo(SIZE, o[1]);
            ctx.moveTo(o[0], 0); ctx.lineTo(o[0], SIZE);
            ctx.stroke();
            ctx.globalAlpha = 1;

            const M = curM();
            const seg = Math.min(Math.floor(pos), keyframes.length - 2);
            const fadeT = pos - seg;          // 0 at chapter start, 1 at the next
            const SHAPE = "#457b9d";

            // Ghost of the shape we're leaving, fading out as we move away —
            // same color as the moving shape, so nothing jumps between hues.
            const ghost = keyframes[seg];
            ctx.beginPath();
            UNIT.forEach((p, i) => {
              const q = tx(applyM(ghost, p));
              if (i === 0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]);
            });
            ctx.closePath();
            ctx.globalAlpha = 0.16 * (1 - fadeT);
            ctx.fillStyle = SHAPE;
            ctx.fill();
            ctx.globalAlpha = 0.5 * (1 - fadeT);
            ctx.strokeStyle = SHAPE;
            ctx.lineWidth = 1.5;
            ctx.stroke();
            ctx.globalAlpha = 1;

            // The moving shape keeps one steady color the whole way.
            ctx.beginPath();
            UNIT.forEach((p, i) => {
              const q = tx(applyM(M, p));
              if (i === 0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]);
            });
            ctx.closePath();
            ctx.globalAlpha = 0.4;
            ctx.fillStyle = SHAPE;
            ctx.fill();
            ctx.globalAlpha = 1;
            ctx.strokeStyle = SHAPE;
            ctx.lineWidth = 2.5;
            ctx.stroke();

            if (document.activeElement !== timeline) timeline.value = String(pos);
            playBtn.textContent = model.get("playing") ? "⏸ pause" : "▶ play";
            const chapter = Math.min(Math.round(pos), 3);
            label.style.color = model.get("fg");
            label.textContent = STAGES[chapter];
          }

          function frame(t) {
            if (!lastT) lastT = t;
            const dt = t - lastT;
            lastT = t;
            if (model.get("playing")) {
              if (hold > 0) {
                hold -= dt;
              } else {
                const interval = 1000 / Math.max(model.get("speed"), 0.1);
                pos += dt / interval;
                if (stopAt !== null && pos >= stopAt) {
                  pos = stopAt;
                  stopAt = null;
                  model.set("playing", false);
                  model.save_changes();
                } else if (pos >= PERIOD) {
                  pos = 0;
                  hold = HOLD_MS;
                }
              }
            }
            draw();
            rafId = requestAnimationFrame(frame);
          }

          function snap(dir) {
            let t = dir > 0 ? Math.floor(pos) + 1 : Math.ceil(pos) - 1;
            t = Math.max(0, Math.min(PERIOD, t));
            pos = t;
            hold = 0;
            stopAt = null;
            model.set("playing", false);
            model.save_changes();
            draw();
          }

          playBtn.addEventListener("click", () => {
            stopAt = null;
            model.set("playing", !model.get("playing"));
            model.save_changes();
          });
          nextBtn.addEventListener("click", () => {
            // Play smoothly up to the next chapter, then pause there.
            let target = Math.floor(pos + 1e-9) + 1;
            if (pos >= PERIOD - 1e-9) { pos = 0; target = 1; }
            stopAt = Math.min(target, PERIOD);
            hold = 0;
            model.set("playing", true);
            model.save_changes();
          });
          backBtn.addEventListener("click", () => snap(-1));
          fwdBtn.addEventListener("click", () => snap(1));
          timeline.addEventListener("input", () => {
            stopAt = null;
            model.set("playing", false);
            model.save_changes();
            pos = parseFloat(timeline.value);
            hold = 0;
            draw();
          });

          setup();
          rafId = requestAnimationFrame(frame);
          model.on("change:a", setup);
          model.on("change:b", setup);
          model.on("change:c", setup);
          model.on("change:d", setup);
          model.on("change:fg", draw);

          return () => { if (rafId) cancelAnimationFrame(rafId); };
        }
        export default { render };
        """

    return (MorphWidget,)


@app.cell(hide_code=True)
def morph_widget(MorphWidget, matrix_ui, mo):
    (ma, mb), (mc, md) = ((float(v) for v in row) for row in matrix_ui.value)
    morph_fg = "#e5e7eb" if mo.app_meta().theme == "dark" else "#111827"
    morph = MorphWidget(a=ma, b=mb, c=mc, d=md, fg=morph_fg)
    mo.vstack([matrix_ui, mo.ui.anywidget(morph)])
    return


if __name__ == "__main__":
    app.run()

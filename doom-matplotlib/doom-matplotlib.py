# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "cydoomgeneric",
#     "numpy",
#     "anywidget==0.9.21",
#     "traitlets==5.14.3",
#     "pillow",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Doom in a Canvas

    This notebook runs **Doom** (the 1993 classic) inside an HTML5 `<canvas>` using
    [`cydoomgeneric`](https://github.com/wojciech-graj/cydoomgeneric) — a Python binding
    for the portable DoomGeneric engine.

    The rendering pipeline: `cydoomgeneric` calls our `draw_frame(pixels)` callback on every
    game tick. We JPEG-encode the frame and send raw bytes to an anywidget, which decodes
    and paints them onto a `<canvas>` element. Keyboard events flow back from JS to Python.

    **Click the game canvas to focus it, then use arrow keys to move, Ctrl to shoot, Space to open doors.**
    """)
    return


@app.cell
def _():
    import cydoomgeneric as cdg
    import numpy as np
    import os
    import urllib.request
    import io

    return cdg, io, os, urllib


@app.cell
def _(mo, os, urllib):
    wad_path = "doom1.wad"
    wad_url = "https://distro.ibiblio.org/slitaz/sources/packages/d/doom1.wad"

    if not os.path.exists(wad_path):
        mo.output.append(mo.md("Downloading `doom1.wad` (shareware)..."))
        urllib.request.urlretrieve(wad_url, wad_path)

    mo.md(f"**WAD file ready:** `{wad_path}` ({os.path.getsize(wad_path) / 1024 / 1024:.1f} MB)")
    return


@app.cell
def _():
    import anywidget
    import traitlets

    class DoomCanvas(anywidget.AnyWidget):
        """Anywidget that renders JPEG frames on a canvas and captures keyboard input."""

        _esm = """
        function render({ model, el }) {
            const wrapper = document.createElement("div");
            wrapper.style.position = "relative";
            wrapper.style.display = "inline-block";

            const canvas = document.createElement("canvas");
            canvas.width = 640;
            canvas.height = 400;
            canvas.tabIndex = 0;
            canvas.style.cursor = "crosshair";
            canvas.style.outline = "none";
            canvas.style.border = "2px solid #333";
            canvas.style.imageRendering = "pixelated";
            canvas.style.background = "#000";
            canvas.style.display = "block";
            const ctx = canvas.getContext("2d");

            // Focus indicator badge
            const badge = document.createElement("div");
            badge.style.position = "absolute";
            badge.style.top = "8px";
            badge.style.right = "8px";
            badge.style.padding = "3px 8px";
            badge.style.borderRadius = "4px";
            badge.style.fontSize = "11px";
            badge.style.fontFamily = "monospace";
            badge.style.pointerEvents = "none";
            badge.style.zIndex = "10";
            badge.style.transition = "opacity 0.15s";

            // Manual active state — don't rely on browser focus at all
            let active = false;
            const pressedKeys = new Set();

            function releaseAllKeys() {
                if (pressedKeys.size === 0) return;
                for (const key of pressedKeys) {
                    localBuffer.push(key + ":0");
                }
                pressedKeys.clear();
                model.set("key_events", localBuffer.slice());
                model.save_changes();
            }

            function setActive(val) {
                if (!val) releaseAllKeys();
                active = val;
                if (active) {
                    canvas.style.border = "2px solid #4a4";
                    badge.textContent = "ACTIVE — press Q to release";
                    badge.style.background = "rgba(40, 120, 40, 0.85)";
                    badge.style.color = "#fff";
                } else {
                    canvas.style.border = "2px solid #333";
                    badge.textContent = "PAUSED — click to play";
                    badge.style.background = "rgba(50, 50, 50, 0.85)";
                    badge.style.color = "#aaa";
                }
            }

            setActive(false);

            wrapper.appendChild(canvas);
            wrapper.appendChild(badge);
            el.appendChild(wrapper);

            // Click anywhere on the canvas to activate
            canvas.addEventListener("click", () => setActive(true));

            // Render JPEG frames from Python (base64-encoded)
            model.on("change:frame_b64", () => {
                const b64 = model.get("frame_b64");
                if (!b64) return;
                const img = new Image();
                img.onload = () => {
                    ctx.drawImage(img, 0, 0, 640, 400);
                };
                img.src = "data:image/jpeg;base64," + b64;
            });

            // Map JS key names to simple names for Python
            // z and f are alternative fire keys since Ctrl is awkward in browsers
            const KEY_MAP = {
                "ArrowLeft": "left",
                "ArrowRight": "right",
                "ArrowUp": "up",
                "ArrowDown": "down",
                "Control": "control",
                "z": "control",
                "f": "control",
                "Shift": "shift",
                "Enter": "enter",
                "Escape": "escape",
                " ": " ",
                ",": ",",
                ".": ".",
            };

            const GAME_KEYS = new Set(Object.keys(KEY_MAP));

            // Local buffer — never read from model, only append and sync
            const localBuffer = [];

            function handleKey(e, pressed) {
                if (!active) return;
                // Q releases the canvas
                if (e.key === "q" || e.key === "Q") {
                    if (pressed) setActive(false);
                    return;
                }
                if (!GAME_KEYS.has(e.key)) return;
                e.preventDefault();
                const key = KEY_MAP[e.key];
                if (pressed) pressedKeys.add(key); else pressedKeys.delete(key);
                localBuffer.push(key + (pressed ? ":1" : ":0"));
                model.set("key_events", localBuffer.slice());
                model.save_changes();
            }

            // Listen at document level — bypasses marimo's event interception
            document.addEventListener("keydown", (e) => handleKey(e, true));
            document.addEventListener("keyup", (e) => handleKey(e, false));
        }
        export default { render };
        """

        frame_b64 = traitlets.Unicode("").tag(sync=True)
        key_events = traitlets.List(traitlets.Unicode(), []).tag(sync=True)

    return (DoomCanvas,)


@app.cell
def _(DoomCanvas, cdg, io, mo):
    RESX = 320
    RESY = 200
    FRAME_SKIP = 2

    canvas_widget = DoomCanvas()

    class DoomRunner:
        """Bridges cydoomgeneric callbacks to the anywidget canvas."""

        def __init__(self, widget):
            self.widget = widget
            self.frame_count = 0
            self._last_event_len = 0

        def draw_frame(self, pixels):
            thread = mo.current_thread()
            if thread is not None and thread.should_exit:
                import sys
                sys.exit(0)

            self.frame_count += 1
            if self.frame_count % FRAME_SKIP != 0:
                return

            # BGR -> RGB
            rgb = pixels[:, :, [2, 1, 0]]

            # Encode as JPEG, then base64 for safe traitlet transport
            import base64
            from PIL import Image
            img = Image.fromarray(rgb)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            self.widget.frame_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        def get_key(self):
            events = self.widget.key_events
            if len(events) <= self._last_event_len:
                return None

            # Process next unread event
            event_str = events[self._last_event_len]
            self._last_event_len += 1

            parts = event_str.split(":")
            if len(parts) != 2:
                return self.get_key()
            key, pressed_str = parts[0], int(parts[1])

            KEYMAP = {
                "left": cdg.Keys.LEFTARROW,
                "right": cdg.Keys.RIGHTARROW,
                "up": cdg.Keys.UPARROW,
                "down": cdg.Keys.DOWNARROW,
                ",": cdg.Keys.STRAFE_L,
                ".": cdg.Keys.STRAFE_R,
                "control": cdg.Keys.FIRE,
                " ": cdg.Keys.USE,
                "shift": cdg.Keys.RSHIFT,
                "enter": cdg.Keys.ENTER,
                "escape": cdg.Keys.ESCAPE,
            }

            if key in KEYMAP:
                return (pressed_str, KEYMAP[key])
            if len(key) == 1:
                return (pressed_str, ord(key.lower()))
            return self.get_key()

        def set_window_title(self, t):
            pass

    doom = DoomRunner(canvas_widget)
    return RESX, RESY, canvas_widget, doom


@app.cell
def _(RESX, RESY, cdg, doom, mo):
    def start_doom():
        cdg.init(
            RESX,
            RESY,
            doom.draw_frame,
            doom.get_key,
            set_window_title=doom.set_window_title,
        )
        doom.running = True
        cdg.main()

    doom_thread = mo.Thread(target=start_doom, daemon=True)
    doom_thread.start()
    return


@app.cell
def _(canvas_widget, mo):
    mo.vstack([
        canvas_widget,
        mo.md("""
    | Action | Key |
    |--------|-----|
    | Move forward / back | `↑` `↓` |
    | Turn left / right | `←` `→` |
    | Strafe left / right | `,` `.` |
    | Open door / use | `Space` |
    | Shoot | `Z` or `F` (or `Ctrl`) |
    | Run | `Shift` |
    | Menu / confirm | `Enter` / `Esc` |
    | **Release keyboard** | `Q` |
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How it works

    ### Rendering: Python → Canvas
    1. `cydoomgeneric` calls `draw_frame(pixels)` with a numpy array every game tick
    2. We convert BGR→RGB and JPEG-encode with Pillow (~1ms for 320×200)
    3. Raw JPEG bytes are sent to the anywidget via a `Bytes` traitlet
    4. JavaScript decodes with `createImageBitmap()` (hardware-accelerated) and paints on `<canvas>`

    ### Input: Canvas → Python
    1. The `<canvas>` has `tabIndex` so it can receive focus and keyboard events
    2. `keydown`/`keyup` events are mapped to simple key names and pushed into a `List` traitlet
    3. `get_key()` reads from that list on each game tick

    ### Why this is faster than matplotlib
    - No `savefig()` rasterization — Pillow JPEG encode is ~10× faster
    - No PNG overhead — JPEG is smaller and faster to encode
    - No matplotlib figure/axes overhead at all
    - `createImageBitmap` decodes on GPU in the browser
    - Canvas `drawImage` is hardware-accelerated

    ### Performance knobs
    - `FRAME_SKIP` — only render every Nth frame (currently 2)
    - JPEG `quality` — lower = smaller bytes over websocket (currently 70)
    - Resolution — 320×200 native, upscaled to 640×400 in CSS with `image-rendering: pixelated`
    """)
    return


if __name__ == "__main__":
    app.run()

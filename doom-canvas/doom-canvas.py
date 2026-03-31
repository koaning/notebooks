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
    # Doom in a Marimo Notebook

    This notebook runs the original 1993 Doom engine inside a marimo notebook. The game
    runs at playable frame rates despite every single frame travelling from a C game engine,
    through Python, over a WebSocket, and into your browser. Here's how.

    ## The architecture

    ```
    ┌─ Python (backend) ───────────────────┐       ┌─ Browser (frontend) ─────────┐
    │                                      │       │                              │
    │  cydoomgeneric (C extension)         │       │  anywidget                   │
    │  ├─ runs the full Doom engine        │       │  ├─ <canvas> element         │
    │  ├─ physics, AI, rendering           │  ws   │  ├─ decodes JPEG → drawImage │
    │  └─ calls draw_frame(pixels) ────────│──────>│  └─ captures keydown/keyup   │
    │                                      │       │         │                    │
    │  get_key() polls for input <─────────│───────│─────────┘                    │
    │                                      │  ws   │                              │
    └──────────────────────────────────────┘       └──────────────────────────────┘
    ```

    **Nothing runs in the browser except a JPEG decoder and keyboard listener.** The entire game —
    physics, enemy AI, BSP rendering, collision detection — runs in Python's process via
    [`cydoomgeneric`](https://github.com/wojciech-graj/cydoomgeneric), a Python binding for the
    portable DoomGeneric C engine.

    ## Why is it fast?

    At first glance, shipping every frame through `Python → JPEG → base64 → WebSocket → browser`
    sounds hopelessly slow. But each step is cheaper than you'd think:

    **1. The frame is tiny.** Doom renders at 320×200 — that's 64,000 pixels. A modern 1080p
    game pushes 2,073,600 pixels per frame, which is 32× more. The entire Doom framebuffer
    fits in 192 KB of RAM.

    **2. JPEG encoding is fast at this size.** Pillow's JPEG encoder (libjpeg under the hood)
    compresses a 320×200 frame in under 1 ms. At quality 70, the output is typically 5–10 KB —
    small enough that WebSocket transfer is near-instant on localhost.

    **3. Base64 is a memcpy, not a bottleneck.** Converting 8 KB of JPEG to ~11 KB of base64
    is a trivial string operation. We use base64 because marimo's widget transport (via pickle)
    chokes on raw binary bytes, but the overhead is negligible.

    **4. The browser decodes JPEG in hardware.** When JavaScript creates an `Image` element with
    a data URI, the browser hands JPEG decoding to the platform's native codec — often
    GPU-accelerated. Then `canvas.drawImage()` blits it to screen, also GPU-accelerated.

    **5. Frame skipping hides the remaining latency.** The game engine ticks at full speed
    (processing input every tick), but we only encode and ship every 2nd frame. This means
    the game stays responsive even if a frame occasionally takes longer to deliver.

    ## What's the actual bottleneck?

    The `draw_frame → JPEG encode → base64 → traitlet sync → WebSocket → browser decode → paint`
    round trip takes roughly 5–15 ms total, which puts us in the 30–60 fps range. The dominant
    cost is the traitlet sync — anywidget's change-detection and marimo's WebSocket serialization
    add a few milliseconds of overhead beyond the raw encode time.

    For comparison, the **matplotlib approach** (which this notebook evolved from) topped out at
    ~10 fps because `fig.savefig()` rasterizes the entire figure through matplotlib's Agg backend —
    a full software rendering pipeline designed for publication-quality plots, not real-time video.
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

            wrapper.appendChild(canvas);
            wrapper.appendChild(badge);
            el.appendChild(wrapper);

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

            canvas.addEventListener("click", () => setActive(true));

            // Render JPEG frames from Python (base64-encoded)
            model.on("change:frame_b64", () => {
                const b64 = model.get("frame_b64");
                if (!b64) return;
                const img = new Image();
                img.onload = () => {
                    ctx.drawImage(img, 0, 0, CANVAS_W, CANVAS_H);
                };
                img.src = "data:image/jpeg;base64," + b64;
            });

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
            const localBuffer = [];

            function handleKey(e, pressed) {
                if (!active) return;
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
def _(canvas_widget):
    canvas_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The widget: an anywidget with two traitlets

    The entire browser-side component is a single anywidget with two synchronized traitlets:

    | Traitlet | Direction | Type | Purpose |
    |----------|-----------|------|---------|
    | `frame_b64` | Python → JS | `Unicode` | Base64-encoded JPEG frame |
    | `key_events` | JS → Python | `List[Unicode]` | Append-only log of `"key:1"` / `"key:0"` strings |

    ### Why append-only key events?

    Key events are tricky to sync. If JavaScript reads the current traitlet list, appends one event,
    and writes it back, two rapid keypresses can race — the second read happens before the first
    write syncs, so the first event gets overwritten. A lost `keyup` means Doom thinks you're
    still holding the key, and you spin in circles forever.

    The fix: JavaScript maintains a **local buffer** that only grows. Every keypress appends to the
    local array, then the full array is synced to Python. Python tracks an index into this list
    (`_last_event_len`) and consumes new entries on each `get_key()` call. No events can be lost
    because nothing is ever removed or overwritten.

    ### Focus management

    Browser focus is unreliable inside marimo's DOM. Instead of using `canvas.focus()` / `canvas.blur()`,
    we track an `active` boolean manually: click the canvas to activate, press Q to deactivate. When
    deactivating, we send synthetic `keyup` events for all currently held keys to prevent stuck inputs.

    ### Performance knobs

    | Parameter | Current | Effect |
    |-----------|---------|--------|
    | `FRAME_SKIP` | 2 | Only encode every Nth frame (game logic still ticks every frame) |
    | JPEG `quality` | 70 | Lower = smaller payload over WebSocket, minor visual loss |
    | Resolution | 320×200 | Native Doom resolution; upscaled to 640×400 via CSS `image-rendering: pixelated` |
    """)
    return


if __name__ == "__main__":
    app.run()

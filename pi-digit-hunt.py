# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "marimo",
#     "manim>=0.18.0",
#     "av>=14.0.0",
#     "mpmath",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hunting for numbers inside $\pi$

    A manim animation. We start with $\pi$ and its decimal expansion
    $3.14159\ldots$, then scroll the digits leftward like a tape past a fixed
    reading frame. We slow down as the sequence **19** arrives, freeze on it, and
    show its position. Then the same hunt — a bit faster — for **1988**, which
    lives much deeper in $\pi$.
    """)
    return


@app.cell
def _():
    from mpmath import mp

    mp.dps = 4900
    # decimals after the "3." — enough to cover 8888 (index 4751) plus context
    decimals = mp.nstr(mp.pi, 4850, strip_zeros=False).split(".")[1][:4800]
    return (decimals,)


@app.cell
def _(decimals):
    import os
    # Add LaTeX to PATH before importing manim (so it finds latex on import)
    os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ.get("PATH", "")

    from manim import (
        Scene, Text, MathTex, Triangle, FadeIn, Write,
        always_redraw, ValueTracker, rate_functions, PI,
        UP, RIGHT, ORIGIN, WHITE, YELLOW,
    )

    class PiDigitHunt(Scene):
        def construct(self):
            full = "3." + decimals          # the digit stream; char 2 onward are decimals
            font_size = 84                  # big digits
            line_y = -2.6                   # the number line sits below the centered pi
            frame_x = 0.0                   # the reading cursor (screen centre)

            # uniform monospace advance, measured once at this font size
            ruler = Text("0123456789", font="Monospace", font_size=font_size)
            w = (ruler[9].get_center()[0] - ruler[0].get_center()[0]) / 9

            t = ValueTracker(0.0)           # fractional char-index of `full` at the cursor
            found = []                      # [(char_start, char_end)] to paint yellow

            # Render only the ~40 digits around the cursor, rebuilt every frame. Avoids
            # ever holding one giant 1500-glyph mobject (which manim fails to re-render).
            def make_line():
                c = t.get_value()
                mid = int(round(c))
                lo, hi = max(0, mid - 20), min(len(full), mid + 20)
                line = Text(full[lo:hi], font="Monospace", font_size=font_size, color=WHITE)
                line.set_y(line_y)
                line.shift(RIGHT * ((frame_x - (c - lo) * w) - line[0].get_center()[0]))
                for cs, ce in found:                       # keep matches lit as they pass
                    for i in range(cs, ce):
                        if lo <= i < hi:
                            line[i - lo].set_color(YELLOW)
                return line

            number_line = always_redraw(make_line)

            # Ticking index counter — the index itself, no explanatory sentence.
            counter = always_redraw(
                lambda: Text(f"digit {max(0, round(t.get_value()) - 1)}",
                             font="Monospace", font_size=46, color=YELLOW)
                .to_corner(UP + RIGHT, buff=0.6)
            )

            pointer = Triangle(color=YELLOW, fill_opacity=1).scale(0.13).rotate(PI)
            pointer.move_to([frame_x, line_y + 0.95, 0])

            pi_sym = MathTex(r"\pi", font_size=240).move_to(ORIGIN)

            # ===== intro: big centred pi, then the number line beneath it =====
            self.play(Write(pi_sym))
            self.wait(0.4)
            head = Text(full[:20], font="Monospace", font_size=font_size, color=WHITE)
            head.set_y(line_y)
            head.shift(RIGHT * (frame_x - head[0].get_center()[0]))
            self.play(Write(head), FadeIn(pointer))
            self.wait(0.8)
            self.remove(head)                              # swap to the live sliding window
            self.add(number_line, counter)

            def lock_on(target, start):                    # light up the match, then hold
                found.append((start, start + len(target)))
                # a brief no-movement play forces a live frame so the new highlight is
                # actually drawn (a bare self.wait would just hold the pre-highlight frame).
                self.play(t.animate.set_value(start), run_time=0.5)
                self.wait(1.5)

            def accel_cruise_decel(a, ramp=0.45):
                # trapezoidal velocity: constant acceleration up, a high-speed cruise,
                # then constant deceleration down — area (total distance) normalised to 1.
                v = 1.0 / (1.0 - ramp)                     # cruise speed
                if a < ramp:                               # ramping up
                    return v * a * a / (2 * ramp)
                if a < 1 - ramp:                           # cruising
                    return v * ramp / 2 + v * (a - ramp)
                b = a - (1 - ramp)                         # ramping down
                p0 = v * ramp / 2 + v * (1 - 2 * ramp)
                return p0 + v * b - v * b * b / (2 * ramp)

            # hunt 1: "19" (37th digit) — slow & steady, decelerate onto it
            s1 = full.find("19")
            self.play(t.animate.set_value(s1 - 8), run_time=3.2, rate_func=rate_functions.linear)
            self.play(t.animate.set_value(s1), run_time=2.2, rate_func=rate_functions.rush_from)
            lock_on("19", s1)

            # hunt 2: "8888" (4751st digit) — accelerate gently, cruise fast, then slow onto it
            s2 = full.find("8888")
            self.play(t.animate.set_value(s2), run_time=11.0, rate_func=accel_cruise_decel)
            lock_on("8888", s2)
            self.wait(1.5)

    return (PiDigitHunt,)


@app.cell
def _(mo):
    # Resolution presets - CLI arg takes precedence
    _cli_res = mo.cli_args().get("resolution")
    _default = {
        "720p": "720p (fast)",
        "1080p": "1080p (high quality)",
        "4k": "4K (maximum quality)",
    }.get(_cli_res, "720p (fast)")

    resolution = mo.ui.dropdown(
        options={
            "720p (fast)": "720p",
            "1080p (high quality)": "1080p",
            "4K (maximum quality)": "4k",
        },
        value=_default,
        label="Resolution"
    )
    return (resolution,)


@app.cell
def _(mo, resolution):
    # In CLI mode (mo.app_meta().mode == "script"), auto-run. In edit mode, use button.
    _is_cli = mo.app_meta().mode == "script"
    render_button = mo.ui.run_button(label="Render Animation")
    mo.hstack([resolution, render_button], justify="start", gap=1) if not _is_cli else mo.md("*Auto-rendering in CLI mode...*")
    return (render_button,)


@app.cell
def _(PiDigitHunt, mo, render_button, resolution):
    from pathlib import Path

    # Auto-run in CLI mode, otherwise wait for button
    _is_cli = mo.app_meta().mode == "script"
    mo.stop(not _is_cli and not render_button.value)

    # Resolution settings
    _resolution_presets = {
        "720p": {"height": 720, "width": 1280, "fps": 30, "folder": "720p30"},
        "1080p": {"height": 1080, "width": 1920, "fps": 60, "folder": "1080p60"},
        "4k": {"height": 2160, "width": 3840, "fps": 60, "folder": "2160p60"},
    }
    _preset = _resolution_presets[resolution.value]

    # Output directory in the same folder as the notebook
    _output_dir = Path(__file__).parent / "media" if "__file__" in dir() else Path.cwd() / "media"
    _output_dir.mkdir(exist_ok=True)

    # Render the scene
    from manim import config
    config.media_dir = str(_output_dir)
    config.pixel_height = _preset["height"]
    config.pixel_width = _preset["width"]
    config.frame_rate = _preset["fps"]

    scene = PiDigitHunt()
    scene.render()

    # Find the final output file (not partial files)
    _video_path = _output_dir / "videos" / _preset["folder"] / "PiDigitHunt.mp4"

    if _video_path.exists():
        with open(_video_path, 'rb') as f:
            _video_data = f.read()
        output = mo.vstack([
            mo.md(f"**Video saved to:** `{_video_path}`"),
            mo.video(src=_video_data)
        ])
    else:
        output = mo.md("**Error:** Could not find rendered video")

    output
    return


if __name__ == "__main__":
    app.run()

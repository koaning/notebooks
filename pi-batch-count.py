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
    # Counting **88** in batches across $\pi$

    A manim animation. We take $\pi$'s endless digit stream, **cut it into batches of
    two**, and slide along counting how often the batch is exactly **88**. We keep a
    running tally, race through the first **1,000,000** digits, then stop and read off
    the count.
    """)
    return


@app.cell
def _():
    from mpmath import mp

    # First 1,000,000 decimal digits of pi (~10s to compute).
    mp.dps = 1_000_050
    decimals = mp.nstr(mp.pi, 1_000_020, strip_zeros=False).split(".")[1][:1_000_000]

    # Non-overlapping batches of size 2; record which batches equal "88".
    hits = [j for j in range(len(decimals) // 2) if decimals[2 * j:2 * j + 2] == "88"]
    count88 = len(hits)
    return count88, decimals, hits


@app.cell
def _(count88, decimals, hits):
    import os
    # Add LaTeX to PATH before importing manim (so it finds latex on import)
    os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ.get("PATH", "")

    import bisect

    from manim import (
        Scene, Text, Triangle, VGroup, FadeIn, FadeOut, Write,
        always_redraw, ValueTracker, rate_functions, PI,
        UP, DOWN, LEFT, RIGHT, ORIGIN, WHITE, YELLOW, GREY, BLUE,
    )

    class PiBatchCount(Scene):
        def construct(self):
            font_size = 72
            line_y = -0.5
            cursor_x = 0.0
            num_batches = len(decimals) // 2

            ruler = Text("0123456789", font="Monospace", font_size=font_size)
            w = (ruler[9].get_center()[0] - ruler[0].get_center()[0]) / 9
            max_gap = 0.9 * w                        # extra space opened between batches

            b = ValueTracker(0.0)                    # batch index sitting at the cursor
            gap = ValueTracker(0.0)                  # 0 = one stream, 1 = split into pairs

            # Render only the ~15 batches around the cursor, rebuilt each frame.
            def make_line():
                bc = b.get_value()
                slot = 2 * w + gap.get_value() * max_gap
                cur = int(bc)
                center = int(round(bc))
                lo, hi = max(0, center - 7), min(num_batches, center + 8)
                group = VGroup()
                for j in range(lo, hi):
                    pair = decimals[2 * j:2 * j + 2]
                    chip = Text(pair, font="Monospace", font_size=font_size, color=WHITE)
                    chip.move_to([cursor_x + (j - bc) * slot, line_y, 0])
                    if pair == "88" and j <= cur:    # a counted match stays lit
                        chip.set_color(YELLOW)
                    elif j == cur:                   # the batch under the cursor
                        chip.set_color(BLUE)
                    group.add(chip)
                return group

            number_line = always_redraw(make_line)

            pointer = Triangle(color=YELLOW, fill_opacity=1).scale(0.12).rotate(PI)
            pointer.move_to([cursor_x, line_y + 0.85, 0])

            digits_counter = always_redraw(lambda: Text(
                f"digits checked: {2 * (int(b.get_value()) + 1):,}",
                font="Monospace", font_size=34, color=WHITE).to_corner(UP + LEFT, buff=0.6))
            found_counter = always_redraw(lambda: Text(
                f"88 found: {bisect.bisect_right(hits, int(b.get_value()))}",
                font="Monospace", font_size=42, color=YELLOW).to_corner(UP + RIGHT, buff=0.6))

            def caption(text):
                # below the number line, clear of the counters in the top corners
                return Text(text, font="Monospace", font_size=36, color=GREY).move_to([0, -2.7, 0])

            # ===== intro: the long stream, then cut into batches of two =====
            self.add(number_line)
            cap = caption("π's digits — one long stream")
            self.play(Write(cap))
            self.wait(1.2)
            cap2 = caption("cut into batches of 2")
            self.play(gap.animate.set_value(1.0), FadeOut(cap), FadeIn(cap2), run_time=1.6)
            self.wait(1.0)

            # ===== bring in the cursor + counters and state the goal =====
            cap3 = caption("count how often a batch is 88")
            self.add(digits_counter, found_counter)
            self.play(FadeIn(pointer), FadeOut(cap2), FadeIn(cap3))
            self.wait(1.2)
            self.play(FadeOut(cap3))

            # ===== check the first few pairs slowly (none are 88) =====
            self.play(b.animate.set_value(6), run_time=3.0, rate_func=rate_functions.linear)

            # ===== land on each of the first three 88s; the tally ticks 1 → 2 → 3 =====
            cap4 = caption("…there's an 88 — tally it")
            self.play(b.animate.set_value(hits[0]), run_time=2.6, rate_func=rate_functions.rush_from)
            self.play(FadeIn(cap4))
            self.wait(1.2)
            self.play(b.animate.set_value(hits[1]), run_time=2.2, rate_func=rate_functions.rush_from)
            self.wait(1.1)
            self.play(b.animate.set_value(hits[2]), run_time=1.8, rate_func=rate_functions.rush_from)
            self.wait(1.1)
            self.play(FadeOut(cap4))

            # ===== fast-forward through the rest, counting every 88 along the way =====
            cap5 = caption("…fast-forward to 1,000,000 digits")
            self.play(FadeIn(cap5))
            self.play(b.animate.set_value(num_batches - 1),
                      run_time=9.0, rate_func=rate_functions.rush_into)
            self.play(FadeOut(cap5))
            self.wait(0.5)

            # ===== stop and read off the count =====
            for mob in (number_line, pointer, digits_counter, found_counter):
                mob.clear_updaters()
            self.play(FadeOut(number_line), FadeOut(pointer),
                      FadeOut(digits_counter), FadeOut(found_counter))
            summary = VGroup(
                Text("first 1,000,000 digits of π", font="Monospace", font_size=40, color=GREY),
                Text(f"88 appears {count88:,} times", font="Monospace", font_size=66, color=YELLOW),
            ).arrange(DOWN, buff=0.5).move_to(ORIGIN)
            self.play(Write(summary))
            self.wait(2.5)

    return (PiBatchCount,)


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
def _(PiBatchCount, mo, render_button, resolution):
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

    scene = PiBatchCount()
    scene.render()

    # Find the final output file (not partial files)
    _video_path = _output_dir / "videos" / _preset["folder"] / "PiBatchCount.mp4"

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the count means

    With non-overlapping batches of two digits there are **500,000** batches in the
    first million digits, and each is equally likely to be any of the 100 values
    `00`–`99`. So **88** should turn up about `500,000 / 100 = 5,000` times — and π
    delivers **4,988**, right on the money. Same flavour of "π behaves like a random
    digit stream" as the coupon-collector notebook, seen through a single value.
    """)
    return


if __name__ == "__main__":
    app.run()

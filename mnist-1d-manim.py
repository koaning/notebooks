# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "marimo",
#     "manim>=0.18.0",
#     "av>=14.0.0",
#     "numpy==2.4.3",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy.ndimage import gaussian_filter

    return gaussian_filter, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MNIST-1D: how each sample is generated

    [MNIST-1D](https://www.alphaxiv.org/abs/2011.14439v4) (Greydanus & Kobak)
    starts from **10 hand-coded length-12 "templates"** — one clean curve per
    digit. Every training sample is one template run through a fixed chain of
    randomized transformations that destroys absolute position information:

    1. **pad** — append a random number of zeros
    2. **dilate** — resample everything to a common length (72)
    3. **scale** — randomly stretch the amplitude
    4. **translate** — circularly shift the signal
    5. **noise** — correlated noise in the padded gaps + fine iid jitter
    6. **shear** — add a linear tilt
    7. **subsample** — resample down to the final 40-point feature vector

    This notebook animates all 10 templates going through that pipeline *at the
    same time*, so you can watch clean curves turn into the messy signals that a
    linear classifier can no longer separate.

    **Prerequisites:** Manim needs `ffmpeg`, `pkg-config`, and LaTeX to render.
    On macOS:

    ```bash
    brew install pkg-config ffmpeg mactex-no-gui
    ```
    """)
    return


@app.cell
def _(gaussian_filter, np):
    # The exact MNIST-1D pipeline, reimplemented inline (see mnist1d/transform.py).
    # Default args from mnist1d.data.get_dataset_args().
    TEMPLATE_LEN = 12
    PADDING = (36, 60)
    SCALE_COEFF = 1.2  # paper uses 0.4; exaggerated here so the stretch is visible
    MAX_TRANSLATION = 48
    CORR_NOISE_SCALE = 0.25
    IID_NOISE_SCALE = 0.02
    SHEAR_SCALE = 0.75
    FINAL_SEQ_LENGTH = 40
    DILATED_LEN = TEMPLATE_LEN + PADDING[1]  # 72

    def get_templates():
        d0 = np.asarray([5, 6, 6.5, 6.75, 7, 7, 7, 7, 6.75, 6.5, 6, 5])
        d1 = np.asarray([5, 3, 3, 3.4, 3.8, 4.2, 4.6, 5, 5.4, 5.8, 5, 5])
        d2 = np.asarray([5, 6, 6.5, 6.5, 6, 5.25, 4.75, 4, 3.5, 3.5, 4, 5])
        d3 = np.asarray([5, 6, 6.5, 6.5, 6, 5, 5, 6, 6.5, 6.5, 6, 5])
        d4 = np.asarray([5, 4.4, 3.8, 3.2, 2.6, 2.6, 5, 5, 5, 5, 5, 5])
        d5 = np.asarray([5, 3, 3, 3, 3, 5, 6, 6.5, 6.5, 6, 4.5, 5])
        d6 = np.asarray([5, 4, 3.5, 3.25, 3, 3, 3, 3, 3.25, 3.5, 4, 5])
        d7 = np.asarray([5, 7, 7, 6.6, 6.2, 5.8, 5.4, 5, 4.6, 4.2, 5, 5])
        d8 = np.asarray([5, 4, 3.5, 3.5, 4, 5, 5, 4, 3.5, 3.5, 4, 5])
        d9 = np.asarray([5, 4, 3.5, 3.5, 4, 5, 5, 5, 5, 4.7, 4.3, 5])
        x = np.stack([d0, d1, d2, d3, d4, d5, d6, d7, d8, d9]).astype(float)
        x -= x.mean(1, keepdims=True)  # whiten
        x /= x.std(1, keepdims=True)
        x -= x[:, :1]  # signal starts at 0
        return x / 6.0

    def pad(x, p):
        return np.concatenate([x, np.zeros(p)])

    def interpolate(x, n):
        old = np.linspace(0, 1, len(x))
        new = np.linspace(0, 1, n)
        return np.interp(new, old, x)

    def scale(x, coeff):
        return x * (1 + coeff)

    def translate(x, k):
        # Circular shift. We roll *left* (signal exits the left edge and wraps in
        # from the right) — visually cleaner than the paper's right-roll, and a
        # circular shift's direction is just a cosmetic convention.
        return np.concatenate([x[k:], x[:k]]) if k else x

    def add_noise(x, rng):
        mask = x != 0
        corr = gaussian_filter(CORR_NOISE_SCALE * rng.standard_normal(x.shape), 2)
        x = mask * x + (1 - mask) * corr
        return x + IID_NOISE_SCALE * rng.standard_normal(x.shape)

    def shear(x, coeff):
        return x - coeff * np.linspace(-0.5, 0.5, len(x))

    def build_states(seed=2):
        # Capture every template's signal after each pipeline stage so the
        # animation can morph stage N -> stage N+1 across all 10 panels at once.
        rng = np.random.default_rng(seed)
        templates = get_templates()
        n = len(templates)

        # Draw per-sample random parameters up front (one set per template).
        pads = rng.integers(PADDING[0], PADDING[1] + 1, size=n)
        scale_coeffs = SCALE_COEFF * (rng.random(n) - 0.5)
        shifts = rng.integers(0, MAX_TRANSLATION, size=n)
        shear_coeffs = SHEAR_SCALE * (rng.random(n) - 0.5)

        s_template, s_pad, s_dilate, s_scale = [], [], [], []
        s_translate, s_noise, s_shear, s_final = [], [], [], []
        for i in range(n):
            x = templates[i]
            s_template.append(x)
            x = pad(x, int(pads[i]))
            s_pad.append(x)
            x = interpolate(x, DILATED_LEN)
            s_dilate.append(x)
            x = scale(x, scale_coeffs[i])
            s_scale.append(x)
            x = translate(x, int(shifts[i]))
            s_translate.append(x)
            x = add_noise(x, rng)
            s_noise.append(x)
            x = shear(x, shear_coeffs[i])
            s_shear.append(x)
            x = interpolate(x, FINAL_SEQ_LENGTH)
            s_final.append(x)

        return {
            "steps": [
                ("10 digit templates", s_template),
                ("pad — append zeros to the right", s_pad),
                ("dilate — stretch to a common length (72)", s_dilate),
                ("scale — stretch the amplitude", s_scale),
                ("translate — slide & wrap around", s_translate),
                ("add noise — correlated in gaps + iid jitter", s_noise),
                ("shear — add a linear tilt", s_shear),
                ("subsample — keep 40 points", s_final),
            ],
            "shifts": shifts,
        }

    _result = build_states()
    states = _result["steps"]
    shifts = _result["shifts"]
    return DILATED_LEN, shifts, states


@app.cell
def _(DILATED_LEN, np, shifts, states):
    import os

    # Add LaTeX to PATH before importing manim (captions use Text, so LaTeX is
    # not strictly required, but keep it available for safety).
    os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ.get("PATH", "")

    from manim import (
        Scene,
        VMobject,
        VGroup,
        Rectangle,
        Text,
        Dot,
        ValueTracker,
        Create,
        ReplacementTransform,
        FadeIn,
        FadeOut,
        Write,
        UP,
        DOWN,
        BLUE,
        YELLOW,
        GREY,
    )

    # Fonts from koaning.io: IBM Plex Sans for prose, JetBrains Mono for the
    # numeric digit labels.
    PROSE_FONT = "IBM Plex Sans"
    MONO_FONT = "JetBrains Mono"

    # Layout: 2 rows x 5 cols of small panels.
    COLS = 5
    BOX_W, BOX_H = 2.15, 1.95
    COL_X = [(c - (COLS - 1) / 2) * 2.65 for c in range(COLS)]
    ROW_Y = [1.35, -1.85]

    # DILATED_LEN (= 72) is the common length signals share once dilated; it is
    # imported from the pipeline cell above so there's a single source of truth.

    # Fixed vertical data range (across every state) so amplitude/shear/scale
    # changes are visible and comparable between steps.
    _all_vals = np.concatenate([sig for _, sigs in states for sig in sigs])
    Y_MIN = float(_all_vals.min()) - 0.05
    Y_MAX = float(_all_vals.max()) + 0.05

    def _bx(cx, frac):
        # Horizontal position: frac in [0, 1] across the panel's drawable width.
        left, right = cx - BOX_W / 2 + 0.08, cx + BOX_W / 2 - 0.08
        return left + frac * (right - left)

    def _by(cy, value):
        bottom, top = cy - BOX_H / 2 + 0.1, cy + BOX_H / 2 - 0.1
        return bottom + (value - Y_MIN) / (Y_MAX - Y_MIN) * (top - bottom)

    def _points(sig, cx, cy, x0=0.0, x1=1.0):
        fracs = np.linspace(x0, x1, len(sig))
        return [np.array([_bx(cx, f), _by(cy, v), 0.0]) for f, v in zip(fracs, sig)]

    def _make_line(sig, cx, cy, x0=0.0, x1=1.0, color=BLUE):
        line = VMobject(stroke_width=2.5, color=color)
        line.set_points_as_corners(_points(sig, cx, cy, x0, x1))
        return line

    def _make_dots(sig, cx, cy, color=YELLOW):
        return VGroup(
            *[Dot(p, radius=0.03, color=color) for p in _points(sig, cx, cy)]
        )

    def _span(n):
        # On the fixed 72-index grid, a length-n signal occupies the left n/72.
        return (n - 1) / (DILATED_LEN - 1)

    def _overshoot(t):
        # ease-out-back: shoots slightly past the target then settles, so the
        # amplitude stretch visibly "springs" instead of easing in unnoticed.
        c = 2.4
        return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2

    class MNIST1DGeneration(Scene):
        def construct(self):
            s = [sigs for _, sigs in states]  # signals per step
            centers = [(COL_X[d % COLS], ROW_Y[d // COLS]) for d in range(10)]

            title = Text(
                "MNIST-1D: how each sample is generated",
                font=PROSE_FONT,
                weight="SEMIBOLD",
                font_size=34,
            )
            title.to_edge(UP, buff=0.3)

            def make_caption(i):
                cap = Text(states[i][0], font=PROSE_FONT, font_size=26, color=YELLOW)
                cap.next_to(title, DOWN, buff=0.22)
                return cap

            caption = make_caption(0)
            self.play(Write(title), Write(caption))

            # Panel boxes + digit labels.
            boxes, labels = VGroup(), VGroup()
            for d in range(10):
                cx, cy = centers[d]
                box = Rectangle(width=BOX_W, height=BOX_H, color=GREY, stroke_width=1.5)
                box.move_to([cx, cy, 0])
                lbl = Text(str(d), font=MONO_FONT, font_size=20, color=GREY)
                lbl.move_to([cx - BOX_W / 2 + 0.2, cy + BOX_H / 2 - 0.2, 0])
                boxes.add(box)
                labels.add(lbl)
            self.play(Create(boxes), FadeIn(labels), run_time=1.2)

            # Step 0 — templates fill each panel.
            lines = VGroup(*[_make_line(s[0][d], *centers[d]) for d in range(10)])
            self.play(Create(lines), run_time=2)
            self.wait(1)

            # Step 1 — pad: the signal squeezes to the left while a flat zero
            # tail is drawn out to the right (zeros appended to the end).
            new_caption = make_caption(1)
            anims, parts = [ReplacementTransform(caption, new_caption)], []
            for d in range(10):
                cx, cy = centers[d]
                tlen, plen = len(s[0][d]), len(s[1][d])
                compressed = _make_line(s[0][d], cx, cy, 0.0, _span(tlen))
                tail = _make_line(
                    np.zeros(plen - tlen), cx, cy, tlen / (DILATED_LEN - 1), _span(plen)
                )
                anims += [ReplacementTransform(lines[d], compressed), Create(tail)]
                parts.append((compressed, tail))
            self.play(*anims, run_time=1.8)
            caption = new_caption
            # Fuse each (compressed, tail) pair into one line for the next step.
            new_lines = VGroup()
            for d in range(10):
                cx, cy = centers[d]
                fused = _make_line(s[1][d], cx, cy, 0.0, _span(len(s[1][d])))
                self.remove(*parts[d])
                self.add(fused)
                new_lines.add(fused)
            lines = new_lines
            self.wait(1)

            # Step 2 — dilate: stretch each padded signal out to the full width.
            new_caption = make_caption(2)
            new_lines = VGroup(*[_make_line(s[2][d], *centers[d]) for d in range(10)])
            self.play(
                ReplacementTransform(caption, new_caption),
                *[ReplacementTransform(lines[d], new_lines[d]) for d in range(10)],
                run_time=1.8,
            )
            caption, lines = new_caption, new_lines
            self.wait(1)

            # Step 3 — scale: each signal stretches vertically about its baseline.
            # Driven by a ValueTracker with a spring overshoot so the amplitude
            # change is clearly visible rather than a near-invisible morph.
            new_caption = make_caption(3)
            self.play(ReplacementTransform(caption, new_caption), run_time=0.5)
            caption = new_caption
            amp = ValueTracker(0.0)

            def make_scale_updater(d):
                cx, cy = centers[d]
                base, target = s[2][d], s[3][d]

                def updater(m):
                    a = amp.get_value()
                    m.set_points_as_corners(
                        _points(base + a * (target - base), cx, cy, 0.0, 1.0)
                    )

                return updater

            for d in range(10):
                lines[d].add_updater(make_scale_updater(d))
            self.play(amp.animate.set_value(1.0), run_time=1.8, rate_func=_overshoot)
            for d in range(10):
                lines[d].clear_updaters()
            new_lines = VGroup(*[_make_line(s[3][d], *centers[d]) for d in range(10)])
            for d in range(10):
                self.remove(lines[d])
                self.add(new_lines[d])
            lines = new_lines
            self.wait(1)

            # Step 4 — translate: a genuine circular slide. A ValueTracker rolls
            # each panel's signal to the left; whatever leaves the left edge wraps
            # back in cleanly from the right.
            new_caption = make_caption(4)
            slide = ValueTracker(0.0)
            up_factor = 8
            grid = np.linspace(0, 1, DILATED_LEN)
            ups = [
                np.interp(np.linspace(0, 1, DILATED_LEN * up_factor), grid, s[3][d])
                for d in range(10)
            ]

            def make_updater(d):
                cx, cy = centers[d]
                up, k = ups[d], int(shifts[d])

                def updater(m):
                    # Negative roll = leftward motion, entering from the right.
                    rolled = np.roll(up, -int(round(slide.get_value() * k * up_factor)))
                    m.set_points_as_corners(_points(rolled, cx, cy, 0.0, 1.0))

                return updater

            for d in range(10):
                lines[d].add_updater(make_updater(d))
            self.play(
                ReplacementTransform(caption, new_caption),
                slide.animate.set_value(1.0),
                run_time=2.2,
            )
            for d in range(10):
                lines[d].clear_updaters()
            caption = new_caption
            new_lines = VGroup(*[_make_line(s[4][d], *centers[d]) for d in range(10)])
            for d in range(10):
                self.remove(lines[d])
                self.add(new_lines[d])
            lines = new_lines
            self.wait(1)

            # Step 5 — noise grows into the signal (correlated in gaps + iid).
            new_caption = make_caption(5)
            new_lines = VGroup(*[_make_line(s[5][d], *centers[d]) for d in range(10)])
            self.play(
                ReplacementTransform(caption, new_caption),
                *[ReplacementTransform(lines[d], new_lines[d]) for d in range(10)],
                run_time=1.8,
            )
            caption, lines = new_caption, new_lines
            self.wait(1)

            # Step 6 — shear: a linear tilt across the whole signal.
            new_caption = make_caption(6)
            new_lines = VGroup(*[_make_line(s[6][d], *centers[d]) for d in range(10)])
            self.play(
                ReplacementTransform(caption, new_caption),
                *[ReplacementTransform(lines[d], new_lines[d]) for d in range(10)],
                run_time=1.6,
            )
            caption, lines = new_caption, new_lines
            self.wait(1)

            # Step 7 — subsample: lay the 40 sample points on top of the curve,
            # then fade the curve away so only the final feature vector remains.
            new_caption = make_caption(7)
            dots = VGroup(*[_make_dots(s[7][d], *centers[d]) for d in range(10)])
            self.play(
                ReplacementTransform(caption, new_caption),
                FadeIn(dots),
                run_time=1.5,
            )
            self.play(FadeOut(lines), run_time=1.0)
            self.wait(2)

    return (MNIST1DGeneration,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Render the animation

    Pick a resolution and click **Render**. Lower resolutions render much faster
    — use 720p while iterating, 1080p/4K for a final.
    """)
    return


@app.cell
def _(mo):
    # Resolution presets - CLI arg takes precedence.
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
        label="Resolution",
    )
    return (resolution,)


@app.cell
def _(mo, resolution):
    # In CLI mode (mo.app_meta().mode == "script"), auto-run. In edit mode, use button.
    _is_cli = mo.app_meta().mode == "script"
    render_button = mo.ui.run_button(label="Render Animation")
    (
        mo.hstack([resolution, render_button], justify="start", gap=1)
        if not _is_cli
        else mo.md("*Auto-rendering in CLI mode...*")
    )
    return (render_button,)


@app.cell
def _(MNIST1DGeneration, mo, render_button, resolution):
    from pathlib import Path

    # Auto-run in CLI mode, otherwise wait for the button.
    _is_cli = mo.app_meta().mode == "script"
    mo.stop(not _is_cli and not render_button.value)

    _resolution_presets = {
        "720p": {"height": 720, "width": 1280, "fps": 30, "folder": "720p30"},
        "1080p": {"height": 1080, "width": 1920, "fps": 60, "folder": "1080p60"},
        "4k": {"height": 2160, "width": 3840, "fps": 60, "folder": "2160p60"},
    }
    _preset = _resolution_presets[resolution.value]

    _output_dir = Path(__file__).parent / "media" if "__file__" in dir() else Path.cwd() / "media"
    _output_dir.mkdir(exist_ok=True)

    from manim import config

    config.media_dir = str(_output_dir)
    config.pixel_height = _preset["height"]
    config.pixel_width = _preset["width"]
    config.frame_rate = _preset["fps"]

    scene = MNIST1DGeneration()
    scene.render()

    _video_path = _output_dir / "videos" / _preset["folder"] / "MNIST1DGeneration.mp4"
    if _video_path.exists():
        with open(_video_path, "rb") as f:
            _video_data = f.read()
        output = mo.vstack(
            [
                mo.md(f"**Video saved to:** `{_video_path}`"),
                mo.video(src=_video_data),
            ]
        )
    else:
        output = mo.md("**Error:** Could not find rendered video")

    output
    return


if __name__ == "__main__":
    app.run()

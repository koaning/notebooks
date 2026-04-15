# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.4.4",
#     "matplotlib",
#     "pillow==12.2.0",
#     "openpiv",
#     "wigglystuff==0.3.2",
#     "mohtml==0.1.11",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from openpiv.filters import replace_outliers
    from openpiv.pyprocess import extended_search_area_piv, get_coordinates
    from openpiv.validation import sig2noise_val
    from PIL import Image
    from wigglystuff import PlaySlider

    return (
        Image,
        Path,
        PlaySlider,
        extended_search_area_piv,
        get_coordinates,
        io,
        mo,
        np,
        plt,
        replace_outliers,
        sig2noise_val,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # OpenPIV on a GIF

    Use [Particle Image Velocimetry](https://en.wikipedia.org/wiki/Particle_image_velocimetry)
    to estimate velocity fields between consecutive GIF frames
    and overlay motion arrows on the original video.

    **Preprocessing** is applied to each grayscale frame before PIV analysis.
    You can enable multiple steps; they are applied in order from top to bottom.

    - *High pass* removes slow-varying background. `sigma` controls the Gaussian blur
      size — higher values remove more background.
    - *Contrast stretch* clips intensity to a percentile range and rescales.
    - *Local variance normalization* equalizes local contrast using two Gaussian filters.

    **PIV settings** control the cross-correlation and visualization.

    - *Window size* — interrogation window in pixels. Larger = more context but coarser.
    - *Overlap* — overlap between neighboring windows. Higher = denser field.
    - *S2N threshold* — signal-to-noise filter. Vectors below this are replaced by interpolation.
    - *Arrow scale* — controls arrow length. Lower = bigger arrows.
    - *Show every Nth arrow* — subsamples the vector field for clarity.
    """)
    return


@app.cell
def _(Path, mo):
    gif_files = sorted(
        p.name
        for p in Path(__file__).parent.glob("*.gif")
        if p.name not in ("output.gif", "preprocessed.gif")
    )
    gif_picker = mo.ui.dropdown(gif_files, value=gif_files[0], label="Source GIF")
    slider_clip_low = mo.ui.slider(
        0, 50, step=1, value=5, label="Clip floor (intensity)"
    )
    return gif_picker, slider_clip_low


@app.cell
def _(mo):
    slider_window = mo.ui.slider(
        16, 128, step=16, value=48, label="Window size (px)"
    )
    slider_overlap = mo.ui.slider(0, 64, step=8, value=24, label="Overlap (px)")
    slider_s2n = mo.ui.slider(
        0.1, 1.5, step=0.01, value=1.05, label="S2N threshold"
    )
    slider_arrow_scale = mo.ui.slider(
        50, 500, step=10, value=150, label="Arrow scale (lower = bigger)"
    )
    slider_arrow_every = mo.ui.slider(
        1, 5, step=1, value=1, label="Show every Nth arrow"
    )
    slider_speed = mo.ui.slider(
        20, 500, step=10, value=100, label="Playback speed (ms/frame)"
    )
    return (
        slider_arrow_every,
        slider_arrow_scale,
        slider_overlap,
        slider_s2n,
        slider_speed,
        slider_window,
    )


@app.cell
def _():
    # No longer needed — preprocessing simplified to intensity clip
    return


@app.cell(hide_code=True)
def _(
    generate_btn,
    gif_picker,
    mo,
    slider_arrow_every,
    slider_arrow_scale,
    slider_clip_low,
    slider_overlap,
    slider_s2n,
    slider_speed,
    slider_window,
):
    preprocessing = mo.md(f"""
    **Preprocessing**

    {gif_picker}

    {slider_clip_low}
    """)

    piv_settings = mo.md(f"""
    **PIV settings**

    {slider_window}
    {slider_overlap}
    {slider_s2n}
    {slider_arrow_scale}
    {slider_arrow_every}
    {slider_speed}
    """)

    mo.sidebar(
        [
            preprocessing,
            piv_settings,
            generate_btn,
        ],
        width="30rem",
    )
    return


@app.cell
def _(Image, Path, gif_picker, np):
    gif = Image.open(Path(__file__).parent / gif_picker.value)
    frames_color = []
    frames_gray = []
    try:
        while True:
            frame = gif.copy()
            frames_color.append(np.array(frame.convert("RGB")))
            frames_gray.append(np.array(frame.convert("L"), dtype=np.float64))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    gif.seek(0)
    frame_duration = gif.info.get("duration", 100)
    frame_count = len(frames_color)
    return frame_count, frames_color, frames_gray


@app.cell
def _(np, slider_clip_low):
    def preprocess(frame):
        clipped = np.clip(frame, slider_clip_low.value, 255)
        return clipped.astype(np.float64)

    return (preprocess,)


@app.cell
def _(PlaySlider, frame_count, mo, slider_speed):
    preview_player = mo.ui.anywidget(
        PlaySlider(
            min_value=0,
            max_value=frame_count - 1,
            step=1,
            interval_ms=slider_speed.value,
            loop=True,
        )
    )
    return (preview_player,)


@app.cell(hide_code=True)
def _(
    Image,
    frames_color,
    frames_gray,
    io,
    mo,
    np,
    preprocess,
    preview_player,
):
    preview_idx = int(preview_player.value["value"])

    # Original frame as PNG bytes
    buf_orig = io.BytesIO()
    Image.fromarray(frames_color[preview_idx]).save(buf_orig, format="PNG")

    # Preprocessed frame
    preprocessed_frame = preprocess(frames_gray[preview_idx])


    def _to_preview_png(frame):
        clipped = np.clip(
            (frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255,
            0,
            255,
        )
        buf = io.BytesIO()
        Image.fromarray(clipped.astype(np.uint8)).save(buf, format="PNG")
        return buf.getvalue()


    buf_pre_bytes = _to_preview_png(preprocessed_frame)

    mo.vstack(
        [
            mo.md("### Preprocessing Preview"),
            preview_player,
            mo.hstack(
                [
                    mo.vstack(
                        [mo.md("**Original**"), mo.image(src=buf_orig.getvalue())]
                    ),
                    mo.vstack(
                        [mo.md("**Preprocessed**"), mo.image(src=buf_pre_bytes)]
                    ),
                ],
                justify="center",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    generate_btn = mo.ui.run_button(label="Generate PIV")
    generate_btn
    return (generate_btn,)


@app.cell
def _(
    Image,
    extended_search_area_piv,
    frame_count,
    frames_color,
    frames_gray,
    generate_btn,
    get_coordinates,
    io,
    mo,
    np,
    plt,
    preprocess,
    replace_outliers,
    sig2noise_val,
    slider_arrow_every,
    slider_arrow_scale,
    slider_overlap,
    slider_s2n,
    slider_window,
):
    mo.stop(
        not generate_btn.value,
        mo.md("Click **Generate PIV** to compute velocity fields for all frames."),
    )

    window_size = slider_window.value
    overlap = min(slider_overlap.value, window_size - 8)
    search_area_size = window_size
    s2n_threshold = slider_s2n.value
    arrow_scale = slider_arrow_scale.value
    arrow_every = slider_arrow_every.value

    processed = [preprocess(f) for f in frames_gray]

    # Compute PIV for each consecutive frame pair
    velocity_fields = []
    for i in range(frame_count - 1):
        u, v, s2n = extended_search_area_piv(
            processed[i],
            processed[i + 1],
            window_size=window_size,
            overlap=overlap,
            dt=1.0,
            search_area_size=search_area_size,
        )
        flags = sig2noise_val(s2n, threshold=s2n_threshold)
        u, v = replace_outliers(
            u, v, flags, method="localmean", max_iter=3, kernel_size=1
        )
        x, y = get_coordinates(
            frames_gray[i].shape,
            search_area_size=search_area_size,
            overlap=overlap,
        )
        velocity_fields.append((x, y, u, v))

    # Duplicate last field so every frame gets arrows
    velocity_fields.append(velocity_fields[-1])

    # Render arrows on each frame and store as PNG bytes
    h, w = frames_color[0].shape[:2]
    dpi = 100
    fig_w, fig_h = w / dpi, h / dpi

    original_bytes = []
    for f in frames_color:
        buf = io.BytesIO()
        Image.fromarray(f).save(buf, format="PNG")
        original_bytes.append(buf.getvalue())


    def _normalize_frame(f):
        clipped = np.clip((f - f.min()) / (f.max() - f.min() + 1e-8) * 255, 0, 255)
        buf = io.BytesIO()
        Image.fromarray(clipped.astype(np.uint8)).save(buf, format="PNG")
        return buf.getvalue()


    preprocessed_bytes = [_normalize_frame(f) for f in processed]

    overlay_bytes = []
    with mo.status.spinner("Rendering frames..."):
        for i in range(frame_count):
            x, y, u, v = velocity_fields[i]
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            ax.imshow(frames_color[i])
            ax.quiver(
                x[::arrow_every, ::arrow_every],
                y[::arrow_every, ::arrow_every],
                u[::arrow_every, ::arrow_every],
                v[::arrow_every, ::arrow_every],
                color="yellow",
                scale=arrow_scale,
                width=0.004,
                headwidth=4,
            )
            ax.set_axis_off()
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

            buf = io.BytesIO()
            fig.savefig(
                buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0
            )
            plt.close(fig)
            overlay_bytes.append(buf.getvalue())
    return original_bytes, overlay_bytes, preprocessed_bytes


@app.cell
def _(PlaySlider, mo, original_bytes, slider_speed):
    result_player = mo.ui.anywidget(
        PlaySlider(
            min_value=0,
            max_value=len(original_bytes) - 1,
            step=1,
            interval_ms=slider_speed.value,
            loop=True,
        )
    )
    return (result_player,)


@app.cell(hide_code=True)
def _(mo, original_bytes, overlay_bytes, preprocessed_bytes, result_player):
    idx = int(result_player.value["value"])

    original = mo.vstack(
        [
            mo.md("### Original"),
            mo.image(src=original_bytes[idx]),
        ]
    )
    preprocessed = mo.vstack(
        [
            mo.md("### Preprocessed"),
            mo.image(src=preprocessed_bytes[idx]),
        ]
    )
    output = mo.vstack(
        [
            mo.md("### Velocity Vectors"),
            mo.image(src=overlay_bytes[idx]),
        ]
    )
    mo.vstack(
        [
            mo.md("### Results"),
            result_player,
            mo.hstack([original, preprocessed, output], justify="center"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()

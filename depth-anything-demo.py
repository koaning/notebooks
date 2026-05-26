# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pillow",
#     "torch",
#     "transformers",
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import io

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from transformers import pipeline

    return Image, ImageDraw, io, mo, np, pipeline, plt, torch


@app.cell
def _(mo):
    mo.Html(
        """
        <h1>Depth Anything V2</h1>
        <p>
          Upload a photo and inspect the model's relative depth estimate.
          Brightness in the depth map is normalized per image, so it should be
          read as relative structure rather than metric distance.
        </p>
        """
    )
    return


@app.cell
def _(mo):
    upload = mo.ui.file(kind="button", label="Upload photo")
    max_side = mo.ui.slider(
        start=256,
        stop=1024,
        step=64,
        value=512,
        label="Max side",
    )
    colormap = mo.ui.dropdown(
        options=["magma", "viridis", "plasma", "inferno", "cividis", "gray"],
        value="magma",
        label="Colormap",
    )
    invert_depth = mo.ui.checkbox(value=False, label="Invert depth")
    return colormap, invert_depth, max_side, upload


@app.cell
def _(colormap, invert_depth, max_side, mo, upload):
    mo.vstack(
        [
            upload,
            mo.hstack([max_side, colormap, invert_depth], justify="start"),
        ]
    )
    return


@app.cell
def _(Image, ImageDraw, np):
    def make_sample_image():
        width, height = 900, 600
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        sky = np.dstack(
            [
                120 + 70 * (1 - y),
                165 + 55 * (1 - y),
                210 + 35 * (1 - y),
            ]
        )
        ground = np.dstack(
            [
                75 + 30 * y,
                115 + 45 * y,
                85 + 20 * y,
            ]
        )
        base = np.zeros((height, width, 3), dtype=np.uint8)
        horizon = int(height * 0.45)
        base[:horizon] = sky[:horizon]
        base[horizon:] = ground[horizon:]

        image = Image.fromarray(base).convert("RGB")
        draw = ImageDraw.Draw(image)

        draw.rectangle([610, 145, 760, 385], fill="#c65f46", outline="#6d3328", width=5)
        draw.polygon([(590, 145), (685, 70), (780, 145)], fill="#8b2f2a")
        draw.rectangle([655, 255, 715, 385], fill="#573126")

        draw.ellipse([95, 285, 245, 435], fill="#d9b36c", outline="#6d5534", width=5)
        draw.ellipse([132, 322, 208, 398], fill="#f2dc9b")

        for x, scale in [(345, 0.8), (430, 1.05), (515, 1.3)]:
            top = int(300 - scale * 95)
            bottom = int(445 + scale * 12)
            draw.rectangle([x, top, x + int(22 * scale), bottom], fill="#6b4a2f")
            draw.ellipse(
                [
                    x - int(38 * scale),
                    top - int(45 * scale),
                    x + int(60 * scale),
                    top + int(50 * scale),
                ],
                fill="#2f7f4f",
            )

        for i in range(12):
            x0 = 35 + i * 75
            y0 = 505 + i * 4
            draw.line([x0, y0, x0 + 70, y0 + 16], fill="#ddd2b2", width=3)

        return image

    return (make_sample_image,)


@app.cell
def _(Image, io, make_sample_image, upload):
    def load_uploaded_image(file_list):
        if file_list:
            return Image.open(io.BytesIO(file_list[0].contents)).convert("RGB")
        return make_sample_image()

    source_image = load_uploaded_image(upload.value)
    return (source_image,)


@app.cell
def _(Image, max_side, source_image):
    def resize_preserving_aspect(image, limit):
        width, height = image.size
        scale = min(1.0, limit / max(width, height))
        resized_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(resized_size, Image.Resampling.LANCZOS)

    inference_image = resize_preserving_aspect(source_image, max_side.value)
    return (inference_image,)


@app.cell
def _(torch):
    def detect_device():
        if torch.cuda.is_available():
            return "cuda", 0
        if torch.backends.mps.is_available():
            return "mps", "mps"
        return "cpu", -1

    device_label, pipeline_device = detect_device()
    return device_label, pipeline_device


@app.cell
def _(mo, pipeline, pipeline_device):
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"

    @mo.cache
    def load_depth_pipeline(device):
        return pipeline(
            task="depth-estimation",
            model=model_id,
            device=device,
        )

    depth_pipe = load_depth_pipeline(pipeline_device)
    return depth_pipe, model_id


@app.cell
def _(depth_pipe, inference_image):
    depth_result = depth_pipe(inference_image)
    raw_depth = depth_result["depth"]
    return (raw_depth,)


@app.cell
def _(Image, colormap, invert_depth, np, plt, raw_depth):
    def depth_to_array(depth):
        if isinstance(depth, Image.Image):
            return np.asarray(depth, dtype=np.float32)
        return np.asarray(depth, dtype=np.float32)

    def colorize_depth(depth, cmap_name, invert):
        depth_arr = depth_to_array(depth)
        depth_min = float(depth_arr.min())
        depth_max = float(depth_arr.max())
        denom = depth_max - depth_min
        if denom == 0:
            normalized = np.zeros_like(depth_arr, dtype=np.float32)
        else:
            normalized = (depth_arr - depth_min) / denom
        if invert:
            normalized = 1.0 - normalized

        rgba = plt.get_cmap(cmap_name)(normalized)
        rgb = (rgba[..., :3] * 255).astype("uint8")
        return Image.fromarray(rgb), depth_min, depth_max

    depth_image, depth_min, depth_max = colorize_depth(
        raw_depth,
        colormap.value,
        invert_depth.value,
    )
    return depth_image, depth_max, depth_min


@app.cell
def _(depth_image, inference_image, mo):
    mo.hstack(
        [
            mo.vstack([mo.md("**Input**"), inference_image]),
            mo.vstack([mo.md("**Relative depth**"), depth_image]),
        ],
        justify="start",
        align="start",
    )
    return


@app.cell
def _(depth_max, depth_min, device_label, inference_image, mo, model_id):
    mo.md(
        f"**Model:** `{model_id}`  \n"
        f"**Device:** `{device_label}`  \n"
        f"**Inference size:** `{inference_image.size[0]} x {inference_image.size[1]}`  \n"
        f"**Raw depth range:** `{depth_min:.3f}` to `{depth_max:.3f}`"
    )
    return


if __name__ == "__main__":
    app.run()

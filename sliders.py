# /// script
# dependencies = [
#     "marimo",
#     "mohtml==0.1.11",
#     "numpy==2.4.3",
#     "pillow==12.1.1",
#     "scipy==1.17.1",
#     "wigglystuff==0.2.40",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="columns", sql_output="polars")


@app.cell
def _():
    import marimo as mo
    from wigglystuff import Paint

    return Paint, mo


@app.cell
def _(mo):
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy.ndimage import map_coordinates

    upload = mo.ui.file(kind="button", label="Upload Image", filetypes=[".png"])
    c_re = mo.ui.slider(0.1, 2.0, step=0.01, value=1.0, label="Scale ($c_{re}$)")
    c_im = mo.ui.slider(-1.0, 1.0, step=0.01, value=0.0, label="Twist ($c_{im}$)")
    zoom = mo.ui.slider(0.0, 10.0, step=0.01, value=0.0, label="Zoom Depth")
    view_mode = mo.ui.radio(["Spiral Space", "Log Space"], value="Spiral Space")

    mo.md(
        f"""
        # 3Blue1Brown Droste Mapping
        Adjust the sliders to "align" the tiles.

        {mo.vstack([upload, view_mode] + [c_re, c_im, zoom], justify="space-around")}
        """
    )
    return (
        Image,
        ImageDraw,
        c_im,
        c_re,
        map_coordinates,
        np,
        upload,
        view_mode,
        zoom,
    )


@app.cell
def _(get_source, upload):
    get_source(upload.value)
    return


@app.cell
def _(Paint, get_source, mo, upload):
    _src = get_source(upload.value)
    widget = mo.ui.anywidget(Paint(init_image=_src))
    widget
    return (widget,)


@app.cell
def _(Image, ImageDraw, mo, upload):
    import io 

    @mo.cache
    def get_source(file_val):
        if file_val:
            raw_data = upload.value[0].contents

            # Use io.BytesIO to make it "look" like a file for PIL
            img = Image.open(io.BytesIO(raw_data)).convert("RGB")
            # Resize to a manageable size for real-time slider performance
            img.thumbnail((600, 600))
            return img

        # Procedural placeholder with alignment markers
        img = Image.new("RGB", (300, 300), (245, 245, 240))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 299, 299], outline="black", width=10) # Outer
        d.rectangle([100, 100, 200, 200], outline="red", width=5) # Inner
        d.text((120, 10), "OUTER", fill="black")
        return img

    return (get_source,)


@app.cell
def _(np, widget):
    # Prepare the image and its metadata once
    img_array = np.array(widget.get_pil())
    h, w, _ = img_array.shape
    return


@app.cell
def _(
    Image,
    c_im,
    c_re,
    get_source,
    map_coordinates,
    mo,
    np,
    upload,
    view_mode,
    zoom,
):
    @mo.cache
    def compute_droste(img, cr, ci, z_depth, mode):
        arr = np.array(img)
        h, w, _ = arr.shape

        # Grid in [-1, 1]
        y, x = np.indices((h, w))
        u, v = (x - w/2) / (w/2), (y - h/2) / (h/2)

        # 1. Log-Polar Coordinates
        # We add z_depth to the radius to "travel" through the spiral
        r_log = 0.5 * np.log(u**2 + v**2 + 1e-9) + (z_depth * 2)
        theta = np.arctan2(v, u)

        # 2. Apply the Complex Linear Map: f(z) = c * z
        # This is the "Bizarre Log Space" transformation
        z_real = cr * r_log - ci * theta
        z_imag = ci * r_log + cr * theta

        if mode == "Log Space":
            # Visualize the tiled grid
            map_x = (z_real * (w / 4)) % w
            map_y = (z_imag / (2 * np.pi) * h) % h
        else:
            # 3. Back to Cartesian with Periodic tiling
            # The modulo on the log-radius creates the infinite nesting
            r_final = np.exp(z_real % 1.0) 
            theta_final = z_imag

            map_x = (r_final * np.cos(theta_final) + 1) / 2 * w
            map_y = (r_final * np.sin(theta_final) + 1) / 2 * h

        # 4. Sampling with boundary handling
        out = np.stack([
            map_coordinates(arr[..., i], [map_y % h, map_x % w], order=1) 
            for i in range(3)
        ], axis=-1)

        return Image.fromarray(out)

    # --- 4. RENDER LOOP ---
    src = get_source(upload.value)
    dst = compute_droste(src, c_re.value, c_im.value, zoom.value, view_mode.value)

    mo.hstack([src, dst], justify="space-around")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

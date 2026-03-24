# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
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

    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt


    def generate_checkerboard(grid_size=512, squares=8):
        """Generates a grayscale checkerboard pattern."""
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        X, Y = np.meshgrid(x, y)

        # Logic: (x_index + y_index) % 2
        # We scale by 'squares' to determine density
        checker = (np.floor(X * squares) + np.floor(Y * squares)) % 2

        # Add a small dot at the origin to track movement
        center_mask = (X**2 + Y**2) < 0.01
        checker = np.where(center_mask, 0.5, checker)

        return checker


    def generate_uv_map(grid_size=512):
        """Generates a color map where Red=X and Green=Y."""
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        X, Y = np.meshgrid(x, y)

        # Create an RGB image (Height, Width, 3)
        uv_image = np.zeros((grid_size, grid_size, 3))
        uv_image[..., 0] = X  # Red channel
        uv_image[..., 1] = Y  # Green channel
        uv_image[..., 2] = 0.5  # Blue (constant)

        return uv_image

    return generate_checkerboard, generate_uv_map, np, plt


@app.cell
def _(generate_checkerboard, generate_uv_map, mo, plt):
    # Generate textures
    check = generate_checkerboard()
    uv = generate_uv_map()

    # Display
    def plot_two_imgs(check, uv):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(check, cmap="gray", extent=[-1, 1, -1, 1])
        ax[0].set_title("Checkerboard (Structure)")
        ax[1].imshow(uv, extent=[0, 1, 0, 1])
        ax[1].set_title("UV Map (Direction/Flow)")
        return mo.as_html(fig)

    plot_two_imgs(check, uv)
    return check, plot_two_imgs, uv


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Field notes: Complex Arrays

    It turns out that numpy supports complex number arrays!
    """)
    return


@app.cell
def _(np):
    _ = np.linspace(1, 10, 10)
    np.exp(_ + 1j)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This lets us do some amazing things.
    """)
    return


@app.cell
def _(np):
    res = 512 
    u = np.linspace(-2, 1, res)      # "Zoom" level (log-radius)
    v = np.linspace(-np.pi, np.pi, res) # Rotation (angle)
    U, V = np.meshgrid(u, v)
    W = U + 1j*V

    Z = np.exp(W)

    # 3. Pull colors from the checkerboard (z-plane)
    z_real, z_imag = np.real(Z), np.imag(Z)
    mask = (np.abs(z_real) <= 1) & (np.abs(z_imag) <= 1)
    mask
    return (mask,)


@app.cell
def _(mask):
    mask
    return


@app.cell
def _(check, np, plot_two_imgs, uv):
    def apply_map(texture, mapper, res=512, *args, **kwargs):
        u = np.linspace(-2, 1, res)      # "Zoom" level (log-radius)
        v = np.linspace(-np.pi, np.pi, res) # Rotation (angle)
        U, V = np.meshgrid(u, v)
        W = U + 1j*V

        # This is where the function is actually applied
        Z = mapper(W, *args, **kwargs)

        z_real, z_imag = np.real(Z), np.imag(Z)
        mask = (np.abs(z_real) <= 1) & (np.abs(z_imag) <= 1)

        tex_y = ((z_imag + 1) / 2 * (texture.shape[0] - 1)).astype(int)
        tex_x = ((z_real + 1) / 2 * (texture.shape[1] - 1)).astype(int)

        output = np.zeros_like(texture)
        output[mask] = texture[tex_y[mask], tex_x[mask]]
        return output

    def apply_log_map(texture, res=512):
        # 1. Create the w-plane (Rectangular canvas)
        # x maps to ln(r), y maps to theta
        u = np.linspace(-2, 1, res)      # "Zoom" level (log-radius)
        v = np.linspace(-np.pi, np.pi, res) # Rotation (angle)
        U, V = np.meshgrid(u, v)
        W = U + 1j*V

        # 2. Inverse map: z = exp(w)
        Z = np.exp(W)

        # 3. Pull colors from the checkerboard (z-plane)
        z_real, z_imag = np.real(Z), np.imag(Z)
        mask = (np.abs(z_real) <= 1) & (np.abs(z_imag) <= 1)

        # Map [-1, 1] coords to [0, res-1] pixel indices
        tex_y = ((z_imag + 1) / 2 * (texture.shape[0] - 1)).astype(int)
        tex_x = ((z_real + 1) / 2 * (texture.shape[1] - 1)).astype(int)

        output = np.zeros_like(texture)
        output[mask] = texture[tex_y[mask], tex_x[mask]]
        return output

    plot_two_imgs(apply_map(check, np.sqrt), apply_map(uv, np.sqrt))
    return


@app.cell
def _(np, plt):
    from wigglystuff import ChartPuck

    # 1. The Mapping Logic
    # We move 'w' (Target) and calculate 'z' (Source)
    def get_source_point(w_complex):
        # Let's use the Square Root mapping as our intuition builder
        return np.sqrt(w_complex)

    def draw_mapping_viz(ax, widget):
        # 1. Clear the main axis to start fresh
        ax.axis('off')
    
        # 2. Create two actual sub-axes inside the main puck area
        # [left, bottom, width, height]
        ax_source = ax.inset_axes([0.0, 0.1, 0.45, 0.8]) # Left side (z)
        ax_target = ax.inset_axes([0.55, 0.1, 0.45, 0.8]) # Right side (w)
    
        # 3. Pull Puck coordinates (w-plane)
        w_x, w_y = widget.x[0], widget.y[0]
        w_val = complex(w_x, w_y)
    
        # 4. Map back to Source (z-plane)
        z_val = np.sqrt(w_val)
    
        # --- TARGET AXIS (Right) ---
        ax_target.set_title("Target Canvas (w)")
        ax_target.set_xlim(-2.5, 2.5); ax_target.set_ylim(-2.5, 2.5)
        ax_target.grid(True, alpha=0.2)
        ax_target.axhline(0, color='black', lw=0.5); ax_target.axvline(0, color='black', lw=0.5)
        # The Puck you move
        ax_target.scatter(w_x, w_y, color='red', s=100, zorder=5)
        # Visual "crosshair" for the puck
        ax_target.axhline(w_y, color='red', alpha=0.2, linestyle='--')
        ax_target.axvline(w_x, color='red', alpha=0.2, linestyle='--')

        # --- SOURCE AXIS (Left) ---
        ax_source.set_title("Source Sticker (z)")
        ax_source.set_xlim(-2.5, 2.5); ax_source.set_ylim(-2.5, 2.5)
        ax_source.grid(True, alpha=0.2)
        ax_source.axhline(0, color='black', lw=0.5); ax_source.axvline(0, color='black', lw=0.5)
    
        # Draw the "Sticker" (The -1 to 1 bounds)
        sticker = plt.Rectangle((-1, -1), 2, 2, facecolor='gray', alpha=0.1, edgecolor='blue')
        ax_source.add_patch(sticker)
    
        # The "Ghost" point being sampled
        ax_source.scatter(z_val.real, z_val.imag, color='blue', s=100, zorder=5)
        # Visual "crosshair" for the sample point
        ax_source.axhline(z_val.imag, color='blue', alpha=0.2, linestyle='--')
        ax_source.axvline(z_val.real, color='blue', alpha=0.2, linestyle='--')
                      
    puck = ChartPuck.from_callback(
        draw_fn=draw_mapping_viz,
        x_bounds=(-2, 2), y_bounds=(-2, 2), # Puck moves in w-space
        figsize=(10, 5), x=1.0, y=0.5
    )
    puck
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

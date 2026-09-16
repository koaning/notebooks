# /// script
# requires-python = "==3.12"
# dependencies = [
#     "manim==0.19.1",
#     "manim-slides==5.5.2",
#     "marimo>=0.24.2",
#     "mohtml==0.1.11",
#     "moterm==0.1.0",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="columns", sql_output="polars")

with app.setup:
    import numpy as np
    from manim import (
        DecimalNumber,
        VGroup,
        MovingCameraScene,
        Write,
        config,
        ORIGIN,
        WHITE,
        YELLOW,
    )
    from manim_slides import Slide

    N = 300  # length of the random vector
    SEED = 0
    FOCUS = N // 2  # the single number we open zoomed in on
    DY = 0.7  # vertical spacing between successive numbers

    ASPECT = config.frame_width / config.frame_height

    def numbers():
        # Deterministic draw so every slide (rendered independently) builds the
        # exact same column and the camera hand-offs line up.
        rng = np.random.default_rng(SEED)
        return rng.normal(size=N)

    def make_column():
        vals = numbers()
        col = VGroup(
            *[
                DecimalNumber(v, num_decimal_places=2, include_sign=True)
                for v in vals
            ]
        )
        for i, num in enumerate(col):
            num.move_to([0, -i * DY, 0])
        # Put the focus number exactly at the origin and paint it yellow, so the
        # opening close-up has a clear subject and the frame stays centred.
        col.shift(-col[FOCUS].get_center())
        col[FOCUS].set_color(YELLOW)
        return col

    def frame_width_for(rows):
        """Camera-frame width that vertically fits `rows` numbers."""
        return rows * DY * ASPECT

    # Zoom breakpoints, in number of visible rows. Each slide's end matches the
    # next slide's start so the stitched deck zooms out continuously.
    ROWS = [1.3, 4, 12, 55, N * 1.03]

    def open_frame(scene, rows):
        scene.camera.frame.move_to(ORIGIN).set(width=frame_width_for(rows))

    def zoom_to(scene, rows, run_time=3):
        scene.play(
            scene.camera.frame.animate.set(width=frame_width_for(rows)),
            run_time=run_time,
        )


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    return Path, mo


@app.class_definition
class Slide1One(Slide, MovingCameraScene):
    def construct(self):
        col = make_column()
        open_frame(self, ROWS[0])
        # Add every number except the focus one, then write the focus in; the
        # neighbours sit just outside the tight frame until we zoom out.
        self.add(*[n for i, n in enumerate(col) if i != FOCUS])
        self.play(Write(col[FOCUS]))
        zoom_to(self, ROWS[1])


@app.class_definition
class Slide2Few(Slide, MovingCameraScene):
    def construct(self):
        col = make_column()
        open_frame(self, ROWS[1])
        self.add(col)
        zoom_to(self, ROWS[2])


@app.class_definition
class Slide3Many(Slide, MovingCameraScene):
    def construct(self):
        col = make_column()
        open_frame(self, ROWS[2])
        self.add(col)
        zoom_to(self, ROWS[3])


@app.class_definition
class Slide4Full(Slide, MovingCameraScene):
    def construct(self):
        col = make_column()
        open_frame(self, ROWS[3])
        self.add(col)
        zoom_to(self, ROWS[4], run_time=4)


@app.cell
def _(Path, mo):
    import re
    import hashlib
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    scene_names = [
        "Slide1One",
        "Slide2Few",
        "Slide3Many",
        "Slide4Full",
    ]

    # Per-scene render cache: key = hash(scene block) + hash(shared setup block),
    # parsed from this file's own text. Unchanged scenes are skipped; editing the
    # setup block re-renders all slides, editing one scene re-renders only it.
    src = Path("random-vector-zoom.py").read_text()
    setup_block = re.search(r"with app\.setup:.*?(?=\n@app\.)", src, re.S).group(0)
    common = hashlib.md5(setup_block.encode()).hexdigest()
    blocks = {
        m.group(1): m.group(0)
        for m in re.finditer(
            r"@app\.class_definition\nclass (\w+).*?(?=\n@app\.|\nif __name__|\Z)",
            src,
            re.S,
        )
    }

    cache = Path(".zoom-cache")
    cache.mkdir(exist_ok=True)
    todo = []
    for name in scene_names:
        digest = hashlib.md5((blocks[name] + common).encode()).hexdigest()
        stamp = cache / f"{name}.hash"
        rendered = Path("slides") / f"{name}.json"
        if not (stamp.exists() and stamp.read_text() == digest and rendered.exists()):
            todo.append((name, digest))

    def render_one(item):
        name, digest = item
        result = subprocess.run(
            f"manim-slides render random-vector-zoom.py {name}",
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            (cache / f"{name}.hash").write_text(digest)
        return name, result.returncode

    # Render the first scene alone so manim creates the shared media directories
    # once; parallel renders then can't race to mkdir them. The rest run 4-wide.
    if todo:
        render_one(todo[0])
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(render_one, todo[1:]))

    subprocess.run(
        "manim-slides convert "
        + " ".join(scene_names)
        + " -c controls=true random_vector_zoom.html --one-file",
        shell=True,
        check=True,
    )

    mo.iframe(Path("random_vector_zoom.html").read_text())
    return


if __name__ == "__main__":
    app.run()

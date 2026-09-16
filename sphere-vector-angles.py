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
        ThreeDAxes,
        Sphere,
        Arrow3D,
        ParametricFunction,
        GrowFromCenter,
        Create,
        ORIGIN,
        BLUE,
        YELLOW,
        RED,
        GREEN,
        ORANGE,
        DEGREES,
    )
    from manim_slides import ThreeDSlide

    def normalize(v):
        v = np.array(v, dtype=float)
        return v / np.linalg.norm(v)

    def slerp(a, b, s):
        """Point fraction s along the great-circle arc from unit a to unit b."""
        omega = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
        so = np.sin(omega)
        return (np.sin((1 - s) * omega) * a + np.sin(s * omega) * b) / so

    def sph(az, el):
        """Unit vector at azimuth az and elevation el (degrees)."""
        a, e = np.radians(az), np.radians(el)
        return normalize([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])

    # Four fixed unit vectors fanned around the sphere: one per xy-quadrant and
    # split across both hemispheres, so the arcs sweep across the whole sphere
    # (some going below the equator) and stay easy to see.
    VECS = [sph(30, 40), sph(130, -25), sph(210, 35), sph(310, -45)]
    COLORS = [YELLOW, RED, GREEN, ORANGE]

    # Each slide does a half turn (180). Chaining the start angles by slide
    # index keeps every hand-off continuous (slide k ends where slide k+1 starts).
    PHI = 65
    ZOOM = 1.5
    BASE_THETA = 45

    def turn_start(idx):
        return BASE_THETA + 180 * idx

    def make_axes():
        return ThreeDAxes(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            z_range=[-1.5, 1.5, 1],
        ).set_opacity(0.35)

    def make_sphere(opacity):
        return Sphere(radius=1, resolution=(24, 24)).set_color(BLUE).set_opacity(opacity)

    def vec(i):
        return Arrow3D(start=ORIGIN, end=VECS[i], color=COLORS[i])

    def arc(j, i):
        # Pairwise angle arc between vectors j and i. Nested radii (by a stable
        # global index) keep the arcs from piling up on each other; coloured by
        # the newer vector i so it is clear which angles it introduced.
        idx = i * (i - 1) // 2 + j
        radius = 0.35 + 0.08 * idx
        angle = ParametricFunction(
            lambda s: radius * slerp(VECS[j], VECS[i], s),
            t_range=[0, 1, 0.02],
            color=COLORS[i],
        )
        angle.set_shade_in_3d(True)
        return angle

    def state_before(i):
        """Vectors 0..i-1 plus every pairwise arc among them (already on screen
        when vector i is about to be added)."""
        items = [vec(k) for k in range(i)]
        for k in range(i):
            for j in range(k):
                items.append(arc(j, k))
        return items

    def add_vector(scene, i):
        """Grow vector i, then draw its arcs to all earlier vectors."""
        scene.play(GrowFromCenter(vec(i)))
        if i > 0:
            scene.play(*[Create(arc(j, i)) for j in range(i)])


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    return Path, mo


@app.class_definition
class Slide1Sphere(ThreeDSlide):
    def construct(self):
        self.set_camera_orientation(phi=PHI * DEGREES, theta=turn_start(0) * DEGREES, zoom=ZOOM)
        self.add(make_axes())
        sphere = make_sphere(1.0)
        self.play(GrowFromCenter(sphere))
        self.move_camera(theta=(turn_start(0) + 180) * DEGREES, run_time=4)


@app.class_definition
class Slide2SeeThrough(ThreeDSlide):
    def construct(self):
        self.set_camera_orientation(phi=PHI * DEGREES, theta=turn_start(1) * DEGREES, zoom=ZOOM)
        sphere = make_sphere(1.0)
        self.add(make_axes(), sphere)
        self.play(sphere.animate.set_opacity(0.15))
        self.move_camera(theta=(turn_start(1) + 180) * DEGREES, run_time=4)


@app.class_definition
class Slide3Pair(ThreeDSlide):
    def construct(self):
        self.set_camera_orientation(phi=PHI * DEGREES, theta=turn_start(2) * DEGREES, zoom=ZOOM)
        self.add(make_axes(), make_sphere(0.15))
        # Start the vector part already showing two vectors and their one arc.
        self.play(GrowFromCenter(vec(0)), GrowFromCenter(vec(1)))
        self.play(Create(arc(0, 1)))
        self.move_camera(theta=(turn_start(2) + 180) * DEGREES, run_time=4)


@app.class_definition
class Slide4Vector3(ThreeDSlide):
    def construct(self):
        self.set_camera_orientation(phi=PHI * DEGREES, theta=turn_start(3) * DEGREES, zoom=ZOOM)
        self.add(make_axes(), make_sphere(0.15), *state_before(2))
        add_vector(self, 2)
        self.move_camera(theta=(turn_start(3) + 180) * DEGREES, run_time=4)


@app.class_definition
class Slide5Vector4(ThreeDSlide):
    def construct(self):
        self.set_camera_orientation(phi=PHI * DEGREES, theta=turn_start(4) * DEGREES, zoom=ZOOM)
        self.add(make_axes(), make_sphere(0.15), *state_before(3))
        add_vector(self, 3)
        self.move_camera(theta=(turn_start(4) + 180) * DEGREES, run_time=4)


@app.cell
def _(Path, mo):
    import re
    import hashlib
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    scene_names = [
        "Slide1Sphere",
        "Slide2SeeThrough",
        "Slide3Pair",
        "Slide4Vector3",
        "Slide5Vector4",
    ]

    # Per-scene render cache: key = hash(scene block) + hash(shared setup block),
    # parsed from this file's own text. Unchanged scenes are skipped; editing the
    # setup block re-renders all slides, editing one scene re-renders only it.
    src = Path("sphere-vector-angles.py").read_text()
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

    cache = Path(".angles-cache")
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
            f"manim-slides render sphere-vector-angles.py {name}",
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
        + " -c controls=true sphere_vector_angles.html --one-file",
        shell=True,
        check=True,
    )

    mo.iframe(Path("sphere_vector_angles.html").read_text())
    return


if __name__ == "__main__":
    app.run()

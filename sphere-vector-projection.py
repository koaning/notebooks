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
        Dot3D,
        Polygon,
        ParametricFunction,
        GrowFromCenter,
        Create,
        FadeIn,
        FadeOut,
        ORIGIN,
        BLUE,
        YELLOW,
        RED,
        GREEN,
        WHITE,
        TEAL,
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

    # Fixed, pleasant tilted pair so the plane is clearly off-axis.
    A = normalize([1.0, 0.3, 0.6])
    B = normalize([0.2, 1.0, 0.5])
    NORMAL = normalize(np.cross(A, B))
    U = A
    W = normalize(np.cross(NORMAL, U))
    THETA = np.arccos(np.clip(np.dot(A, B), -1.0, 1.0))

    # Camera keyframes per slide (start -> end). Each slide's end matches the
    # next slide's start so the stitched deck rotates gently and continuously.
    PHI = 60
    THETAS = [200, 206, 212, 218, 224, 230]  # slide i runs THETAS[i] -> THETAS[i+1]

    def nearest_angle(target, ref):
        """Co-terminal value of target (degrees) within +/-180 of ref, so an
        animated move_camera takes the short way round instead of spinning."""
        return target + 360.0 * round((ref - target) / 360.0)

    # Face-on camera: look straight down the plane's normal, so the plane reads
    # as a flat 2D picture and the great circle looks like a true circle. Snap
    # theta near slide 5's end angle so the rotation into face-on is a short
    # tilt, not a near-full spin around the circle.
    FACE_PHI = np.degrees(np.arccos(NORMAL[2]))
    FACE_THETA = nearest_angle(np.degrees(np.arctan2(NORMAL[1], NORMAL[0])), THETAS[4])

    def camera_rotation(phi, theta, gamma):
        """manim's world->screen rotation for a ThreeDCamera (radians)."""
        cz = lambda a: np.array(
            [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
        )
        cx = lambda a: np.array(
            [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
        )
        return cz(gamma) @ cx(-phi) @ cz(-theta - np.pi / 2)

    # Camera roll that makes the plane's edges (along U and W) horizontal and
    # vertical. Since the yellow vector A == U is one of those edges, this same
    # roll also lands the yellow vector on the +x (screen-horizontal) axis.
    u_on_screen = camera_rotation(np.radians(FACE_PHI), np.radians(FACE_THETA), 0.0) @ U
    GAMMA_FLAT = -np.degrees(np.arctan2(u_on_screen[1], u_on_screen[0]))

    # Builders — each scene reconstructs identical mobjects so slides can be
    # rendered independently and cached one at a time.
    def make_axes():
        return ThreeDAxes(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            z_range=[-1.5, 1.5, 1],
        ).set_opacity(0.35)

    def make_sphere(opacity):
        return Sphere(radius=1, resolution=(24, 24)).set_color(BLUE).set_opacity(opacity)

    def make_arrows():
        return (
            Arrow3D(start=ORIGIN, end=A, color=YELLOW),
            Arrow3D(start=ORIGIN, end=B, color=RED),
        )

    def make_angle_arc():
        arc = ParametricFunction(
            lambda s: 0.45 * slerp(A, B, s),
            t_range=[0, 1, 0.02],
            color=GREEN,
        )
        arc.set_shade_in_3d(True)
        return arc

    def make_dots():
        return (
            Dot3D(point=ORIGIN, color=WHITE),
            Dot3D(point=A, color=YELLOW),
            Dot3D(point=B, color=RED),
        )

    def make_plane():
        half = 1.3
        corners = [
            half * U + half * W,
            -half * U + half * W,
            -half * U - half * W,
            half * U - half * W,
        ]
        plane = Polygon(*corners, color=TEAL, fill_opacity=0.25, stroke_opacity=0.6)
        plane.set_shade_in_3d(True)
        return plane

    def make_circle():
        # Unit circle living in the plane: where the plane meets the sphere.
        circle = ParametricFunction(
            lambda t: np.cos(t) * U + np.sin(t) * W,
            t_range=[0, 2 * np.pi, 0.02],
            color=WHITE,
        )
        circle.set_shade_in_3d(True)
        return circle

    # Parameterized versions of the builders, used by the closing recap where
    # each cycle grabs a fresh random pair of vectors.
    def frame_for(a, b):
        a, b = normalize(a), normalize(b)
        nrm = normalize(np.cross(a, b))
        u = a
        w = normalize(np.cross(nrm, u))
        face_phi = np.degrees(np.arccos(nrm[2]))
        face_theta = np.degrees(np.arctan2(nrm[1], nrm[0]))
        return u, w, face_phi, face_theta

    def random_pair(seed):
        rng = np.random.default_rng(seed)
        return normalize(rng.normal(size=3)), normalize(rng.normal(size=3))

    def arrows_for(a, b):
        return (
            Arrow3D(start=ORIGIN, end=normalize(a), color=YELLOW),
            Arrow3D(start=ORIGIN, end=normalize(b), color=RED),
        )

    def plane_for(u, w):
        half = 1.3
        corners = [
            half * u + half * w,
            -half * u + half * w,
            -half * u - half * w,
            half * u - half * w,
        ]
        plane = Polygon(*corners, color=TEAL, fill_opacity=0.25, stroke_opacity=0.6)
        plane.set_shade_in_3d(True)
        return plane

    def circle_for(u, w):
        circle = ParametricFunction(
            lambda t: np.cos(t) * u + np.sin(t) * w,
            t_range=[0, 2 * np.pi, 0.02],
            color=WHITE,
        )
        circle.set_shade_in_3d(True)
        return circle

    def angle_arc_for(a, b):
        arc = ParametricFunction(
            lambda s: 0.45 * slerp(normalize(a), normalize(b), s),
            t_range=[0, 1, 0.02],
            color=GREEN,
        )
        arc.set_shade_in_3d(True)
        return arc

    def gamma_flat_for(u, face_phi, face_theta):
        """Camera roll that lands the yellow vector (== u) on the +x axis."""
        us = camera_rotation(np.radians(face_phi), np.radians(face_theta), 0.0) @ u
        return -np.degrees(np.arctan2(us[1], us[0]))

    def recap_cycle(scene, seed):
        """One recap beat: random pair -> angle -> plane -> rotate face-on ->
        circle -> roll the yellow vector onto the x-axis. Each cycle fades in
        from black at a fixed 3/4 view and holds on its 2D result, so there is
        no content teleport between cycles (the camera reset happens while the
        screen is black at the start of the next cycle)."""
        a, b = random_pair(seed)
        u, w, face_phi, face_theta = frame_for(a, b)
        gamma_flat = gamma_flat_for(u, face_phi, face_theta)
        scene.set_camera_orientation(phi=65 * DEGREES, theta=45 * DEGREES, zoom=1.3)
        scene.play(FadeIn(make_axes()), FadeIn(make_sphere(0.12)))
        arrow_a, arrow_b = arrows_for(a, b)
        scene.play(GrowFromCenter(arrow_a), GrowFromCenter(arrow_b))
        scene.play(Create(angle_arc_for(a, b)))
        scene.play(FadeIn(plane_for(u, w)))
        scene.move_camera(
            phi=face_phi * DEGREES,
            theta=nearest_angle(face_theta, 45) * DEGREES,
            zoom=1.5,
            run_time=3,
        )
        scene.play(Create(circle_for(u, w)))
        # Rotate in 2D so the yellow vector aligns with the x-axis.
        scene.move_camera(gamma=gamma_flat * DEGREES, run_time=2)


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    return Path, mo


@app.class_definition
class Slide1Sphere(ThreeDSlide):
    def construct(self):
        self.set_camera_orientation(phi=PHI * DEGREES, theta=THETAS[0] * DEGREES, zoom=1.0)
        self.add(make_axes())
        sphere = make_sphere(1.0)
        self.play(GrowFromCenter(sphere))
        self.move_camera(theta=THETAS[1] * DEGREES)


@app.class_definition
class Slide2SeeThrough(ThreeDSlide):
    def construct(self):
        self.set_camera_orientation(phi=PHI * DEGREES, theta=THETAS[1] * DEGREES, zoom=1.0)
        sphere = make_sphere(1.0)
        self.add(make_axes(), sphere)
        self.play(sphere.animate.set_opacity(0.15))
        self.move_camera(theta=THETAS[2] * DEGREES)


@app.class_definition
class Slide3Vectors(ThreeDSlide):
    def construct(self):
        self.set_camera_orientation(phi=PHI * DEGREES, theta=THETAS[2] * DEGREES, zoom=1.0)
        self.add(make_axes(), make_sphere(0.15))
        arrow_a, arrow_b = make_arrows()
        self.play(GrowFromCenter(arrow_a), GrowFromCenter(arrow_b))
        self.move_camera(theta=THETAS[3] * DEGREES, zoom=1.8)


@app.class_definition
class Slide4Angle(ThreeDSlide):
    def construct(self):
        self.set_camera_orientation(phi=PHI * DEGREES, theta=THETAS[3] * DEGREES, zoom=1.8)
        arrow_a, arrow_b = make_arrows()
        self.add(make_axes(), make_sphere(0.15), arrow_a, arrow_b)
        self.play(Create(make_angle_arc()))
        self.move_camera(theta=THETAS[4] * DEGREES)


@app.class_definition
class Slide5Plane(ThreeDSlide):
    def construct(self):
        self.set_camera_orientation(phi=PHI * DEGREES, theta=THETAS[4] * DEGREES, zoom=1.8)
        arrow_a, arrow_b = make_arrows()
        self.add(make_axes(), make_sphere(0.15), arrow_a, arrow_b, make_angle_arc())
        dot_o, dot_a, dot_b = make_dots()
        self.play(FadeIn(dot_o), FadeIn(dot_a), FadeIn(dot_b))
        # A full 360 orbit before the plane appears, to show the two vectors
        # sitting in 3D. It ends exactly where it started, so the hand-off into
        # the next slide has no cut.
        self.move_camera(theta=(THETAS[4] + 360) * DEGREES, run_time=6)
        self.play(FadeIn(make_plane()))


@app.class_definition
class Slide6FaceOn(ThreeDSlide):
    def construct(self):
        self.set_camera_orientation(phi=PHI * DEGREES, theta=THETAS[4] * DEGREES, zoom=1.8)
        arrow_a, arrow_b = make_arrows()
        dot_o, dot_a, dot_b = make_dots()
        self.add(
            make_axes(),
            make_sphere(0.15),
            arrow_a,
            arrow_b,
            make_angle_arc(),
            dot_o,
            dot_a,
            dot_b,
            make_plane(),
        )
        # Rotate the whole scene so we look straight down the plane's normal.
        self.move_camera(
            phi=FACE_PHI * DEGREES,
            theta=FACE_THETA * DEGREES,
            zoom=1.5,
            run_time=4,
        )
        # Trace the circle where the plane cuts the sphere.
        self.play(Create(make_circle()))


@app.class_definition
class Slide7Flatten(ThreeDSlide):
    def construct(self):
        self.set_camera_orientation(
            phi=FACE_PHI * DEGREES, theta=FACE_THETA * DEGREES, zoom=1.5
        )
        arrow_a, arrow_b = make_arrows()
        dot_o, dot_a, dot_b = make_dots()
        axes = make_axes()
        sphere = make_sphere(0.15)
        self.add(
            axes,
            sphere,
            arrow_a,
            arrow_b,
            make_angle_arc(),
            dot_o,
            dot_a,
            dot_b,
            make_plane(),
            make_circle(),
        )
        # Fade out the 3D scaffolding, leaving the flat 2D circle picture.
        self.play(FadeOut(sphere), FadeOut(axes))


@app.class_definition
class Slide8AlignSquare(ThreeDSlide):
    def construct(self):
        self.set_camera_orientation(
            phi=FACE_PHI * DEGREES, theta=FACE_THETA * DEGREES, zoom=1.5
        )
        arrow_a, arrow_b = make_arrows()
        dot_o, dot_a, dot_b = make_dots()
        self.add(
            arrow_a,
            arrow_b,
            make_angle_arc(),
            dot_o,
            dot_a,
            dot_b,
            make_plane(),
            make_circle(),
        )
        # Roll the camera so the square sits upright in view (which also lands
        # the yellow vector on the +x axis, since it runs along a square edge).
        self.move_camera(gamma=GAMMA_FLAT * DEGREES, run_time=3)


@app.class_definition
class Slide9FadeSquare(ThreeDSlide):
    def construct(self):
        self.set_camera_orientation(
            phi=FACE_PHI * DEGREES,
            theta=FACE_THETA * DEGREES,
            gamma=GAMMA_FLAT * DEGREES,
            zoom=1.5,
        )
        arrow_a, arrow_b = make_arrows()
        dot_o, dot_a, dot_b = make_dots()
        plane = make_plane()
        self.add(
            arrow_a,
            arrow_b,
            make_angle_arc(),
            dot_o,
            dot_a,
            dot_b,
            plane,
            make_circle(),
        )
        # Fade out the square, leaving the circle with the two vectors, yellow
        # on the x-axis.
        self.play(FadeOut(plane))


@app.class_definition
class Slide10RecapA(ThreeDSlide):
    def construct(self):
        recap_cycle(self, seed=1)


@app.class_definition
class Slide11RecapB(ThreeDSlide):
    def construct(self):
        recap_cycle(self, seed=7)


@app.class_definition
class Slide12RecapC(ThreeDSlide):
    def construct(self):
        recap_cycle(self, seed=42)


@app.cell
def _(Path, mo):
    import re
    import hashlib
    import subprocess

    # Slides in deck order. Each is a ThreeDSlide scene defined above.
    scene_names = [
        "Slide1Sphere",
        "Slide2SeeThrough",
        "Slide3Vectors",
        "Slide4Angle",
        "Slide5Plane",
        "Slide6FaceOn",
        "Slide7Flatten",
        "Slide8AlignSquare",
        "Slide9FadeSquare",
        "Slide10RecapA",
        "Slide11RecapB",
        "Slide12RecapC",
    ]

    # Cache key per scene comes from parsing this file's own text (inspect is
    # unreliable on marimo class_definition classes). The shared setup block
    # feeds every key, so editing a builder/constant re-renders all slides;
    # editing one scene's body re-renders only that slide.
    src = Path("sphere-vector-projection.py").read_text()
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

    cache = Path(".slide-cache")
    cache.mkdir(exist_ok=True)
    for name in scene_names:
        digest = hashlib.md5((blocks[name] + common).encode()).hexdigest()
        stamp = cache / f"{name}.hash"
        rendered = Path("slides") / f"{name}.json"
        if not (stamp.exists() and stamp.read_text() == digest and rendered.exists()):
            subprocess.run(
                f"manim-slides render sphere-vector-projection.py {name}",
                shell=True,
                check=True,
            )
            stamp.write_text(digest)

    subprocess.run(
        "manim-slides convert "
        + " ".join(scene_names)
        + " -c controls=true sphere_vector_projection.html --one-file",
        shell=True,
        check=True,
    )

    mo.iframe(Path("sphere_vector_projection.html").read_text())
    return


if __name__ == "__main__":
    app.run()

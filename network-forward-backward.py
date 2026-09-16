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
        Circle,
        Line,
        Text,
        VGroup,
        MovingCameraScene,
        FadeIn,
        Indicate,
        LaggedStart,
        ShowPassingFlash,
        ManimColor,
        interpolate_color,
        config,
        ORIGIN,
        BLACK,
    )
    from manim_slides import Slide

    ASPECT = config.frame_width / config.frame_height

    # Warm parchment palette from the reference image: hollow cream rings that
    # fill in when a node lights up, on black.
    CREAM = ManimColor("#EDE6C8")
    WINNER_COL = ManimColor("#F2D64B")  # brighter yellow for the winning token
    GRAD = ManimColor("#4FB6E6")  # cool blue for the backward (gradient) pass

    NODE_R = 0.11

    # Column sizes: 5 input tokens -> hidden columns -> 6-word output vocab.
    LAYERS = [5, 8, 8, 8, 8, 8, 6]
    DX = 2.2  # horizontal spacing between layers
    DY = 0.55  # vertical spacing between nodes in a column

    INPUT_TOKENS = ["The", "capital", "of", "France", "is"]
    # Output vocab column, with the winning next token emphasised.
    OUTPUT_TOKENS = ["a", "the", "Paris", "France", "home", "located"]
    WINNER = "Paris"

    def layer_x(layer):
        return layer * DX - (len(LAYERS) - 1) / 2 * DX

    def node_pos(layer, i):
        size = LAYERS[layer]
        y = (size - 1) / 2 * DY - i * DY
        return np.array([layer_x(layer), y, 0.0])

    def node_value(layer, i):
        # Deterministic per-node activation in [0.12, 1], so every independently
        # rendered slide fills the same nodes to the same brightness.
        return float(np.random.default_rng(1000 * layer + i).uniform(0.12, 1.0))

    def make_node(layer, i):
        # An opaque black disc masks the edges crossing behind it, with a cream
        # ring on top. node[0] is the disc (its colour == the node's value),
        # node[1] is the outline. At rest the disc is black, so the ring reads
        # as empty; lighting up brightens the disc towards cream.
        p = node_pos(layer, i)
        disc = Circle(
            radius=NODE_R, stroke_width=0, fill_color=BLACK, fill_opacity=1.0
        ).move_to(p)
        ring = Circle(
            radius=NODE_R, stroke_color=CREAM, stroke_width=2.2, fill_opacity=0.0
        ).move_to(p)
        return VGroup(disc, ring)

    def make_nodes():
        # One VGroup per layer, so slides can address whole columns by index.
        cols = VGroup()
        for layer, size in enumerate(LAYERS):
            cols.add(VGroup(*[make_node(layer, i) for i in range(size)]))
        return cols

    def value_color(layer, i):
        # Disc colour == activation: black (dim) -> cream (bright).
        return interpolate_color(BLACK, CREAM, node_value(layer, i))

    def fill_nodes(cols):
        """Colour every disc to its activation value (used to seed a slide that
        opens after the forward pass already ran)."""
        for layer, col in enumerate(cols):
            for i, node in enumerate(col):
                node[0].set_fill(value_color(layer, i), opacity=1.0)

    def fill_column_anims(col, layer):
        return [
            col[i][0].animate.set_fill(value_color(layer, i), opacity=1.0)
            for i in range(len(col))
        ]

    def gap_edges(gap):
        lines = VGroup()
        for i in range(LAYERS[gap]):
            for j in range(LAYERS[gap + 1]):
                lines.add(
                    Line(
                        node_pos(gap, i),
                        node_pos(gap + 1, j),
                        stroke_width=1.1,
                        color=CREAM,
                    ).set_opacity(0.28)
                )
        return lines

    def edges_by_gap():
        return [gap_edges(g) for g in range(len(LAYERS) - 1)]

    def make_edges():
        edges = VGroup()
        for g in range(len(LAYERS) - 1):
            edges.add(*gap_edges(g))
        return edges

    def flash_gap(lines, color, reverse=False):
        """Signal travelling along every edge of one gap (left->right, or
        right->left when reverse)."""
        anims = []
        for line in lines:
            bright = line.copy().set_stroke(color, width=2.4, opacity=1.0)
            if reverse:
                bright.reverse_points()
            anims.append(ShowPassingFlash(bright, time_width=0.5))
        return LaggedStart(*anims, lag_ratio=0.003)

    def make_input_labels():
        labels = VGroup()
        for i, word in enumerate(INPUT_TOKENS):
            t = Text(word, font_size=22, color=CREAM)
            t.next_to(node_pos(0, i), direction=np.array([-1, 0, 0]), buff=0.35)
            labels.add(t)
        return labels

    def make_output_labels():
        labels = VGroup()
        for i, word in enumerate(OUTPUT_TOKENS):
            color = WINNER_COL if word == WINNER else CREAM
            t = Text(word, font_size=22, color=color)
            t.set_opacity(1.0 if word == WINNER else 0.55)
            t.next_to(node_pos(len(LAYERS) - 1, i), direction=np.array([1, 0, 0]), buff=0.35)
            labels.add(t)
        return labels

    # Camera helpers. The wide frame shows the whole network plus label margins.
    WIDE_W = (len(LAYERS) - 1) * DX + 6.5
    NGAPS = len(LAYERS) - 1
    ROT_STEP = np.radians(0.7)  # gentle roll added per layer during a pass
    ZOOM_STEP = 0.985  # slight zoom-in per layer during the forward pass

    def open_wide(scene):
        scene.camera.frame.move_to(ORIGIN).set(width=WIDE_W)

    def set_focus(scene):
        """Camera state the forward pass ends on (and the backward pass starts
        from): the whole-network wide frame, rolled and zoomed a touch."""
        frame = scene.camera.frame
        frame.move_to(ORIGIN).set(width=WIDE_W)
        frame.scale(ZOOM_STEP ** NGAPS)
        frame.rotate(ROT_STEP * NGAPS)


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    return Path, mo


@app.class_definition
class Slide1Build(Slide, MovingCameraScene):
    def construct(self):
        open_wide(self)
        # Network is present on the opening frame (no black hold); labels fade in.
        self.add(make_edges(), make_nodes())
        self.wait(0.3)
        self.play(FadeIn(make_input_labels()), FadeIn(make_output_labels()))
        self.wait(0.3)


@app.class_definition
class Slide2Forward(Slide, MovingCameraScene):
    def construct(self):
        open_wide(self)  # hands off from Slide1's wide frame
        gaps = edges_by_gap()
        cols = make_nodes()
        self.add(VGroup(*gaps), cols, make_input_labels(), make_output_labels())
        # Light the input column, then push a signal forward gap by gap: each
        # step flashes the edges then fills the next column to its values, while
        # the camera rolls and zooms a touch for life.
        self.play(*fill_column_anims(cols[0], 0), run_time=0.5)
        for g in range(NGAPS):
            self.play(
                flash_gap(gaps[g], WINNER_COL),
                *fill_column_anims(cols[g + 1], g + 1),
                self.camera.frame.animate.rotate(ROT_STEP).scale(ZOOM_STEP),
                run_time=0.6,
            )
        winner_idx = OUTPUT_TOKENS.index(WINNER)
        self.play(Indicate(cols[-1][winner_idx][1], color=WINNER_COL, scale_factor=1.7))
        self.wait(0.3)


@app.class_definition
class Slide3Backward(Slide, MovingCameraScene):
    def construct(self):
        set_focus(self)  # hands off from where the forward pass ended
        gaps = edges_by_gap()
        cols = make_nodes()
        fill_nodes(cols)  # network already carries the forward-pass activations
        self.add(VGroup(*gaps), cols, make_input_labels(), make_output_labels())
        # Gradients flow back right->left: blue signal along the edges, each
        # column's rings pulsing blue as it arrives; camera unrolls to wide.
        for g in reversed(range(NGAPS)):
            self.play(
                flash_gap(gaps[g], GRAD, reverse=True),
                *[Indicate(node[1], color=GRAD, scale_factor=1.3) for node in cols[g]],
                self.camera.frame.animate.rotate(-ROT_STEP).scale(1 / ZOOM_STEP),
                run_time=0.6,
            )
        self.wait(0.3)


@app.cell
def _(Path, mo):
    import re
    import hashlib
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    scene_names = [
        "Slide1Build",
        "Slide2Forward",
        "Slide3Backward",
    ]

    # Per-scene render cache: key = hash(scene block) + hash(shared setup block),
    # parsed from this file's own text. Unchanged scenes are skipped; editing the
    # setup block re-renders all slides, editing one scene re-renders only it.
    src = Path("network-forward-backward.py").read_text()
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

    cache = Path(".network-cache")
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
            f"manim-slides render network-forward-backward.py {name}",
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
        + " -c controls=true network_forward_backward.html --one-file",
        shell=True,
        check=True,
    )

    mo.iframe(Path("network_forward_backward.html").read_text())
    return


if __name__ == "__main__":
    app.run()

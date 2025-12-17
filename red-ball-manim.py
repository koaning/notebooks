# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "marimo",
#     "manim>=0.18.0",
#     "av>=14.0.0",
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
    mo.md("""
    ## The Red Ball Riddle - Manim Animation

    This notebook creates a manim animation visualizing the counterintuitive "red ball riddle":

    > You have 99 red balls and 1 blue ball (99% red).
    > How many red balls must you remove to get 98% red?

    **Answer: 50 balls!** (leaving 49 red + 1 blue = 50 balls, where 49/50 = 98%)
    """)
    return


@app.cell
def _():
    import os
    # Add LaTeX to PATH before importing manim (so it finds latex on import)
    os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ.get("PATH", "")

    from manim import (
        Scene, Text, Circle, VGroup, FadeIn, FadeOut, Write, Create, Transform,
        UP, DOWN, LEFT, RIGHT, ORIGIN, RED, BLUE, WHITE, YELLOW, GREY,
        Axes, Line, Dot, DashedLine, ReplacementTransform, Indicate, MathTex, Tex
    )
    import numpy as np

    class RedBallRiddle(Scene):
        def construct(self):
            ball_radius = 0.12
            spacing = 0.3

            # Helper to create ball grid
            def create_balls(n_red, n_blue, center_x=0, center_y=0):
                balls = VGroup()
                total = n_red + n_blue
                cols = 10 if total == 100 else int(np.ceil(np.sqrt(total)))
                for i in range(total):
                    row = i // cols
                    col = i % cols
                    x = center_x + (col - (cols - 1) / 2) * spacing
                    y = center_y + (row - (total // cols - 1) / 2) * spacing
                    color = RED if i < n_red else BLUE
                    ball = Circle(radius=ball_radius, fill_opacity=1, color=color, stroke_width=1)
                    ball.move_to([x, y, 0])
                    balls.add(ball)
                return balls

            # ===== SCENE 1: The Setup =====
            balls = create_balls(99, 1, center_y=-0.5)
            pct_label = Tex("99\\%", font_size=42)
            pct_label.to_edge(UP + RIGHT, buff=0.5)

            self.play(FadeIn(balls), run_time=2)
            self.play(Write(pct_label))
            self.wait(1)

            # ===== SCENE 2: The Question =====
            blue_ball = balls[99]
            self.play(Indicate(blue_ball, scale_factor=1.5, color=YELLOW), run_time=1)
            self.wait(0.5)

            question_text = Tex("98\\%?", font_size=56, color=YELLOW)
            question_text.next_to(pct_label, DOWN, buff=0.3)
            self.play(Write(question_text))
            self.wait(1)

            # ===== SCENE 3: The Intuition (Wrong) =====
            # Remove 1 ball and show percentage barely changes
            self.play(FadeOut(balls[98]), run_time=0.5)
            new_pct = Tex("98.99\\%", font_size=42)
            new_pct.move_to(pct_label)
            self.play(Transform(pct_label, new_pct))
            self.wait(1)

            # ===== SCENE 4: The Reveal =====
            # Remove 2 more balls (3 total) to show it's still not enough
            counter = Tex("Red: 98", font_size=36, color=RED)
            counter.to_edge(LEFT + UP, buff=0.5)
            self.play(Write(counter))

            for i in [97, 96]:
                self.play(FadeOut(balls[i]), run_time=0.3)
                new_count = i
                pct = new_count / (new_count + 1) * 100
                new_counter = Tex(f"Red: {new_count}", font_size=36, color=RED)
                new_counter.to_edge(LEFT + UP, buff=0.5)
                new_pct_text = Tex(f"{pct:.2f}\\%", font_size=42)
                new_pct_text.move_to(pct_label)
                self.play(
                    Transform(counter, new_counter),
                    Transform(pct_label, new_pct_text),
                    run_time=0.3
                )

            self.wait(1)

            # Clear the balls scene
            remaining_balls = VGroup(*[balls[i] for i in range(96)], balls[99])
            self.play(
                FadeOut(remaining_balls),
                FadeOut(pct_label),
                FadeOut(question_text),
                FadeOut(counter)
            )
            self.wait(0.5)

            # ===== SCENE 5: The Reframe (Formula) =====
            formula = MathTex(r"\frac{R}{R + B}", "=", "98\\%", font_size=72)
            formula.move_to(ORIGIN)
            self.play(Write(formula))
            self.wait(2)

            # ===== SCENE 6: Formula to Line Chart =====
            formula_label = MathTex(r"\frac{R}{R + 1}", font_size=42)
            formula_label.to_edge(UP + LEFT, buff=0.5)

            axes = Axes(
                x_range=[0, 100, 20],
                y_range=[0, 1, 0.2],
                x_length=8,
                y_length=4,
                axis_config={"include_numbers": False},
            )
            axes.move_to(ORIGIN).shift(DOWN * 0.5)

            x_label = Tex("$R$ (red balls)", font_size=32)
            x_label.next_to(axes.x_axis, DOWN, buff=0.3)
            y_label = MathTex(r"\frac{R}{R+1}", font_size=32)
            y_label.next_to(axes.y_axis, LEFT, buff=0.3)

            # The curve: R / (R + 1) with B = 1
            curve = axes.plot(lambda r: r / (r + 1) if r > 0 else 0, x_range=[1, 99], color=RED)

            self.play(
                ReplacementTransform(formula, formula_label),
                Create(axes),
                run_time=1.5
            )
            self.play(Write(x_label), Write(y_label))
            self.play(Create(curve), run_time=2)
            self.wait(1)

            # ===== SCENE 7: The Insight (y=0.98 line) =====
            target_line = DashedLine(
                axes.c2p(0, 0.98),
                axes.c2p(100, 0.98),
                color=YELLOW
            )
            target_label = Tex("98\\%", font_size=32, color=YELLOW)
            target_label.next_to(target_line, RIGHT, buff=0.2)

            self.play(Create(target_line), Write(target_label))

            # Find intersection point (R=49 gives 49/50 = 0.98)
            intersection_x = 49
            intersection_y = 49 / 50
            intersection_dot = Dot(axes.c2p(intersection_x, intersection_y), color=YELLOW)

            vertical_line = DashedLine(
                axes.c2p(intersection_x, 0),
                axes.c2p(intersection_x, intersection_y),
                color=YELLOW
            )

            r_label = MathTex("R = 49", font_size=32, color=YELLOW)
            r_label.next_to(axes.c2p(intersection_x, intersection_y), UP + RIGHT, buff=0.2)

            self.play(Create(vertical_line), Create(intersection_dot))
            self.play(Write(r_label))
            self.wait(2)

            # ===== SCENE 8: The Formula Solution =====
            # Clear chart and show the algebra
            self.play(
                FadeOut(axes), FadeOut(curve), FadeOut(target_line), FadeOut(target_label),
                FadeOut(vertical_line), FadeOut(intersection_dot), FadeOut(r_label),
                FadeOut(x_label), FadeOut(y_label), FadeOut(formula_label)
            )

            # Aligned equations using MathTex with aligned environment
            solve_steps = MathTex(
                r"\frac{R}{R + 1} &= 0.98 \\",
                r"R &= 0.98 \cdot (R + 1) \\",
                r"R &= 0.98R + 0.98 \\",
                r"0.02R &= 0.98 \\",
                r"R &= 49",
                font_size=48
            )
            solve_steps.move_to(ORIGIN + LEFT * 1.5)

            self.play(Write(solve_steps), run_time=4)
            self.wait(1)

            # Show balls being removed: 99 -> 49
            result_text = Tex("Remove 50 balls!", font_size=42, color=YELLOW)
            result_text.to_edge(RIGHT).shift(UP * 0.5)
            self.play(Write(result_text))

            # Small ball animation on the right
            small_balls = create_balls(99, 1, center_x=3.5, center_y=-1.5)
            small_balls.scale(0.5)
            self.play(FadeIn(small_balls), run_time=1)

            # Remove 50 balls quickly
            balls_to_remove = VGroup(*[small_balls[i] for i in range(49, 99)])
            self.play(FadeOut(balls_to_remove), run_time=1)

            # Rearrange remaining
            final_balls = VGroup(*[small_balls[i] for i in range(49)], small_balls[99])
            self.wait(1)

            self.play(FadeOut(solve_steps), FadeOut(result_text), FadeOut(final_balls))

            # ===== SCENE 9: The Asymmetry =====
            axes2 = Axes(
                x_range=[50, 100, 10],
                y_range=[0, 100, 20],
                x_length=8,
                y_length=4,
                axis_config={"include_numbers": False},
            )
            axes2.move_to(ORIGIN).shift(DOWN * 0.5)

            x_label2 = Tex("Target \\% Red", font_size=32)
            x_label2.next_to(axes2.x_axis, DOWN, buff=0.3)
            y_label2 = Tex("Balls to change", font_size=32)
            y_label2.next_to(axes2.y_axis, LEFT, buff=0.3)

            self.play(Create(axes2), Write(x_label2), Write(y_label2))

            # Remove red balls curve: to get P% red with 1 blue, need R = P/(100-P)
            # balls to remove = 99 - P/(100-P)
            remove_red_curve = axes2.plot(
                lambda p: max(0, 99 - p / (100 - p)) if p < 99.5 else 0,
                x_range=[50, 99],
                color=RED
            )
            remove_red_label = Tex("Remove red", font_size=28, color=RED)
            remove_red_label.next_to(axes2.c2p(70, 50), UP)

            self.play(Create(remove_red_curve), run_time=2)
            self.play(Write(remove_red_label))
            self.wait(1)

            # Add blue balls curve: to get P% red with 99 red, need blue = 99*(100-P)/P
            add_blue_curve = axes2.plot(
                lambda p: 99 * (100 - p) / p if p > 50 else 99,
                x_range=[50, 99],
                color=BLUE
            )
            add_blue_label = Tex("Add blue", font_size=28, color=BLUE)
            add_blue_label.next_to(axes2.c2p(80, 20), DOWN)

            self.play(Create(add_blue_curve), run_time=2)
            self.play(Write(add_blue_label))
            self.wait(3)

    return (RedBallRiddle,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Run the Animation

    Click the button below to render the animation. This will create an MP4 file.

    **Prerequisites:** Manim requires `ffmpeg`, `pkg-config`, and LaTeX to render videos. On macOS, install them with:

    ```bash
    brew install pkg-config ffmpeg mactex-no-gui
    ```

    After installing MacTeX, you may need to restart your terminal or run `eval "$(/usr/libexec/path_helper)"` to update your PATH.
    """)
    return


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
def _(RedBallRiddle, mo, render_button, resolution):
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

    scene = RedBallRiddle()
    scene.render()

    # Find the final output file (not partial files)
    _video_path = _output_dir / "videos" / _preset["folder"] / "RedBallRiddle.mp4"

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
    ## The Math Behind the Riddle

    The key insight is that with a fixed number of blue balls ($B=1$), the percentage of red balls is:

    $$P = \frac{R}{R + B} = \frac{R}{R + 1}$$

    To achieve 98% red ($P = 0.98$):

    $$0.98 = \frac{R}{R + 1}$$

    $$0.98(R + 1) = R$$

    $$0.98R + 0.98 = R$$

    $$0.98 = R - 0.98R = 0.02R$$

    $$R = \frac{0.98}{0.02} = 49$$

    So we need 49 red balls, meaning we remove $99 - 49 = 50$ balls!
    """)
    return


if __name__ == "__main__":
    app.run()

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
    from manim import Scene, Text, Circle, VGroup, Cross, FadeIn, FadeOut, Write, Create, Transform, UP, DOWN, LEFT, RIGHT, RED, BLUE, GREEN, YELLOW

    class RedBallRiddle(Scene):
        def construct(self):
            # Title
            title = Text("The Red Ball Riddle", font_size=48)
            self.play(Write(title))
            self.wait(1)
            self.play(title.animate.to_edge(UP))

            # Initial setup text
            setup_text = Text("You have 99 red balls and 1 blue ball", font_size=32)
            setup_text.next_to(title, DOWN, buff=0.5)
            self.play(Write(setup_text))
            self.wait(1)

            # Create initial grid of balls (10x10 grid)
            balls = VGroup()
            ball_radius = 0.12
            spacing = 0.3

            for i in range(100):
                row = i // 10
                col = i % 10
                x = (col - 4.5) * spacing
                y = (row - 4.5) * spacing - 1

                if i < 99:
                    ball = Circle(radius=ball_radius, fill_opacity=1, color=RED, stroke_width=1)
                else:
                    ball = Circle(radius=ball_radius, fill_opacity=1, color=BLUE, stroke_width=1)

                ball.move_to([x, y, 0])
                balls.add(ball)

            self.play(FadeIn(balls), run_time=2)
            self.wait(1)

            # Show percentage
            pct_text = Text("99% are red", font_size=36, color=RED)
            pct_text.to_edge(RIGHT).shift(UP * 2)
            self.play(Write(pct_text))
            self.wait(1)

            # Question
            self.play(FadeOut(setup_text))
            question = Text("How many red balls to remove\nto get 98% red?", font_size=28)
            question.next_to(title, DOWN, buff=0.5)
            self.play(Write(question))
            self.wait(2)

            # Show wrong intuition
            wrong_answer = Text("Just 1 ball?", font_size=32, color=YELLOW)
            wrong_answer.to_edge(LEFT).shift(UP * 2)
            self.play(Write(wrong_answer))
            self.wait(1)

            cross = Cross(wrong_answer, stroke_width=4)
            self.play(Create(cross))
            self.wait(1)

            # Reveal answer
            self.play(FadeOut(wrong_answer), FadeOut(cross))
            answer = Text("Answer: 50 balls!", font_size=36, color=GREEN)
            answer.to_edge(LEFT).shift(UP * 2)
            self.play(Write(answer))
            self.wait(1)

            # Animate removing 50 balls
            balls_to_remove = VGroup(*[balls[i] for i in range(50, 99)])

            self.play(FadeOut(pct_text))

            # Show counter (using Text to avoid LaTeX dependency)
            counter = Text("Red balls: 99", font_size=32, color=RED)
            counter.to_edge(RIGHT).shift(UP * 2)
            self.play(Write(counter))

            # Remove balls one by one (faster animation)
            for i, ball in enumerate(balls_to_remove):
                new_count = 99 - i - 1
                new_counter = Text(f"Red balls: {new_count}", font_size=32, color=RED)
                new_counter.to_edge(RIGHT).shift(UP * 2)
                self.play(
                    ball.animate.set_opacity(0).scale(0.1),
                    Transform(counter, new_counter),
                    run_time=0.05
                )

            self.wait(1)

            # Rearrange remaining balls
            remaining_balls = VGroup(*[balls[i] for i in range(50)], balls[99])

            # New positions for 50 balls (7x8 grid approximately)
            new_positions = []
            for i in range(50):
                row = i // 7
                col = i % 7
                x = (col - 3) * spacing
                y = (row - 3.5) * spacing - 1
                new_positions.append([x, y, 0])

            # Move remaining red balls
            animations = []
            for i in range(49):
                animations.append(balls[i].animate.move_to(new_positions[i]))

            # Move blue ball to last position
            animations.append(balls[99].animate.move_to(new_positions[49]))

            self.play(*animations, run_time=1.5)
            self.wait(1)

            # Final percentage
            final_pct = Text("49 red + 1 blue = 50 balls\n49/50 = 98% red", font_size=28, color=GREEN)
            final_pct.to_edge(DOWN)
            self.play(Write(final_pct))
            self.wait(2)

            # Key insight
            self.play(FadeOut(question), FadeOut(answer), FadeOut(counter))
            insight = Text(
                "1% change in percentage\nrequires removing 50% of red balls!",
                font_size=32,
                color=YELLOW
            )
            insight.next_to(title, DOWN, buff=0.5)
            self.play(Write(insight))
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
    # In CLI mode (mo.app_meta().mode == "script"), auto-run. In edit mode, use button.
    _is_cli = mo.app_meta().mode == "script"
    render_button = mo.ui.run_button(label="Render Animation")
    render_button if not _is_cli else mo.md("*Auto-rendering in CLI mode...*")
    return (render_button,)


@app.cell
def _(RedBallRiddle, mo, render_button):
    import os
    from pathlib import Path

    # Auto-run in CLI mode, otherwise wait for button
    _is_cli = mo.app_meta().mode == "script"
    mo.stop(not _is_cli and not render_button.value)

    # Output directory in the same folder as the notebook
    _output_dir = Path(__file__).parent / "media" if "__file__" in dir() else Path.cwd() / "media"
    _output_dir.mkdir(exist_ok=True)

    # Render the scene
    from manim import config
    config.media_dir = str(_output_dir)
    config.pixel_height = 720
    config.pixel_width = 1280
    config.frame_rate = 30

    scene = RedBallRiddle()
    scene.render()

    # Find the final output file (not partial files)
    _video_path = _output_dir / "videos" / "720p30" / "RedBallRiddle.mp4"

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

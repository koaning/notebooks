# /// script
# requires-python = "==3.12"
# dependencies = [
#     "manim==0.19.1",
#     "manim-slides==5.5.2",
#     "mohtml==0.1.11",
#     "moterm==0.1.0",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="columns", sql_output="polars")

with app.setup:
    from manim import Dot, Circle, MoveAlongPath, GrowFromCenter, BLUE, ORIGIN, linear
    from manim_slides import Slide, ThreeDSlide


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    return Path, mo


@app.class_definition
class BasicExample(Slide):
    def construct(self):
        circle = Circle(radius=3, color=BLUE)
        dot = Dot()

        self.play(GrowFromCenter(circle))

        self.next_slide(loop=True)
        self.play(MoveAlongPath(dot, circle), run_time=2, rate_func=linear)
        self.next_slide()

        self.play(dot.animate.move_to(ORIGIN))


@app.cell
def _(Path, mo):
    from moterm import Kmd

    Kmd("manim-slides render manim-slidez.py BasicExample")
    Kmd("manim-slides convert BasicExample -c controls=true basic_example.html --one-file")

    mo.iframe(Path("basic_example.html").read_text())
    return


if __name__ == "__main__":
    app.run()

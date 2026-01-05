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
app = marimo.App(sql_output="polars")

with app.setup:
    from manim import (
        Dot,
        Circle,
        Square,
        VGroup,
        Text,
        MathTex,
        Axes,
        FadeIn,
        FadeOut,
        Create,
        Write,
        Transform,
        ReplacementTransform,
        MoveToTarget,
        AnimationGroup,
        Succession,
        BLUE,
        RED,
        GREEN,
        WHITE,
        YELLOW,
        ORIGIN,
        UP,
        DOWN,
        LEFT,
        RIGHT,
        linear,
        smooth,
        config,
    )
    from manim_slides import Slide
    import numpy as np


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    return Path, mo


@app.class_definition
class LanchesterSlides(Slide):
    """Minimal slides for Lanchester's Laws - designed for voiceover"""

    def construct(self):
        # Slide 1: Two similar-sized armies
        self.show_balanced_armies()
        self.next_slide()

        # Slide 2: Very unbalanced armies
        self.show_unbalanced_armies()
        self.next_slide()

        # Slide 3: Empty axes (time vs armies)
        self.show_axes()
        self.next_slide()

        # Slide 4: Introduce differential equations
        self.show_diff_eq()
        self.next_slide()

        # Slide 5: Simplify with alpha = beta
        self.show_simplify()
        self.next_slide()

        # Slide 6: Derive the invariant
        self.show_invariant()
        self.next_slide()

        # Slide 7: Solve for survivors
        self.show_survivors()
        self.next_slide()

        # Slide 8: Final formula
        self.show_final_formula()

    def make_army(self, n, color, cols=5):
        """Helper to create a grid of dots representing an army"""
        return VGroup(
            *[
                Dot(color=color, radius=0.12).move_to(
                    RIGHT * (i % cols) * 0.4 + DOWN * (i // cols) * 0.4
                )
                for i in range(n)
            ]
        )

    def show_balanced_armies(self):
        """Two armies, A slightly bigger than B"""
        # Army A: 25 soldiers
        army_a = self.make_army(25, BLUE)
        army_a.move_to(LEFT * 3)
        label_a = MathTex("A = 25", color=BLUE, font_size=36)
        label_a.next_to(army_a, UP, buff=0.5)

        # Army B: 20 soldiers
        army_b = self.make_army(20, RED)
        army_b.move_to(RIGHT * 3)
        label_b = MathTex("B = 20", color=RED, font_size=36)
        label_b.next_to(army_b, UP, buff=0.5)

        self.play(Create(army_a), Create(army_b))
        self.play(Write(label_a), Write(label_b))

    def show_unbalanced_armies(self):
        """Large army vs small army - to provoke thinking"""
        self.clear()

        # Army A: 50 soldiers
        army_a = self.make_army(50, BLUE, cols=10)
        army_a.move_to(LEFT * 3)
        label_a = MathTex("A = 50", color=BLUE, font_size=36)
        label_a.next_to(army_a, UP, buff=0.5)

        # Army B: 10 soldiers
        army_b = self.make_army(10, RED, cols=5)
        army_b.move_to(RIGHT * 3)
        label_b = MathTex("B = 10", color=RED, font_size=36)
        label_b.next_to(army_b, UP, buff=0.5)

        self.play(Create(army_a), Create(army_b))
        self.play(Write(label_a), Write(label_b))

    def show_axes(self):
        """Show empty axes: time vs army size"""
        self.clear()

        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 50, 10],
            x_length=10,
            y_length=5,
            axis_config={"include_tip": True},
        )

        x_label = MathTex("t", font_size=36)
        x_label.next_to(axes.x_axis, RIGHT)
        y_label = MathTex("\\text{soldiers}", font_size=30)
        y_label.next_to(axes.y_axis, UP)

        self.play(Create(axes))
        self.play(Write(x_label), Write(y_label))

    def show_diff_eq(self):
        """Introduce the differential equations"""
        self.clear()

        eq1 = MathTex(r"\frac{dA}{dt} = -\alpha \cdot B", font_size=56)
        eq2 = MathTex(r"\frac{dB}{dt} = -\alpha \cdot A", font_size=56)

        eq1.move_to(UP * 1)
        eq2.move_to(DOWN * 1)

        self.play(Write(eq1))
        self.play(Write(eq2))

    def show_simplify(self):
        """Show the trick: multiply and subtract"""
        self.clear()

        # Start with equations
        eq1 = MathTex(r"\frac{dA}{dt} = -\alpha B", font_size=48)
        eq2 = MathTex(r"\frac{dB}{dt} = -\alpha A", font_size=48)
        eq1.move_to(UP * 2)
        eq2.next_to(eq1, DOWN, buff=0.5)

        self.play(Write(eq1), Write(eq2))
        self.next_slide()

        # Multiply first by A, second by B
        eq1_mult = MathTex(r"A \frac{dA}{dt} = -\alpha A B", font_size=48)
        eq2_mult = MathTex(r"B \frac{dB}{dt} = -\alpha A B", font_size=48)
        eq1_mult.move_to(UP * 2)
        eq2_mult.next_to(eq1_mult, DOWN, buff=0.5)

        self.play(Transform(eq1, eq1_mult), Transform(eq2, eq2_mult))

    def show_invariant(self):
        """Show that A² - B² is constant"""
        self.clear()

        # Subtract the equations
        subtract = MathTex(
            r"A \frac{dA}{dt} - B \frac{dB}{dt} = 0",
            font_size=48
        )
        subtract.move_to(UP * 1.5)

        self.play(Write(subtract))
        self.next_slide()

        # This means d/dt of something is zero
        derivative = MathTex(
            r"\frac{d}{dt}\left( A^2 - B^2 \right) = 0",
            font_size=48
        )
        derivative.move_to(ORIGIN)

        self.play(Write(derivative))
        self.next_slide()

        # So A² - B² is constant
        invariant = MathTex(
            r"A^2 - B^2 = A_0^2 - B_0^2",
            font_size=56
        )
        invariant.move_to(DOWN * 1.5)

        self.play(Write(invariant))

    def show_survivors(self):
        """Derive the number of survivors"""
        self.clear()

        # When B = 0, A = A_final
        condition = MathTex(r"\text{When } B = 0:", font_size=42)
        condition.move_to(UP * 2)

        eq = MathTex(r"A_{\text{final}}^2 - 0 = A_0^2 - B_0^2", font_size=48)
        eq.move_to(UP * 0.5)

        self.play(Write(condition))
        self.play(Write(eq))
        self.next_slide()

        # Solve
        solve = MathTex(r"A_{\text{final}}^2 = A_0^2 - B_0^2", font_size=48)
        solve.move_to(DOWN * 0.5)

        self.play(Write(solve))

    def show_final_formula(self):
        """Show the final formula"""
        self.clear()

        formula = MathTex(
            r"A_{\text{final}} = \sqrt{A_0^2 - B_0^2}",
            font_size=64
        )

        self.play(Write(formula))
        self.next_slide()

        # Example: A=50, B=10
        example = MathTex(
            r"\sqrt{50^2 - 10^2} = \sqrt{2500 - 100} = \sqrt{2400} \approx 49",
            font_size=42
        )
        example.next_to(formula, DOWN, buff=1)

        self.play(Write(example))


@app.cell
def _():
    from moterm import Kmd

    Kmd("manim-slides render lanchester-slides.py LanchesterSlides")
    return (Kmd,)


@app.cell
def _(Kmd):
    Kmd(
        "manim-slides convert LanchesterSlides -c controls=true lanchester.html --one-file"
    )
    return


@app.cell
def _(Path, mo):
    mo.iframe(Path("lanchester.html").read_text())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

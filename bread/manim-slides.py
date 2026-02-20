# /// script
# requires-python = "==3.12"
# dependencies = [
#     "manim==0.19.1",
#     "manim-slides==5.5.2",
#     "marimo>=0.19.11",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    from manim import (
        Axes, Tex, Dot, DashedLine, VGroup,
        Write, Create, FadeOut, FadeIn, GrowFromCenter, Transform,
        BLUE, ORANGE, GRAY, WHITE, UP, DOWN, RIGHT, LEFT,
    )
    from manim_slides import Slide


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    return Path, mo


@app.class_definition
class BreadStory(Slide):
    def _make_axes(self, n_weeks):
        axes = Axes(
            x_range=[0, n_weeks, 26],
            y_range=[0, 2000, 500],
            x_length=10,
            y_length=6,
            axis_config={
                "include_numbers": False,
                "include_ticks": True,
                "include_tip": True,
                "tip_width": 0.2,
                "tip_height": 0.2,
            },
        )
        x_label = Tex("Weeks", font_size=22).next_to(axes.x_axis, DOWN, buff=0.4)
        y_label = Tex(r"\$", font_size=28).next_to(axes.y_axis, LEFT, buff=0.3)
        return axes, VGroup(axes, x_label, y_label)

    def _bake_mobjects(self, axes, machine, rate, unc, n_weeks):
        """Build line + band + label for a given baking rate/uncertainty."""
        line = axes.plot(lambda x: machine + rate * x, x_range=[0, n_weeks], color=ORANGE)
        lo = axes.plot(lambda x: machine + (rate - unc) * x, x_range=[0, n_weeks])
        hi = axes.plot(lambda x: machine + (rate + unc) * x, x_range=[0, n_weeks])
        band = axes.get_area(hi, bounded_graph=lo, x_range=(0, n_weeks), color=ORANGE, opacity=0.15)
        label = Tex(
            f"\\${machine:.0f} + \\${rate:.2f}/week $\\pm$ \\${unc:.2f}",
            color=ORANGE, font_size=28,
        )
        label.to_corner(UP + RIGHT)
        return line, band, label

    def construct(self):
        buy_rate = 8.0
        buy_unc = 2.0
        bake_rate = 2.75
        bake_unc = 0.50
        machine = 200.0
        n_weeks = 156
        bw = machine / (buy_rate - bake_rate)
        bc = buy_rate * bw

        # Single set of axes, shifted up for bottom padding
        axes, axes_group = self._make_axes(n_weeks)
        axes_group.shift(UP * 0.4)
        self.play(Create(axes_group))
        self.next_slide()

        # --- Slide 1: Buying bread ---
        buy_line = axes.plot(lambda x: buy_rate * x, x_range=[0, n_weeks], color=BLUE)
        buy_lo = axes.plot(lambda x: (buy_rate - buy_unc) * x, x_range=[0, n_weeks])
        buy_hi = axes.plot(lambda x: (buy_rate + buy_unc) * x, x_range=[0, n_weeks])
        buy_band = axes.get_area(
            buy_hi, bounded_graph=buy_lo, x_range=(0, n_weeks), color=BLUE, opacity=0.15
        )
        buy_label = Tex(r"\$8/week $\pm$ \$2", color=BLUE, font_size=28)
        buy_label.to_corner(UP + RIGHT)

        self.play(Create(buy_line), FadeIn(buy_band), Write(buy_label))
        self.next_slide()

        # Demo: change buying slope (steeper then back)
        steep_buy = axes.plot(lambda x: 12.0 * x, x_range=[0, n_weeks], color=BLUE)
        steep_buy_lo = axes.plot(lambda x: (12.0 - buy_unc) * x, x_range=[0, n_weeks])
        steep_buy_hi = axes.plot(lambda x: (12.0 + buy_unc) * x, x_range=[0, n_weeks])
        steep_buy_band = axes.get_area(
            steep_buy_hi, bounded_graph=steep_buy_lo, x_range=(0, n_weeks), color=BLUE, opacity=0.15
        )
        steep_buy_label = Tex(r"\$12/week $\pm$ \$2", color=BLUE, font_size=28)
        steep_buy_label.to_corner(UP + RIGHT)
        self.play(
            Transform(buy_line, steep_buy),
            Transform(buy_band, steep_buy_band),
            Transform(buy_label, steep_buy_label),
        )
        self.next_slide()

        orig_buy = axes.plot(lambda x: buy_rate * x, x_range=[0, n_weeks], color=BLUE)
        orig_buy_lo = axes.plot(lambda x: (buy_rate - buy_unc) * x, x_range=[0, n_weeks])
        orig_buy_hi = axes.plot(lambda x: (buy_rate + buy_unc) * x, x_range=[0, n_weeks])
        orig_buy_band = axes.get_area(
            orig_buy_hi, bounded_graph=orig_buy_lo, x_range=(0, n_weeks), color=BLUE, opacity=0.15
        )
        orig_buy_label = Tex(r"\$8/week $\pm$ \$2", color=BLUE, font_size=28)
        orig_buy_label.to_corner(UP + RIGHT)
        self.play(
            Transform(buy_line, orig_buy),
            Transform(buy_band, orig_buy_band),
            Transform(buy_label, orig_buy_label),
        )
        self.next_slide()

        # --- Slide 2: Baking bread appears alongside buying ---
        self.play(FadeOut(buy_label))
        bake_line, bake_band, bake_label = self._bake_mobjects(
            axes, machine, bake_rate, bake_unc, n_weeks
        )
        self.play(Create(bake_line), FadeIn(bake_band), Write(bake_label))
        self.next_slide()

        # Demo: change slope (steeper then back)
        steep_line, steep_band, steep_label = self._bake_mobjects(
            axes, machine, 5.0, bake_unc, n_weeks
        )
        self.play(
            Transform(bake_line, steep_line),
            Transform(bake_band, steep_band),
            Transform(bake_label, steep_label),
        )
        self.next_slide()

        orig_line, orig_band, orig_label = self._bake_mobjects(
            axes, machine, bake_rate, bake_unc, n_weeks
        )
        self.play(
            Transform(bake_line, orig_line),
            Transform(bake_band, orig_band),
            Transform(bake_label, orig_label),
        )
        self.next_slide()

        # Demo: change variance (wider then back)
        wide_line, wide_band, wide_label = self._bake_mobjects(
            axes, machine, bake_rate, 1.50, n_weeks
        )
        self.play(
            Transform(bake_band, wide_band),
            Transform(bake_label, wide_label),
        )
        self.next_slide()

        narrow_line, narrow_band, narrow_label = self._bake_mobjects(
            axes, machine, bake_rate, bake_unc, n_weeks
        )
        self.play(
            Transform(bake_band, narrow_band),
            Transform(bake_label, narrow_label),
        )
        self.next_slide()

        # --- Slide 3: Breakeven annotation ---
        dashed = DashedLine(axes.c2p(bw, 0), axes.c2p(bw, bc), color=GRAY)
        dot = Dot(axes.c2p(bw, bc), color=WHITE, radius=0.08)

        self.play(Create(dashed), GrowFromCenter(dot))
        self.next_slide()


@app.cell
def _():
    import subprocess

    def run(cmd):
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout + result.stderr

    return (run,)


@app.cell
def _(mo, run):
    _out = run("manim-slides render bread/manim-slides.py BreadStory")
    mo.md(f"```\n{_out}\n```") if _out.strip() else mo.md("Render complete.")
    return


@app.cell
def _(mo, run):
    _out = run("manim-slides convert BreadStory -c controls=true bread_story.html --one-file")
    mo.md(f"```\n{_out}\n```") if _out.strip() else mo.md("Convert complete.")
    return


@app.cell
def _(Path, mo):
    _html = Path("bread_story.html")
    mo.iframe(_html.read_text()) if _html.exists() else mo.md("**Render failed** — check the output from the cells above.")
    return


if __name__ == "__main__":
    app.run()

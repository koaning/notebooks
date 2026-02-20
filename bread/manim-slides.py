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
    import numpy as np
    from manim import (
        Axes, Tex, Dot, DashedLine, VGroup, MovingCameraScene,
        Write, Create, FadeOut, FadeIn, GrowFromCenter, Transform,
        LaggedStart, Polygon, Rectangle,
        BLUE, ORANGE, GRAY, WHITE, RED, UP, DOWN, RIGHT, LEFT,
    )
    from manim_slides import Slide


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    return Path, mo


@app.class_definition
class BreadStory(Slide, MovingCameraScene):
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

    def _buy_mobjects(self, axes, rate, unc, n_weeks):
        """Build line + band for a given buying rate/uncertainty."""
        line = axes.plot(lambda x: rate * x, x_range=[0, n_weeks], color=BLUE)
        lo = axes.plot(lambda x: (rate - unc) * x, x_range=[0, n_weeks])
        hi = axes.plot(lambda x: (rate + unc) * x, x_range=[0, n_weeks])
        band = axes.get_area(hi, bounded_graph=lo, x_range=(0, n_weeks), color=BLUE, opacity=0.15)
        return line, band

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

        # --- Slide 3: Reshape for simulation (steeper blue, flatter orange) ---
        sim_buy_rate, sim_buy_unc = 12.0, 2.0
        sim_bake_rate, sim_bake_unc = 1.50, 1.25
        sim_machine = 400.0
        sim_bw = sim_machine / (sim_buy_rate - sim_bake_rate)
        sim_bc = sim_buy_rate * sim_bw

        sim_buy_line, sim_buy_band = self._buy_mobjects(axes, sim_buy_rate, sim_buy_unc, n_weeks)
        sim_bake_line, sim_bake_band, sim_bake_label = self._bake_mobjects(
            axes, sim_machine, sim_bake_rate, sim_bake_unc, n_weeks
        )
        self.play(
            Transform(buy_line, sim_buy_line),
            Transform(buy_band, sim_buy_band),
            Transform(bake_line, sim_bake_line),
            Transform(bake_band, sim_bake_band),
            Transform(bake_label, sim_bake_label),
        )
        self.next_slide()

        # --- Slide 4: Breakeven annotation ---
        dashed = DashedLine(axes.c2p(sim_bw, 0), axes.c2p(sim_bw, sim_bc), color=GRAY)
        dot = Dot(axes.c2p(sim_bw, sim_bc), color=WHITE, radius=0.08)

        self.play(Create(dashed), GrowFromCenter(dot))
        self.next_slide()

        # --- Slide 5: Zoom into the breakeven area ---
        self.play(FadeOut(bake_label))
        self.play(
            self.camera.frame.animate.scale(0.4).move_to(axes.c2p(sim_bw, sim_bc))
        )
        self.next_slide()

        # --- Slide 6: Simulate sample lines within bands ---
        # Scenario A: blue high (13.5), orange low (1.0) → early breakeven
        sa_buy, sa_bake = 13.5, 1.0
        sa_bw = sim_machine / (sa_buy - sa_bake)
        sa_bc = sa_buy * sa_bw
        sa_buy_line = axes.plot(lambda x: sa_buy * x, x_range=[0, n_weeks], color=BLUE, stroke_width=2)
        sa_bake_line = axes.plot(lambda x: sim_machine + sa_bake * x, x_range=[0, n_weeks], color=ORANGE, stroke_width=2)
        sa_dashed = DashedLine(axes.c2p(sa_bw, 0), axes.c2p(sa_bw, sa_bc), color=GRAY)
        sa_dot = Dot(axes.c2p(sa_bw, sa_bc), color=WHITE, radius=0.05)
        self.play(
            Transform(buy_line, sa_buy_line),
            Transform(bake_line, sa_bake_line),
            Transform(dashed, sa_dashed),
            Transform(dot, sa_dot),
            self.camera.frame.animate.move_to(axes.c2p(sa_bw, sa_bc)),
        )
        self.next_slide()

        # Scenario B: blue low (10.5), orange high (2.0) → late breakeven
        sb_buy, sb_bake = 10.5, 2.0
        sb_bw = sim_machine / (sb_buy - sb_bake)
        sb_bc = sb_buy * sb_bw
        sb_buy_line = axes.plot(lambda x: sb_buy * x, x_range=[0, n_weeks], color=BLUE, stroke_width=2)
        sb_bake_line = axes.plot(lambda x: sim_machine + sb_bake * x, x_range=[0, n_weeks], color=ORANGE, stroke_width=2)
        sb_dashed = DashedLine(axes.c2p(sb_bw, 0), axes.c2p(sb_bw, sb_bc), color=GRAY)
        sb_dot = Dot(axes.c2p(sb_bw, sb_bc), color=WHITE, radius=0.05)
        self.play(
            Transform(buy_line, sb_buy_line),
            Transform(bake_line, sb_bake_line),
            Transform(dashed, sb_dashed),
            Transform(dot, sb_dot),
            self.camera.frame.animate.move_to(axes.c2p(sb_bw, sb_bc)),
        )
        self.next_slide()

        # Scenario C: back to center
        sc_buy_line = axes.plot(lambda x: sim_buy_rate * x, x_range=[0, n_weeks], color=BLUE, stroke_width=2)
        sc_bake_line = axes.plot(lambda x: sim_machine + sim_bake_rate * x, x_range=[0, n_weeks], color=ORANGE, stroke_width=2)
        sc_dashed = DashedLine(axes.c2p(sim_bw, 0), axes.c2p(sim_bw, sim_bc), color=GRAY)
        sc_dot = Dot(axes.c2p(sim_bw, sim_bc), color=WHITE, radius=0.05)
        self.play(
            Transform(buy_line, sc_buy_line),
            Transform(bake_line, sc_bake_line),
            Transform(dashed, sc_dashed),
            Transform(dot, sc_dot),
            self.camera.frame.animate.move_to(axes.c2p(sim_bw, sim_bc)),
        )
        self.next_slide()

        # --- Slide 7: Highlight the diamond intersection region ---
        buy_hi_rate = sim_buy_rate + sim_buy_unc   # 14
        buy_lo_rate = sim_buy_rate - sim_buy_unc   # 10
        bake_hi_rate = sim_bake_rate + sim_bake_unc  # 2.75
        bake_lo_rate = sim_bake_rate - sim_bake_unc  # 0.25

        # Four corners where band edges cross
        def _cross(buy_r, bake_r):
            x = sim_machine / (buy_r - bake_r)
            return x, buy_r * x

        d_left  = _cross(buy_hi_rate, bake_lo_rate)   # earliest breakeven
        d_top   = _cross(buy_hi_rate, bake_hi_rate)
        d_right = _cross(buy_lo_rate, bake_hi_rate)    # latest breakeven
        d_bot   = _cross(buy_lo_rate, bake_lo_rate)

        diamond = Polygon(
            axes.c2p(*d_left), axes.c2p(*d_top),
            axes.c2p(*d_right), axes.c2p(*d_bot),
            color=RED, fill_opacity=0.25, stroke_width=2,
        )
        self.play(Create(diamond))
        self.next_slide()

        # --- Slide 8: Clear lines, keep diamond as focus ---
        self.play(
            FadeOut(buy_line), FadeOut(bake_line),
            FadeOut(dashed), FadeOut(dot),
        )
        self.next_slide()

        # --- Slide 9: Sample many dots inside diamond + zoom into diamond ---
        center_x = (d_left[0] + d_right[0]) / 2
        center_y = (d_bot[1] + d_top[1]) / 2

        rng = np.random.default_rng(42)
        n_samples = 30_000
        b_rates = rng.uniform(buy_lo_rate, buy_hi_rate, size=n_samples)
        k_rates = rng.uniform(bake_lo_rate, bake_hi_rate, size=n_samples)
        x_bw = sim_machine / (b_rates - k_rates)
        y_bw = b_rates * x_bw

        dots = VGroup(*[
            Dot(axes.c2p(x_bw[i], y_bw[i]), color=WHITE, radius=0.001,
                fill_opacity=0.35)
            for i in range(n_samples)
        ])

        # Fade in chunks of 3000 for progressive reveal
        chunk_size = 3000
        chunks = [
            VGroup(*dots[i:min(i + chunk_size, n_samples)])
            for i in range(0, n_samples, chunk_size)
        ]
        self.play(
            LaggedStart(*[FadeIn(c) for c in chunks], lag_ratio=0.15, run_time=5),
            self.camera.frame.animate.scale(0.6).move_to(
                axes.c2p(center_x, center_y)
            ),
        )
        self.next_slide()

        # --- Slide 10: Sand falls → histogram, background fades away ---
        n_bins = 20
        bin_lo = np.floor(x_bw.min())
        bin_hi = np.ceil(x_bw.max())
        bin_edges = np.linspace(bin_lo, bin_hi, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        bin_idx = np.digitize(x_bw, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        baseline_y = center_y - 60
        dot_spacing_y = 0.04

        bin_counts = [0] * n_bins
        target_positions = []
        for i in range(n_samples):
            b = bin_idx[i]
            tx = bin_centers[b]
            ty = baseline_y + bin_counts[b] * dot_spacing_y
            target_positions.append(axes.c2p(tx, ty))
            bin_counts[b] += 1

        # Build target VGroup and use single Transform (much faster than 30K anims)
        hist_dots = VGroup(*[
            Dot(target_positions[i], color=WHITE, radius=0.001, fill_opacity=0.35)
            for i in range(n_samples)
        ])

        self.play(
            Transform(dots, hist_dots, run_time=5),
            FadeOut(buy_band), FadeOut(bake_band),
            FadeOut(diamond), FadeOut(axes_group),
        )
        self.next_slide()

        # --- Slide 11: Restore original view (no dots) ---
        self.play(
            FadeOut(dots),
            FadeIn(buy_band), FadeIn(bake_band),
            FadeIn(axes_group),
            self.camera.frame.animate.scale(1 / 0.6).move_to(axes.c2p(sim_bw, sim_bc)),
        )
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

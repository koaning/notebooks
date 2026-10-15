# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.25.1",
#     "matplotlib==3.11.2",
#     "numpy==2.5.3",
#     "polars==2.0.0",
#     "wigglystuff==0.5.33",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The inspection paradox

    How big is a typical college class? You can ask the teachers, or you can
    ask the students. You'd think both give the same answer.

    Say the number of students in a class follows the distribution below. Most
    classes are somewhere near the average, with the odd giant lecture. Hover
    the slider to make classes bigger or smaller.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import polars as pl
    from wigglystuff import EsmWidget, HoverSlider

    if mo.app_meta().theme == "dark":
        plt.rcParams.update({"figure.facecolor": "#181a19", "axes.facecolor": "#181a19", "savefig.facecolor": "#181a19"})
    return EsmWidget, HoverSlider, mo, np, pl, plt


@app.cell
def _(HoverSlider, mo):
    class_size = mo.ui.anywidget(
        HoverSlider(start=10, stop=80, step=5, value=40, label="average students per class", width=320)
    )
    class_size
    return (class_size,)


@app.cell(hide_code=True)
def _(class_size, np, plt):
    mean_students = float(class_size.value["hover_value"])
    students_grid = np.linspace(0, 320, 321)
    gamma_scale = mean_students / 2
    density = students_grid * np.exp(-students_grid / gamma_scale) / gamma_scale**2

    fig_dist, ax_dist = plt.subplots(figsize=(10, 2.5))
    ax_dist.fill_between(students_grid, density, color="#4C78A8", alpha=0.4)
    ax_dist.plot(students_grid, density, color="#4C78A8")
    ax_dist.set_xlim(0, 320)
    ax_dist.set_ylim(0, density.max() * 1.1)
    ax_dist.set_yticks([])
    ax_dist.set_xlabel("students in the class")
    for side in ["top", "right", "left"]:
        ax_dist.spines[side].set_visible(False)
    fig_dist.tight_layout()
    fig_dist
    return (mean_students,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now sample 100 classes from it. Each teacher adds one answer to the top
    histogram. But a class with 40 students also has 40 people we could ask,
    and every one of them says "40". Those answers go in the bottom histogram.
    """)
    return


@app.cell(hide_code=True)
def _(EsmWidget, mo):
    class_anim = EsmWidget(
        r"""
        import { animate, stagger } from "https://cdn.jsdelivr.net/npm/motion@12/+esm";
        import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.17/+esm";

        const W = 720, H = 480, ML = 34, MR = 20, MT = 8;
        const INNER_H = 130;
        const TOP_Y = 116, BOT_Y = 292, TICKS_H = 24, AXIS_H = 42;
        const BOX_W = 130, BOX_H = 76, BOX_TOP = 32;
        const BLUE = "#4C78A8", RED = "#E45756";
        // 3Blue1Brown-style ease in and out.
        const SMOOTH = [0.65, 0, 0.35, 1];

        const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

        function div(parent, css, text) {
          const d = document.createElement("div");
          d.style.cssText = "position:absolute;" + css;
          if (text !== undefined) d.textContent = text;
          parent.appendChild(d);
          return d;
        }

        export default {
          render({ model, el }) {
            el.style.font = "12px system-ui, sans-serif";

            // Bumped on every button click or data change, so an older loop knows to quit.
            let gen = 0;
            let mode = "stopped";
            let st = null;
            // Elements and counts of the class currently being animated.
            const transient = new Set();
            let pending = null;

            function setup() {
              el.replaceChildren();
              transient.clear();
              pending = null;
              const { counts, mean, bin_width: bw, n_bins: nb } = model.get("data") || {};
              if (!counts) return false;

              const buttons = {};
              const bar = div(el, `right:${MR}px;top:0px;display:flex;gap:6px`);
              for (const name of ["replay", "slow", "fast", "superfast", "stop"]) {
                const btn = document.createElement("div");
                btn.textContent = name;
                btn.style.cssText = "padding:1px 9px;border:1px solid #ccc;border-radius:4px;cursor:pointer;user-select:none";
                btn.onclick = () => (name === "replay" ? replay() : start(name === "stop" ? "stopped" : name));
                bar.appendChild(btn);
                buttons[name] = btn;
              }

              // Rotated y-axis labels, centred on each histogram.
              for (const [y, color, text] of [[TOP_Y, BLUE, "Teachers say"], [BOT_Y, RED, "Students say"]]) {
                div(el, `left:${ML / 2 - 60}px;top:${y + MT + INNER_H / 2 - 8}px;width:120px;text-align:center;transform:rotate(-90deg);color:${color};font-weight:500`, text);
              }

              const zeros = () => new Array(nb).fill(0);
              st = {
                counts, mean, bw, nb, buttons,
                sampled: div(el, `left:${ML}px;top:3px;font-size:11px;opacity:0.6`),
                topAvg: div(el, `right:${MR}px;top:${TOP_Y + MT}px;font-size:11px;opacity:0.6;z-index:1`),
                botAvg: div(el, `right:${MR}px;top:${BOT_Y + MT}px;font-size:11px;opacity:0.6;z-index:1`),
                topPlot: div(el, `left:0px;top:${TOP_Y}px`),
                botPlot: div(el, `left:0px;top:${BOT_Y}px`),
                topCount: zeros(), botCount: zeros(),
                // What the bars show right now, and where they are heading.
                shown: { top: zeros(), bot: zeros(), yMax: 1 },
                target: { top: zeros(), bot: zeros(), yMax: 1 },
                tween: null,
                nClasses: 0, classSum: 0, nStudents: 0, sumSquares: 0,
              };
              updateBars(0);
              return true;
            }

            const xPx = (v) => ML + (v / (st.bw * st.nb)) * (W - ML - MR);
            const binOf = (n) => Math.min(Math.floor(n / st.bw), st.nb - 1);
            const binMid = (b) => xPx((b + 0.5) * st.bw);
            // Pixel y of the top of a bottom-histogram bar once the current tween lands.
            const botBarTop = (b) => BOT_Y + MT + INNER_H * (1 - st.target.bot[b] / st.target.yMax);
            // Gamma with shape 2 is the sum of two exponentials.
            const sample = () => Math.round(-(st.mean / 2) * (Math.log(1 - Math.random()) + Math.log(1 - Math.random())));

            function histogram(shares, yMax, color, withLabel) {
              const bins = shares.map((share, b) => ({ x1: b * st.bw, x2: (b + 1) * st.bw, share }));
              return Plot.plot({
                width: W,
                height: MT + INNER_H + (withLabel ? AXIS_H : TICKS_H),
                marginLeft: ML, marginRight: MR, marginTop: MT, marginBottom: withLabel ? AXIS_H : TICKS_H,
                x: withLabel
                  ? { domain: [0, st.bw * st.nb], label: "students in the class", labelAnchor: "center", labelArrow: "none" }
                  : { domain: [0, st.bw * st.nb], label: null },
                y: { domain: [0, yMax], axis: null },
                marks: [
                  Plot.rectY(bins, { x1: "x1", x2: "x2", y: "share", fill: color, insetLeft: 1, insetRight: 1 }),
                  Plot.ruleY([0], { stroke: "#aaa" }),
                ],
              });
            }

            function draw() {
              st.topPlot.replaceChildren(histogram(st.shown.top, st.shown.yMax, BLUE, false));
              st.botPlot.replaceChildren(histogram(st.shown.bot, st.shown.yMax, RED, true));
            }

            function updateBars(duration) {
              const top = st.topCount.map((c) => (st.nClasses ? c / st.nClasses : 0));
              const bot = st.botCount.map((c) => (st.nStudents ? c / st.nStudents : 0));
              st.target = { top, bot, yMax: Math.max(...top, ...bot, 1e-9) };

              const classAvg = st.nClasses ? st.classSum / st.nClasses : 0;
              const studentAvg = st.nStudents ? st.sumSquares / st.nStudents : 0;
              st.sampled.textContent = `${st.nClasses.toLocaleString("en-US")} class${st.nClasses === 1 ? "" : "es"} sampled`;
              st.topAvg.textContent = `average ${classAvg.toFixed(1)}`;
              st.botAvg.textContent = `average ${studentAvg.toFixed(1)}`;

              if (st.tween) st.tween.stop();
              if (duration < 0.1) {
                st.shown = st.target;
                draw();
                return;
              }
              const from = st.shown, to = st.target;
              const mix = (a, b, t) => a + (b - a) * t;
              st.tween = animate(0, 1, {
                duration,
                ease: SMOOTH,
                onUpdate: (t) => {
                  st.shown = {
                    top: from.top.map((v, b) => mix(v, to.top[b], t)),
                    bot: from.bot.map((v, b) => mix(v, to.bot[b], t)),
                    yMax: mix(from.yMax, to.yMax, t),
                  };
                  draw();
                },
              });
            }

            function commitTop(n) {
              st.topCount[binOf(n)] += 1;
              st.nClasses += 1;
              st.classSum += n;
            }

            function commitBot(n) {
              st.botCount[binOf(n)] += n;
              st.nStudents += n;
              st.sumSquares += n * n;
            }

            // Drop whatever class is mid-animation, but keep its answers.
            function finishInFlight() {
              transient.forEach((e) => e.remove());
              transient.clear();
              if (pending) {
                if (!pending.topDone) commitTop(pending.n);
                if (!pending.botDone) commitBot(pending.n);
                pending = null;
                updateBars(0.2);
              }
            }

            function start(newMode) {
              if (!st) return;
              gen++;
              finishInFlight();
              mode = newMode;
              for (const [name, btn] of Object.entries(st.buttons)) {
                const active = name === mode || (name === "stop" && mode === "stopped");
                btn.style.borderColor = active ? BLUE : "#ccc";
                btn.style.color = active ? BLUE : "";
              }
              if (mode !== "stopped") loop(gen);
            }

            function replay() {
              gen++;
              if (setup()) start("intro");
            }

            async function loop(myGen) {
              const alive = () => myGen === gen;
              while (alive()) {
                if (mode === "intro") {
                  const i = st.nClasses;
                  if (i >= st.counts.length) return start("stopped");
                  if (i < 3) await slowStep(st.counts[i], 1, true, alive);
                  else if (i < 15) await slowStep(st.counts[i], 0.35, false, alive);
                  else await fastStep([st.counts[i]], 70);
                } else if (mode === "slow") {
                  await slowStep(sample(), 1, true, alive);
                } else if (mode === "fast") {
                  await fastStep([sample(), sample(), sample()], 60);
                } else {
                  await fastStep(Array.from({ length: 25 }, sample), 25);
                }
              }
            }

            async function fastStep(ns, ms) {
              for (const n of ns) {
                commitTop(n);
                commitBot(n);
              }
              updateBars(0);
              await sleep(ms);
            }

            function temp(css, text) {
              const d = div(el, css, text);
              transient.add(d);
              return d;
            }

            async function slowStep(n, f, explain, alive) {
              const b = binOf(n);
              pending = { n, topDone: false, botDone: false };
              const barLeft = xPx(b * st.bw) + 1;
              const barInner = xPx(st.bw) - xPx(0) - 2;
              const topBarTop = () => TOP_Y + MT + INNER_H * (1 - st.target.top[b] / st.target.yMax);

              // The class appears above its bin, its students popping in one by one.
              const boxLeft = Math.max(ML, Math.min(W - MR - BOX_W, binMid(b) - BOX_W / 2));
              const boxCss = `left:${boxLeft}px;top:${BOX_TOP}px;width:${BOX_W}px;height:${BOX_H}px;box-sizing:border-box;border:1.5px solid ${BLUE};border-radius:6px;background-color:rgba(76,120,168,0)`;
              const box = temp(boxCss + ";opacity:0");
              const cols = Math.max(1, Math.ceil(Math.sqrt((n * (BOX_W - 10)) / (BOX_H - 10))));
              const rows = Math.max(1, Math.ceil(n / cols));
              const cell = Math.min(9, (BOX_W - 10) / cols, (BOX_H - 10) / rows);
              const size = Math.max(2, cell - 1.5);
              const dots = [];
              for (let k = 0; k < n; k++) {
                const x = boxLeft + 5 + (k % cols) * cell;
                const y = BOX_TOP + 5 + Math.floor(k / cols) * cell;
                const dot = temp(`left:${x}px;top:${y}px;width:${size}px;height:${size}px;border-radius:50%;background:${RED};opacity:0`);
                dots.push({ dot, x, y });
              }
              const dotEls = dots.map((d) => d.dot);
              animate(box, { opacity: [0, 1], scale: [0.9, 1] }, { duration: 0.5 * f, ease: SMOOTH });
              if (n > 0) {
                await animate(dotEls, { opacity: [0, 1], scale: [0, 1] }, { duration: 0.3 * f, delay: stagger((0.5 * f) / n), ease: SMOOTH });
              } else {
                await sleep(500 * f);
              }
              if (!alive()) return;
              if (explain) await sleep(400);
              if (!alive()) return;

              // A copy of the class morphs into one block on top of its bar.
              const copy = temp(boxCss);
              await animate(
                copy,
                { left: `${barLeft}px`, top: `${topBarTop() - 8}px`, width: `${barInner}px`, height: "8px", borderRadius: "1px", backgroundColor: BLUE },
                { duration: 0.8 * f, ease: SMOOTH }
              );
              if (!alive()) return;
              commitTop(n);
              pending.topDone = true;
              updateBars(0.4 * f);
              await animate(copy, { top: `${topBarTop()}px`, height: "0px" }, { duration: 0.4 * f, ease: SMOOTH });
              if (!alive()) return;
              copy.remove();
              if (explain) await sleep(500);
              if (!alive()) return;

              // Every student flies down, in order, and they stack into a strip on top of their bar.
              const botTop = botBarTop(b);
              const landCell = Math.max(2, Math.min(size + 1.5, Math.sqrt((barInner * 40) / Math.max(n, 1))));
              const landCols = Math.max(1, Math.floor(barInner / landCell));
              const stripH = Math.ceil(n / landCols) * landCell;
              const pad = (barInner - landCols * landCell) / 2;
              const flights = dots.map(({ dot, x, y }, k) => {
                const cx = barLeft + pad + (k % landCols) * landCell + landCell / 2;
                const cy = botTop - (Math.floor(k / landCols) + 0.5) * landCell;
                return animate(
                  dot,
                  { x: cx - (x + size / 2), y: cy - (y + size / 2), scale: (landCell - 0.5) / size },
                  { duration: 0.9 * f, delay: (k / Math.max(n, 1)) * (explain ? 0.9 : 0.4) * f, ease: SMOOTH }
                );
              });
              animate(box, { opacity: 0, scale: 0.9 }, { duration: 0.5 * f, delay: 0.4 * f, ease: SMOOTH });
              await Promise.all(flights);
              if (!alive()) return;

              // The stack of students melts into a solid strip, which the bar then absorbs.
              const strip = temp(`left:${barLeft}px;top:${botTop - stripH}px;width:${barInner}px;height:${stripH}px;background:${RED};opacity:0`);
              if (n > 0) animate(dotEls, { opacity: 0 }, { duration: 0.3 * f, ease: SMOOTH });
              await animate(strip, { opacity: 1 }, { duration: 0.3 * f, ease: SMOOTH });
              if (!alive()) return;
              commitBot(n);
              pending.botDone = true;
              updateBars(0.5 * f);
              await animate(strip, { top: `${botBarTop(b)}px`, height: "0px" }, { duration: 0.5 * f, ease: SMOOTH });
              if (!alive()) return;
              if (explain) await sleep(700);
              if (!alive()) return;
              transient.forEach((e) => e.remove());
              transient.clear();
              pending = null;
            }

            replay();
            model.on("change:data", replay);
            return () => {
              gen++;
              if (st && st.tween) st.tween.stop();
              model.off("change:data", replay);
            };
          },
        };
        """,
        width=720,
        height=480,
    )
    mo.center(class_anim).style(padding="24px 0")
    return (class_anim,)


@app.cell(hide_code=True)
def _(class_anim, mean_students, np):
    sim_bin_width = max(1, round(mean_students / 5))
    class_anim.data = {
        "mean": mean_students,
        "counts": np.random.default_rng(16).gamma(2, mean_students / 2, size=100).round().astype(int).tolist(),
        "bin_width": sim_bin_width,
        "n_bins": int(np.ceil(4 * mean_students / sim_bin_width)),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The teachers average out to the slider's value. The students land around
    1.5 times that, because a big lecture gets counted once for every student
    in it, while a small seminar barely gets counted at all.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Brothers and sisters

    Same trick, different room. Ask parents how many kids they have, and some
    say zero. Now turn to a room full of people and ask how many kids their
    parents had. Nobody answers zero, because childless families send nobody
    to the room. And a family with three kids shows up three times.
    """)
    return


@app.cell
def _(HoverSlider, mo):
    avg_kids = mo.ui.anywidget(
        HoverSlider(start=0.5, stop=3.0, step=0.1, value=1.6, label="average kids per family", width=320)
    )
    avg_kids
    return (avg_kids,)


@app.cell
def _(avg_kids, mo, np, plt):
    kids = np.random.default_rng(3).poisson(float(avg_kids.value["hover_value"]), size=5000)
    sizes = np.arange(0, 8)
    share_parents = np.array([(kids == size).mean() for size in sizes])
    share_room = np.array([kids[kids == size].sum() for size in sizes]) / kids.sum()

    fig4, ax_kids = plt.subplots(figsize=(10, 4))
    ax_kids.bar(sizes - 0.2, share_parents, width=0.4, color="#4C78A8", label="ask the parents")
    ax_kids.bar(sizes + 0.2, share_room, width=0.4, color="#E45756", label="ask the room")
    ax_kids.set_xticks(sizes)
    ax_kids.set_xlabel("kids in the family")
    ax_kids.set_yticks([])
    ax_kids.legend(frameon=False)
    fig4.tight_layout()

    room_avg = (kids**2).sum() / kids.sum()
    mo.vstack(
        [
            mo.md(
                f"Average family: **{kids.mean():.1f}** kids. "
                f"Average person in the room grew up with **{room_avg:.1f}** kids in the family, "
                f"so **{room_avg - 1:.1f}** brothers and sisters."
            ),
            fig4,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Who's online?

    Now some real data. Between 2006 and 2009, researchers logged into a World
    of Warcraft server every 10 minutes and wrote down every player who was
    online. That gives one row per player per snapshot.

    Suppose you want to know how much a typical player plays. You could go
    through the list of every player who ever showed up. Or you could log in at
    a random moment, walk up to someone online, and ask. That second option is
    how an in-game survey or a website popup works.
    """)
    return


@app.cell
def _(pl):
    wow = pl.read_parquet(
        "https://media.githubusercontent.com/media/koaning/wow-avatar-datasets/main/wow-lite.parquet"
    )
    daily = (
        wow.with_columns(
            hours=pl.int_range(1, pl.len() + 1).over("player_id", order_by="datetime") / 6,
            day=pl.col("datetime").dt.date(),
        )
        .group_by("player_id", "day")
        .agg(pl.col("hours").max())
        .with_columns(days_played=pl.int_range(1, pl.len() + 1).over("player_id", order_by="day"))
    )
    activity = daily.group_by("day").len().sort("day")
    days = activity["day"]
    return activity, daily, days


@app.cell
def _(HoverSlider, days, mo):
    day_index = mo.ui.anywidget(
        HoverSlider(start=0, stop=len(days) - 1, step=1, value=len(days) // 2, label="survey day", width=320)
    )
    day_index
    return (day_index,)


@app.cell(hide_code=True, expand_output=True)
def _(activity, daily, day_index, days, mo, np, pl, plt):
    survey_day = days[int(day_index.value["hover_value"])]
    surveyed = daily.filter(pl.col("day") == survey_day)
    registry = daily.filter(pl.col("day") <= survey_day).group_by("player_id").agg(
        pl.col("hours").max(), per_day=pl.col("hours").max() / pl.col("days_played").max()
    )

    snapshot_edges = np.unique(np.round(np.logspace(0, np.log10(daily["hours"].max() * 6), 40)))
    hour_bins = (np.append(snapshot_edges - 0.5, snapshot_edges[-1] + 0.5) / 6).tolist()
    share_players = registry["hours"].hist(bins=hour_bins).select(pl.col("count") / pl.col("count").sum())
    share_online = surveyed["hours"].hist(bins=hour_bins).select(pl.col("count") / pl.col("count").sum())

    before_day = activity.filter(pl.col("day") <= survey_day)
    fig_survey, (ax_activity, ax_today) = plt.subplots(2, 1, figsize=(10, 6.5), height_ratios=[3, 3.5])
    ax_activity.plot(activity["day"], activity["len"], color="#888888", linewidth=0.8)
    ax_activity.fill_between(before_day["day"], before_day["len"], color="#4C78A8", alpha=0.5, label="counted in \"every player so far\"")
    ax_activity.axvline(survey_day, color="#E45756", linewidth=2, label="survey day")
    ax_activity.set_ylabel("players online")
    ax_activity.set_ylim(0, None)
    for activity_side in ["top", "right"]:
        ax_activity.spines[activity_side].set_visible(False)
    ax_activity.legend(frameon=False, loc="upper left")


    per_day_bins = np.arange(0, 12.25, 0.25).tolist()
    per_day_players = registry["per_day"].hist(bins=per_day_bins).select(pl.col("count") / pl.col("count").sum())
    per_day_online = (surveyed["hours"] / surveyed["days_played"]).hist(bins=per_day_bins).select(pl.col("count") / pl.col("count").sum())
    ax_today.stairs(per_day_players["count"], per_day_bins, fill=True, color="#4C78A8", alpha=0.5, label="ask every player so far")
    ax_today.stairs(per_day_online["count"], per_day_bins, fill=True, color="#E45756", alpha=0.5, label="ask who's online that day")
    ax_today.set_xlim(0, 12)
    ax_today.set_xlabel("hours per day played, so far")
    ax_today.set_yticks([])
    for today_side in ["top", "right", "left"]:
        ax_today.spines[today_side].set_visible(False)
    ax_today.legend(frameon=False)
    fig_survey.tight_layout()

    online_per_day = (surveyed["hours"] / surveyed["days_played"]).mean()
    mo.vstack(
        [
            fig_survey,
            mo.md(
                f"Average hours per day played: every player so far **{registry['per_day'].mean():.1f}h**, "
                f"who's online that day **{online_per_day:.1f}h**."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Most players tried the game for an evening and left. They're the bulk of
    the player list, but they were almost never online when you looked. The
    regulars, meanwhile, show up in snapshot after snapshot.

    The dataset itself has a mild version of the same problem: a session
    shorter than 10 minutes can slip between two snapshots and never show up.
    """)
    return


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.25.1",
#     "matplotlib==3.11.2",
#     "numpy==2.5.3",
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

    How busy is a typical restaurant? You can ask the restaurants, or you can
    ask the customers. You'd think both give the same answer.

    Say the number of customers a restaurant gets on a night follows the
    distribution below. Most nights are somewhere near the average, with the
    odd packed night. Hover the slider to make the town busier or quieter.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from wigglystuff import EsmWidget, HoverSlider

    return EsmWidget, HoverSlider, mo, np, plt


@app.cell
def _(HoverSlider, mo):
    busyness = mo.ui.anywidget(
        HoverSlider(start=10, stop=80, step=5, value=40, label="average customers per night", width=320)
    )
    busyness
    return (busyness,)


@app.cell(hide_code=True)
def _(busyness, np, plt):
    mean_customers = float(busyness.value["hover_value"])
    customers_grid = np.linspace(0, 320, 321)
    gamma_scale = mean_customers / 2
    density = customers_grid * np.exp(-customers_grid / gamma_scale) / gamma_scale**2

    fig_dist, ax_dist = plt.subplots(figsize=(10, 2.5))
    ax_dist.fill_between(customers_grid, density, color="#4C78A8", alpha=0.4)
    ax_dist.plot(customers_grid, density, color="#4C78A8")
    ax_dist.set_xlim(0, 320)
    ax_dist.set_ylim(0, density.max() * 1.1)
    ax_dist.set_yticks([])
    ax_dist.set_xlabel("customers in a restaurant that night")
    for side in ["top", "right", "left"]:
        ax_dist.spines[side].set_visible(False)
    fig_dist.tight_layout()
    fig_dist
    return (mean_customers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now sample 100 restaurants from it. Each restaurant adds one answer to the
    top histogram. But a restaurant with 40 customers also has 40 people we could
    ask, and every one of them says "40". Those answers go in the bottom
    histogram.
    """)
    return


@app.cell(hide_code=True)
def _(EsmWidget):
    crowd_anim = EsmWidget(
        r"""
        import { animate } from "https://cdn.jsdelivr.net/npm/motion@12/+esm";

        const W = 720, H = 510, L = 40, R = 20, PW = W - L - R;
        const HIST_H = 130, TOP_BASE = 270, BOT_BASE = 450;
        const BOX_W = 130, BOX_H = 76, BOX_TOP = 34;
        const BLUE = "#4C78A8", RED = "#E45756";

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
            el.style.fontFamily = "system-ui, sans-serif";
            el.style.fontSize = "13px";

            // Bumped on every button click or data change, so an older loop knows to quit.
            let gen = 0;
            let mode = "stopped";
            let st = null;
            // Elements and counts of the restaurant currently being animated.
            const transient = new Set();
            let pending = null;

            function setup() {
              el.replaceChildren();
              transient.clear();
              pending = null;
              const { counts, mean, bin_width: bw, n_bins: nb } = model.get("data") || {};
              if (!counts) return false;
              const binW = PW / nb;

              for (const base of [TOP_BASE, BOT_BASE]) {
                div(el, `left:${L}px;top:${base}px;width:${PW}px;height:1px;background:#999`);
              }
              const tickStep = bw * Math.ceil(nb / 5);
              for (let v = 0; v <= bw * nb; v += tickStep) {
                div(el, `left:${L + (v / (bw * nb)) * PW - 20}px;top:${BOT_BASE + 4}px;width:40px;text-align:center;opacity:0.7`, String(v));
              }
              div(el, `left:${L}px;top:${BOT_BASE + 24}px;width:${PW}px;text-align:center;opacity:0.7`, "customers in the restaurant that night");

              const buttons = {};
              const bar = div(el, `right:${R}px;top:2px;display:flex;gap:6px`);
              for (const name of ["replay", "slow", "fast", "stop"]) {
                const btn = document.createElement("div");
                btn.textContent = name;
                btn.style.cssText = "padding:2px 10px;border:1px solid #999;border-radius:4px;cursor:pointer;user-select:none";
                btn.onclick = () => (name === "replay" ? replay() : start(name === "stop" ? "stopped" : name));
                bar.appendChild(btn);
                buttons[name] = btn;
              }

              const makeBars = (base, color) =>
                Array.from({ length: nb }, (_, b) =>
                  div(el, `left:${L + b * binW + 1}px;bottom:${H - base}px;width:${binW - 2}px;height:0px;background:${color}`)
                );
              st = {
                counts, mean, bw, nb, binW, buttons,
                topTitle: div(el, `left:${L}px;top:${TOP_BASE - HIST_H - 22}px;color:${BLUE};font-weight:600`),
                botTitle: div(el, `left:${L}px;top:${BOT_BASE - HIST_H - 22}px;color:${RED};font-weight:600`),
                status: div(el, `left:${L}px;top:6px;opacity:0.8`),
                topBars: makeBars(TOP_BASE, BLUE),
                botBars: makeBars(BOT_BASE, RED),
                topCount: new Array(nb).fill(0),
                botCount: new Array(nb).fill(0),
                botHeights: new Array(nb).fill(0),
                nRestaurants: 0, restaurantSum: 0, nCustomers: 0, sumSquares: 0,
              };
              updateBars(0);
              return true;
            }

            const binOf = (n) => Math.min(Math.floor(n / st.bw), st.nb - 1);
            const binMid = (b) => L + (b + 0.5) * st.binW;
            // Gamma with shape 2 is the sum of two exponentials.
            const sample = () => Math.round(-(st.mean / 2) * (Math.log(1 - Math.random()) + Math.log(1 - Math.random())));

            function updateBars(duration) {
              const topShare = st.topCount.map((c) => (st.nRestaurants ? c / st.nRestaurants : 0));
              const botShare = st.botCount.map((c) => (st.nCustomers ? c / st.nCustomers : 0));
              const yMax = Math.max(...topShare, ...botShare, 1e-9);
              topShare.forEach((s, b) => animate(st.topBars[b], { height: `${(s / yMax) * HIST_H}px` }, { duration }));
              botShare.forEach((s, b) => {
                st.botHeights[b] = (s / yMax) * HIST_H;
                animate(st.botBars[b], { height: `${st.botHeights[b]}px` }, { duration });
              });
              const restAvg = st.nRestaurants ? st.restaurantSum / st.nRestaurants : 0;
              const customerAvg = st.nCustomers ? st.sumSquares / st.nCustomers : 0;
              const answers = (k) => `${k} answer${k === 1 ? "" : "s"}`;
              st.topTitle.textContent = `Ask the restaurants (${answers(st.nRestaurants)}): average ${restAvg.toFixed(0)}`;
              st.botTitle.textContent = `Ask the customers (${answers(st.nCustomers)}): average ${customerAvg.toFixed(0)}`;
            }

            function commitTop(n) {
              st.topCount[binOf(n)] += 1;
              st.nRestaurants += 1;
              st.restaurantSum += n;
            }

            function commitBot(n) {
              st.botCount[binOf(n)] += n;
              st.nCustomers += n;
              st.sumSquares += n * n;
            }

            function showStatus(k, n) {
              const of = mode === "intro" ? ` of ${st.counts.length}` : "";
              st.status.textContent = `restaurant ${k}${of}: ${n} customers`;
            }

            // Drop whatever restaurant is mid-animation, but keep its answers.
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
                btn.style.fontWeight = active ? "600" : "400";
                btn.style.borderColor = active ? BLUE : "#999";
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
                  const i = st.nRestaurants;
                  if (i >= st.counts.length) {
                    start("stopped");
                    st.status.textContent = `${st.counts.length} restaurants sampled`;
                    return;
                  }
                  if (i < 3) await slowStep(st.counts[i], 1, true, alive);
                  else if (i < 15) await slowStep(st.counts[i], 0.35, false, alive);
                  else await fastStep([st.counts[i]], 70);
                } else if (mode === "slow") {
                  await slowStep(sample(), 1, true, alive);
                } else {
                  await fastStep([sample(), sample(), sample()], 60);
                }
              }
            }

            async function fastStep(ns, ms) {
              for (const n of ns) {
                commitTop(n);
                commitBot(n);
              }
              showStatus(st.nRestaurants, ns[ns.length - 1]);
              updateBars(0.05);
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
              showStatus(st.nRestaurants + 1, n);

              // The restaurant appears above its bin, full of its customers.
              const boxLeft = Math.max(L, Math.min(W - R - BOX_W, binMid(b) - BOX_W / 2));
              const box = temp(`left:${boxLeft}px;top:${BOX_TOP}px;width:${BOX_W}px;height:${BOX_H}px;border:1.5px solid ${BLUE};border-radius:6px;opacity:0`);
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
              await animate([box, ...dots.map((d) => d.dot)], { opacity: 1 }, { duration: 0.4 * f });
              if (!alive()) return;
              if (explain) await sleep(400);
              if (!alive()) return;

              // One answer from the restaurant drops into the top histogram.
              const vote = temp(`left:${binMid(b) - 6}px;top:${BOX_TOP + BOX_H}px;width:12px;height:12px;background:${BLUE};border-radius:2px`);
              if (explain) {
                temp(`left:${binMid(b) + st.binW / 2 + 6}px;top:${TOP_BASE - HIST_H + 10}px;color:${BLUE};white-space:nowrap`, `1 restaurant says ${n}`);
              }
              await animate(vote, { y: TOP_BASE - BOX_TOP - BOX_H - 12 }, { duration: 0.6 * f });
              if (!alive()) return;
              vote.remove();
              commitTop(n);
              pending.topDone = true;
              updateBars(0.3 * f);
              if (explain) await sleep(700);
              if (!alive()) return;

              // Every customer inside flies down and gives the same answer.
              if (explain) {
                temp(`left:${binMid(b) + st.binW / 2 + 6}px;top:${BOT_BASE - HIST_H + 10}px;color:${RED};white-space:nowrap`, `${n} customers each say ${n}`);
              }
              const targetY = BOT_BASE - st.botHeights[b] - size;
              const flights = dots.map(({ dot, x, y }) =>
                animate(
                  dot,
                  { x: binMid(b) - x + (Math.random() - 0.5) * (st.binW - 4) - size / 2, y: targetY - y - Math.random() * 6 },
                  { duration: 0.9 * f, delay: Math.random() * (explain ? 0.8 : 0.3) * f }
                )
              );
              await Promise.all(flights);
              if (!alive()) return;
              commitBot(n);
              pending.botDone = true;
              updateBars(0.3 * f);
              await animate([box, ...dots.map((d) => d.dot)], { opacity: 0 }, { duration: 0.3 * f });
              if (!alive()) return;
              if (explain) await sleep(900);
              if (!alive()) return;
              transient.forEach((e) => e.remove());
              transient.clear();
              pending = null;
            }

            replay();
            model.on("change:data", replay);
            return () => {
              gen++;
              model.off("change:data", replay);
            };
          },
        };
        """,
        width=720,
        height=510,
    )
    crowd_anim
    return (crowd_anim,)


@app.cell(hide_code=True)
def _(crowd_anim, mean_customers, np):
    sim_bin_width = max(1, round(mean_customers / 5))
    crowd_anim.data = {
        "mean": mean_customers,
        "counts": np.random.default_rng(16).gamma(2, mean_customers / 2, size=100).round().astype(int).tolist(),
        "bin_width": sim_bin_width,
        "n_bins": int(np.ceil(4 * mean_customers / sim_bin_width)),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The restaurants average out to the slider's value. The customers land around
    1.5 times that, because a busy restaurant gets counted once for every
    person inside it, while a quiet one barely gets counted at all.
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


if __name__ == "__main__":
    app.run()

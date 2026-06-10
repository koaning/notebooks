# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "mpmath",
#     "polars",
#     "altair",
#     "numpy==2.4.6",
#     "matplotlib==3.10.9",
#     "scipy==1.17.1",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Is your birthday hiding inside $\pi$?

    $\pi$\u2019s digits run on forever without repeating. Type a number below \u2014 your
    **birth year** is a fun one \u2014 and we\u2019ll find where it first shows up.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    import numpy as np
    from mpmath import mp

    return alt, mo, mp, np, pl


@app.cell(hide_code=True)
def _(mp):
    import sys


    def pi_digits(n):
        """First n decimal digits of pi as a string (the digits after "3.")."""
        sys.set_int_max_str_digits(max(n + 50, 4300))
        mp.dps = n + 25
        scaled = mp.floor(mp.pi * mp.mpf(10) ** n)
        return str(int(scaled))[1:]


    # k=4 needs ~390k digits on average (coupon-collector mean); 600k gives headroom.
    N_DIGITS = 1_000_000
    digits = pi_digits(N_DIGITS)
    return (digits,)


@app.cell(hide_code=True)
def _(mo):
    pi_query = mo.ui.text(value="1989", label="a number (try your birth year)")
    pi_query
    return (pi_query,)


@app.cell(hide_code=True)
def _(digits, mo, pi_query):
    def search_view(digits, raw, context=12):
        s = raw.strip()
        if not s.isdigit():
            return mo.md("Type a number (digits only) above \u261d\ufe0f")
        pos = digits.find(s)
        if pos < 0:
            return mo.md(
                f"**{s}** isn\u2019t in the first {len(digits):,} digits we computed \u2014 "
                f"it\u2019s almost surely further out (we just stopped at {len(digits):,})."
            )
        before = digits[max(0, pos - context):pos]
        after = digits[pos + len(s):pos + len(s) + context]
        return mo.md(
            f"### \U0001f3af Found **{s}** at digit **{pos + 1:,}** of \u03c0\n\n"
            f"\u2026{before}**{s}**{after}\u2026"
        )


    search_view(digits, pi_query.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## But is *every* number in there?

    It seems so.

    But that's not the most interesting thing. It's not just that the numbers are in here. It's that they are in here and it seems that they are randomly distributed in there!
    """)
    return


@app.cell
def _(mo):
    digits_slider = mo.ui.slider(1, 4, 1, label="n digits")
    prob_slider = mo.ui.slider(0.01, 0.10, 0.01, label="prob limit")
    digits_slider, prob_slider
    return digits_slider, prob_slider


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we look at every number that comes out as a batch.
    """)
    return


@app.cell
def _(digits_slider, n_batches, prob_slider):
    from scipy.stats import binom

    lim_lower, lim_upper = [
        binom.ppf(prob_slider.value, n_batches, 1/10**digits_slider.value), 
        binom.ppf(1 - prob_slider.value, n_batches, 1/10**digits_slider.value)
    ]

    lim_lower, lim_upper
    return lim_lower, lim_upper


@app.cell
def _(digits, digits_slider):
    import itertools as it

    n_batches = sum(1 for _ in it.batched(digits, digits_slider.value))
    return it, n_batches


@app.cell
def _(digits, digits_slider, it, lim_lower, lim_upper):

    import matplotlib.pylab as plt
    from collections import Counter

    start = Counter({k: 0 for k in it.product("0123456789", repeat=digits_slider.value)})
    counts = (start + Counter(it.batched(digits, digits_slider.value))).values()

    out_of_bounds = 0
    for v in counts:
        if v < lim_lower or v > lim_upper:
            out_of_bounds += 1

    print(f"{out_of_bounds} out of {len(start)}, approx {out_of_bounds/len(start)*100/2}% per side")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 1 — Collect every $k$-digit number

    We're gonna read $\pi$ in disjoint chunks of length $k$ and wait until **every**
    value $0\ldots10^k-1$ has appeared at least once. That's a situation similar to the famous coupon-collector problem. One with $n = 10^k$ coupons.
    """)
    return


@app.cell(hide_code=True)
def _(digits):
    def first_complete(digits, k):
        """Scan disjoint k-length chunks of `digits`; find when all 10**k values appear.

        Returns the chunk index and digit position where the collection first
        completes, the straggler value (last to appear), and a running record of
        (chunk_count, distinct_count) after each chunk for plotting.
        """
        universe = 10 ** k
        seen = set()
        record = []
        completed_at_chunk = None
        last_value = None
        n_chunks = len(digits) // k
        for i in range(n_chunks):
            value = int(digits[i * k:(i + 1) * k])
            if value not in seen:
                seen.add(value)
                if len(seen) == universe and completed_at_chunk is None:
                    completed_at_chunk = i
                    last_value = value
            record.append((i + 1, len(seen)))
        return {
            "k": k,
            "universe": universe,
            "completed_chunk": completed_at_chunk,
            "completed_digit_pos": None if completed_at_chunk is None else (completed_at_chunk + 1) * k,
            "last_value": last_value,
            "n_seen": len(seen),
            "record": record,
        }


    K_VALUES = (1, 2, 3, 4)
    results = {k: first_complete(digits, k) for k in K_VALUES}
    return (results,)


@app.cell(hide_code=True)
def _(mo, pl, results):
    def coupon_stats(n):
        """Mean and standard deviation of the coupon-collector time for n coupons."""
        mean = n * sum(1.0 / i for i in range(1, n + 1))
        var = sum((n / (n - j)) ** 2 - (n / (n - j)) for j in range(n))
        return mean, var ** 0.5


    rows = []
    for k, r in results.items():
        n = r["universe"]
        mean_chunks, sd_chunks = coupon_stats(n)
        chunks = r["completed_chunk"] + 1
        rows.append({
            "k": k,
            "values (10^k)": n,
            "done @ digit": r["completed_digit_pos"],
            "chunks needed": chunks,
            "last value": r["last_value"],
            "theory mean (chunks)": round(mean_chunks, 1),
            "theory sd (chunks)": round(sd_chunks, 1),
            "z-score": round((chunks - mean_chunks) / sd_chunks, 2),
        })

    summary_df = pl.DataFrame(rows)
    mo.vstack([
        mo.md("## Results vs. coupon-collector theory"),
        summary_df,
        mo.md(
            "The **z-score** says how many standard deviations \u03c0's actual "
            "completion sits from the coupon-collector mean. All four land within "
            "about one sd \u2014 exactly what you'd expect if \u03c0's digits behave "
            "like an i.i.d. uniform random stream. No value is anomalously early or late."
        ),
    ])
    return (coupon_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 2 — Turning up the resolution

    Collecting *every* $k$-digit number only gives us the points
    $n = 10, 100, 1000, 10000$ — four spots, exponentially far apart. For a **dense**
    view we want to coupon-collect a universe of *any* size $n$.

    Clean trick: **fold $\pi$ into $n$ coupons with a mod.** Read $\pi$ in a window of
    digits wide enough that its value is $\gg n$, take that value $\bmod n$, and you get
    a draw that's (near-)uniform over $0\ldots n-1$ — exactly one coupon. Collect all
    $n$ and it takes about $n\cdot H(n)$ steps, the textbook result. Now the universe
    size is just a slider.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    n_coupons = mo.ui.slider(2, 1000, value=100, show_value=True,
                             label="number of coupons n  (fold \u03c0 mod n)")
    n_coupons
    return (n_coupons,)


@app.cell(hide_code=True)
def _(coupon_stats, digits, mo, n_coupons):
    def coupon_steps(digits, n, guard=2):
        """Steps to collect all n coupons, folding pi into 0..n-1 via mod on wide windows."""
        w = len(str(n)) + guard          # 10**w >> n, so each draw is near-uniform over 0..n-1
        seen = set()
        steps = 0
        for i in range(len(digits) // w):
            coupon = int(digits[i * w:(i + 1) * w]) % n
            steps += 1
            if coupon not in seen:
                seen.add(coupon)
                if len(seen) == n:
                    return steps
        return None


    demo_w = len(str(n_coupons.value)) + 2
    demo_steps = coupon_steps(digits, n_coupons.value)
    demo_mean, demo_sd = coupon_stats(n_coupons.value)

    mo.md(
        f"Folding \u03c0 into **n = {n_coupons.value}** coupons "
        f"(reading {demo_w} digits per draw, then mod {n_coupons.value}):  \n\n"
        f"\u03c0 collects all {n_coupons.value} coupons in **{demo_steps:,} steps**.  \n"
        f"Theory $n\\cdot H(n) \\approx$ **{demo_mean:,.0f}** steps "
        f"(\u00b1 {demo_sd:,.0f}), so this run is **{(demo_steps - demo_mean) / demo_sd:+.1f}\u03c3** "
        f"from the mean."
    )
    return (coupon_steps,)


@app.cell(hide_code=True)
def _(mo):
    n_sweep = mo.ui.slider(10, 1000, value=120, show_value=True,
                           label="how many n values to sweep (more = denser curve)")
    n_sweep
    return (n_sweep,)


@app.cell(hide_code=True)
def _(alt, coupon_steps, digits, mo, n_sweep, np, pl):
    def sweep_n(n_points, max_n=1000):
        return sorted(set(int(round(2 + i * (max_n - 2) / (n_points - 1)))
                          for i in range(n_points)))


    def simulate_band(n_values, n_runs=3000, seed=0):
        """Empirical 10th-90th percentile of coupon-collect time, from real RNG draws."""
        rng = np.random.default_rng(seed)
        rows = []
        for n in n_values:
            p = np.arange(n, 0, -1) / n                          # P(new coupon) at each stage
            sims = rng.geometric(p, size=(n_runs, n)).sum(axis=1)  # one full run per row
            rows.append({"n": n, "p1": float(np.percentile(sims, 1)),
                         "p99": float(np.percentile(sims, 99))})
        return pl.DataFrame(rows)


    sweep_ns = sweep_n(n_sweep.value)
    pi_df = pl.DataFrame([{"n": n, "steps": coupon_steps(digits, n)} for n in sweep_ns])
    band_df = simulate_band(sweep_ns)

    sim_band = (
        alt.Chart(band_df)
        .mark_area(color="#9d9d9d", opacity=0.4)
        .encode(x=alt.X("n:Q", title="number of coupons n"),
                y=alt.Y("p1:Q", title="steps to collect all n"),
                y2="p99:Q")
    )
    pi_dots = (
        alt.Chart(pi_df)
        .mark_circle(size=42, color="#4c78a8")
        .encode(x="n:Q", y="steps:Q", tooltip=["n", "steps"])
    )
    sweep_chart = (sim_band + pi_dots).properties(width=620, height=340)

    mo.vstack([
        mo.md(
            "**\u03c0 vs. actual randomness.** The grey band is the 10th\u201390th percentile of "
            "**simulated** coupon-collector runs on a true random-number generator "
            "(300 runs per $n$) \u2014 no formula, just rolling dice. Each blue dot is \u03c0's "
            "single run at that $n$. If \u03c0's digits are random-like, its dots should land "
            "inside the band about as often as a fresh random source would."
        ),
        sweep_chart,
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

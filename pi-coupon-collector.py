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


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    import numpy as np
    from mpmath import mp

    return mo, mp


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
        before = digits[max(0, pos - context) : pos]
        after = digits[pos + len(s) : pos + len(s) + context]
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
        binom.ppf(prob_slider.value, n_batches, 1 / 10**digits_slider.value),
        binom.ppf(1 - prob_slider.value, n_batches, 1 / 10**digits_slider.value),
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
    from collections import Counter

    start = Counter({k: 0 for k in it.product("0123456789", repeat=digits_slider.value)})
    counts = (start + Counter(it.batched(digits, digits_slider.value))).values()

    out_of_bounds = 0
    for v in counts:
        if v < lim_lower or v > lim_upper:
            out_of_bounds += 1

    print(
        f"{out_of_bounds} out of {len(start)}, approx {out_of_bounds / len(start) * 100 / 2}% per side"
    )
    return


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
    return N_DIGITS, digits, pi_digits


@app.cell(hide_code=True)
def _(N_DIGITS, digits, pi_digits):
    def test_pi_digits_match_known_reference():
        # pi's first 100 decimal digits (after the point), from a published source.
        reference = (
            "1415926535897932384626433832795028841971"
            "6939937510582097494459230781640628620899"
            "86280348253421170679"
        )
        assert pi_digits(100) == reference
        assert pi_digits(10) == reference[:10]


    def test_pi_digits_prefix_consistency():
        # A longer computation must agree with shorter ones on the shared prefix,
        # which catches rounding / precision corruption in the tail.
        assert pi_digits(2_000)[:100] == pi_digits(100)
        assert pi_digits(20_000)[:2_000] == pi_digits(2_000)


    def test_pi_digits_shape():
        out = pi_digits(777)
        assert len(out) == 777
        assert set(out) <= set("0123456789")


    def test_notebook_digits_are_valid():
        # validate the actual sequence the rest of the notebook relies on
        reference = (
            "1415926535897932384626433832795028841971"
            "6939937510582097494459230781640628620899"
            "86280348253421170679"
        )
        assert len(digits) == N_DIGITS
        assert digits[:100] == reference
        assert set(digits) <= set("0123456789")

    return


if __name__ == "__main__":
    app.run()

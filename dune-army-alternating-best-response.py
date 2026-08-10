# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "altair<5.5",
#     "httpx",
#     "marimo",
#     "mohtml==0.1.11",
#     "numpy==2.5.1",
#     "polars",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Robust C&C-family army via *alternating best response*

    Companion to the fictitious-play notebook. Same game, same payoff matrix `M`
    from real duels on [datasette.exe.xyz](https://datasette.exe.xyz/cnc_units) —
    but a **different solver**, to see whether it lands on the same army.

    Method: **alternating best-response dynamics**. `x` plays a pure best
    response to `y`'s *current* army, then `y` best-responds to `x`'s current
    army, and so on. No history-averaging (that is what fictitious play does).
    In a zero-sum game this tends to **cycle** rather than settle — unless we
    damp each move with a **stepsize** `α`: `α=1` is the full jump (cycles),
    smaller `α` pulls `x` toward the fictitious-play army.
    """)
    return


@app.cell
def _():
    import io

    import altair as alt
    import httpx
    import marimo as mo
    import numpy as np
    import polars as pl
    from mohtml import div, img, span

    return alt, div, httpx, img, io, mo, np, pl, span


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Data — game, budget, units

    Pick the **game** (which roster and matchups to use) and the **budget**.
    Army size sets the scale: combat is non-linear, so both armies fielding more
    units changes the matchups — `M` is loaded from duels at exactly that budget.
    Use the expandable panel to drop units from the roster. Everything downstream
    rebuilds on any change.
    """)
    return


@app.cell
def _(mo):
    game_dd = mo.ui.dropdown(
        options={
            "Dune 2000 (d2k)": "d2k",
            "Command & Conquer (cnc)": "cnc",
            "Red Alert (ra)": "ra",
        },
        value="Dune 2000 (d2k)",
        label="game",
    )
    budget_dd = mo.ui.dropdown(
        options={f"{b:,} credits": b for b in (2000, 10000, 20000, 50000, 100000)},
        value="100,000 credits",
        label="simulation budget (matrix scale)",
    )
    mo.hstack([game_dd, budget_dd], justify="start", gap=2)
    return budget_dd, game_dd


@app.cell
def _(budget_dd, game_dd, httpx, io, pl):
    def load_csv(sql: str) -> pl.DataFrame:
        resp = httpx.get(
            "https://datasette.exe.xyz/cnc_units.csv",
            params={"sql": sql, "_size": "max"},
            timeout=30,
        )
        resp.raise_for_status()
        return pl.read_csv(io.StringIO(resp.text))

    def load_json(sql: str) -> list[dict]:
        resp = httpx.get(
            "https://datasette.exe.xyz/cnc_units.json",
            params={"sql": sql, "_shape": "array", "_size": "max"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    mod = game_dd.value
    budget = budget_dd.value

    # Full roster = the units that actually appear in this game's duels.
    all_units = [
        r["code"]
        for r in load_json(
            f"select distinct attacker as code from duels where mod='{mod}' "
            "order by attacker"
        )
    ]

    duels_df = load_csv(
        "select attacker, defender, atk_hp_left, atk_hp_max, atk_cost_max, "
        "def_hp_left, def_hp_max, def_cost_max "
        f"from duels where mod='{mod}' and budget={budget}"
    )

    # Unit metadata keyed by lowercased code (cnc duels are UPPERCASE, the units
    # table is lowercase — match case-insensitively so all games resolve).
    meta = {
        r["code"].lower(): {
            "name": r["name"],
            "cost": float(r["cost"]),
            "icon": f"data:image/png;base64,{r['icon']['encoded']}",
        }
        for r in load_json(f"select code, name, cost, icon from units where mod='{mod}'")
    }
    return all_units, budget, duels_df, meta


@app.cell
def _(all_units, mo):
    unit_select = mo.ui.multiselect(
        options=all_units, value=all_units, label="units in play"
    )
    mo.accordion({"⚙️ Units — deselect to exclude from the matrix": unit_select})
    return (unit_select,)


@app.cell
def _(all_units, meta, mo, np, unit_select):
    chosen = set(unit_select.value)
    units = [u for u in all_units if u in chosen]
    mo.stop(len(units) < 2, mo.md("**Select at least two units.**"))

    cost = np.array([meta[u.lower()]["cost"] for u in units], dtype=float)
    icons = {u: meta[u.lower()]["icon"] for u in units}
    names = {u: meta[u.lower()]["name"] for u in units}
    return cost, icons, names, units


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Payoff matrix `M`

    We score a duel by **how much of each army's cost survives**, since credits
    are the common currency across units (raw HP isn't comparable — units have
    wildly different HP pools). HP is just how we *normalise* survival: a unit at
    40% HP counts as 40% of its credit value. So each side's surviving value is
    `hp_left / hp_max × credits_fielded`, and the margin from row unit *i* is

    `(value_i − value_j) / budget`.

    (This also penalises units that can't spend the whole budget — some hit a
    unit cap, so they field fewer credits than the budget allows.) Each duel
    contributes to both `M[i,j]` and `M[j,i]`, mixing attacker and defender roles
    → a role-neutral, ~anti-symmetric zero-sum matrix.

    Combat is non-linear in army size, so `M` is **scale-dependent** — it is
    built from duels fought at our target budget. Blue = row unit wins the
    matchup, red = it loses.
    """)
    return


@app.cell
def _(budget, duels_df, np, units):
    idx = {u: k for k, u in enumerate(units)}
    n = len(units)
    _sums = np.zeros((n, n))
    _counts = np.zeros((n, n))

    for row in duels_df.iter_rows(named=True):
        if row["attacker"] not in idx or row["defender"] not in idx:
            continue  # a deselected unit
        a, d = idx[row["attacker"]], idx[row["defender"]]
        # Surviving army *value* in credits (hp fraction × credits fielded),
        # as a fraction of budget. This also penalises units that cannot spend
        # the whole budget (cost_max < budget due to unit caps).
        va = row["atk_hp_left"] / row["atk_hp_max"] * row["atk_cost_max"] / budget
        vd = row["def_hp_left"] / row["def_hp_max"] * row["def_cost_max"] / budget
        margin = va - vd
        _sums[a, d] += margin
        _counts[a, d] += 1
        _sums[d, a] += -margin
        _counts[d, a] += 1

    M = np.divide(_sums, _counts, out=np.zeros_like(_sums), where=_counts > 0)
    return (M,)


@app.cell
def _(M, div, icons, img, mo, names, span, units):
    ICON = 58
    COL = ICON + 20

    def payoff_style(v):
        # v in [-1, 1]: blue = row unit wins, red = row unit loses.
        base = (33, 102, 172) if v >= 0 else (178, 24, 43)
        t = min(abs(v), 1.0)
        rgb = tuple(int(255 + (b - 255) * t) for b in base)
        fg = "#fff" if t > 0.55 else "#1a1a1a"
        return f"background:rgb{rgb};color:{fg};"

    def header(code):
        # Unit name label sitting just above its (pixel-art) icon.
        return div(
            span(
                names[code],
                style="font:600 10px system-ui;color:#555;text-align:center;"
                "line-height:1.15;",
            ),
            img(
                src=icons[code],
                width=str(ICON),
                height=str(ICON),
                title=code,
                style="image-rendering:pixelated;display:block;margin:3px auto 0;",
            ),
            style=f"width:{COL}px;display:flex;flex-direction:column;"
            "align-items:center;justify-content:flex-end;",
        )

    _cell = (
        f"width:{COL}px;height:{COL}px;display:flex;align-items:center;"
        "justify-content:center;font:600 14px system-ui;border-radius:7px;"
    )

    _kids = [div(span("i \\ j", style="font:600 11px system-ui;color:#999;"),
                 style=f"width:{COL}px;display:flex;align-items:flex-end;"
                 "justify-content:center;padding-bottom:6px;")]
    _kids += [header(u) for u in units]
    for i, ui in enumerate(units):
        _kids.append(header(ui))
        for j, uj in enumerate(units):
            v = float(M[i, j])
            if i == j:
                _kids.append(div("–", style=_cell + "background:#f2f2f2;color:#ccc;"))
            else:
                _kids.append(
                    div(
                        f"{v:+.2f}",
                        style=_cell + payoff_style(v),
                        title=f"{ui} vs {uj}: {v:+.2f}",
                    )
                )

    matrix = div(
        *_kids,
        style=(
            f"display:grid;grid-template-columns:repeat({len(units) + 1},{COL}px);"
            "gap:6px;padding:16px;width:fit-content;align-items:end;"
        ),
    )
    # Wrap in a horizontally scrollable container for wide rosters.
    scroller = div(matrix, style="overflow-x:auto;max-width:100%;")
    mo.accordion({f"🔲 Payoff matrix M ({len(units)}×{len(units)})": scroller})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Alternating best response — the solver

    Same game as fictitious play: pick a *spend distribution* $x$, the enemy
    picks $y$, our score is $x^\top M y$. The equilibrium $x^\star$ is the robust
    army. But we reach for it differently here.

    **The dynamics.** No history, no averaging — the two sides just take turns
    playing a pure best response to the *current* opponent:

    1. **Our move.** Against the enemy's current army $y$, field the single unit
       with the best matchup: $i^\star = \arg\max_i (My)_i$, so $x = e_{i^\star}$.
    2. **Enemy move.** Against our *new* army $x$, the enemy fields the single
       unit that hurts us most: $j^\star = \arg\min_j (x^\top M)_j$, so
       $y = e_{j^\star}$.
    3. Repeat.

    **What to expect.** For a zero-sum game these unaveraged dynamics generally
    **do not converge** — they orbit the equilibrium (think matching-pennies:
    you switch, so I switch, so you switch back…). The **exploitability gap**

    $$\max_i (My)_i \;-\; \min_j (x^\top M)_j$$

    of the *current* pure strategies keeps bouncing instead of shrinking to $0$.

    **Stepsize.** Instead of jumping all the way to the best-response vertex,
    take a damped step $x \leftarrow (1-\alpha)\,x + \alpha\,e_{i^\star}$ (and
    likewise for $y$). Now $x$ itself is a mixed army. $\alpha = 1$ is the full
    jump above (it cycles); shrinking $\alpha$ damps the move and, on a zero-sum
    game, pulls $x$ toward the Nash equilibrium — recovering the fictitious-play
    army in the limit. The slider interpolates between "misses by cycling" and
    "converges."
    """)
    return


@app.cell
def _(np):
    def alternating_best_response(M, T, alpha, rng):
        # Alternating best-response dynamics with a stepsize `alpha`. Each round
        # x takes a convex step toward its pure best response to the enemy's
        # current army, then y steps toward its best response to x:
        #   x <- (1-alpha)*x + alpha*e_i,   y <- (1-alpha)*y + alpha*e_j.
        # alpha=1 is a full jump to the best-response vertex — unaveraged best
        # response, which cycles. Smaller alpha damps the move and, on a
        # zero-sum game, pulls x toward the Nash equilibrium. The current mixed
        # strategy x IS the army — no history-averaging.
        n = M.shape[0]
        x_cur = rng.dirichlet(np.ones(n))  # random start (decays at rate 1-alpha)
        y_cur = rng.dirichlet(np.ones(n))
        gaps, values = [], []
        x_hist, y_hist = np.zeros((T, n)), np.zeros((T, n))  # x(t), y(t)
        x_pick = np.zeros(T, dtype=int)  # best-response target x steps toward
        y_pick = np.zeros(T, dtype=int)

        for t in range(T):
            i = int(np.argmax(M @ y_cur))
            x_cur = (1.0 - alpha) * x_cur
            x_cur[i] += alpha

            j = int(np.argmin(x_cur @ M))
            y_cur = (1.0 - alpha) * y_cur
            y_cur[j] += alpha

            x_pick[t] = i
            y_pick[t] = j
            x_hist[t] = x_cur
            y_hist[t] = y_cur
            # Exploitability of the current mixed strategies.
            gaps.append(float((M @ y_cur).max() - (x_cur @ M).min()))
            values.append(float(x_cur @ M @ y_cur))

        return {
            "x_final": x_cur,  # the final mixed army
            "y_final": y_cur,
            "gaps": np.array(gaps),
            "values": np.array(values),
            "x_hist": x_hist,
            "y_hist": y_hist,
            "x_pick": x_pick,
            "y_pick": y_pick,
        }

    return (alternating_best_response,)


@app.cell
def _(mo):
    T_slider = mo.ui.slider(
        10, 30000, value=800, step=10, label="iterations T", show_value=True
    )
    restarts_slider = mo.ui.slider(
        1, 12, value=6, step=1, label="random restarts", show_value=True
    )
    alpha_slider = mo.ui.slider(
        0.001, 1.0, value=1.0, step=0.001, label="stepsize α", show_value=True
    )
    mo.hstack([T_slider, restarts_slider, alpha_slider], justify="start", gap=2)
    return T_slider, alpha_slider, restarts_slider


@app.cell
def _(
    M,
    T_slider,
    alpha_slider,
    alternating_best_response,
    np,
    restarts_slider,
):
    # Run alternating best response from several random starting armies.
    runs = [
        alternating_best_response(
            M, T_slider.value, alpha_slider.value, np.random.default_rng(s)
        )
        for s in range(restarts_slider.value)
    ]
    fp = runs[0]
    x_final = fp["x_final"]
    y_final = fp["y_final"]
    return fp, runs, x_final, y_final


@app.cell
def _(mo):
    log_gap = mo.ui.checkbox(value=True, label="log scale for exploitability gap")
    log_gap
    return (log_gap,)


@app.cell
def _(alt, log_gap, mo, np, pl, runs):
    _step = max(1, len(runs[0]["gaps"]) // 250)  # subsample so Altair stays light
    _long = pl.concat(
        [
            pl.DataFrame(
                {
                    "iteration": range(1, len(r["gaps"]) + 1, _step),
                    "gap": r["gaps"][::_step],
                    "restart": f"seed {s}",
                }
            )
            for s, r in enumerate(runs)
        ]
    )
    _chart = (
        alt.Chart(_long)
        .mark_line(opacity=0.75)
        .encode(
            x=alt.X("iteration:Q", title="iteration"),
            y=alt.Y(
                "gap:Q",
                title="exploitability gap (current strategies)",
                scale=alt.Scale(type="symlog", constant=0.01)
                if log_gap.value
                else alt.Scale(type="linear"),
            ),
            color=alt.Color("restart:N", title="random start"),
        )
        .properties(width=620, height=280, title="Exploitability of the current pair")
        .interactive()
    )

    _final_gaps = np.array([r["gaps"][-1] for r in runs])
    _caption = mo.md(
        f"**{len(runs)} random restarts** of alternating best response. "
        f"Worst final exploitability gap **{_final_gaps.max():.2e}** — "
        + (
            "still large: the current pure strategies keep **cycling**, they do "
            "not settle to an equilibrium (unlike fictitious play)."
            if _final_gaps.max() > 1e-3
            else "small: on this matrix the dynamics happen to settle."
        )
    )
    mo.vstack([_chart, _caption])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The trajectory

    The top chart shows the **best-response target each round** — the vertex $x$
    steps toward. At $\alpha = 1$ it jumps straight there and cycles; at smaller
    $\alpha$ it still flips, but $x$ only edges toward it. The bottom chart is the
    **army mix $x(t)$** itself: with damping it smooths out and converges. Compare
    it to the fictitious-play notebook's equilibrium mix.
    """)
    return


@app.cell
def _(alt, fp, names, pl, units):
    _picks = fp["x_pick"]
    _win = min(len(_picks), 40)  # a short window makes the zig-zag legible
    _df = pl.DataFrame(
        {
            "iteration": range(1, _win + 1),
            "unit": [names[units[i]] for i in _picks[:_win]],
        }
    )
    # Connect consecutive picks so a period-2 flip reads as an obvious sawtooth
    # rather than two static rows of dots.
    _base = alt.Chart(_df).encode(
        x=alt.X("iteration:Q", title=f"iteration (first {_win})"),
        y=alt.Y("unit:N", title="best-response target"),
    )
    (
        (
            _base.mark_line(color="#4a90a4", opacity=0.6)
            + _base.mark_circle(size=60, color="#4a90a4").encode(
                tooltip=["iteration", "unit"]
            )
        ).properties(
            width=560,
            height=len(units) * 26 + 20,
            title="Best-response target i(t) — the vertex x steps toward",
        )
    )
    return


@app.cell
def _(alt, fp, names, pl, units):
    _hist = fp["x_hist"]
    _step = max(1, _hist.shape[0] // 300)  # subsample so Altair stays light
    _rows = [
        {"iteration": t + 1, "unit": names[units[k]], "share": float(_hist[t, k])}
        for t in range(0, _hist.shape[0], _step)
        for k in range(len(units))
    ]
    _df = pl.DataFrame(_rows)
    (
        alt.Chart(_df)
        .mark_area()
        .encode(
            x=alt.X("iteration:Q", title="iteration"),
            y=alt.Y("share:Q", stack="normalize", title="spend share"),
            color=alt.Color("unit:N", title="unit"),
            order=alt.Order("unit:N"),
        )
        .properties(width=560, height=280, title="Army mix x(t)")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Spend shares → army

    Convert the time-average spend distribution to unit counts, then greedily
    spend leftover credits on the affordable unit with the best marginal
    payoff against the enemy's average mix.

    The **army budget** below is independent of the simulation budget above: the
    mix comes from the dynamics, this just decides how many credits we actually
    spend fielding it.
    """)
    return


@app.cell
def _(mo):
    army_budget = mo.ui.number(
        start=1000,
        stop=1_000_000,
        step=1000,
        value=100_000,
        label="army budget (credits)",
    )
    army_budget
    return (army_budget,)


@app.cell
def _(M, army_budget, cost, np, pl, units, x_final, y_final):
    def build_army(x, total, cost, M, y):
        counts = np.floor(total * x / cost).astype(int)
        leftover = total - int((counts * cost).sum())
        marginal = M @ y  # value per unit vs enemy mix
        while True:
            affordable = np.where(cost <= leftover)[0]
            if len(affordable) == 0:
                break
            pick = affordable[int(np.argmax(marginal[affordable]))]
            counts[pick] += 1
            leftover -= int(cost[pick])
        return counts, leftover

    counts, leftover = build_army(x_final, army_budget.value, cost, M, y_final)

    army = pl.DataFrame(
        {
            "unit": units,
            "cost": cost.astype(int),
            "count": counts,
            "spend": counts * cost.astype(int),
            "spend_share": np.round(counts * cost / max(int((counts * cost).sum()), 1), 3),
        }
    ).sort("spend", descending=True)
    return army, leftover


@app.cell
def _(army, leftover, mo):
    mo.md(f"""
    **Total spend:** {int(army['spend'].sum()):,} · **leftover:** {leftover}
    """)
    return


@app.cell
def _(alt, army):
    (
        alt.Chart(army)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="unit count"),
            y=alt.Y("unit:N", sort="-x", title=None),
            color=alt.Color("unit:N", legend=None),
            tooltip=["unit", "count", "cost", "spend"],
        )
        .properties(width=560, height=200, title="Final army")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What beats this army?

    Now **lock in the army above** and let the enemy counter it. Since the payoff
    is linear in the enemy's choice, their most optimal counter is always a **pure
    spam of a single unit** — the column that hurts our *fielded* mix the most. So
    the best possible counter-army is just "all-in" on one unit.

    Below: every unit ranked by how it fares against our army (red = it beats us),
    then the single most optimal counter-army.
    """)
    return


@app.cell
def _(M, army, np, units):
    # Realized spend-share of the army we actually field (integer counts).
    _spend = {r["unit"]: r["spend"] for r in army.iter_rows(named=True)}
    _total = max(sum(_spend.values()), 1)
    x_field = np.array([_spend[u] / _total for u in units])

    # Our payoff if the enemy fields a pure spam of unit j; most negative = best
    # counter for them.
    vs_counter = x_field @ M
    counter = int(np.argmin(vs_counter))
    return counter, vs_counter


@app.cell
def _(alt, counter, names, pl, units, vs_counter):
    _df = pl.DataFrame(
        {
            "unit": [names[u] for u in units],
            "our_payoff": [float(v) for v in vs_counter],
            "outcome": [
                "beats our army" if v < -1e-9 else "loses to our army"
                for v in vs_counter
            ],
            "best": [k == counter for k in range(len(units))],
        }
    ).sort("our_payoff")
    (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            x=alt.X("our_payoff:Q", title="our payoff vs a pure spam of this unit"),
            y=alt.Y("unit:N", sort="x", title=None),
            color=alt.Color(
                "outcome:N",
                scale=alt.Scale(
                    domain=["beats our army", "loses to our army"],
                    range=["#c0392b", "#4a90a4"],
                ),
                title=None,
            ),
            tooltip=["unit", alt.Tooltip("our_payoff:Q", format="+.3f"), "outcome"],
        )
        .properties(
            width=560,
            height=len(units) * 26 + 20,
            title="How each unit fares against our army",
        )
    )
    return


@app.cell
def _(
    budget,
    cost,
    counter,
    div,
    icons,
    img,
    mo,
    names,
    span,
    units,
    vs_counter,
):
    _j = counter
    _margin = -float(vs_counter[_j])  # enemy's advantage against our army
    _n_counter = int(budget // cost[_j])

    _card = div(
        img(
            src=icons[units[_j]],
            width="72",
            height="72",
            style="image-rendering:pixelated;",
        ),
        div(
            span(names[units[_j]], style="font:700 18px system-ui;"),
            span(
                f"{_n_counter:,} units  ·  {budget:,} credits",
                style="font:13px system-ui;color:#666;",
            ),
            style="display:flex;flex-direction:column;gap:2px;",
        ),
        style="display:flex;align-items:center;gap:14px;padding:14px 18px;"
        "border:1px solid #eee;border-radius:12px;width:fit-content;",
    )
    _msg = (
        f"**Most optimal counter:** all-in on **{names[units[_j]]}**, "
        f"beating our army by **{_margin:+.3f}** of budget."
        if _margin > 1e-9
        else "**This army is unexploitable** — the best single-unit counter only "
        "ties it (payoff ≈ 0)."
    )
    mo.vstack([_card, mo.md(_msg)])
    return


if __name__ == "__main__":
    app.run()

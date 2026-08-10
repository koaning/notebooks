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
    # Robust C&C-family army via fictitious play

    We look for a robust army for a chosen game and budget, using a 1v1 payoff
    matrix `M` derived from real duel data on
    [datasette.exe.xyz](https://datasette.exe.xyz/cnc_units).

    Method: **fictitious play** (averaged best-response dynamics) on the
    symmetric zero-sum payoff matrix.
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
    (equilibrium, army, validation) rebuilds on any change.
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

    # pl.DataFrame(
    #     {"unit": [names[u] for u in units], "code": units, "cost": cost.astype(int)}
    # ).sort("cost")
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
    ## Fictitious play — how we find the robust army

    **The game.** Think of army-building as a two-player zero-sum game. We pick a
    *spend distribution* $x$ (what fraction of our budget goes to each unit); the
    enemy picks theirs, $y$. Our score is

    $$\text{payoff}(x, y) = x^\top M y = \sum_{i,j} x_i \, M_{ij} \, y_j,$$

    the credit-weighted average of the pairwise matchups. A **robust** army is one
    that does well against the enemy's *best possible* response — i.e. the
    equilibrium $x^\star$ of this game. There is no single "best unit"; the answer
    is generally a *mix*, because the enemy adapts to whatever we field.

    **The algorithm.** Fictitious play finds that equilibrium by having both sides
    learn from each other's history. Start with a guess for the enemy, then repeat:

    1. **Our move.** Look at the enemy's *average* army so far, $\bar y$. Field the
       single unit with the best expected matchup against it:
       $i^\star = \arg\max_i (M\bar y)_i$.
    2. **Enemy move.** The enemy looks at *our* average so far, $\bar x$, and fields
       the single unit that hurts us most: $j^\star = \arg\min_j (\bar x^\top M)_j$.
    3. Record both one-unit picks and update the running averages.

    Each step is a trivial pure best response, but the **time-averages** $\bar x,
    \bar y$ mix over many rounds. For a zero-sum game they are guaranteed to
    converge to a Nash equilibrium (Robinson, 1951). We take $\bar x$ as our army.

    **Are we there yet?** The **exploitability gap** measures it:

    $$\underbrace{\max_i (M\bar y)_i}_{\text{best we could do vs }\bar y}
      \;-\;
      \underbrace{\min_j (\bar x^\top M)_j}_{\text{worst the enemy can do to }\bar x}.$$

    Both bracket the game's value; the gap shrinks to $0$ at equilibrium. When it is
    near zero, no enemy composition meaningfully beats $\bar x$ — that is the robust
    army we are after.
    """)
    return


@app.cell
def _(np):
    def fictitious_play(M, T, rng):
        # Each round: we best-respond to the enemy's average (argmax M@y_avg),
        # the enemy best-responds to ours (argmin x_avg@M). The running averages
        # x_final / y_final converge to the zero-sum Nash equilibrium.
        n = M.shape[0]
        # Random start: seed each side's history with one virtual move drawn
        # uniformly from the simplex. Its weight is 1, so it fades like 1/T and
        # cannot affect the limit — it only decides where the walk begins.
        x_sum = rng.dirichlet(np.ones(n))
        y_sum = rng.dirichlet(np.ones(n))
        x0 = x_sum.copy()  # the raw random start (t=0), before any best response
        x_count = y_count = 1
        gaps, values = [], []
        x_hist, y_hist = np.zeros((T, n)), np.zeros((T, n))  # running mixes
        y_used = np.zeros(n)  # how often each enemy unit was a best response

        for t in range(T):
            y_avg = y_sum / y_count
            i = int(np.argmax(M @ y_avg))
            x_sum[i] += 1.0
            x_count += 1

            x_avg = x_sum / x_count
            j = int(np.argmin(x_avg @ M))
            y_sum[j] += 1.0
            y_count += 1
            y_used[j] += 1

            x_hist[t] = x_avg
            y_hist[t] = y_sum / y_count
            gaps.append(float((M @ y_avg).max() - (x_avg @ M).min()))
            values.append(float(x_avg @ M @ y_avg))

        return {
            "x_final": x_sum / x_count,
            "y_final": y_sum / y_count,
            "gaps": np.array(gaps),
            "values": np.array(values),
            "x_hist": x_hist,
            "y_hist": y_hist,
            "x0": x0,
            "y_used": y_used,
        }

    return (fictitious_play,)


@app.cell
def _(mo):
    T_slider = mo.ui.slider(
        10, 30000, value=800, step=10, label="iterations T", show_value=True
    )
    restarts_slider = mo.ui.slider(
        1, 12, value=6, step=1, label="random restarts", show_value=True
    )
    mo.hstack([T_slider, restarts_slider], justify="start", gap=2)
    return T_slider, restarts_slider


@app.cell
def _(M, T_slider, fictitious_play, np, restarts_slider):
    # Run fictitious play from several random starting beliefs. In a zero-sum
    # game every run must converge to the same game value, so this is our
    # convergence check — all gap curves should fall to ~0 together.
    runs = [
        fictitious_play(M, T_slider.value, np.random.default_rng(s))
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
def _(M, alt, log_gap, mo, np, pl, runs):
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
                title="exploitability gap",
                scale=alt.Scale(type="symlog", constant=0.01)
                if log_gap.value
                else alt.Scale(type="linear"),
            ),
            color=alt.Color("restart:N", title="random start"),
        )
        .properties(width=620, height=280, title="Convergence from random starts")
        .interactive()
    )

    _values = np.array([float(r["x_final"] @ M @ r["y_final"]) for r in runs])
    _final_gaps = np.array([r["gaps"][-1] for r in runs])
    _caption = mo.md(
        f"**{len(runs)} random restarts** → game value "
        f"**{_values.mean():+.4f}** (spread {np.ptp(_values):.1e}); "
        f"worst final exploitability gap **{_final_gaps.max():.2e}**. "
        "All starts collapse to the same value ⇒ converged."
    )
    mo.vstack([_chart, _caption])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How the mix settles

    The running spend shares $\bar x_i(t)$ as our army evolves over the iterations
    of the first restart — watch them flatten out as the game reaches equilibrium.
    (The game is symmetric, so the enemy's equilibrium mix $\bar y$ coincides with
    ours; one chart says it all.)

    Play starts from a **random** point on the simplex; tick the box to show it as
    $t=0$. It carries weight 1, so it washes out within a step or two — which is
    exactly why the equilibrium doesn't depend on where we start.
    """)
    return


@app.cell
def _(mo):
    show_t0 = mo.ui.checkbox(value=False, label="show random init (t=0)")
    show_t0
    return (show_t0,)


@app.cell
def _(alt, fp, names, pl, show_t0, units):
    _hist = fp["x_hist"]
    _step = max(1, _hist.shape[0] // 300)  # subsample so Altair stays light
    _rows = [
        {"iteration": t + 1, "unit": names[units[k]], "share": float(_hist[t, k])}
        for t in range(0, _hist.shape[0], _step)
        for k in range(len(units))
    ]
    if show_t0.value:
        # Prepend the raw random allocation before any best response.
        _rows = [
            {"iteration": 0, "unit": names[units[k]], "share": float(fp["x0"][k])}
            for k in range(len(units))
        ] + _rows
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
        .properties(width=560, height=280, title="Our army mix x̄(t)")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Spend shares → army

    Convert the equilibrium spend distribution to unit counts, then greedily
    spend leftover credits on the affordable unit with the best marginal
    payoff against the enemy's equilibrium mix.

    The **army budget** below is independent of the simulation budget above: the
    equilibrium mix comes from the matrix, this just decides how many credits we
    actually spend fielding it.
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
        "ties it (payoff ≈ 0). That is exactly what the equilibrium guarantees."
    )
    mo.vstack([_card, mo.md(_msg)])
    return


if __name__ == "__main__":
    app.run()

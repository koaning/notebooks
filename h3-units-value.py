# /// script
# dependencies = [
#     "marimo",
#     "moutils[db]",
#     "polars",
#     "altair",
#     "mohtml==0.1.11",
#     "playwright==1.62.0",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", sql_output="polars")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # HoMM3 units: best bang for your buck 🪙

    Which base unit gives the most value per gold? We pull units from the
    [`h3_units`](https://datasette.exe.xyz/h3_units/units) datasette and compare a
    few value-per-gold lenses. **Pick a level** below (and optionally hide
    flying/ranged units, since the combat model is melee-only).

    *The neutral Peasant and the duplicate Factory Halfling are excluded.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    level_sel = mo.ui.dropdown(
        options=[str(i) for i in range(1, 8)], value="1", label="Unit level"
    )
    melee_only = mo.ui.switch(value=False, label="Melee only (exclude flying & ranged)")
    mo.hstack([level_sel, melee_only], justify="start", gap=2)
    return level_sel, melee_only


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    import math

    from moutils.db.datasette import DatasetteConnection
    from mohtml import table, tr, th, td, div, span, img

    return (
        DatasetteConnection,
        alt,
        div,
        img,
        math,
        mo,
        pl,
        span,
        table,
        td,
        th,
        tr,
    )


@app.cell
def _(DatasetteConnection):
    datasette = DatasetteConnection("https://datasette.exe.xyz", "h3_units")
    return (datasette,)


@app.cell
def _(datasette, level_sel, melee_only, mo):
    _conds = [
        f"level = '{level_sel.value}'",
        "name NOT IN ('Peasant', 'Halfling (Factory)')",
    ]
    if melee_only.value:
        _conds += [
            "COALESCE(special, '') NOT LIKE '%Flying%'",
            "COALESCE(special, '') NOT LIKE '%Ranged%'",
        ]
    _where = " AND ".join(_conds)
    units = mo.sql(
        f"""
        SELECT name, town, level, gold, attack, defense, min_dmg, max_dmg,
               hp, speed, growth, ai_value, special, icon
        FROM units
        WHERE {_where}
        ORDER BY gold
        """,
        engine=datasette,
    )
    return (units,)


@app.cell(hide_code=True)
def _(pl, units):
    # Everything here is built from ai_value — the game's own aggregate power score —
    # so there are no hand-picked stat weights. Three ai_value-based lenses:
    #   value_per_gold        = ai_value / gold          (per-coin efficiency)
    #   weekly_value          = ai_value * growth        (dwelling throughput)
    #   value_growth_per_gold = ai_value * growth / gold (efficiency x production)
    scored = units.with_columns(
        icon=pl.lit("data:image/png;base64,") + pl.col("icon").struct.field("encoded"),
    ).with_columns(
        value_per_gold=(pl.col("ai_value") / pl.col("gold")).round(2),
        weekly_value=pl.col("ai_value") * pl.col("growth"),
        value_growth_per_gold=(
            pl.col("ai_value") * pl.col("growth") / pl.col("gold")
        ).round(1),
    )

    scored.select(
        "name", "town", "gold", "ai_value", "growth",
        "value_per_gold", "weekly_value", "value_growth_per_gold",
    ).sort("value_growth_per_gold", descending=True)
    return (scored,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Three lenses, all built on `ai_value`

    I dropped the earlier `combat_score` (attack + defense + avg_dmg + hp) — you were
    right to distrust it: it sums stats on different scales with weights I invented and
    ignores abilities. These three lenses use only **`ai_value`**, the game's own power score:

    - **Value per gold** = `ai_value ÷ gold` — power per coin. Growth-independent.
    - **Weekly value** = `ai_value × growth` — power *one dwelling* fields per week (absolute).
    - **Value × growth per gold** = `ai_value × growth ÷ gold` — your metric: per-coin
      efficiency **weighted by weekly production**, so cheap high-growth units rise
      (equals `value_per_gold × growth`).

    Pick a lens below to re-rank.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    METRICS = {
        "Value per gold  (ai_value ÷ gold)": "value_per_gold",
        "Weekly value  (ai_value × growth)": "weekly_value",
        "Value × growth per gold  (ai_value × growth ÷ gold)": "value_growth_per_gold",
    }
    metric = mo.ui.dropdown(
        options=METRICS,
        value="Value × growth per gold  (ai_value × growth ÷ gold)",
        label="Rank by",
    )
    metric
    return METRICS, metric


@app.cell(hide_code=True)
def _(METRICS, alt, metric, mo, scored):
    col = metric.value
    label = next(k for k, v in METRICS.items() if v == col)
    ranked = scored.sort(col, descending=True)
    top_name = ranked["name"][0]

    bars = (
        alt.Chart(ranked)
        .mark_bar()
        .encode(
            x=alt.X(col, title=label),
            y=alt.Y("name", sort="-x", title=None),
            color=alt.condition(
                alt.datum.name == top_name,
                alt.value("#2563eb"),
                alt.value("#cbd5e1"),
            ),
            tooltip=[
                "name", "town", "gold", "ai_value", "growth",
                "value_per_gold", "weekly_value", "value_growth_per_gold",
            ],
        )
        .properties(height=360, title=f"lvl-1 base units ranked by {label}")
    )
    mo.ui.altair_chart(bars)
    return


@app.cell(hide_code=True)
def _(metric, mo, scored):
    mo.ui.table(
        scored.sort(metric.value, descending=True).select(
            "icon", "name", "town", "gold", "ai_value", "growth",
            "value_per_gold", "weekly_value", "value_growth_per_gold",
        ),
        format_mapping={
            "icon": lambda v: mo.image(
                v, width=48, style={"image-rendering": "pixelated"}
            )
        },
    )
    return


@app.cell(hide_code=True)
def _(level_sel, melee_only, mo, scored):
    _v = scored.sort("value_per_gold", descending=True).row(0, named=True)
    _w = scored.sort("weekly_value", descending=True).row(0, named=True)
    _g = scored.sort("value_growth_per_gold", descending=True).row(0, named=True)
    _tag = " · melee only" if melee_only.value else ""
    mo.md(
        f"""
        ### Verdict — level {level_sel.value}{_tag}  ({len(scored)} units)

        - 🥇 **Value per gold → {_v['name']} ({_v['town']})** — {_v['ai_value']} ÷ {_v['gold']} = **{_v['value_per_gold']}**.
        - 🏭 **Weekly throughput → {_w['name']} ({_w['town']})** — {_w['ai_value']} × {_w['growth']} = **{_w['weekly_value']:,}**/week.
        - ⚖️ **Value × growth per gold → {_g['name']} ({_g['town']})** — **{_g['value_growth_per_gold']}**.
        """
    )
    return


@app.cell(hide_code=True)
def _(math, units):
    # --- HoMM3 melee combat model (deterministic, average damage) ---
    # Stats come straight from `units`; keyed "Name (Town)" to disambiguate the
    # two Halflings. Damage uses the real H3 attack/defense modifier:
    #   attack > defense -> +5% per point (cap +300%)
    #   attack < defense -> -2.5% per point (cap -70%)
    # simulate() records a per-swing timeline so we can chart the brawl, and takes
    # `first` to force initiative: "faster" (by speed), "A", or "B".
    unit_stats = {r["name"]: r for r in units.to_dicts()}


    def avg_dmg(s):
        return (s["min_dmg"] + s["max_dmg"]) / 2


    def raw_damage(att, n, dfd):
        """Average damage n attacking creatures deal to a defender stack."""
        base = avg_dmg(att) * n
        diff = att["attack"] - dfd["defense"]
        if diff >= 0:
            mod = 1 + min(0.05 * diff, 3.0)
        else:
            mod = 1 - min(0.025 * (-diff), 0.7)
        return base * mod


    def apply_damage(stack, dmg):
        """Subtract dmg from a stack's HP pool; return creatures killed."""
        hp = stack["s"]["hp"]
        pool = (stack["count"] - 1) * hp + stack["tophp"] - dmg
        before = stack["count"]
        if pool <= 0:
            stack["count"], stack["tophp"] = 0, 0
        else:
            stack["count"] = math.ceil(pool / hp)
            stack["tophp"] = pool - (stack["count"] - 1) * hp
        return before - stack["count"]


    def simulate(a_key, a_n, d_key, d_n, first="faster", max_rounds=200):
        a, d = unit_stats[a_key], unit_stats[d_key]
        st = {
            "A": {"s": a, "count": a_n, "tophp": a["hp"], "key": a_key},
            "D": {"s": d, "count": d_n, "tophp": d["hp"], "key": d_key},
        }
        if first == "A":
            order = ["A", "D"]
        elif first == "B":
            order = ["D", "A"]
        else:  # faster stack acts first; tie -> A (the "attacker")
            order = sorted(["A", "D"], key=lambda k: st[k]["s"]["speed"], reverse=True)

        log = []
        # timeline: one row per state change, incl. the starting full stacks
        timeline = [{"step": 0, "round": 0, "actor": None, "event": "start",
                     "target": None, "dmg": 0, "killed": 0,
                     "A": st["A"]["count"], "D": st["D"]["count"]}]
        step, rnd = 0, 0

        def record(rnd, actor, event, target, dmg, killed):
            nonlocal step
            step += 1
            timeline.append({"step": step, "round": rnd, "actor": actor,
                             "event": event, "target": target,
                             "dmg": dmg, "killed": killed,
                             "A": st["A"]["count"], "D": st["D"]["count"]})

        for rnd in range(1, max_rounds + 1):
            for atk in order:
                dfk = "D" if atk == "A" else "A"
                if st[atk]["count"] == 0 or st[dfk]["count"] == 0:
                    continue
                dmg = raw_damage(st[atk]["s"], st[atk]["count"], st[dfk]["s"])
                killed = apply_damage(st[dfk], dmg)
                record(rnd, atk, "attack", dfk, dmg, killed)
                log.append(
                    f"R{rnd}: {st[atk]['s']['name']} hit {st[dfk]['s']['name']} "
                    f"for {dmg:.0f} -> {killed} killed ({st[dfk]['count']} left)"
                )
                if st[dfk]["count"] > 0:  # the struck stack retaliates once
                    rdmg = raw_damage(st[dfk]["s"], st[dfk]["count"], st[atk]["s"])
                    rkilled = apply_damage(st[atk], rdmg)
                    record(rnd, dfk, "retaliate", atk, rdmg, rkilled)
                    log.append(
                        f"     retal: {st[dfk]['s']['name']} for {rdmg:.0f} -> "
                        f"{rkilled} killed ({st[atk]['count']} left)"
                    )
            if st["A"]["count"] == 0 or st["D"]["count"] == 0:
                break
        winner = "A" if st["A"]["count"] > 0 else ("D" if st["D"]["count"] > 0 else "draw")
        return {"state": st, "log": log, "rounds": rnd,
                "winner": winner, "order": order, "timeline": timeline}

    return simulate, unit_stats


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## ⚔️ Simulate a fight — both orderings at once

    Set **one gold budget**; each side auto-buys as many as it can afford, so this
    is a pure bang-for-buck duel — gold is the only knob. Because who strikes first
    can decide a close fight, we **always run both orderings** (A-first *and* B-first)
    side by side rather than guessing initiative.

    Each chart traces **creatures alive after every swing**: each downward step is a
    hit landing, ▽ = an **attack**, ◆ = a **retaliation** (a struck stack hits back once).
    Try a mirror match like **Goblin vs Goblin** (the default): dead symmetric, so
    whoever swings first wins — the banner flags when initiative is decisive.

    *Model: average-damage H3 melee — real attack/defense modifier, per-stack HP pools.*
    """)
    return


@app.cell(hide_code=True)
def _(mo, unit_stats):
    keys = sorted(unit_stats, key=lambda k: (unit_stats[k]["town"], k))
    labels = {f'{k}  ·  {unit_stats[k]["town"]}': k for k in keys}
    _opts = list(labels)
    a_unit = mo.ui.dropdown(options=labels, value=_opts[0], label="Stack A")
    d_unit = mo.ui.dropdown(options=labels, value=_opts[0], label="Stack B")
    budget = mo.ui.number(
        start=50, stop=200000, step=50, value=1000, label="Gold budget (each side)"
    )

    mo.vstack([
        mo.hstack([a_unit, d_unit], justify="start", gap=2),
        budget,
    ])
    return a_unit, budget, d_unit


@app.cell(hide_code=True)
def _(a_unit, alt, budget, d_unit, mo, pl, simulate, unit_stats):
    ga = unit_stats[a_unit.value]["gold"]
    gd = unit_stats[d_unit.value]["gold"]
    a_name, d_name = a_unit.value, d_unit.value
    # side labels keep mirror matchups (e.g. Goblin vs Goblin) distinguishable
    a_lab, d_lab = f"{a_name} (A)", f"{d_name} (B)"
    # gold is the only knob: counts derive from the shared budget automatically
    an = max(1, budget.value // ga)
    dn = max(1, budget.value // gd)

    # Run BOTH initiative orderings, not a single "faster" guess.
    scenarios = [("A", f"{a_lab} strikes first"), ("B", f"{d_lab} strikes first")]
    count_rows, hit_rows, verdicts = [], [], []
    for first, sc in scenarios:
        r = simulate(a_name, an, d_name, dn, first=first)
        win = a_lab if r["winner"] == "A" else (d_lab if r["winner"] == "D" else "Draw")
        verdicts.append((sc, win, r["state"]["A"]["count"], r["state"]["D"]["count"], r["rounds"]))
        for e in r["timeline"]:
            count_rows.append({"scenario": sc, "swing": e["step"], "stack": a_lab, "alive": e["A"]})
            count_rows.append({"scenario": sc, "swing": e["step"], "stack": d_lab, "alive": e["D"]})
            if e["event"] in ("attack", "retaliate"):
                tgt = a_lab if e["target"] == "A" else d_lab
                act = a_lab if e["actor"] == "A" else d_lab
                hit_rows.append({
                    "scenario": sc, "swing": e["step"], "stack": tgt,
                    "alive": e[e["target"]], "event": e["event"], "by": act,
                    "dmg": round(e["dmg"]), "killed": e["killed"],
                })

    flips = verdicts[0][1] != verdicts[1][1]
    banner = (
        "⚠️ **Initiative decides this** — winner flips with who swings first."
        if flips else
        "✅ **Initiative-proof** — same winner either way."
    )
    summary = mo.md(
        f"{banner}  ·  **{budget.value:,}g** → {an}× {a_lab} vs {dn}× {d_lab}"
    )

    domain, rng = [a_lab, d_lab], ["#2563eb", "#dc2626"]
    counts_df, hits_df = pl.DataFrame(count_rows), pl.DataFrame(hit_rows)


    def scenario_chart(sc):
        cdf = counts_df.filter(pl.col("scenario") == sc)
        hdf = hits_df.filter(pl.col("scenario") == sc)
        lines = (
            alt.Chart(cdf)
            .mark_line(interpolate="step-after", point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("swing:Q", title="swing #", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("alive:Q", title="creatures alive"),
                color=alt.Color(
                    "stack:N",
                    scale=alt.Scale(domain=domain, range=rng),
                    legend=alt.Legend(title="stack (survivor line)"),
                ),
                tooltip=["swing", "stack", "alive"],
            )
        )
        marks = (
            alt.Chart(hdf)
            .mark_point(size=130, filled=True, opacity=0.95, stroke="white", strokeWidth=1)
            .encode(
                x="swing:Q", y="alive:Q",
                shape=alt.Shape(
                    "event:N",
                    scale=alt.Scale(domain=["attack", "retaliate"], range=["triangle-down", "diamond"]),
                    title="hit type",
                ),
                color=alt.Color("stack:N", scale=alt.Scale(domain=domain, range=rng), legend=None),
                tooltip=["by", "event", "dmg", "killed"],
            )
        )
        return (lines + marks).properties(width=300, height=300, title=sc)


    battle_chart = alt.hconcat(
        *[scenario_chart(sc) for _, sc in scenarios]
    ).resolve_scale(color="shared", shape="shared")

    mo.vstack([summary, battle_chart])
    return


@app.cell(hide_code=True)
def _(
    budget,
    div,
    img,
    mo,
    scored,
    simulate,
    span,
    table,
    td,
    th,
    tr,
    unit_stats,
):
    # Round-robin at the same gold budget, ROW UNIT ATTACKS (strikes first).
    # This is deliberately asymmetric: cell (i, j) is the fight where i swings first,
    # which is a different fight from (j, i) where j swings first. Number = the
    # winner's surviving % of its own army (how decisive); green = the attacking ROW
    # unit wins, red = the attacker loses anyway. Compare a cell with its mirror:
    # both green => whoever attacks wins (initiative decides). Uses the `budget` knob.
    mbud = budget.value
    _names0 = list(unit_stats)
    icons = dict(zip(scored["name"].to_list(), scored["icon"].to_list()))


    def _icon(n, px=24):
        return img(src=icons[n], width=str(px), height=str(px), title=n,
                   style="image-rendering:pixelated;display:block")


    def _cnt(k):
        return max(1, mbud // unit_stats[k]["gold"])


    def _fight(i, j):
        # ROW i strikes first; returns (row_won, winner_surviving_fraction)
        ni, nj = _cnt(i), _cnt(j)
        r = simulate(i, ni, j, nj, first="A")
        if r["winner"] == "A":
            return True, r["state"]["A"]["count"] / ni
        if r["winner"] == "D":
            return False, r["state"]["D"]["count"] / nj
        return None, 0.0


    grid = {i: {j: _fight(i, j) for j in _names0} for i in _names0}
    # "attack wins": how many opponents this unit beats when it strikes first
    awins = {i: sum(1 for j in _names0 if i != j and grid[i][j][0] is True) for i in _names0}
    order = sorted(_names0, key=lambda k: (-awins[k], unit_stats[k]["gold"]))

    CELL = "text-align:center;width:34px;height:30px;font-size:10px;font-weight:700"


    def _cell(i, j):
        won, m = grid[i][j]
        if won is None:
            return td("–", title=f"{i} vs {j}: mutual annihilation",
                      style=f"background:#e5e7eb;color:#374151;{CELL}")
        alpha = 0.20 + 0.80 * m
        rgb = "22,163,74" if won else "220,38,38"
        fg = "white" if alpha > 0.55 else ("#14532d" if won else "#7f1d1d")
        who = "attacker wins" if won else "attacker LOSES"
        tip = f"{i} attacks {j}: {who}, winner keeps {m * 100:.0f}%"
        return td(f"{m * 100:.0f}%", title=tip,
                  style=f"background:rgba({rgb},{alpha:.2f});color:{fg};{CELL}")


    head = tr(
        th(div("attacker ↓", style="font-size:10px;color:#6b7280;transform:rotate(-90deg)"),
           style="padding:2px"),
        *[th(_icon(n, 26), style="padding:2px;vertical-align:bottom") for n in order],
        th("W", title="opponents beaten when attacking",
           style="font-size:10px;padding:2px 6px;color:#6b7280"),
    )
    rows_html = [
        tr(
            th(div(_icon(i, 22), span(i, style="font-size:11px"),
                   style="display:flex;align-items:center;gap:6px;justify-content:flex-end;white-space:nowrap"),
               style="padding:2px 8px"),
            *[_cell(i, j) for j in order],
            td(str(awins[i]),
               style="text-align:center;font-weight:700;font-size:11px;background:#f3f4f6;width:26px"),
        )
        for i in order
    ]
    grid_tbl = table(head, *rows_html, style="border-collapse:collapse;font-family:ui-sans-serif,system-ui")

    legend = div(
        "Row unit ATTACKS (strikes first). Number = winner's surviving %.  ",
        span("■", style="color:#16a34a;font-size:14px"), " attacker wins   ",
        span("■", style="color:#dc2626;font-size:14px"), " attacker loses anyway   ",
        "· mirror cell also green ⇒ initiative decides.",
        style="font-size:12px;margin-top:8px;color:#374151",
    )

    # initiative-decided pairs: i beats j attacking AND j beats i attacking
    initiative = [
        (order[a], order[b])
        for a in range(len(order)) for b in range(a + 1, len(order))
        if grid[order[a]][order[b]][0] is True and grid[order[b]][order[a]][0] is True
    ]

    mo.vstack([
        mo.md(
            f"## Who beats whom when they **attack**, at **{mbud:,}g** each\n"
            f"Asymmetric on purpose — striking first is an advantage. **{len(initiative)}** "
            f"matchup(s) are decided purely by who attacks (both directions green)."
        ),
        mo.Html(str(grid_tbl)),
        mo.Html(str(legend)),
    ])
    return


if __name__ == "__main__":
    app.run()

# /// script
# dependencies = [
#     "marimo",
#     "moutils[db]",
#     "polars",
#     "altair",
#     "mohtml==0.1.11",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", sql_output="polars")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # HoMM3 lvl-1 units: best bang for your buck 🪙

    Which **tier-1 base unit** gives the most value per gold? We pull every
    level-1 base unit from the [`h3_units`](https://datasette.exe.xyz/h3_units/units)
    datasette and compare a few value-per-gold lenses side by side.

    *Scope: the lvl-1 base units of the 9 original towns plus HotA's Cove (Nymph)
    and Bulwark (Kobold) and the neutral Halfling — the Peasant and the duplicate
    Factory Halfling are excluded.*
    """)
    return


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
def _(datasette, mo):
    units = mo.sql(
        f"""
        SELECT name, town, level, gold, attack, defense, min_dmg, max_dmg,
               hp, speed, growth, ai_value, icon
        FROM units
        WHERE level = '1'
          AND name NOT IN ('Peasant', 'Halfling (Factory)')
        ORDER BY gold
        """,
        engine=datasette
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
def _(mo):
    mo.md("""
    ### Verdict

    - 🥇 **Value per gold → Pixie (Conflux).** `55 ÷ 25 = 2.20` — best raw efficiency per coin.
    - 🏭 **Weekly throughput → Centaur (Rampart).** `100 × 14 = 1400`/week — most absolute power one dwelling fields (but ~3× the price each).
    - ⚖️ **Value × growth per gold (your metric) → Pixie again**, `55 × 20 ÷ 25 = 44.0`, with
      **Halfling** 2nd (`75 × 15 ÷ 40 = 28.1`) and **Nymph** 3rd (26.1).

    **Bottom line:** across every ai_value-based lens the **Pixie** is the best bang for
    your buck — cheap, efficient, *and* high-growth. The only unit that beats it on
    anything is the Centaur, and only on absolute weekly power because it costs far more per head.
    """)
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
    # label shows the town, value is the plain unit name (the unit_stats key)
    labels = {f'{k}  ·  {unit_stats[k]["town"]}': k for k in keys}
    a_unit = mo.ui.dropdown(options=labels, value="Goblin  ·  Stronghold", label="Stack A")
    d_unit = mo.ui.dropdown(options=labels, value="Goblin  ·  Stronghold", label="Stack B")
    budget = mo.ui.number(
        start=50, stop=100000, step=50, value=1000, label="Gold budget (each side)"
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
    # Round-robin at the same gold budget. Each cell shows the WINNER'S surviving %
    # (how decisive), averaged over both initiative orderings. Green = the ROW unit
    # wins both orderings, red = it loses both, gray = depends on who swings first.
    # Colour shade scales with the margin. Uses the same `budget` knob above.
    mbud = budget.value
    _names0 = list(unit_stats)
    icons = dict(zip(scored["name"].to_list(), scored["icon"].to_list()))


    def _icon(n, px=24):
        return img(src=icons[n], width=str(px), height=str(px), title=n,
                   style="image-rendering:pixelated;display:block")


    def _cnt(k):
        return max(1, mbud // unit_stats[k]["gold"])


    def _battle(i, j):
        ni, nj = _cnt(i), _cnt(j)
        out = []
        for first in ("A", "B"):
            r = simulate(i, ni, j, nj, first=first)
            w = r["winner"]
            if w == "A":
                frac = r["state"]["A"]["count"] / ni
            elif w == "D":
                frac = r["state"]["D"]["count"] / nj
            else:
                frac = 0.0
            out.append((w, frac))
        return out


    def _classify(i, j):
        b = _battle(i, j)
        winners = {w for w, _ in b}
        margin = sum(f for _, f in b) / len(b)  # avg winner surviving fraction
        if winners == {"A"}:
            return "W", margin
        if winners == {"D"}:
            return "L", margin
        return "~", None


    grid = {i: {j: _classify(i, j) for j in _names0} for i in _names0}
    mwins = {i: sum(1 for j in _names0 if i != j and grid[i][j][0] == "W") for i in _names0}
    order = sorted(_names0, key=lambda k: (-mwins[k], unit_stats[k]["gold"]))

    cycles = [
        (a, b, c)
        for a in order for b in order for c in order
        if len({a, b, c}) == 3
        and grid[a][b][0] == "W" and grid[b][c][0] == "W" and grid[c][a][0] == "W"
    ]

    CELL = "text-align:center;width:34px;height:30px;font-size:10px;font-weight:700"


    def _cell(i, j):
        if i == j:
            return td("·", style=f"background:#374151;color:#9ca3af;{CELL}")
        o, m = grid[i][j]
        if o == "~":
            return td("~", title=f"{i} vs {j}: depends on initiative",
                      style=f"background:#e5e7eb;color:#374151;{CELL}")
        alpha = 0.20 + 0.80 * m
        rgb = "22,163,74" if o == "W" else "220,38,38"
        fg = "white" if alpha > 0.55 else ("#14532d" if o == "W" else "#7f1d1d")
        tip = f"{i} vs {j}: winner keeps {m * 100:.0f}% of its army"
        return td(f"{m * 100:.0f}%", title=tip,
                  style=f"background:rgba({rgb},{alpha:.2f});color:{fg};{CELL}")


    head = tr(
        th("", style="padding:2px 6px"),
        *[th(_icon(n, 26), style="padding:2px;vertical-align:bottom") for n in order],
        th("W", style="font-size:10px;padding:2px 6px;color:#6b7280"),
    )
    rows_html = [
        tr(
            th(div(_icon(i, 22), span(i, style="font-size:11px"),
                   style="display:flex;align-items:center;gap:6px;justify-content:flex-end;white-space:nowrap"),
               style="padding:2px 8px"),
            *[_cell(i, j) for j in order],
            td(str(mwins[i]),
               style="text-align:center;font-weight:700;font-size:11px;background:#f3f4f6;width:26px"),
        )
        for i in order
    ]
    grid_tbl = table(head, *rows_html, style="border-collapse:collapse;font-family:ui-sans-serif,system-ui")

    legend = div(
        "Number = winner's surviving % (bigger = more decisive).  ",
        span("■", style="color:#16a34a;font-size:14px"), " row wins both orderings   ",
        span("■", style="color:#dc2626;font-size:14px"), " row loses both   ",
        span("■", style="color:#9ca3af;font-size:14px"), " depends on who swings first",
        style="font-size:12px;margin-top:8px;color:#374151",
    )

    verdict = (
        f"\U0001faa8\U0001f4c4✂️ **{len(cycles)} rock-paper-scissors cycle(s)** — non-transitive!"
        if cycles else
        f"\U0001f4ca **No 3-cycle — it's a pecking order.** Top dog: **{order[0]}** "
        f"(beats {mwins[order[0]]}/{len(order) - 1}). Only the gray cells break the ranking."
    )

    mo.vstack([
        mo.md(f"## Who beats whom — and by how much — at **{mbud:,}g** each\n{verdict}"),
        mo.Html(str(grid_tbl)),
        mo.Html(str(legend)),
    ])
    return


if __name__ == "__main__":
    app.run()

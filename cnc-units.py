# /// script
# dependencies = [
#     "marimo",
#     "moutils[db]",
#     "polars",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", sql_output="polars")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from moutils.db.datasette import DatasetteConnection

    return DatasetteConnection, mo, pl


@app.cell
def _(DatasetteConnection):
    datasette = DatasetteConnection("https://datasette.exe.xyz", "cnc_units")
    return (datasette,)


@app.cell
def _(datasette, mo):
    units = mo.sql(
        f"""
        SELECT *
        FROM units
        """,
        engine=datasette
    )
    return (units,)


@app.cell
def _(pl, units):
    # datasette returns blobs as {"$base64": true, "encoded": "..."} structs
    with_icons = units.with_columns(
        icon=pl.lit("data:image/png;base64,") + pl.col("icon").struct.field("encoded"),
        # "gdi|nod" means buildable by both, so treat faction as a list
        factions=pl.col("faction").str.split("|"),
    )
    return (with_icons,)


@app.cell
def _(mo, with_icons):
    games = sorted(with_icons["mod"].unique())
    game = mo.ui.multiselect(options=games, value=games, label="game")
    cost = mo.ui.range_slider(
        start=with_icons["cost"].min(),
        stop=with_icons["cost"].max(),
        step=50,
        value=[with_icons["cost"].min(), with_icons["cost"].max()],
        label="cost",
        show_value=True,
    )
    return cost, game


@app.cell
def _(game, pl, with_icons):
    # each selector only offers values that still exist upstream of it,
    # and starts with all of them selected
    by_game = with_icons.filter(pl.col("mod").is_in(game.value))
    return (by_game,)


@app.cell
def _(by_game, mo):
    factions = sorted({f for row in by_game["factions"] for f in row if f != "any"})
    faction = mo.ui.multiselect(options=factions, value=factions, label="faction")
    return (faction,)


@app.cell
def _(by_game, faction, mo, pl):
    # "any" units are buildable by everyone, so they ride along with whatever
    # real factions are selected; "gdi|nod" units match either one
    allowed = list(faction.value) + ["any"] if faction.value else []
    by_faction = by_game.filter(
        pl.col("factions").list.eval(pl.element().is_in(allowed)).list.any()
    )

    categories = sorted(by_faction["category"].unique())
    category = mo.ui.multiselect(options=categories, value=categories, label="category")
    return by_faction, category


@app.cell
def _(category, cost, faction, game, mo):
    mo.hstack([game, faction, category, cost], justify="start", gap=2)
    return


@app.cell
def _(by_faction, category, cost, mo, pl, with_icons):
    selected = by_faction.filter(
        pl.col("cost").is_between(*cost.value)
        & pl.col("category").is_in(category.value)
    ).select("icon", "name", "mod")

    mo.vstack([
        mo.md(f"**{len(selected)}** of {len(with_icons)} units"),
        mo.ui.table(
            selected,
            # sprites are 64x48, so scale up without smoothing
            format_mapping={
                "icon": lambda v: mo.image(
                    v, width=128, style={"image-rendering": "pixelated"}
                )
            },
            column_widths={"icon": 160},
        ),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

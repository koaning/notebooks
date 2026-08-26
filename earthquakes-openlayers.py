# /// script
# dependencies = [
#     "marimo",
#     "moutils[db]",
#     "polars",
#     "openlayers==0.1.6",
#     "wigglystuff==0.5.27",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", sql_output="polars")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 🌍 Earthquakes on OpenLayers

    Points pulled live from the [`earthquakes`](https://datasette.exe.xyz/earthquakes/earthquakes)
    Datasette and rendered on a real map with the `openlayers` package. Hover a point
    for its place, magnitude and time.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import polars as pl
    import openlayers as ol
    import math

    from moutils.db.datasette import DatasetteConnection
    from wigglystuff import Knob

    return DatasetteConnection, Knob, math, mo, ol


@app.cell(hide_code=True)
def _(DatasetteConnection):
    datasette = DatasetteConnection("https://datasette.exe.xyz", "earthquakes")
    return (datasette,)


@app.cell(hide_code=True)
def _(datasette, mo):
    regions_df = mo.sql(
        f"""
        SELECT DISTINCT region FROM earthquakes
        WHERE region IS NOT NULL AND region != ''
        ORDER BY region
        """,
        output=False,
        engine=datasette
    )
    return (regions_df,)


@app.cell(hide_code=True)
def _(Knob, mo, regions_df):
    min_mag = mo.ui.slider(
        start=5.0, stop=8.8, step=0.1, value=5.0, label="Min magn", show_value=True
    )
    max_mag = mo.ui.slider(
        start=5.0, stop=8.8, step=0.1, value=8.8, label="Max magn", show_value=True
    )
    region = mo.ui.dropdown(
        options=["All"] + regions_df["region"].to_list(),
        value="All",
        label="Region",
        searchable=True,
    )
    zoom_knob = mo.ui.anywidget(
        Knob(value=2.0, min_value=1.0, max_value=12.0, step=0.5, label="Zoom", midi=True)
    )
    mo.hstack(
        [min_mag, max_mag, region, zoom_knob], justify="start", gap=2, align="center"
    )
    return max_mag, min_mag, region, zoom_knob


@app.cell(hide_code=True)
def _(m):
    m
    return


@app.cell(hide_code=True)
def _(datasette, max_mag, min_mag, mo, region):
    selected_region = region.value
    region_clause = (
        ""
        if selected_region == "All"
        else f"AND region = '{selected_region.replace(chr(39), chr(39) * 2)}'"
    )
    df = mo.sql(
        f"""
        SELECT latitude, longitude, mag, place, time
        FROM earthquakes
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
          AND mag BETWEEN {min_mag.value} AND {max_mag.value}
          {region_clause}
        ORDER BY mag DESC
        LIMIT 500
        """,
        output=False,
        engine=datasette,
    )
    return (df,)


@app.cell(hide_code=True)
def _(df):
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row["longitude"], row["latitude"]],
                },
                "properties": {
                    "mag": row["mag"],
                    "place": row["place"],
                    "time": str(row["time"]),
                },
            }
            for row in df.iter_rows(named=True)
        ],
    }
    return (fc,)


@app.cell(hide_code=True)
def _(ol):
    vector = ol.VectorLayer(
        source=ol.VectorSource(geojson={"type": "FeatureCollection", "features": []})
    )
    m = ol.MapWidget(
        layers=[ol.BasemapLayer(), vector],
        view=ol.View(center=(0.0, 0.0), zoom=1.0),
    )
    m.add_tooltip()
    return m, vector


@app.cell(hide_code=True)
def _(df, fc, m, math, ol, vector):
    m.set_source(vector.id, ol.VectorSource(geojson=fc))
    lons = df["longitude"].to_list()
    lats = df["latitude"].to_list()
    if lons:
        m.set_center((min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2)
        span = max(max(lons) - min(lons), max(lats) - min(lats), 0.5)
        m.set_zoom(max(1.0, min(12.0, math.log2(360 / span) - 0.5)))
    return


@app.cell(hide_code=True)
def _(m, zoom_knob):
    m.set_zoom(zoom_knob.value["value"])
    return


if __name__ == "__main__":
    app.run()

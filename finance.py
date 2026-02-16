# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==6.0.0",
#     "marimo",
#     "polars==1.37.1",
#     "justetf-scraping @ git+https://github.com/druzsan/justetf-scraping.git",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(
    width="medium",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from pathlib import Path

    return Path, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # EU ETF Portfolio Tracker

    This notebook tracks the value of your EU ETF portfolio over time. It reads transactions from `transactions.csv` and fetches daily NAV data via [justETF](https://www.justetf.com/).
    """)
    return


@app.cell
def _(pl):
    raw = pl.read_csv("transactions.csv").rename({
        "Datum": "Date",
        "Aantal": "Quantity",
        "Totaal EUR": "Investment",
    })

    isin_names = dict(
        raw.select(["ISIN", "Product"]).unique(subset=["ISIN"]).iter_rows()
    )

    investments = (
        raw.select(["Date", "ISIN", "Quantity", "Investment"])
        .with_columns(
            Date=pl.col("Date").str.strptime(pl.Date, "%d-%m-%Y").dt.strftime("%Y-%m-%d"),
            Investment=pl.col("Investment").str.replace_all(",", ".").cast(pl.Float64).abs(),
            Quantity=pl.col("Quantity").cast(pl.Int64),
        )
    )
    investments
    return investments, isin_names


@app.cell
def _(Path, investments):
    import justetf_scraping

    parent_folder = Path("etf-data")
    parent_folder.mkdir(exist_ok=True)

    for _isin in investments["ISIN"].unique():
        csv_path = parent_folder / f"{_isin}.csv"
        if not csv_path.exists():
            df = justetf_scraping.load_chart(_isin)
            df = df.reset_index()
            date_col = df.columns[0]
            df = df.rename(columns={date_col: "Date"})
            df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
            df["ISIN"] = _isin
            df[["Date", "ISIN", "quote"]].to_csv(csv_path, index=False)
    return (parent_folder,)


@app.cell(hide_code=True)
def _(investments, parent_folder, pl):
    import altair as alt


    def clean_data(dataf, investment_df):
        return (
            dataf.select(["Date", "ISIN", "quote"])
            .join(
                investment_df.select(["Date", "ISIN", "Quantity", "Investment"]),
                on=["Date", "ISIN"],
                how="left",
            )
            .with_columns(
                pl.col("Investment").fill_null(0),
                pl.col("Quantity").fill_null(0),
            )
            .sort("ISIN", "Date")
        )


    def calculate_portfolio_value(dataf):
        return (
            dataf.with_columns(
                CumInvestment=pl.col("Investment").cum_sum().over("ISIN"),
                TotalShares=pl.col("Quantity").cum_sum().over("ISIN"),
            )
            .filter(pl.col("CumInvestment") > 0)
            .with_columns(
                PortfolioValue=pl.col("TotalShares") * pl.col("quote"),
                PnL=(pl.col("TotalShares") * pl.col("quote")) - pl.col("CumInvestment"),
            )
        )


    def calculate_performance(dataf):
        return (
            dataf.group_by("Date")
            .agg(
                [
                    pl.sum("CumInvestment").alias("TotalInvested"),
                    pl.sum("PortfolioValue").alias("TotalValue"),
                    pl.sum("PnL").alias("TotalPnL"),
                ]
            )
            .with_columns(
                Date=pl.col("Date").str.to_date(),
                ReturnPct=((pl.col("TotalValue") / pl.col("TotalInvested")) - 1) * 100,
            )
            .sort("Date")
        )


    def make_chart(dataf):
        portfolio_chart = (
            alt.Chart(dataf)
            .mark_area(line=True, opacity=0.3)
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("TotalValue:Q", title="Value (€)"),
            )
        )

        invested_line = (
            alt.Chart(dataf)
            .mark_line(strokeDash=[5, 5], color="black", strokeWidth=2)
            .encode(x="Date:T", y="TotalInvested:Q")
        )

        return portfolio_chart + invested_line


    per_etf = (
        pl.read_csv(f"{parent_folder}/*", glob=True)
        .pipe(clean_data, investment_df=investments)
        .pipe(calculate_portfolio_value)
    )

    cached = per_etf.pipe(calculate_performance)
    return cached, make_chart, per_etf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Investment value over time
    """)
    return


@app.cell
def _(cached):
    cached
    return


@app.cell
def _(cached, make_chart):
    cached.pipe(make_chart)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Returns over time
    """)
    return


@app.cell
def _(cached):
    cached.plot.line("Date", "ReturnPct")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Per ETF
    """)
    return


@app.cell
def _(isin_names, mo, per_etf):
    isin_options = {
        f"{isin_names.get(isin, isin)} ({isin})": isin
        for isin in sorted(per_etf["ISIN"].unique().to_list())
    }
    etf_dropdown = mo.ui.dropdown(
        isin_options,
        value=list(isin_options.keys())[0],
        label="Select ETF",
    )
    etf_dropdown
    return (etf_dropdown,)


@app.cell
def _(etf_dropdown, make_chart, per_etf, pl):
    selected = etf_dropdown.value
    single_etf = (
        per_etf.filter(pl.col("ISIN") == selected)
        .with_columns(Date=pl.col("Date").str.to_date())
        .rename({"CumInvestment": "TotalInvested", "PortfolioValue": "TotalValue"})
        .sort("Date")
    )
    single_etf.pipe(make_chart)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Data for download
    """)
    return


@app.cell
def _(cached):
    cached
    return


if __name__ == "__main__":
    app.run()

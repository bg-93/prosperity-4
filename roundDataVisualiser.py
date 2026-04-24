from __future__ import annotations

import textwrap
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "ROUND_1"
DEFAULT_SYMBOL = "INTARIAN_PEPPER_ROOT"
DISPLAY_NAME_BY_SYMBOL = {
    "INTARIAN_PEPPER_ROOT": "Intarian Pepper Root",
    "ASH_COATED_OSMIUM": "Ash Coated Osmium",
}
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
}


st.set_page_config(
    page_title="INTARIAN_PEPPER_ROOT CSV Dashboard",
    layout="wide",
)


def wrap_label(label: str, width: int = 16) -> str:
    return "<br>".join(textwrap.wrap(label, width=width)) or label


def pretty_symbol(symbol: str) -> str:
    return DISPLAY_NAME_BY_SYMBOL.get(symbol, symbol.replace("_", " ").title())


def read_uploaded_csv(file: BytesIO) -> pd.DataFrame:
    return pd.read_csv(file, sep=";")


@st.cache_data(show_spinner=False)
def load_default_csvs(kind: str) -> tuple[pd.DataFrame, ...]:
    pattern = f"{kind}_round_1_day_*.csv"
    return tuple(pd.read_csv(path, sep=";") for path in sorted(DEFAULT_DATA_DIR.glob(pattern)))


def prepare_price_data(frames: list[pd.DataFrame], symbol: str) -> pd.DataFrame:
    filtered_frames: list[pd.DataFrame] = []
    for index, frame in enumerate(frames):
        df = frame.copy()
        df = df[df["product"] == symbol].copy()
        if df.empty:
            continue

        if "day" not in df.columns:
            raise ValueError("Prices CSV must contain a `day` column.")

        for column in df.columns:
            if column != "product":
                df[column] = pd.to_numeric(df[column], errors="coerce")
        empty_book = df["bid_price_1"].isna() & df["ask_price_1"].isna() & (df["mid_price"] == 0)
        df.loc[empty_book, "mid_price"] = np.nan
        df["source_file"] = f"prices_upload_{index + 1}.csv"
        filtered_frames.append(df)

    if not filtered_frames:
        raise ValueError(f"No price rows found for symbol `{symbol}`.")

    prices = pd.concat(filtered_frames, ignore_index=True).sort_values(["day", "timestamp"]).reset_index(drop=True)
    prices["global_ts"] = (prices["day"] - prices["day"].min()) * 1_000_000 + prices["timestamp"]
    prices["spread_1"] = prices["ask_price_1"] - prices["bid_price_1"]
    prices["mid_from_quotes"] = (prices["bid_price_1"] + prices["ask_price_1"]) / 2
    total_top_level = prices["bid_volume_1"] + prices["ask_volume_1"]
    prices["microprice_l1"] = (
        prices["ask_price_1"] * prices["bid_volume_1"] + prices["bid_price_1"] * prices["ask_volume_1"]
    ) / total_top_level.replace(0, np.nan)
    prices["book_imbalance"] = (
        (prices["bid_volume_1"] - prices["ask_volume_1"]) / total_top_level.replace(0, np.nan)
    )
    prices["mid_change_1"] = prices["mid_price"].diff()
    prices["mid_change_10"] = prices["mid_price"].diff(10)
    prices["rolling_mid_200"] = prices["mid_price"].rolling(200, min_periods=20).mean()
    return prices


def prepare_trade_data(frames: list[pd.DataFrame], symbol: str, min_day: int) -> pd.DataFrame:
    filtered_frames: list[pd.DataFrame] = []
    inferred_day = min_day

    for index, frame in enumerate(frames):
        df = frame.copy()
        df = df[df["symbol"] == symbol].copy()
        if df.empty:
            continue

        if "day" not in df.columns:
            df["day"] = inferred_day
            inferred_day += 1

        for column in ["timestamp", "price", "quantity", "day"]:
            if column not in df.columns:
                raise ValueError(f"Trades CSV must contain a `{column}` column.")
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df["source_file"] = f"trades_upload_{index + 1}.csv"
        filtered_frames.append(df)

    if not filtered_frames:
        raise ValueError(f"No trade rows found for symbol `{symbol}`.")

    trades = pd.concat(filtered_frames, ignore_index=True).sort_values(["day", "timestamp"]).reset_index(drop=True)
    trades["global_ts"] = (trades["day"] - min_day) * 1_000_000 + trades["timestamp"]
    trades["notional"] = trades["price"] * trades["quantity"]
    return trades


def build_trade_context(prices: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    trade_context = pd.merge_asof(
        trades.sort_values("global_ts"),
        prices[
            [
                "global_ts",
                "day",
                "timestamp",
                "bid_price_1",
                "bid_volume_1",
                "ask_price_1",
                "ask_volume_1",
                "mid_price",
                "mid_from_quotes",
                "microprice_l1",
                "spread_1",
                "book_imbalance",
            ]
        ].sort_values("global_ts"),
        on="global_ts",
        direction="backward",
        suffixes=("_trade", "_quote"),
    )
    trade_context["trade_vs_mid"] = trade_context["price"] - trade_context["mid_price"]
    trade_context["trade_vs_microprice"] = trade_context["price"] - trade_context["microprice_l1"]
    return trade_context


def build_overview(prices: pd.DataFrame, trades: pd.DataFrame, trade_context: pd.DataFrame) -> pd.DataFrame:
    vwap = trades["notional"].sum() / trades["quantity"].sum() if trades["quantity"].sum() else np.nan
    overview = pd.DataFrame(
        {
            "metric": [
                "Quote rows",
                "Trade rows",
                "Invalid placeholder mids removed",
                "Best bid availability",
                "Best ask availability",
                "Median quoted spread",
                "Mean quoted spread",
                "Median trade price",
                "VWAP",
                "Mean trade quantity",
                "Mean book imbalance",
                "Trade vs mid (mean)",
                "Trade vs microprice (mean)",
                "1-step up moves share",
                "10-step up moves share",
                "Mid autocorr lag 1",
                "Mid autocorr lag 10",
            ],
            "value": [
                len(prices),
                len(trades),
                prices["mid_price"].isna().sum(),
                prices["bid_price_1"].notna().mean(),
                prices["ask_price_1"].notna().mean(),
                prices["spread_1"].median(),
                prices["spread_1"].mean(),
                trades["price"].median(),
                vwap,
                trades["quantity"].mean(),
                prices["book_imbalance"].mean(),
                trade_context["trade_vs_mid"].mean(),
                trade_context["trade_vs_microprice"].mean(),
                (prices["mid_change_1"] > 0).mean(),
                (prices["mid_change_10"] > 0).mean(),
                prices["mid_price"].autocorr(1),
                prices["mid_price"].autocorr(10),
            ],
        }
    )
    return overview


def build_day_summary(prices: pd.DataFrame, trade_context: pd.DataFrame) -> pd.DataFrame:
    return prices.groupby("day").agg(
        quote_rows=("timestamp", "count"),
        bid_available=("bid_price_1", lambda s: s.notna().mean()),
        ask_available=("ask_price_1", lambda s: s.notna().mean()),
        spread_mean=("spread_1", "mean"),
        spread_median=("spread_1", "median"),
        mid_mean=("mid_price", "mean"),
        mid_first=("mid_price", "first"),
        mid_last=("mid_price", "last"),
        imbalance_mean=("book_imbalance", "mean"),
    ).join(
        trade_context.groupby("day_trade").agg(
            trade_rows=("price", "count"),
            trade_price_mean=("price", "mean"),
            trade_price_median=("price", "median"),
            trade_qty_mean=("quantity", "mean"),
            trade_vs_mid_mean=("trade_vs_mid", "mean"),
        )
    )


def build_bias_lines(symbol: str, prices: pd.DataFrame, trade_context: pd.DataFrame) -> list[str]:
    median_spread = prices["spread_1"].median()
    mean_imbalance = prices["book_imbalance"].mean()
    trade_bias_mid = trade_context["trade_vs_mid"].mean()
    trade_bias_micro = trade_context["trade_vs_microprice"].mean()
    up_move_1 = (prices["mid_change_1"] > 0).mean()
    up_move_10 = (prices["mid_change_10"] > 0).mean()
    autocorr_10 = prices["mid_price"].autocorr(10)
    day_zero = sorted(prices["day"].dropna().unique())
    reference_day = 0 if 0 in day_zero else day_zero[0]
    day_bias_mid = trade_context.loc[trade_context["day_trade"] == reference_day, "trade_vs_mid"].mean()
    valid_mid = prices["mid_price"].dropna()
    start_mid = valid_mid.iloc[0]
    end_mid = valid_mid.iloc[-1]
    net_move = end_mid - start_mid
    direction = "upward" if net_move > 0 else "downward" if net_move < 0 else "flat"
    persistence_text = "continuation" if abs(up_move_10 - 0.5) > 0.08 else "mean reversion / chop"
    trade_print_text = "below" if trade_bias_mid < 0 else "above" if trade_bias_mid > 0 else "at"
    inventory_skew = "buy-leaning" if net_move > 0 else "sell-leaning" if net_move < 0 else "balanced"
    product_name = pretty_symbol(symbol)

    return [
        f"{product_name} market-making bias overview.",
        f"The market trends {direction}: mid moves from {start_mid:.1f} to {end_mid:.1f} across the sample.",
        f"Median level-1 spread is {median_spread:.1f} ticks and mean imbalance is {mean_imbalance:.3f}.",
        (
            f"Trend persistence is visible: {up_move_1:.1%} of 1-step moves and {up_move_10:.1%} of 10-step moves "
            f"are upward, with lag-10 autocorrelation {autocorr_10:.3f}, which looks more like {persistence_text}."
        ),
        (
            f"Trades print {trade_print_text} mid on average: {trade_bias_mid:.2f} ticks vs mid and "
            f"{trade_bias_micro:.2f} ticks vs microprice."
        ),
        f"On day {reference_day}, trades print {day_bias_mid:.2f} ticks vs mid on average.",
        f"Practical takeaway: keep a {inventory_skew} directional bias when your signal agrees, but let inventory control cap exposure.",
        "Execution takeaway: let trade-location information stop you from paying through the spread too aggressively.",
    ]


def base_layout(title: str, height: int) -> dict[str, Any]:
    return {
        "title": {"text": title, "x": 0.01, "y": 0.98},
        "template": "plotly_white",
        "height": height,
        "margin": {"l": 70, "r": 30, "t": 120, "b": 70},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    }


def build_dashboard_figure(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    trade_context: pd.DataFrame,
    symbol: str,
) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=2,
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
        subplot_titles=[
            "Best Quotes and Trades",
            "Trend in Mid Price",
            "Quoted Spread Over Time",
            "Level-1 Book Imbalance",
            "Where Trades Print vs Mid",
            "10-Tick Mid Price Changes",
        ],
    )

    fig.add_trace(
        go.Scatter(x=prices["global_ts"], y=prices["bid_price_1"], mode="lines", name="Best bid", line={"width": 1.2}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=prices["global_ts"], y=prices["ask_price_1"], mode="lines", name="Best ask", line={"width": 1.2}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=trades["global_ts"],
            y=trades["price"],
            mode="markers",
            name="Trade price",
            marker={"size": 6, "color": "black", "opacity": 0.35},
            customdata=np.stack([trades["day"], trades["timestamp"], trades["quantity"]], axis=-1),
            hovertemplate=(
                "Day: %{customdata[0]:.0f}<br>"
                "Timestamp: %{customdata[1]:.0f}<br>"
                "Trade price: %{y:.2f}<br>"
                "Quantity: %{customdata[2]:.0f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=prices["global_ts"], y=prices["mid_price"], mode="lines", name="Mid price", line={"width": 1.6}),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=prices["global_ts"],
            y=prices["rolling_mid_200"],
            mode="lines",
            name="Rolling 200-tick mean",
            line={"width": 2.2},
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=prices["global_ts"],
            y=prices["spread_1"],
            mode="lines",
            name="Spread",
            line={"width": 1.2, "color": "#d62728"},
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=prices["spread_1"].median(),
        line_width=1,
        line_dash="dash",
        line_color="black",
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=prices["book_imbalance"].dropna(),
            nbinsx=40,
            name="Book imbalance",
            marker={"color": "#1f77b4"},
            opacity=0.8,
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_vline(
        x=prices["book_imbalance"].mean(),
        line_width=1,
        line_dash="dash",
        line_color="black",
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Histogram(
            x=trade_context["trade_vs_mid"].dropna(),
            nbinsx=40,
            name="Trade vs mid",
            marker={"color": "#2ca02c"},
            opacity=0.8,
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="black", row=3, col=1)
    fig.add_vline(
        x=trade_context["trade_vs_mid"].mean(),
        line_width=2,
        line_dash="dot",
        line_color="#ff7f0e",
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=prices["mid_change_10"].dropna(),
            nbinsx=40,
            name="10-tick mid changes",
            marker={"color": "#9467bd"},
            opacity=0.8,
            showlegend=False,
        ),
        row=3,
        col=2,
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="black", row=3, col=2)

    fig.update_layout(**base_layout(f"{pretty_symbol(symbol)} Dashboard", height=1180))
    fig.update_annotations(font={"size": 14})
    fig.update_xaxes(title_text="Synthetic round timeline", row=1, col=1, automargin=True)
    fig.update_yaxes(title_text="Price", row=1, col=1, automargin=True)
    fig.update_xaxes(title_text="Synthetic round timeline", row=1, col=2, automargin=True)
    fig.update_yaxes(title_text="Mid price", row=1, col=2, automargin=True)
    fig.update_xaxes(title_text="Synthetic round timeline", row=2, col=1, automargin=True)
    fig.update_yaxes(title_text="Spread", row=2, col=1, automargin=True)
    fig.update_xaxes(title_text="(bid vol - ask vol) / (bid vol + ask vol)", row=2, col=2, automargin=True)
    fig.update_yaxes(title_text="Count", row=2, col=2, automargin=True)
    fig.update_xaxes(title_text="Trade price - quoted mid", row=3, col=1, automargin=True)
    fig.update_yaxes(title_text="Count", row=3, col=1, automargin=True)
    fig.update_xaxes(title_text="mid_price[t] - mid_price[t-10]", row=3, col=2, automargin=True)
    fig.update_yaxes(title_text="Count", row=3, col=2, automargin=True)
    return fig


def build_price_trade_trajectory_figure(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    trade_context: pd.DataFrame,
    symbol: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=prices["global_ts"],
            y=prices["mid_price"],
            mode="lines",
            name="Mid price",
            line={"width": 2.2, "color": "#1f77b4"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prices["global_ts"],
            y=prices["bid_price_1"],
            mode="lines",
            name="Best bid",
            line={"width": 1.1, "color": "#2ca02c"},
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prices["global_ts"],
            y=prices["ask_price_1"],
            mode="lines",
            name="Best ask",
            line={"width": 1.1, "color": "#d62728"},
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trades["global_ts"],
            y=trades["price"],
            mode="markers",
            name="Trades",
            marker={
                "size": np.clip(trades["quantity"].fillna(0) * 1.6, 6, 18),
                "color": trade_context["trade_vs_mid"],
                "colorscale": "RdYlGn",
                "colorbar": {"title": "Trade vs mid"},
                "line": {"width": 0.5, "color": "white"},
                "opacity": 0.9,
            },
            customdata=np.stack(
                [
                    trades["day"],
                    trades["timestamp"],
                    trades["quantity"],
                    trade_context["trade_vs_mid"],
                    trade_context["trade_vs_microprice"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "Day: %{customdata[0]:.0f}<br>"
                "Timestamp: %{customdata[1]:.0f}<br>"
                "Trade price: %{y:.2f}<br>"
                "Quantity: %{customdata[2]:.0f}<br>"
                "Trade vs mid: %{customdata[3]:.2f}<br>"
                "Trade vs microprice: %{customdata[4]:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(**base_layout(f"{pretty_symbol(symbol)} Price Trajectory With Trade Prints", height=580))
    fig.update_xaxes(title_text="Synthetic round timeline", rangeslider={"visible": True}, automargin=True)
    fig.update_yaxes(title_text="Price", automargin=True)
    return fig


def render_header() -> None:
    st.title("Round 1 Product Analytics")
    st.caption(
        "Upload Round 1 `prices` and `trades` CSVs to recreate the notebook analysis and switch between separate pages for each product."
    )


def render_inputs() -> tuple[list[pd.DataFrame], list[pd.DataFrame], str]:
    default_price_frames = list(load_default_csvs("prices"))
    default_trade_frames = list(load_default_csvs("trades"))

    price_upload = st.file_uploader(
        "Upload a `prices` CSV file",
        type=["csv"],
        accept_multiple_files=False,
        help="Uploading a new file replaces the previous prices upload. If left empty, the app uses the default Round 1 prices files.",
    )
    trade_upload = st.file_uploader(
        "Upload a `trades` CSV file",
        type=["csv"],
        accept_multiple_files=False,
        help="Uploading a new file replaces the previous trades upload. If left empty, the app uses the default Round 1 trades files.",
    )

    if price_upload is not None:
        price_frames = [read_uploaded_csv(price_upload)]
    else:
        price_frames = default_price_frames

    if trade_upload is not None:
        trade_frames = [read_uploaded_csv(trade_upload)]
    else:
        trade_frames = default_trade_frames

    product_candidates: set[str] = set()
    for frame in price_frames:
        if "product" in frame.columns:
            product_candidates.update(frame["product"].dropna().astype(str).tolist())
    for frame in trade_frames:
        if "symbol" in frame.columns:
            product_candidates.update(frame["symbol"].dropna().astype(str).tolist())

    symbol_options = sorted(product_candidates) or [DEFAULT_SYMBOL]
    page_labels = [pretty_symbol(symbol) for symbol in symbol_options]
    label_to_symbol = dict(zip(page_labels, symbol_options))
    default_label = pretty_symbol(DEFAULT_SYMBOL) if DEFAULT_SYMBOL in symbol_options else page_labels[0]
    selected_label = st.sidebar.radio("Analytics Page", page_labels, index=page_labels.index(default_label))
    symbol = label_to_symbol[selected_label]
    st.sidebar.info(f"Loaded {len(price_frames)} price file(s) and {len(trade_frames)} trade file(s).")
    return price_frames, trade_frames, symbol


def render_metric_cards(prices: pd.DataFrame, trades: pd.DataFrame, trade_context: pd.DataFrame) -> None:
    columns = st.columns(4)
    columns[0].metric("Quote rows", f"{len(prices):,}")
    columns[1].metric("Trade rows", f"{len(trades):,}")
    columns[2].metric("Median spread", f"{prices['spread_1'].median():.2f}")
    columns[3].metric("VWAP", f"{(trades['notional'].sum() / trades['quantity'].sum()):.2f}")

    columns = st.columns(4)
    columns[0].metric("Trade vs mid", f"{trade_context['trade_vs_mid'].mean():.2f}")
    columns[1].metric("Trade vs microprice", f"{trade_context['trade_vs_microprice'].mean():.2f}")
    columns[2].metric("1-step up share", f"{(prices['mid_change_1'] > 0).mean():.1%}")
    columns[3].metric("Lag-10 autocorr", f"{prices['mid_price'].autocorr(10):.3f}")


def main() -> None:
    render_header()
    price_frames, trade_frames, symbol = render_inputs()

    st.markdown(
        """
        <style>
        div[data-testid="stFileUploaderDropzone"] {
            padding: 1rem;
            border-radius: 16px;
        }
        div[data-testid="stMetric"] {
            background: #fafafa;
            border: 1px solid #ececec;
            border-radius: 14px;
            padding: 0.75rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        prices = prepare_price_data(price_frames, symbol)
        trades = prepare_trade_data(trade_frames, symbol, int(prices["day"].min()))
        trade_context = build_trade_context(prices, trades)
    except Exception as exc:
        st.error(f"Could not build the dashboard from the uploaded CSVs. Details: {exc}")
        st.stop()

    overview = build_overview(prices, trades, trade_context)
    day_summary = build_day_summary(prices, trade_context)
    latest_quotes = prices[
        ["day", "timestamp", "bid_price_1", "bid_volume_1", "ask_price_1", "ask_volume_1", "mid_price", "spread_1"]
    ].tail(12)
    largest_trades = trades[["day", "timestamp", "price", "quantity", "notional"]].sort_values(
        ["quantity", "notional"], ascending=[False, False]
    ).head(12)

    st.header(f"{pretty_symbol(symbol)} Analytics")
    render_metric_cards(prices, trades, trade_context)

    st.subheader("Bias Overview")
    for line in build_bias_lines(symbol, prices, trade_context):
        st.markdown(f"- {line}")

    st.subheader("Overview")
    st.dataframe(overview.round(4), use_container_width=True, hide_index=True)

    st.subheader("Day Summary")
    st.dataframe(day_summary.round(4), use_container_width=True)

    st.subheader("Notebook Dashboard")
    st.plotly_chart(
        build_dashboard_figure(prices, trades, trade_context, symbol),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    st.subheader("Price Path With Trade Timing")
    st.plotly_chart(
        build_price_trade_trajectory_figure(prices, trades, trade_context, symbol),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    left, right = st.columns(2)
    left.subheader("Latest Quotes")
    left.dataframe(latest_quotes.round(2), use_container_width=True, hide_index=True)
    right.subheader("Largest Trades")
    right.dataframe(largest_trades.round(2), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

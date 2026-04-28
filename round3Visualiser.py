from __future__ import annotations

import math
import re
import textwrap
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data"
DEFAULT_BUNDLED_DATASET = "ROUND_4" if (DATA_ROOT / "ROUND_4").exists() else "ROUND_3"
DEFAULT_UNDERLYING = "VELVETFRUIT_EXTRACT"
DEFAULT_PAIR_LEFT = "VELVETFRUIT_EXTRACT"
DEFAULT_PAIR_RIGHT = "HYDROGEL_PACK"
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
}
DISPLAY_NAME_BY_SYMBOL = {
    "VELVETFRUIT_EXTRACT": "Velvetfruit Extract",
    "HYDROGEL_PACK": "Hydrogel Pack",
}
OPTION_PATTERN = re.compile(r"VEV_(\d+)$")
NORMAL = NormalDist(mu=0.0, sigma=1.0)


st.set_page_config(page_title="Market Visualiser", layout="wide")


@dataclass(frozen=True)
class PairAnalysis:
    left_symbol: str
    right_symbol: str
    sample_size: int
    alpha: float
    beta: float
    spread_mean: float
    spread_std: float
    latest_spread: float
    latest_zscore: float
    correlation: float
    r_squared: float
    adf_style_stat: float
    adf_style_phi: float
    adf_style_label: str
    half_life: float | None
    zero_crossings: int
    spread_frame: pd.DataFrame


def wrap_label(label: str, width: int = 16) -> str:
    return "<br>".join(textwrap.wrap(label, width=width)) or label


def pretty_symbol(symbol: str) -> str:
    return DISPLAY_NAME_BY_SYMBOL.get(symbol, symbol.replace("_", " ").title())


def is_option_symbol(symbol: str) -> bool:
    return OPTION_PATTERN.match(symbol) is not None


def extract_strike(symbol: str) -> int | None:
    match = OPTION_PATTERN.match(symbol)
    return int(match.group(1)) if match else None


def read_uploaded_csv(file: BytesIO) -> pd.DataFrame:
    return pd.read_csv(file, sep=";")


def parse_day_hint(filename: str | None) -> int | None:
    if not filename:
        return None
    match = re.search(r"day[_-](\d+)", filename)
    return int(match.group(1)) if match else None


@st.cache_data(show_spinner=False)
def list_bundled_datasets() -> tuple[str, ...]:
    datasets = sorted(path.name for path in DATA_ROOT.glob("ROUND_*") if path.is_dir())
    if not datasets:
        raise ValueError(f"No bundled ROUND_* datasets were found in `{DATA_ROOT}`.")
    return tuple(datasets)


@st.cache_data(show_spinner=False)
def load_default_csvs(kind: str, dataset_name: str) -> tuple[pd.DataFrame, ...]:
    data_dir = DATA_ROOT / dataset_name
    paths = sorted(data_dir.glob(f"{kind}_*.csv"))
    if not paths:
        raise ValueError(f"No `{kind}` CSV files were found in `{data_dir}`.")
    return tuple(pd.read_csv(path, sep=";") for path in paths)


@st.cache_data(show_spinner=False)
def load_market_data(
    price_payloads: tuple[tuple[str, bytes], ...],
    trade_payloads: tuple[tuple[str, bytes], ...],
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = DATA_ROOT / dataset_name
    if price_payloads:
        price_frames = []
        for filename, payload in price_payloads:
            frame = pd.read_csv(BytesIO(payload), sep=";")
            frame["source_file"] = filename
            price_frames.append(frame)
    else:
        price_frames = []
        for path in sorted(data_dir.glob("prices_*.csv")):
            frame = pd.read_csv(path, sep=";")
            frame["source_file"] = path.name
            price_frames.append(frame)

    if trade_payloads:
        trade_frames = []
        for filename, payload in trade_payloads:
            frame = pd.read_csv(BytesIO(payload), sep=";")
            frame["source_file"] = filename
            trade_frames.append(frame)
    else:
        trade_frames = []
        for path in sorted(data_dir.glob("trades_*.csv")):
            frame = pd.read_csv(path, sep=";")
            frame["source_file"] = path.name
            trade_frames.append(frame)

    prices = prepare_prices_table(price_frames)
    trades = prepare_trades_table(trade_frames, prices)
    return prices, trades


def prepare_prices_table(frames: list[pd.DataFrame]) -> pd.DataFrame:
    prepared_frames: list[pd.DataFrame] = []
    for index, frame in enumerate(frames):
        df = frame.copy()
        if "product" not in df.columns:
            raise ValueError("Prices CSV must contain a `product` column.")
        if "timestamp" not in df.columns:
            raise ValueError("Prices CSV must contain a `timestamp` column.")

        source_name = str(df.get("source_file", pd.Series([f"prices_upload_{index + 1}.csv"])).iloc[0])
        day_hint = parse_day_hint(source_name)

        if "day" not in df.columns:
            if day_hint is None:
                raise ValueError("Prices CSV must contain a `day` column or include `day_<n>` in the filename.")
            df["day"] = day_hint

        for column in df.columns:
            if column not in {"product", "source_file"}:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        df["product"] = df["product"].astype(str)
        df["source_file"] = source_name
        prepared_frames.append(df)

    if not prepared_frames:
        raise ValueError("No prices data could be loaded.")

    prices = pd.concat(prepared_frames, ignore_index=True)
    prices = prices.sort_values(["day", "timestamp", "product"]).reset_index(drop=True)

    empty_book = prices["bid_price_1"].isna() & prices["ask_price_1"].isna() & (prices["mid_price"] == 0)
    prices.loc[empty_book, "mid_price"] = np.nan
    prices["global_ts"] = (prices["day"] - prices["day"].min()) * 1_000_000 + prices["timestamp"]
    prices["spread_1"] = prices["ask_price_1"] - prices["bid_price_1"]
    prices["mid_from_quotes"] = (prices["bid_price_1"] + prices["ask_price_1"]) / 2
    top_level_volume = prices["bid_volume_1"] + prices["ask_volume_1"]
    prices["microprice_l1"] = (
        prices["ask_price_1"] * prices["bid_volume_1"] + prices["bid_price_1"] * prices["ask_volume_1"]
    ) / top_level_volume.replace(0, np.nan)
    prices["book_imbalance"] = (
        (prices["bid_volume_1"] - prices["ask_volume_1"]) / top_level_volume.replace(0, np.nan)
    )
    prices["strike"] = prices["product"].map(extract_strike)
    prices["is_option"] = prices["strike"].notna()
    return prices


def prepare_trades_table(trade_frames: list[pd.DataFrame], prices: pd.DataFrame) -> pd.DataFrame:
    prepared_frames: list[pd.DataFrame] = []
    inferred_day = int(prices["day"].min())

    for index, frame in enumerate(trade_frames):
        df = frame.copy()
        source_name = str(df.get("source_file", pd.Series([f"trades_upload_{index + 1}.csv"])).iloc[0])
        day_hint = parse_day_hint(source_name)

        if "symbol" not in df.columns:
            raise ValueError("Trades CSV must contain a `symbol` column.")

        if "day" not in df.columns:
            if day_hint is not None:
                df["day"] = day_hint
            else:
                df["day"] = inferred_day
                inferred_day += 1

        for column in ["timestamp", "price", "quantity", "day"]:
            if column not in df.columns:
                raise ValueError(f"Trades CSV must contain a `{column}` column.")
            df[column] = pd.to_numeric(df[column], errors="coerce")

        for column in ["buyer", "seller", "currency"]:
            if column not in df.columns:
                df[column] = ""
            df[column] = df[column].where(df[column].notna(), "").astype(str).str.strip()

        df["symbol"] = df["symbol"].astype(str)
        df["source_file"] = source_name
        prepared_frames.append(df)

    if not prepared_frames:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "buyer",
                "seller",
                "symbol",
                "currency",
                "price",
                "quantity",
                "day",
                "source_file",
                "global_ts",
                "notional",
            ]
        )

    trades = pd.concat(prepared_frames, ignore_index=True)
    trades = trades.sort_values(["day", "timestamp", "symbol"]).reset_index(drop=True)
    trades["global_ts"] = (trades["day"] - prices["day"].min()) * 1_000_000 + trades["timestamp"]
    trades["notional"] = trades["price"] * trades["quantity"]
    trades["buyer_display"] = trades["buyer"].replace("", "Unknown buyer")
    trades["seller_display"] = trades["seller"].replace("", "Unknown seller")
    return trades


def get_product_view(prices: pd.DataFrame, product: str) -> pd.DataFrame:
    product_prices = prices.loc[prices["product"] == product].copy()
    if product_prices.empty:
        raise ValueError(f"No prices found for `{product}`.")
    product_prices = product_prices.sort_values(["day", "timestamp"]).reset_index(drop=True)
    product_prices["mid_change_1"] = product_prices["mid_price"].diff()
    product_prices["mid_change_10"] = product_prices["mid_price"].diff(10)
    product_prices["rolling_mid_200"] = product_prices["mid_price"].rolling(200, min_periods=20).mean()
    product_prices["rolling_mid_50"] = product_prices["mid_price"].rolling(50, min_periods=10).mean()
    product_prices["normalized_mid"] = normalize_series(product_prices["mid_price"])
    product_prices["zscore_mid"] = zscore_series(product_prices["mid_price"])
    return product_prices


def get_trade_view(trades: pd.DataFrame, product: str) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    return trades.loc[trades["symbol"] == product].copy().sort_values(["day", "timestamp"]).reset_index(drop=True)


def build_trade_hover_customdata(trades: pd.DataFrame) -> np.ndarray:
    return trades[["day", "timestamp", "quantity", "notional", "buyer_display", "seller_display"]].to_numpy()


def normalize_series(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    start = valid.iloc[0]
    if start == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return series / start * 100.0


def zscore_series(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - mean) / std


def merge_mid_series(
    prices: pd.DataFrame,
    left_symbol: str,
    right_symbol: str,
) -> pd.DataFrame:
    left = get_product_view(prices, left_symbol)[["global_ts", "day", "timestamp", "mid_price"]].rename(
        columns={"mid_price": "left_mid"}
    )
    right = get_product_view(prices, right_symbol)[["global_ts", "mid_price"]].rename(columns={"mid_price": "right_mid"})
    merged = pd.merge(left, right, on="global_ts", how="inner").dropna(subset=["left_mid", "right_mid"])
    return merged.sort_values("global_ts").reset_index(drop=True)


def compute_pair_analysis(prices: pd.DataFrame, left_symbol: str, right_symbol: str) -> PairAnalysis:
    merged = merge_mid_series(prices, left_symbol, right_symbol)
    if len(merged) < 30:
        raise ValueError("Need at least 30 overlapping quote points to run the pairs analysis.")

    x = merged["right_mid"].to_numpy(dtype=float)
    y = merged["left_mid"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(merged)), x])
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    alpha, beta = coeffs.tolist()
    fitted = alpha + beta * x
    spread = y - fitted
    spread_mean = float(np.mean(spread))
    spread_std = float(np.std(spread, ddof=0))
    latest_spread = float(spread[-1])
    latest_zscore = float((latest_spread - spread_mean) / spread_std) if spread_std else float("nan")
    correlation = float(np.corrcoef(y, x)[0, 1])
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1 - ss_res / ss_tot) if ss_tot else float("nan")

    spread_frame = merged.copy()
    spread_frame["fitted_left"] = fitted
    spread_frame["spread"] = spread
    spread_frame["spread_zscore"] = (
        (spread_frame["spread"] - spread_mean) / spread_std if spread_std else np.nan
    )

    adf_style_stat, adf_style_phi = compute_adf_style_stat(spread_frame["spread"])
    if adf_style_stat <= -3.43:
        adf_style_label = "strong mean reversion"
    elif adf_style_stat <= -2.86:
        adf_style_label = "moderate mean reversion"
    elif adf_style_stat <= -2.57:
        adf_style_label = "weak mean reversion"
    else:
        adf_style_label = "little residual stationarity"

    half_life = estimate_half_life(spread_frame["spread"])
    signs = np.sign(spread_frame["spread"].to_numpy(dtype=float))
    zero_crossings = int(np.sum(signs[1:] * signs[:-1] < 0))

    return PairAnalysis(
        left_symbol=left_symbol,
        right_symbol=right_symbol,
        sample_size=len(spread_frame),
        alpha=float(alpha),
        beta=float(beta),
        spread_mean=spread_mean,
        spread_std=spread_std,
        latest_spread=latest_spread,
        latest_zscore=latest_zscore,
        correlation=correlation,
        r_squared=r_squared,
        adf_style_stat=adf_style_stat,
        adf_style_phi=adf_style_phi,
        adf_style_label=adf_style_label,
        half_life=half_life,
        zero_crossings=zero_crossings,
        spread_frame=spread_frame,
    )


def compute_adf_style_stat(series: pd.Series) -> tuple[float, float]:
    clean = series.dropna().astype(float)
    if len(clean) < 20:
        return float("nan"), float("nan")

    lagged = clean.shift(1).dropna()
    delta = clean.diff().dropna()
    aligned = pd.DataFrame({"delta": delta, "lagged": lagged}).dropna()
    if len(aligned) < 10:
        return float("nan"), float("nan")

    x = aligned["lagged"].to_numpy(dtype=float)
    y = aligned["delta"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(aligned)), x])
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coeffs
    dof = len(y) - design.shape[1]
    if dof <= 0:
        return float("nan"), float("nan")

    sigma2 = float((residuals @ residuals) / dof)
    xtx_inv = np.linalg.inv(design.T @ design)
    se_beta = math.sqrt(max(sigma2 * xtx_inv[1, 1], 0.0))
    phi = float(coeffs[1])
    stat = float(phi / se_beta) if se_beta else float("nan")
    return stat, phi


def estimate_half_life(series: pd.Series) -> float | None:
    clean = series.dropna().astype(float)
    if len(clean) < 20:
        return None
    lagged = clean.shift(1).dropna()
    delta = clean.diff().dropna()
    aligned = pd.DataFrame({"delta": delta, "lagged": lagged}).dropna()
    if len(aligned) < 10:
        return None
    x = aligned["lagged"].to_numpy(dtype=float)
    y = aligned["delta"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(aligned)), x])
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    phi = float(coeffs[1])
    if phi >= 0:
        return None
    return float(-math.log(2) / phi)


def bs_call(spot: float, strike: float, tte: float, rate: float, sigma: float) -> float:
    if tte <= 0:
        return max(spot - strike, 0.0)
    if sigma <= 0:
        return max(spot - strike * math.exp(-rate * tte), 0.0)

    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma**2) * tte) / (sigma * math.sqrt(tte))
    d2 = d1 - sigma * math.sqrt(tte)
    return spot * NORMAL.cdf(d1) - strike * math.exp(-rate * tte) * NORMAL.cdf(d2)


def vega(spot: float, strike: float, tte: float, rate: float, sigma: float) -> float:
    if tte <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma**2) * tte) / (sigma * math.sqrt(tte))
    return spot * math.sqrt(tte) * NORMAL.pdf(d1)


def implied_volatility(
    target_price: float,
    spot: float,
    strike: float,
    tte: float,
    rate: float = 0.0,
    low: float = 1e-4,
    high: float = 5.0,
) -> float | None:
    if any(value <= 0 for value in [spot, strike]) or target_price < 0:
        return None

    sigma = 0.2
    lo = low
    hi = high

    for _ in range(100):
        price = bs_call(spot, strike, tte, rate, sigma)
        diff = target_price - price
        if abs(diff) < 1e-8:
            return sigma

        current_vega = vega(spot, strike, tte, rate, sigma)
        if abs(current_vega) < 1e-6:
            if price < target_price:
                lo = sigma
            else:
                hi = sigma
            sigma = (lo + hi) / 2
            continue

        new_sigma = sigma + diff / current_vega
        if new_sigma <= lo or new_sigma >= hi or not np.isfinite(new_sigma):
            if price < target_price:
                lo = sigma
            else:
                hi = sigma
            sigma = (lo + hi) / 2
        else:
            sigma = new_sigma

    return sigma if np.isfinite(sigma) else None


@st.cache_data(show_spinner=False)
def build_option_analytics(prices: pd.DataFrame, underlying_symbol: str, round_number: int) -> pd.DataFrame:
    underlying = get_product_view(prices, underlying_symbol)[["global_ts", "day", "timestamp", "mid_price"]].rename(
        columns={"mid_price": "underlying_mid"}
    )
    option_symbols = sorted([symbol for symbol in prices["product"].unique().tolist() if is_option_symbol(symbol)])
    tte_years = max(1e-6, (8 - round_number) / 365)

    analytics_frames: list[pd.DataFrame] = []
    for symbol in option_symbols:
        strike = extract_strike(symbol)
        if strike is None:
            continue
        option_df = get_product_view(prices, symbol)[["global_ts", "day", "timestamp", "mid_price"]].rename(
            columns={"mid_price": "option_mid"}
        )
        merged = pd.merge(underlying, option_df, on=["global_ts", "day", "timestamp"], how="inner").dropna()
        if merged.empty:
            continue

        merged["symbol"] = symbol
        merged["strike"] = strike
        merged["tte_years"] = tte_years
        merged["intrinsic_value"] = np.maximum(merged["underlying_mid"] - strike, 0.0)
        merged["time_value"] = merged["option_mid"] - merged["intrinsic_value"]
        merged["implied_vol"] = merged.apply(
            lambda row: implied_volatility(
                target_price=float(row["option_mid"]),
                spot=float(row["underlying_mid"]),
                strike=float(strike),
                tte=float(tte_years),
                rate=0.0,
            ),
            axis=1,
        )
        merged["fair_iv_rolling"] = merged["implied_vol"].rolling(100, min_periods=5).mean()
        merged["bs_fair_value"] = merged.apply(
            lambda row: bs_call(
                spot=float(row["underlying_mid"]),
                strike=float(strike),
                tte=float(tte_years),
                rate=0.0,
                sigma=float(row["fair_iv_rolling"]) if pd.notna(row["fair_iv_rolling"]) else np.nan,
            )
            if pd.notna(row["fair_iv_rolling"])
            else np.nan,
            axis=1,
        )
        merged["mispricing"] = merged["option_mid"] - merged["bs_fair_value"]
        analytics_frames.append(merged)

    if not analytics_frames:
        return pd.DataFrame(
            columns=[
                "global_ts",
                "day",
                "timestamp",
                "underlying_mid",
                "option_mid",
                "symbol",
                "strike",
                "tte_years",
                "intrinsic_value",
                "time_value",
                "implied_vol",
                "fair_iv_rolling",
                "bs_fair_value",
                "mispricing",
            ]
        )

    return pd.concat(analytics_frames, ignore_index=True).sort_values(["strike", "day", "timestamp"]).reset_index(drop=True)


def base_layout(title: str, height: int) -> dict[str, Any]:
    return {
        "title": {"text": title, "x": 0.01, "y": 0.98},
        "template": "plotly_white",
        "height": height,
        "margin": {"l": 65, "r": 30, "t": 100, "b": 60},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    }


def build_product_dashboard(prices: pd.DataFrame, trades: pd.DataFrame, product: str) -> go.Figure:
    product_prices = get_product_view(prices, product)
    product_trades = get_trade_view(trades, product)
    trade_context = build_trade_context(product_prices, product_trades)

    fig = make_subplots(
        rows=3,
        cols=2,
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
        subplot_titles=[
            "Best Quotes and Trades",
            "Mid and Rolling Means",
            "Quoted Spread",
            "Book Imbalance",
            "Trade Location vs Mid",
            "10-Step Mid Change",
        ],
    )

    fig.add_trace(
        go.Scatter(x=product_prices["global_ts"], y=product_prices["bid_price_1"], mode="lines", name="Best bid"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=product_prices["global_ts"], y=product_prices["ask_price_1"], mode="lines", name="Best ask"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=product_prices["global_ts"], y=product_prices["mid_price"], mode="lines", name="Mid", line={"width": 2}),
        row=1,
        col=1,
    )
    if not product_trades.empty:
        fig.add_trace(
            go.Scatter(
                x=product_trades["global_ts"],
                y=product_trades["price"],
                mode="markers",
                name="Trades",
                marker={"size": np.clip(product_trades["quantity"].fillna(0) * 1.5, 6, 18), "opacity": 0.45, "color": "black"},
                customdata=build_trade_hover_customdata(product_trades),
                hovertemplate=(
                    "Day %{customdata[0]}<br>"
                    "Timestamp %{customdata[1]}<br>"
                    "Trade price %{y:.2f}<br>"
                    "Quantity %{customdata[2]:.0f}<br>"
                    "Notional %{customdata[3]:.2f}<br>"
                    "Buyer %{customdata[4]}<br>"
                    "Seller %{customdata[5]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(x=product_prices["global_ts"], y=product_prices["mid_price"], mode="lines", name="Mid price"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=product_prices["global_ts"],
            y=product_prices["rolling_mid_50"],
            mode="lines",
            name="Rolling 50",
            line={"width": 2},
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=product_prices["global_ts"],
            y=product_prices["rolling_mid_200"],
            mode="lines",
            name="Rolling 200",
            line={"width": 2},
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=product_prices["global_ts"],
            y=product_prices["spread_1"],
            mode="lines",
            name="Spread",
            line={"color": "#d62728"},
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=product_prices["book_imbalance"].dropna(),
            nbinsx=40,
            name="Imbalance",
            marker={"color": "#1f77b4"},
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Histogram(
            x=trade_context["trade_vs_mid"].dropna(),
            nbinsx=40,
            name="Trade vs mid",
            marker={"color": "#2ca02c"},
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=product_prices["mid_change_10"].dropna(),
            nbinsx=40,
            name="10-step change",
            marker={"color": "#9467bd"},
            showlegend=False,
        ),
        row=3,
        col=2,
    )

    fig.update_layout(**base_layout(f"{pretty_symbol(product)} Dashboard", 1120))
    fig.update_xaxes(title_text="Synthetic round timeline", row=1, col=1)
    fig.update_xaxes(title_text="Synthetic round timeline", row=1, col=2)
    fig.update_xaxes(title_text="Synthetic round timeline", row=2, col=1)
    fig.update_xaxes(title_text="(bid vol - ask vol) / total vol", row=2, col=2)
    fig.update_xaxes(title_text="Trade price - quoted mid", row=3, col=1)
    fig.update_xaxes(title_text="mid[t] - mid[t-10]", row=3, col=2)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Mid", row=1, col=2)
    fig.update_yaxes(title_text="Spread", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=2)
    return fig


def build_trade_context(prices: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["trade_vs_mid", "trade_vs_microprice"])
    context = pd.merge_asof(
        trades.sort_values("global_ts"),
        prices[["global_ts", "mid_price", "microprice_l1"]].sort_values("global_ts"),
        on="global_ts",
        direction="backward",
    )
    context["trade_vs_mid"] = context["price"] - context["mid_price"]
    context["trade_vs_microprice"] = context["price"] - context["microprice_l1"]
    return context


@st.cache_data(show_spinner=False)
def build_trader_position_ledger(prices: pd.DataFrame, trades: pd.DataFrame, product: str) -> pd.DataFrame:
    product_prices = get_product_view(prices, product)[["global_ts", "mid_price"]].sort_values("global_ts")
    product_trades = get_trade_view(trades, product)

    if product_trades.empty:
        return pd.DataFrame(
            columns=[
                "global_ts",
                "day",
                "timestamp",
                "trader",
                "side",
                "price",
                "quantity",
                "signed_quantity",
                "cash_flow",
                "mark_price",
                "position",
                "inventory_mtm",
                "total_pnl",
            ]
        )

    buyer_legs = product_trades.loc[product_trades["buyer"].ne("")].copy()
    buyer_legs["trader"] = buyer_legs["buyer"]
    buyer_legs["side"] = "Buy"
    buyer_legs["signed_quantity"] = buyer_legs["quantity"]
    buyer_legs["cash_flow"] = -buyer_legs["notional"]

    seller_legs = product_trades.loc[product_trades["seller"].ne("")].copy()
    seller_legs["trader"] = seller_legs["seller"]
    seller_legs["side"] = "Sell"
    seller_legs["signed_quantity"] = -seller_legs["quantity"]
    seller_legs["cash_flow"] = seller_legs["notional"]

    ledger = pd.concat([buyer_legs, seller_legs], ignore_index=True)
    if ledger.empty:
        return pd.DataFrame(
            columns=[
                "global_ts",
                "day",
                "timestamp",
                "trader",
                "side",
                "price",
                "quantity",
                "signed_quantity",
                "cash_flow",
                "mark_price",
                "position",
                "inventory_mtm",
                "total_pnl",
            ]
        )

    ledger = ledger.sort_values(["global_ts", "trader", "side"]).reset_index(drop=True)
    ledger = pd.merge_asof(
        ledger,
        product_prices.rename(columns={"mid_price": "mark_price"}).dropna(subset=["mark_price"]),
        on="global_ts",
        direction="backward",
    )
    ledger["mark_price"] = ledger["mark_price"].where(ledger["mark_price"].notna(), ledger["price"])
    ledger["position"] = ledger.groupby("trader")["signed_quantity"].cumsum()
    ledger["inventory_mtm"] = ledger["position"] * ledger["mark_price"]
    ledger["total_pnl"] = ledger.groupby("trader")["cash_flow"].cumsum() + ledger["inventory_mtm"]
    return ledger[
        [
            "global_ts",
            "day",
            "timestamp",
            "trader",
            "side",
            "price",
            "quantity",
            "signed_quantity",
            "cash_flow",
            "mark_price",
            "position",
            "inventory_mtm",
            "total_pnl",
        ]
    ]


@st.cache_data(show_spinner=False)
def build_trader_pnl_summary(prices: pd.DataFrame, trades: pd.DataFrame, product: str) -> pd.DataFrame:
    ledger = build_trader_position_ledger(prices, trades, product)
    if ledger.empty:
        return pd.DataFrame(
            columns=[
                "trader",
                "trade_count",
                "buy_quantity",
                "sell_quantity",
                "buy_vwap",
                "sell_vwap",
                "net_position",
                "cash_flow",
                "mark_price",
                "inventory_mtm",
                "total_pnl",
            ]
        )

    rows: list[dict[str, Any]] = []
    for trader, trader_ledger in ledger.groupby("trader", sort=True):
        buys = trader_ledger.loc[trader_ledger["signed_quantity"] > 0]
        sells = trader_ledger.loc[trader_ledger["signed_quantity"] < 0]

        buy_quantity = float(buys["signed_quantity"].sum())
        sell_quantity = float((-sells["signed_quantity"]).sum())
        buy_vwap = (buys["price"] * buys["signed_quantity"]).sum() / buy_quantity if buy_quantity else np.nan
        sell_vwap = (sells["price"] * (-sells["signed_quantity"])).sum() / sell_quantity if sell_quantity else np.nan
        last_row = trader_ledger.iloc[-1]

        rows.append(
            {
                "trader": trader,
                "trade_count": int(len(trader_ledger)),
                "buy_quantity": buy_quantity,
                "sell_quantity": sell_quantity,
                "buy_vwap": buy_vwap,
                "sell_vwap": sell_vwap,
                "net_position": float(last_row["position"]),
                "cash_flow": float(trader_ledger["cash_flow"].sum()),
                "mark_price": float(last_row["mark_price"]),
                "inventory_mtm": float(last_row["inventory_mtm"]),
                "total_pnl": float(last_row["total_pnl"]),
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values(["total_pnl", "trade_count"], ascending=[False, False]).reset_index(drop=True)


def build_trader_pnl_bar_chart(summary: pd.DataFrame, product: str) -> go.Figure:
    fig = go.Figure()
    if summary.empty:
        return fig

    colors = np.where(summary["total_pnl"] >= 0, "#2ca02c", "#d62728")
    fig.add_trace(
        go.Bar(
            x=summary["trader"],
            y=summary["total_pnl"],
            marker={"color": colors},
            customdata=summary[["net_position", "trade_count", "mark_price"]].to_numpy(),
            hovertemplate=(
                "Trader %{x}<br>"
                "Total PnL %{y:.2f}<br>"
                "Net position %{customdata[0]:.0f}<br>"
                "Trade count %{customdata[1]:.0f}<br>"
                "Mark price %{customdata[2]:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(**base_layout(f"Trader PnL: {pretty_symbol(product)}", 420))
    fig.update_xaxes(title_text="Trader")
    fig.update_yaxes(title_text="PnL")
    return fig


def build_trader_pnl_timeseries(ledger: pd.DataFrame, trader: str, product: str) -> go.Figure:
    fig = go.Figure()
    trader_ledger = ledger.loc[ledger["trader"] == trader].copy()
    if trader_ledger.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=trader_ledger["global_ts"],
            y=trader_ledger["total_pnl"],
            mode="lines+markers",
            name="Total PnL",
            line={"width": 2.2, "color": "#1f77b4"},
            customdata=trader_ledger[["day", "timestamp", "side", "price", "quantity", "position"]].to_numpy(),
            hovertemplate=(
                "Day %{customdata[0]}<br>"
                "Timestamp %{customdata[1]}<br>"
                "Side %{customdata[2]}<br>"
                "Trade price %{customdata[3]:.2f}<br>"
                "Trade qty %{customdata[4]:.0f}<br>"
                "Position %{customdata[5]:.0f}<br>"
                "PnL %{y:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(**base_layout(f"{trader} PnL Path: {pretty_symbol(product)}", 420))
    fig.update_xaxes(title_text="Synthetic round timeline")
    fig.update_yaxes(title_text="PnL")
    return fig


def build_overlay_figure(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    symbols: list[str],
    value_mode: str,
    show_trades: bool,
) -> go.Figure:
    fig = go.Figure()
    color_cycle = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"]

    for index, symbol in enumerate(symbols):
        product_prices = get_product_view(prices, symbol)
        color = color_cycle[index % len(color_cycle)]
        if value_mode == "Mid price":
            y = product_prices["mid_price"]
            yaxis_title = "Price"
        elif value_mode == "Normalized to 100":
            y = product_prices["normalized_mid"]
            yaxis_title = "Indexed level"
        else:
            y = product_prices["zscore_mid"]
            yaxis_title = "Z-score"

        fig.add_trace(
            go.Scatter(
                x=product_prices["global_ts"],
                y=y,
                mode="lines",
                name=pretty_symbol(symbol),
                line={"width": 2.1, "color": color},
            )
        )

        if show_trades and value_mode == "Mid price":
            product_trades = get_trade_view(trades, symbol)
            if not product_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=product_trades["global_ts"],
                        y=product_trades["price"],
                        mode="markers",
                        name=f"{pretty_symbol(symbol)} trades",
                        marker={"size": 7, "color": color, "opacity": 0.3},
                        customdata=build_trade_hover_customdata(product_trades),
                        hovertemplate=(
                            "Product "
                            + pretty_symbol(symbol)
                            + "<br>Day %{customdata[0]}<br>"
                            "Timestamp %{customdata[1]}<br>"
                            "Trade price %{y:.2f}<br>"
                            "Quantity %{customdata[2]:.0f}<br>"
                            "Buyer %{customdata[4]}<br>"
                            "Seller %{customdata[5]}<extra></extra>"
                        ),
                    )
                )

    fig.update_layout(**base_layout(f"Multi-Product Overlay: {value_mode}", 560))
    fig.update_xaxes(title_text="Synthetic round timeline", rangeslider={"visible": True})
    fig.update_yaxes(title_text=yaxis_title)
    return fig


def build_pairs_figure(pair: PairAnalysis) -> go.Figure:
    frame = pair.spread_frame
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Two Mid Prices on the Same Axis",
            "Regression Spread",
            "Spread Z-Score",
        ],
    )
    fig.add_trace(
        go.Scatter(x=frame["global_ts"], y=frame["left_mid"], mode="lines", name=pretty_symbol(pair.left_symbol)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=frame["global_ts"], y=frame["right_mid"], mode="lines", name=pretty_symbol(pair.right_symbol)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=frame["global_ts"], y=frame["spread"], mode="lines", name="Spread", line={"color": "#d62728"}),
        row=2,
        col=1,
    )
    fig.add_hline(y=pair.spread_mean, line_width=1, line_dash="dash", line_color="black", row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=frame["global_ts"],
            y=frame["spread_zscore"],
            mode="lines",
            name="Spread z-score",
            line={"color": "#9467bd"},
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black", row=3, col=1)
    fig.add_hline(y=2, line_width=1, line_dash="dot", line_color="#ff7f0e", row=3, col=1)
    fig.add_hline(y=-2, line_width=1, line_dash="dot", line_color="#ff7f0e", row=3, col=1)
    fig.update_layout(**base_layout(f"Pairs / Cointegration Screen: {pair.left_symbol} vs {pair.right_symbol}", 940))
    fig.update_xaxes(title_text="Synthetic round timeline", row=3, col=1)
    fig.update_yaxes(title_text="Mid price", row=1, col=1)
    fig.update_yaxes(title_text="Spread", row=2, col=1)
    fig.update_yaxes(title_text="Z-score", row=3, col=1)
    return fig


def build_options_curve_figure(option_analytics: pd.DataFrame, timestamp_mode: str) -> go.Figure:
    if option_analytics.empty:
        return go.Figure()

    if timestamp_mode == "Latest":
        latest_ts = option_analytics["global_ts"].max()
        curve = option_analytics.loc[option_analytics["global_ts"] == latest_ts].copy()
        title_suffix = "Latest Snapshot"
    else:
        curve = option_analytics.groupby("symbol", as_index=False).agg(
            strike=("strike", "first"),
            option_mid=("option_mid", "mean"),
            bs_fair_value=("bs_fair_value", "mean"),
            implied_vol=("implied_vol", "mean"),
        )
        title_suffix = "Mean Across Loaded Sample"

    curve = curve.sort_values("strike")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=["Price by Strike", "Implied Vol by Strike"])
    fig.add_trace(
        go.Scatter(x=curve["strike"], y=curve["option_mid"], mode="lines+markers", name="Market mid"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=curve["strike"], y=curve["bs_fair_value"], mode="lines+markers", name="BS fair"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=curve["strike"], y=curve["implied_vol"], mode="lines+markers", name="Implied vol"),
        row=2,
        col=1,
    )
    fig.update_layout(**base_layout(f"Voucher Surface: {title_suffix}", 760))
    fig.update_xaxes(title_text="Strike", row=2, col=1)
    fig.update_yaxes(title_text="Voucher price", row=1, col=1)
    fig.update_yaxes(title_text="Implied vol", row=2, col=1)
    return fig


def build_option_timeseries_figure(option_analytics: pd.DataFrame, option_symbol: str) -> go.Figure:
    series = option_analytics.loc[option_analytics["symbol"] == option_symbol].copy()
    series["underlying_indexed"] = normalize_series(series["underlying_mid"])
    series["option_indexed"] = normalize_series(series["option_mid"])
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=["Underlying and Voucher Paths (Indexed to 100)", "Implied Vol and Rolling Fair IV", "Observed vs Fair Voucher Price"],
    )
    fig.add_trace(
        go.Scatter(x=series["global_ts"], y=series["underlying_indexed"], mode="lines", name="Underlying"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=series["global_ts"], y=series["option_indexed"], mode="lines", name=option_symbol),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=series["global_ts"], y=series["implied_vol"], mode="lines", name="Market IV"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=series["global_ts"], y=series["fair_iv_rolling"], mode="lines", name="Rolling fair IV"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=series["global_ts"], y=series["option_mid"], mode="lines", name="Observed option mid"),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=series["global_ts"], y=series["bs_fair_value"], mode="lines", name="BS fair value"),
        row=3,
        col=1,
    )
    fig.update_layout(**base_layout(f"Voucher Diagnostics: {option_symbol}", 980))
    fig.update_xaxes(title_text="Synthetic round timeline", row=3, col=1)
    fig.update_yaxes(title_text="Indexed level", row=1, col=1)
    fig.update_yaxes(title_text="Volatility", row=2, col=1)
    fig.update_yaxes(title_text="Option price", row=3, col=1)
    return fig


@st.cache_data(show_spinner=False)
def build_pair_ranking_table(prices: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(symbols):
        for right in symbols[i + 1 :]:
            try:
                pair = compute_pair_analysis(prices, left, right)
            except Exception:
                continue
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "corr": pair.correlation,
                    "r_squared": pair.r_squared,
                    "adf_style_stat": pair.adf_style_stat,
                    "half_life": pair.half_life,
                    "latest_zscore": pair.latest_zscore,
                    "zero_crossings": pair.zero_crossings,
                    "signal_strength": abs(pair.latest_zscore) if np.isfinite(pair.latest_zscore) else np.nan,
                }
            )
    if not rows:
        return pd.DataFrame()
    ranking = pd.DataFrame(rows)
    return ranking.sort_values(["adf_style_stat", "signal_strength"], ascending=[True, False]).reset_index(drop=True)


def render_header() -> None:
    st.title("Prosperity Market Visualiser")
    st.caption(
        "Inspect bundled or uploaded Prosperity `prices` and `trades` CSVs, compare products, inspect voucher mispricings against Velvetfruit Extract, and review per-trader PnL on named-counterparty rounds."
    )


def render_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    bundled_datasets = list_bundled_datasets()

    with st.sidebar:
        st.header("Data")
        dataset_name = st.selectbox(
            "Bundled dataset",
            bundled_datasets,
            index=bundled_datasets.index(DEFAULT_BUNDLED_DATASET) if DEFAULT_BUNDLED_DATASET in bundled_datasets else 0,
            help="Used when you do not upload files manually.",
        )
        price_uploads = st.file_uploader(
            "Drop one or more `prices` CSVs",
            type=["csv"],
            accept_multiple_files=True,
            help="If empty, the app uses the selected bundled price files.",
        )
        trade_uploads = st.file_uploader(
            "Drop one or more `trades` CSVs",
            type=["csv"],
            accept_multiple_files=True,
            help="If empty, the app uses the selected bundled trade files.",
        )

    price_payloads = tuple((uploaded.name, uploaded.getvalue()) for uploaded in price_uploads) if price_uploads else tuple()
    trade_payloads = tuple((uploaded.name, uploaded.getvalue()) for uploaded in trade_uploads) if trade_uploads else tuple()

    prices, trades = load_market_data(price_payloads, trade_payloads, dataset_name)

    with st.sidebar:
        default_prices = load_default_csvs("prices", dataset_name)
        default_trades = load_default_csvs("trades", dataset_name)
        price_count = len(price_payloads) if price_payloads else len(default_prices)
        trade_count = len(trade_payloads) if trade_payloads else len(default_trades)
        data_source = "uploaded files" if price_payloads or trade_payloads else dataset_name
        st.info(f"Loaded {price_count} price file(s) and {trade_count} trade file(s) from {data_source}.")

    return prices, trades


def render_metric_cards(prices: pd.DataFrame, trades: pd.DataFrame, option_analytics: pd.DataFrame) -> None:
    quote_rows = len(prices)
    trade_rows = len(trades)
    unique_products = prices["product"].nunique()
    option_symbols = int(prices["is_option"].sum() > 0) * prices.loc[prices["is_option"], "product"].nunique()

    cols = st.columns(5)
    cols[0].metric("Quote rows", f"{quote_rows:,}")
    cols[1].metric("Trade rows", f"{trade_rows:,}")
    cols[2].metric("Products", f"{unique_products}")
    cols[3].metric("Voucher strikes", f"{option_symbols}")
    cols[4].metric("Days loaded", f"{prices['day'].nunique()}")

    if not option_analytics.empty:
        latest_snapshot = option_analytics.loc[option_analytics["global_ts"] == option_analytics["global_ts"].max()]
        cols = st.columns(4)
        cols[0].metric("Mean IV", f"{option_analytics['implied_vol'].dropna().mean():.3f}")
        cols[1].metric("Latest avg mispricing", f"{latest_snapshot['mispricing'].dropna().mean():.2f}")
        cols[2].metric("Largest abs mispricing", f"{latest_snapshot['mispricing'].abs().dropna().max():.2f}")
        cols[3].metric("Underlying last mid", f"{get_product_view(prices, DEFAULT_UNDERLYING)['mid_price'].dropna().iloc[-1]:.2f}")


def main() -> None:
    render_header()

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
        prices, trades = render_inputs()
    except Exception as exc:
        st.error(f"Could not load the uploaded CSVs. Details: {exc}")
        st.stop()

    products = sorted(prices["product"].dropna().astype(str).unique().tolist())
    option_symbols = sorted([symbol for symbol in products if is_option_symbol(symbol)], key=lambda s: extract_strike(s) or 0)
    non_option_symbols = [symbol for symbol in products if not is_option_symbol(symbol)]

    round_number = st.sidebar.number_input("Round number for option TTE", min_value=1, max_value=8, value=3, step=1)
    option_analytics = build_option_analytics(prices, DEFAULT_UNDERLYING, int(round_number))

    render_metric_cards(prices, trades, option_analytics)

    product_tab, trader_tab, compare_tab, options_tab, pairs_tab, tables_tab = st.tabs(
        ["Product View", "Trader PnL", "Compare Products", "Options", "Pairs / Cointegration", "Data Tables"]
    )

    with product_tab:
        default_product_index = products.index(DEFAULT_UNDERLYING) if DEFAULT_UNDERLYING in products else 0
        selected_product = st.selectbox("Product", products, index=default_product_index)
        product_prices = get_product_view(prices, selected_product)
        product_trades = get_trade_view(trades, selected_product)
        trade_context = build_trade_context(product_prices, product_trades)

        cols = st.columns(4)
        cols[0].metric("Last mid", f"{product_prices['mid_price'].dropna().iloc[-1]:.2f}")
        cols[1].metric("Median spread", f"{product_prices['spread_1'].dropna().median():.2f}")
        cols[2].metric("Book imbalance mean", f"{product_prices['book_imbalance'].dropna().mean():.3f}")
        cols[3].metric("Trade vs mid mean", f"{trade_context['trade_vs_mid'].dropna().mean():.2f}" if not trade_context.empty else "n/a")

        st.plotly_chart(build_product_dashboard(prices, trades, selected_product), use_container_width=True, config=PLOTLY_CONFIG)

        left, right = st.columns(2)
        left.subheader("Latest Quotes")
        left.dataframe(
            product_prices[
                ["day", "timestamp", "bid_price_1", "bid_volume_1", "ask_price_1", "ask_volume_1", "mid_price", "spread_1"]
            ].tail(15).round(3),
            use_container_width=True,
            hide_index=True,
        )
        right.subheader("Largest Trades")
        if product_trades.empty:
            right.info("No trades loaded for this product.")
        else:
            right.dataframe(
                product_trades[["day", "timestamp", "price", "quantity", "notional", "buyer_display", "seller_display"]]
                .sort_values(["quantity", "notional"], ascending=[False, False])
                .head(15)
                .round(3),
                use_container_width=True,
                hide_index=True,
            )

    with trader_tab:
        default_product_index = products.index(DEFAULT_UNDERLYING) if DEFAULT_UNDERLYING in products else 0
        trader_product = st.selectbox("Product for trader PnL", products, index=default_product_index, key="trader_product")
        trader_summary = build_trader_pnl_summary(prices, trades, trader_product)
        trader_ledger = build_trader_position_ledger(prices, trades, trader_product)

        if trader_summary.empty:
            st.info("No named traders were found for this product in the loaded trade data.")
        else:
            cols = st.columns(4)
            cols[0].metric("Named traders", f"{len(trader_summary)}")
            cols[1].metric("Best PnL", f"{trader_summary['total_pnl'].max():.2f}")
            cols[2].metric("Worst PnL", f"{trader_summary['total_pnl'].min():.2f}")
            cols[3].metric("Latest mark", f"{trader_summary['mark_price'].iloc[0]:.2f}")

            st.caption(
                "PnL is computed as net cash flow from fills plus remaining inventory marked to the latest available quoted mid for the selected product."
            )
            st.plotly_chart(build_trader_pnl_bar_chart(trader_summary, trader_product), use_container_width=True, config=PLOTLY_CONFIG)

            selected_trader = st.selectbox("Trader to inspect", trader_summary["trader"].tolist(), key="selected_trader")
            st.plotly_chart(
                build_trader_pnl_timeseries(trader_ledger, selected_trader, trader_product),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

            left, right = st.columns(2)
            left.subheader("Trader PnL Table")
            left.dataframe(trader_summary.round(4), use_container_width=True, hide_index=True)
            right.subheader(f"{selected_trader} Trade Ledger")
            right.dataframe(
                trader_ledger.loc[trader_ledger["trader"] == selected_trader][
                    ["day", "timestamp", "side", "price", "quantity", "signed_quantity", "position", "mark_price", "total_pnl"]
                ].round(4),
                use_container_width=True,
                hide_index=True,
            )

    with compare_tab:
        compare_defaults = [DEFAULT_UNDERLYING, option_symbols[0]] if option_symbols else products[:2]
        selected_symbols = st.multiselect(
            "Products to overlay",
            products,
            default=[symbol for symbol in compare_defaults if symbol in products],
            max_selections=4,
        )
        value_mode = st.radio("Overlay mode", ["Mid price", "Normalized to 100", "Z-score"], horizontal=True)
        show_trades = st.checkbox("Show trade markers when plotting raw mid prices", value=False)

        if len(selected_symbols) < 2:
            st.info("Select at least two products to compare.")
        else:
            st.plotly_chart(
                build_overlay_figure(prices, trades, selected_symbols, value_mode, show_trades),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
            st.caption(
                "Use raw mids when products trade on comparable scales. Use normalized or z-scored overlays when you want shape comparison across the underlying, Hydrogel, and vouchers."
            )

    with options_tab:
        if option_analytics.empty:
            st.info("No option-style `VEV_*` products were found in the loaded prices data.")
        else:
            default_option = "VEV_5300" if "VEV_5300" in option_symbols else option_symbols[0]
            selected_option = st.selectbox("Voucher to inspect", option_symbols, index=option_symbols.index(default_option))
            snapshot_mode = st.radio("Strike surface view", ["Latest", "Average"], horizontal=True)

            selected_series = option_analytics.loc[option_analytics["symbol"] == selected_option].dropna(subset=["option_mid"])
            last_row = selected_series.iloc[-1]
            cols = st.columns(5)
            cols[0].metric("Strike", f"{int(last_row['strike'])}")
            cols[1].metric("Last option mid", f"{last_row['option_mid']:.2f}")
            cols[2].metric("Last IV", f"{last_row['implied_vol']:.3f}" if pd.notna(last_row["implied_vol"]) else "n/a")
            cols[3].metric("BS fair value", f"{last_row['bs_fair_value']:.2f}" if pd.notna(last_row["bs_fair_value"]) else "n/a")
            cols[4].metric("Mispricing", f"{last_row['mispricing']:.2f}" if pd.notna(last_row["mispricing"]) else "n/a")

            st.plotly_chart(
                build_option_timeseries_figure(option_analytics, selected_option),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
            st.plotly_chart(
                build_options_curve_figure(option_analytics, snapshot_mode),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
            st.caption("Time to expiry uses the same assumption as your strategy code: `(8 - round_number) / 365` years.")

            latest_surface = (
                option_analytics.loc[option_analytics["global_ts"] == option_analytics["global_ts"].max()]
                .sort_values("strike")[
                    ["symbol", "strike", "option_mid", "intrinsic_value", "time_value", "implied_vol", "bs_fair_value", "mispricing"]
                ]
                .round(4)
            )
            st.subheader("Latest Voucher Surface Table")
            st.dataframe(latest_surface, use_container_width=True, hide_index=True)

    with pairs_tab:
        left_col, right_col = st.columns(2)
        left_default = products.index(DEFAULT_PAIR_LEFT) if DEFAULT_PAIR_LEFT in products else 0
        right_default = products.index(DEFAULT_PAIR_RIGHT) if DEFAULT_PAIR_RIGHT in products else min(1, len(products) - 1)
        pair_left = left_col.selectbox("Left product", products, index=left_default, key="pair_left")
        pair_right_choices = [symbol for symbol in products if symbol != pair_left]
        pair_right_default_symbol = DEFAULT_PAIR_RIGHT if DEFAULT_PAIR_RIGHT in pair_right_choices else pair_right_choices[0]
        pair_right = right_col.selectbox(
            "Right product",
            pair_right_choices,
            index=pair_right_choices.index(pair_right_default_symbol),
            key="pair_right",
        )

        try:
            pair = compute_pair_analysis(prices, pair_left, pair_right)
            cols = st.columns(6)
            cols[0].metric("Correlation", f"{pair.correlation:.3f}")
            cols[1].metric("Beta", f"{pair.beta:.4f}")
            cols[2].metric("Latest z-score", f"{pair.latest_zscore:.2f}")
            cols[3].metric("ADF-style stat", f"{pair.adf_style_stat:.2f}")
            cols[4].metric("Half-life", f"{pair.half_life:.1f}" if pair.half_life is not None else "n/a")
            cols[5].metric("Zero crossings", f"{pair.zero_crossings}")

            st.plotly_chart(build_pairs_figure(pair), use_container_width=True, config=PLOTLY_CONFIG)
            st.caption(
                f"Hedge equation: `{pair.left_symbol} ~= {pair.alpha:.3f} + {pair.beta:.4f} * {pair.right_symbol}`. "
                "The ADF-style residual stat is a lightweight stationarity proxy: more negative suggests a tighter mean-reverting spread."
            )

            ranking_universe = st.multiselect(
                "Pairs ranking universe",
                products,
                default=[symbol for symbol in [DEFAULT_UNDERLYING, DEFAULT_PAIR_RIGHT, *option_symbols[:4]] if symbol in products],
            )
            if len(ranking_universe) >= 2:
                ranking = build_pair_ranking_table(prices, tuple(ranking_universe))
                if ranking.empty:
                    st.info("No valid overlapping pairs were found in the selected universe.")
                else:
                    st.subheader("Pairs Ranking")
                    st.dataframe(ranking.round(4), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not compute the pairs analysis. Details: {exc}")

    with tables_tab:
        st.subheader("Products")
        st.dataframe(
            pd.DataFrame(
                {
                    "product": products,
                    "display_name": [pretty_symbol(symbol) for symbol in products],
                    "type": ["voucher" if is_option_symbol(symbol) else "underlying/spot" for symbol in products],
                    "strike": [extract_strike(symbol) for symbol in products],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Latest Quotes Across All Products")
        latest_quotes = (
            prices.sort_values(["global_ts", "product"])
            .groupby("product", as_index=False)
            .tail(1)[["product", "day", "timestamp", "bid_price_1", "ask_price_1", "mid_price", "spread_1", "source_file"]]
            .sort_values("product")
        )
        st.dataframe(latest_quotes.round(4), use_container_width=True, hide_index=True)

        if not trades.empty:
            st.subheader("Largest Trades Across All Products")
            largest_trades = trades[
                ["day", "timestamp", "symbol", "price", "quantity", "notional", "buyer_display", "seller_display", "source_file"]
            ].sort_values(["quantity", "notional"], ascending=[False, False])
            st.dataframe(largest_trades.head(30).round(4), use_container_width=True, hide_index=True)
        else:
            st.info("No trades are loaded.")


if __name__ == "__main__":
    main()

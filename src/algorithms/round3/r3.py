import json
import math
from abc import abstractmethod
from collections import deque
from statistics import NormalDist, median
from typing import TypeAlias

from datamodel import Order, Symbol, TradingState


JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

NORMAL = NormalDist(mu=0.0, sigma=1.0)

UNDERLYING = "VELVETFRUIT_EXTRACT"

# Backtest -2 / historical day 2: 6.0
# Round 3 final/live submission: 5.0
INITIAL_DAYS_TO_EXPIRY = 6.0

DAYS_PER_YEAR = 365
DAY_LENGTH = 1_000_000

OPTION_LIMIT = 300
UNDERLYING_LIMIT = 200

SURFACE_STRIKES: dict[Symbol, int] = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}

TRADE_STRIKES: dict[Symbol, int] = {
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
}

SMILE_COEFFS = [0.02721888, 0.00285325, 0.27239067]

SOFT_OPTION_LIMIT = 220
ENABLE_DELTA_HEDGE = False


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return S * NORMAL.cdf(d1) - K * math.exp(-r * T) * NORMAL.cdf(d2)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return NORMAL.cdf(d1)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return S * math.sqrt(T) * NORMAL.pdf(d1)


def implied_vol(
    target_price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    low: float = 0.0001,
    high: float = 5.0,
) -> float | None:
    if target_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None

    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    if target_price < intrinsic - 1e-9 or target_price > S + 1e-9:
        return None

    low_price = bs_call(S, K, T, r, low)
    high_price = bs_call(S, K, T, r, high)

    if target_price <= low_price:
        return low
    if target_price >= high_price:
        return high

    lo = low
    hi = high

    for _ in range(55):
        mid = 0.5 * (lo + hi)
        price = bs_call(S, K, T, r, mid)
        if price < target_price:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def smile_moneyness(S: float, K: float, T: float) -> float:
    if S <= 0 or K <= 0 or T <= 0:
        return 0.0
    return math.log(K / S) / math.sqrt(T)


def fitted_smile_iv(S: float, K: float, T: float) -> float:
    m = smile_moneyness(S, K, T)
    a, b, c = SMILE_COEFFS
    return max(0.01, min(3.0, a * m * m + b * m + c))


def get_tte_years(state: TradingState) -> float:
    timestamp = int(getattr(state, "timestamp", 0))
    fraction_elapsed_today = min(1.0, max(0.0, timestamp / DAY_LENGTH))
    days_left = INITIAL_DAYS_TO_EXPIRY - fraction_elapsed_today
    return max(1e-6, days_left / DAYS_PER_YEAR)


def get_book(state: TradingState, symbol: Symbol) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    if symbol not in state.order_depths:
        return None

    depth = state.order_depths[symbol]
    buys = sorted(depth.buy_orders.items(), reverse=True)
    sells = sorted(depth.sell_orders.items())

    if not buys or not sells:
        return None

    return buys, sells


def get_mid(state: TradingState, symbol: Symbol) -> float | None:
    book = get_book(state, symbol)
    if book is None:
        return None

    buys, sells = book
    return (buys[0][0] + sells[0][0]) / 2.0


class LiveSurface:
    def __init__(self) -> None:
        self.T: float = 1e-6
        self.S: float | None = None
        self.base_iv: dict[Symbol, float] = {}
        self.mid_iv: dict[Symbol, float] = {}
        self.fair_iv: dict[Symbol, float] = {}
        self.fair_price: dict[Symbol, float] = {}
        self.residual: dict[Symbol, float] = {}
        self.delta: dict[Symbol, float] = {}
        self.vega: dict[Symbol, float] = {}
        self.noise: float = 0.01
        self.global_shift: float = 0.0

    @staticmethod
    def build(state: TradingState) -> "LiveSurface":
        surface = LiveSurface()
        surface.T = get_tte_years(state)

        S = get_mid(state, UNDERLYING)
        if S is None:
            return surface

        surface.S = S
        T = surface.T
        r = 0.0

        raw: list[tuple[Symbol, float, float]] = []

        for symbol, strike in SURFACE_STRIKES.items():
            book = get_book(state, symbol)
            if book is None:
                continue

            buys, sells = book
            mid_price = (buys[0][0] + sells[0][0]) / 2.0

            K = float(strike)
            base_iv = fitted_smile_iv(S, K, T)
            mid_iv = implied_vol(mid_price, S, K, T, r)

            if mid_iv is None:
                continue

            delta = bs_delta(S, K, T, r, base_iv)
            vega = bs_vega(S, K, T, r, base_iv)

            surface.base_iv[symbol] = base_iv
            surface.mid_iv[symbol] = mid_iv
            surface.delta[symbol] = delta
            surface.vega[symbol] = vega

            if vega >= 0.5 and 0.04 <= delta <= 0.96:
                raw.append((symbol, mid_iv, base_iv))

        if not raw:
            return surface

        shifts_by_symbol = {
            symbol: mid_iv - base_iv
            for symbol, mid_iv, base_iv in raw
        }

        all_shifts = list(shifts_by_symbol.values())
        surface.global_shift = median(all_shifts)

        abs_resids: list[float] = []

        for symbol, strike in SURFACE_STRIKES.items():
            if symbol not in surface.base_iv:
                continue

            other_shifts = [
                shift for other_symbol, shift in shifts_by_symbol.items()
                if other_symbol != symbol
            ]

            live_shift = median(other_shifts) if len(other_shifts) >= 2 else surface.global_shift

            base_iv = surface.base_iv[symbol]
            fair_iv = max(0.01, min(3.0, base_iv + live_shift))
            K = float(strike)

            surface.fair_iv[symbol] = fair_iv
            surface.fair_price[symbol] = bs_call(S, K, T, r, fair_iv)

            if symbol in surface.mid_iv:
                resid = surface.mid_iv[symbol] - fair_iv
                surface.residual[symbol] = resid
                abs_resids.append(abs(resid))

        if abs_resids:
            surface.noise = max(0.004, median(abs_resids))

        return surface


class Strategy:
    def __init__(self, symbol: Symbol, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders: list[Order] = []

    @abstractmethod
    def act(self, state: TradingState, surface: LiveSurface) -> None:
        raise NotImplementedError

    def run(self, state: TradingState, surface: LiveSurface) -> tuple[list[Order], int]:
        self.orders = []
        self.act(state, surface)
        return self.orders, 0

    def buy(self, price: int, quantity: int) -> None:
        if quantity > 0:
            self.orders.append(Order(self.symbol, int(price), int(quantity)))

    def sell(self, price: int, quantity: int) -> None:
        if quantity > 0:
            self.orders.append(Order(self.symbol, int(price), -int(quantity)))

    def position(self, state: TradingState) -> int:
        return state.position.get(self.symbol, 0)

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass


class VoucherRelativeIVStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, strike: int) -> None:
        super().__init__(symbol, limit)
        self.strike = strike
        self.resid_history: deque[float] = deque(maxlen=80)

    def act(self, state: TradingState, surface: LiveSurface) -> None:
        if surface.S is None:
            return

        if self.symbol not in surface.fair_iv or self.symbol not in surface.fair_price:
            return

        book = get_book(state, self.symbol)
        if book is None:
            return

        buys, sells = book
        best_bid, best_bid_vol = buys[0]
        best_ask, best_ask_vol = sells[0]

        S = surface.S
        T = surface.T
        K = float(self.strike)
        r = 0.0

        fair_iv = surface.fair_iv[self.symbol]
        fair_price = surface.fair_price[self.symbol]
        delta = bs_delta(S, K, T, r, fair_iv)
        vega = bs_vega(S, K, T, r, fair_iv)

        if vega < 0.35:
            return

        if delta < 0.04 or delta > 0.96:
            return

        bid_iv = implied_vol(best_bid, S, K, T, r)
        ask_iv = implied_vol(best_ask, S, K, T, r)

        if bid_iv is None or ask_iv is None:
            return

        resid = surface.residual.get(self.symbol, 0.0)
        self.resid_history.append(resid)

        position = self.position(state)
        to_buy = min(self.limit - position, SOFT_OPTION_LIMIT - position)
        to_sell = min(self.limit + position, SOFT_OPTION_LIMIT + position)

        iv_edge = self.iv_open_edge(surface, vega, delta)
        close_iv_edge = max(0.0025, 0.35 * iv_edge)

        price_edge = max(1.0, 0.75 * vega * iv_edge)
        close_price_edge = max(1.0, 0.35 * price_edge)

        max_clip = self.max_clip(vega, delta)

        bid_dev_iv = bid_iv - fair_iv
        ask_dev_iv = ask_iv - fair_iv

        bid_edge_price = best_bid - fair_price
        ask_edge_price = fair_price - best_ask

        if position > 0.88 * SOFT_OPTION_LIMIT:
            qty = min(position, best_bid_vol, max_clip)
            if qty > 0 and bid_edge_price >= -close_price_edge:
                self.sell(best_bid, qty)
            return

        if position < -0.88 * SOFT_OPTION_LIMIT:
            qty = min(-position, -best_ask_vol, max_clip)
            if qty > 0 and ask_edge_price >= -close_price_edge:
                self.buy(best_ask, qty)
            return

        if (
            to_sell > 0
            and resid > 0.4 * iv_edge
            and bid_dev_iv > iv_edge
            and bid_edge_price > price_edge
        ):
            conviction = max(
                bid_dev_iv / iv_edge,
                bid_edge_price / price_edge,
            )
            qty_left = min(to_sell, self.size_from_conviction(conviction, max_clip, position, side=-1))

            for bid, bid_vol in buys:
                if qty_left <= 0:
                    break

                this_iv = implied_vol(bid, S, K, T, r)
                if this_iv is None:
                    break

                this_edge = bid - fair_price
                if this_iv - fair_iv <= iv_edge or this_edge <= price_edge:
                    break

                qty = min(bid_vol, qty_left)
                if qty > 0:
                    self.sell(bid, qty)
                    qty_left -= qty
                    position -= qty

        elif position > 0 and (bid_dev_iv >= -close_iv_edge or bid_edge_price >= -close_price_edge):
            qty = min(position, best_bid_vol, max(1, max_clip // 2))
            if qty > 0:
                self.sell(best_bid, qty)
                position -= qty

        if (
            to_buy > 0
            and resid < -0.4 * iv_edge
            and ask_dev_iv < -iv_edge
            and ask_edge_price > price_edge
        ):
            conviction = max(
                abs(ask_dev_iv) / iv_edge,
                ask_edge_price / price_edge,
            )
            qty_left = min(to_buy, self.size_from_conviction(conviction, max_clip, position, side=1))

            for ask, ask_vol in sells:
                if qty_left <= 0:
                    break

                this_iv = implied_vol(ask, S, K, T, r)
                if this_iv is None:
                    break

                this_edge = fair_price - ask
                if this_iv - fair_iv >= -iv_edge or this_edge <= price_edge:
                    break

                qty = min(-ask_vol, qty_left)
                if qty > 0:
                    self.buy(ask, qty)
                    qty_left -= qty
                    position += qty

        elif position < 0 and (ask_dev_iv <= close_iv_edge or ask_edge_price >= -close_price_edge):
            qty = min(-position, -best_ask_vol, max(1, max_clip // 2))
            if qty > 0:
                self.buy(best_ask, qty)
                position += qty

        self.directional_passive_quote(
            best_bid=best_bid,
            best_ask=best_ask,
            fair_price=fair_price,
            resid=resid,
            iv_edge=iv_edge,
            price_edge=price_edge,
            position=position,
        )

    def directional_passive_quote(
        self,
        best_bid: int,
        best_ask: int,
        fair_price: float,
        resid: float,
        iv_edge: float,
        price_edge: float,
        position: int,
    ) -> None:
        to_buy = min(self.limit - position, SOFT_OPTION_LIMIT - position)
        to_sell = min(self.limit + position, SOFT_OPTION_LIMIT + position)

        passive_clip = 5
        inv_ratio = position / SOFT_OPTION_LIMIT if SOFT_OPTION_LIMIT else 0.0

        if resid < -0.75 * iv_edge and to_buy > 0:
            quote = min(best_bid + 1, int(round(fair_price - 0.5 * price_edge)))
            if quote < best_ask:
                size = int(round(passive_clip * max(0.25, 1.0 - max(0.0, inv_ratio))))
                self.buy(quote, min(max(1, size), to_buy))

        elif resid > 0.75 * iv_edge and to_sell > 0:
            quote = max(best_ask - 1, int(round(fair_price + 0.5 * price_edge)))
            if quote > best_bid:
                size = int(round(passive_clip * max(0.25, 1.0 + min(0.0, inv_ratio))))
                self.sell(quote, min(max(1, size), to_sell))

    def iv_open_edge(self, surface: LiveSurface, vega: float, delta: float) -> float:
        edge = max(0.006, 1.15 * surface.noise)

        if vega < 1.0:
            edge += 0.005

        if delta < 0.12 or delta > 0.88:
            edge += 0.006
        elif delta < 0.20 or delta > 0.80:
            edge += 0.003

        return edge

    def max_clip(self, vega: float, delta: float) -> int:
        if delta < 0.12 or delta > 0.88:
            return 8

        if self.strike == 5300:
            return 28

        if self.strike in (5200, 5400):
            return 20

        if self.strike in (5100, 5500):
            return 14

        return 8

    def size_from_conviction(self, conviction: float, max_clip: int, position: int, side: int) -> int:
        conviction = max(1.0, min(4.0, conviction))
        size = int(round(max_clip * conviction / 2.2))
        size = max(1, min(max_clip, size))

        inv_ratio = position / SOFT_OPTION_LIMIT if SOFT_OPTION_LIMIT else 0.0

        if side == 1 and inv_ratio > 0:
            size = int(round(size * max(0.25, 1.0 - inv_ratio)))

        if side == -1 and inv_ratio < 0:
            size = int(round(size * max(0.25, 1.0 + inv_ratio)))

        return max(1, size)

    def save(self) -> JSON:
        return {"resid_history": list(self.resid_history)}

    def load(self, data: JSON) -> None:
        if not isinstance(data, dict):
            return

        vals = data.get("resid_history", [])
        if isinstance(vals, list):
            self.resid_history = deque(
                [float(x) for x in vals if isinstance(x, (int, float))],
                maxlen=80,
            )


def hedge_delta(
    state: TradingState,
    surface: LiveSurface,
    option_orders: dict[Symbol, list[Order]],
) -> list[Order]:
    if not ENABLE_DELTA_HEDGE:
        return []

    book = get_book(state, UNDERLYING)
    if book is None:
        return []

    buys, sells = book
    best_bid, best_bid_vol = buys[0]
    best_ask, best_ask_vol = sells[0]

    total_delta = 0.0

    for symbol in TRADE_STRIKES:
        if symbol not in surface.delta:
            continue

        pos = state.position.get(symbol, 0)
        pending = sum(order.quantity for order in option_orders.get(symbol, []))
        total_delta += (pos + pending) * surface.delta[symbol]

    current_pos = state.position.get(UNDERLYING, 0)
    target_pos = int(round(-total_delta))
    target_pos = max(-UNDERLYING_LIMIT, min(UNDERLYING_LIMIT, target_pos))

    needed = target_pos - current_pos

    if abs(needed) < 50:
        return []

    max_clip = 25
    orders: list[Order] = []

    if needed > 0:
        qty = min(needed, -best_ask_vol, UNDERLYING_LIMIT - current_pos, max_clip)
        if qty > 0:
            orders.append(Order(UNDERLYING, int(best_ask), int(qty)))

    elif needed < 0:
        qty = min(-needed, best_bid_vol, UNDERLYING_LIMIT + current_pos, max_clip)
        if qty > 0:
            orders.append(Order(UNDERLYING, int(best_bid), -int(qty)))

    return orders


class Trader:
    def __init__(self) -> None:
        self.strategies: dict[Symbol, VoucherRelativeIVStrategy] = {
            symbol: VoucherRelativeIVStrategy(symbol, OPTION_LIMIT, strike)
            for symbol, strike in TRADE_STRIKES.items()
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        try:
            old_data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            old_data = {}

        if not isinstance(old_data, dict):
            old_data = {}

        surface = LiveSurface.build(state)

        result: dict[Symbol, list[Order]] = {}
        new_data: dict[str, JSON] = {}

        for symbol, strategy in self.strategies.items():
            saved = old_data.get(symbol)
            if saved is not None:
                strategy.load(saved)

            if symbol in state.order_depths and UNDERLYING in state.order_depths:
                orders, _ = strategy.run(state, surface)
                if orders:
                    result[symbol] = orders

            new_data[symbol] = strategy.save()

        hedge_orders = hedge_delta(state, surface, result)
        if hedge_orders:
            result[UNDERLYING] = hedge_orders

        trader_data = json.dumps(new_data, separators=(",", ":"))
        return result, 0, trader_data

import json
from typing import cast
from abc import abstractmethod
from collections import deque
from typing import Any, TypeAlias
import numpy as np
import math
from statistics import NormalDist


from datamodel import Order, OrderDepth, Symbol, TradingState

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

INITIAL_DAYS_TO_EXPIRY = 5.0 #cuz 2nd day, 5 for 3rd day

DAYS_PER_YEAR = 365
DAY_LENGTH = 1_000_000

OPTION_STRIKES: dict[Symbol, int] = {
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

def DELTA(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return NormalDist(mu=0.0, sigma=1.0).cdf(d1)

def get_moneyness(S: float, K: float, T: float) -> float:
    if S <= 0 or K <= 0 or T <= 0:
        return 0.0


    return math.log(K / S) / math.sqrt(T)

def BS_CALL(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)

    if sigma <= 0:
        return max(S - K * math.exp(-r * T), 0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    normal = NormalDist(mu=0.0, sigma=1.0)

    return S * normal.cdf(d1) - K * math.exp(-r * T) * normal.cdf(d2)

def VEGA(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * math.sqrt(T) * NormalDist(0, 1).pdf(d1)

def IV(target_price, S, K, T, r, low=0.001, high=5.0):
    if T<0 or S<0 or K<0:
        return -1

    low, high = 0.0001, 5.0
    sigma = 0.2  # Initial guess

    for _ in range(100):
        price = BS_CALL(S, K, T, r, sigma)
        vega = VEGA(S, K, T, r, sigma)

        # 1. Check for "Poor" Vega
        if abs(vega) < 1e-6:
            # Fallback to Bisection step
            if price < target_price:
                low = sigma
            else:
                high = sigma
            sigma = (low + high) / 2
        else:
            # 2. Proceed with Newton step
            diff = target_price - price
            if abs(diff) < 1e-8:
                return sigma

            new_sigma = sigma + diff / vega

            # 3. Boundary Check (Stay within bounds)
            if new_sigma <= low or new_sigma >= high:
                sigma = (low + high) / 2 # Midpoint fallback
            else:
                sigma = new_sigma

    return sigma

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders: list[Order] = []
        self.conversions = 0

        self.act(state)

        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def sell_item(self,symbol:str ,price:int, quantity: int) ->None:
        self.orders.append(Order(symbol, price, -quantity))

    def buy_item(self,symbol:str ,price:int, quantity: int) ->None:
        self.orders.append(Order(symbol, price, quantity))

    def get_book(self, state:TradingState, symbol:str) :
        return state.order_depths[symbol]

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

    ################HELPER#METHODS###################
    def get_position(self, state: TradingState) -> int:
        return state.position.get(self.symbol, 0)

    def get_best_bid_ask(self, state: TradingState):
        order_depth = state.order_depths[self.symbol]

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        if not buy_orders or not sell_orders:
            return None

        best_bid, best_bid_vol = buy_orders[0]
        best_ask, best_ask_vol = sell_orders[0]

        return best_bid, best_bid_vol, best_ask, best_ask_vol

    def get_mid(self, state: TradingState, symbol: Symbol | None = None) -> float | None:
        symbol = symbol or self.symbol
        if symbol not in state.order_depths:
            return None

        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        if not buy_orders or not sell_orders:
            return None

        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]

        return (best_bid + best_ask) / 2

class HydrogelPackStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        self.mid_history: deque[float] = deque(maxlen=200)
        self.last_mid: float | None = None

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        if not buy_orders or not sell_orders:
            return

        best_bid, best_bid_vol = buy_orders[0]
        best_ask, best_ask_vol = sell_orders[0]

        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        self.mid_history.append(mid)

        if len(self.mid_history) < 20:
            self.last_mid = mid
            return

        rolling_mean = sum(self.mid_history) / len(self.mid_history)

        # -----------------------------
        # 1. Microprice adjustment
        # -----------------------------
        bid_vol = max(1, best_bid_vol)
        ask_vol = max(1, -best_ask_vol)

        microprice = (
            best_ask * bid_vol + best_bid * ask_vol
        ) / (bid_vol + ask_vol)

        micro_signal = microprice - mid

        # -----------------------------
        # 2. Mean reversion signal
        # -----------------------------
        mean_reversion_signal = rolling_mean - mid

        # Small momentum filter.
        # If price is moving hard in one direction, quote less aggressively.
        momentum = 0.0
        if self.last_mid is not None:
            momentum = mid - self.last_mid

        self.last_mid = mid

        # -----------------------------
        # 3. Fair value
        # -----------------------------
        inventory_skew = 2.0 * (position / self.limit)

        fair = (
            mid
            + 0* micro_signal
            + 0.58 * mean_reversion_signal
            - inventory_skew
        )

        # -----------------------------
        # 4. Do not trade bad spreads
        # -----------------------------
        # Hydrogel earns money from spread capture.
        # If spread is too tight, there is not enough edge.
        if spread < 4:
            return

        # -----------------------------
        # 5. Rare liquidity-taking only on huge dislocations
        # -----------------------------
        take_edge = 6.0
        max_take_clip = 8

        # Buy only if ask is clearly below fair.
        if best_ask <= fair - take_edge and to_buy > 0:
            qty = min(to_buy, -best_ask_vol, max_take_clip)
            if qty > 0:
                self.buy(best_ask, qty)
                to_buy -= qty

        # Sell only if bid is clearly above fair.
        if best_bid >= fair + take_edge and to_sell > 0:
            qty = min(to_sell, best_bid_vol, max_take_clip)
            if qty > 0:
                self.sell(best_bid, qty)
                to_sell -= qty

        # -----------------------------
        # 6. Passive market making
        # -----------------------------
        base_clip = 8
        pos_ratio = position / self.limit

        buy_size = int(base_clip * max(0.1, 1.0 - max(pos_ratio, 0.0)))
        sell_size = int(base_clip * max(0.1, 1.0 + min(pos_ratio, 0.0)))

        buy_size = min(buy_size, to_buy)
        sell_size = min(sell_size, to_sell)

        if spread >= 3:
            bid_quote = best_bid + 1
            ask_quote = best_ask - 1

            # If inventory is getting large, stop adding to the bad side.
            if position > 0.5 * self.limit:
                buy_size = 0
            if position < -0.5 * self.limit:
                sell_size = 0

            # If very long, sell more aggressively.
            if position > 0.75 * self.limit:
                ask_quote = best_ask - 2

            # If very short, buy more aggressively.
            if position < -0.75 * self.limit:
                bid_quote = best_bid + 2

            if bid_quote < ask_quote:
                if buy_size > 0:
                    self.buy(bid_quote, buy_size)
                if sell_size > 0:
                    self.sell(ask_quote, sell_size)

    def save(self) -> JSON:
        return {
            "mid_history": list(self.mid_history),
            "last_mid": self.last_mid,
        }

    def load(self, data: JSON) -> None:
        if not isinstance(data, dict):
            return

        mids = data.get("mid_history", [])
        if isinstance(mids, list):
            self.mid_history = deque([float(x) for x in mids], maxlen=200)

        last_mid = data.get("last_mid", None)
        if isinstance(last_mid, int | float):
            self.last_mid = float(last_mid)

class VelvetFruitExtractStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.mid_history: deque[float] = deque(maxlen=20)

    def act(self, state: TradingState) -> None:
        ...
    def save(self) -> JSON:
        return {
            "mid_history": list(self.mid_history)
        }

    def load(self, data: JSON) -> None:
        if not isinstance(data, dict):
            return

        mids = data.get("mid_history", [])
        if isinstance(mids, list):
            self.mid_history = deque([float(x) for x in mids], maxlen=100)


class VelvetFruitExtractVoucherStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, strike_price: int) -> None:
        super().__init__(symbol, limit)
        self.strike_price = strike_price
        self.iv_history: deque[float] = deque(maxlen=100)
        # Your fitted IV smile coefficients:
        # fair_iv = a * moneyness^2 + b * moneyness + c
        self.smile_a = 0.02721888
        self.smile_b = 0.00285325
        self.smile_c = 0.27239067

    def collect_iv_points(self, state: TradingState) -> list[dict[str, float | str]]:
        points = []
        UNDERLYING = "VELVETFRUIT_EXTRACT"

        S = self.get_mid(state, UNDERLYING)
        if S is None:
            return points
        T = self.get_tte_years(state)
        r = 0.0

        for symbol, strike in OPTION_STRIKES.items():
            book = self.get_book(state, symbol)

            if book is None:
                continue

            buy_orders, sell_orders = book

            best_bid = buy_orders[0][0]
            best_ask = sell_orders[0][0]
            mid_price = (best_bid + best_ask) / 2

            K = float(strike)

            mid_iv = IV(mid_price, S, K, T, r)
            bid_iv = IV(best_bid, S, K, T, r)
            ask_iv = IV(best_ask, S, K, T, r)

            if mid_iv is None or bid_iv is None or ask_iv is None:
                continue

            m = get_moneyness(S, K, T)

            # Filter out useless extreme options.
            #delta = bs_delta(S, K, T, r, mid_iv)
            vega = VEGA(S, K, T, r, mid_iv)

            if vega < 0.25:
                continue

            #if delta < 0.03 or delta > 0.97:
            #    continue

            points.append({
                "symbol": symbol,
                "strike": K,
                "moneyness": m,
                "mid_iv": mid_iv,
                "bid_iv": bid_iv,
                "ask_iv": ask_iv,
                #"delta": delta,
                "vega": vega,
            })

        return points

    def fair_iv_from_smile(self, moneyness: float) -> float:
        fair_iv = (
            self.smile_a * (moneyness ** 2)
            + self.smile_b * moneyness
            + self.smile_c
        )

        # Safety clamp so the model does not produce nonsense IVs.
        return max(0.001, min(5.0, fair_iv))

    def get_clip(self, vega: float) -> int:
        return 4

    def hedge_delta_incremental(
        self,
        state: TradingState,
        option_qty: int,
        delta: float,
    ) -> None:
        """
        option_qty > 0 means we bought calls.
        option_qty < 0 means we sold calls.

        For calls:
        bought calls  => positive delta => sell underlying
        sold calls    => negative delta => buy underlying
        """
        UNDERLYING = "VELVETFRUIT_EXTRACT"
        UNDERLYING_LIMIT = 200

        if UNDERLYING not in state.order_depths:
            return

        hedge_qty = int(round(option_qty * delta))

        if hedge_qty == 0:
            return

        book = state.order_depths[UNDERLYING]
        buy_orders = sorted(book.buy_orders.items(), reverse=True)
        sell_orders = sorted(book.sell_orders.items())

        if not buy_orders or not sell_orders:
            return

        underlying_pos = state.position.get(UNDERLYING, 0)

        # If hedge_qty > 0, our option trade added positive delta,
        # so we SELL underlying.
        if hedge_qty > 0:
            qty_left = min(hedge_qty, UNDERLYING_LIMIT + underlying_pos)

            for bid, bid_vol in buy_orders:
                if qty_left <= 0:
                    break

                available = bid_vol
                qty = min(available, qty_left)

                if qty > 0:
                    self.sell_item(UNDERLYING, bid, qty)
                    qty_left -= qty

        # If hedge_qty < 0, our option trade added negative delta,
        # so we BUY underlying.
        elif hedge_qty < 0:
            qty_left = min(-hedge_qty, UNDERLYING_LIMIT - underlying_pos)

            for ask, ask_vol in sell_orders:
                if qty_left <= 0:
                    break

                available = -ask_vol
                qty = min(available, qty_left)

                if qty > 0:
                    self.buy_item(UNDERLYING, ask, qty)
                    qty_left -= qty

    def act(self, state: TradingState) -> None:
        UNDERLYING = "VELVETFRUIT_EXTRACT"
        r = 0.0

        # -----------------------------
        # 1. Get underlying mid price
        # -----------------------------
        S = self.get_mid(state, UNDERLYING)
        if S is None:
            return

        # -----------------------------
        # 2. Get voucher order book
        # -----------------------------
        if self.symbol not in state.order_depths:
            return

        order_depth = state.order_depths[self.symbol]

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        if not buy_orders or not sell_orders:
            return

        best_bid, best_bid_vol = buy_orders[0]
        best_ask, best_ask_vol = sell_orders[0]

        # -----------------------------
        # 3. Compute executable IVs
        # -----------------------------
        K = float(self.strike_price)
        T = self.get_tte_years(state)

        bid_iv = IV(best_bid, S, K, T, r)
        ask_iv = IV(best_ask, S, K, T, r)
        mid_iv = IV((best_bid + best_ask) / 2, S, K, T, r)

        if (
            bid_iv <= 0 or ask_iv <= 0 or mid_iv <= 0
            or not math.isfinite(bid_iv)
            or not math.isfinite(ask_iv)
            or not math.isfinite(mid_iv)
        ):
            return

        self.iv_history.append(mid_iv)

        # -----------------------------
        # 4. Compute fair IV from smile
        # -----------------------------
        moneyness = get_moneyness(S, K, T)
        fair_iv = self.fair_iv_from_smile(moneyness)

        vega = VEGA(S, K, T, r, fair_iv)
        delta = DELTA(S, K, T, r, fair_iv)

        if vega < 0.20:
            return

        # -----------------------------
        # 5. Trade only executable edge
        # -----------------------------
        threshold = 0.005

        position = self.get_position(state)
        to_buy = self.limit - position
        to_sell = self.limit + position

        max_clip = self.get_clip(vega)

        if max_clip <= 0:
            return

        # Use the executable edge, not mid edge.
        buy_edge = fair_iv - ask_iv
        sell_edge = bid_iv - fair_iv

        conviction = 1.0

        if buy_edge > threshold:
            conviction = min(3.0, max(1.0, buy_edge / threshold))
        elif sell_edge > threshold:
            conviction = min(3.0, max(1.0, sell_edge / threshold))

        target_qty = int(max_clip * conviction)

        # -----------------------------
        # 6. Buy only if ask IV is cheap
        # -----------------------------
        if ask_iv < fair_iv - threshold and to_buy > 0:
            qty_left = min(to_buy, target_qty)

            for ask, ask_vol in sell_orders:
                if qty_left <= 0:
                    break

                available = -ask_vol
                qty = min(available, qty_left)

                if qty > 0:
                    self.buy(ask, qty)
                    #self.hedge_delta_incremental(state, option_qty=qty, delta=delta)
                    qty_left -= qty

        # -----------------------------
        # 7. Sell only if bid IV is rich
        # -----------------------------
        elif bid_iv > fair_iv + threshold and to_sell > 0:
            qty_left = min(to_sell, target_qty)

            for bid, bid_vol in buy_orders:
                if qty_left <= 0:
                    break

                available = bid_vol
                qty = min(available, qty_left)

                if qty > 0:
                    self.sell(bid, qty)
                    #self.hedge_delta_incremental(state, option_qty=-qty, delta=delta)
                    qty_left -= qty

    def get_tte_years(self, state: TradingState) -> float:
        timestamp = int(getattr(state, "timestamp", 0))
        fraction_elapsed_today = min(1.0, max(0.0, timestamp / DAY_LENGTH))
        days_left = INITIAL_DAYS_TO_EXPIRY - fraction_elapsed_today
        return max(1e-6, days_left / DAYS_PER_YEAR)

    def save(self) -> JSON:
        return {
            "iv_history": list(self.iv_history)
        }

    def load(self, data: JSON) -> None:
        if not isinstance(data, dict):
            return

        iv_history = data.get("iv_history", [])
        if isinstance(iv_history, list):
            self.iv_history = deque(
                [float(x) for x in iv_history],
                maxlen=100
            )


class Trader:
    def __init__(self) -> None:

        self.strategies: dict[Symbol, Strategy] = {
            "HYDROGEL_PACK": HydrogelPackStrategy("HYDROGEL_PACK", 200),

            #"VEV_4000": VelvetFruitExtractVoucherStrategy("VEV_4000", 300, 4000),
            #"VEV_4500": VelvetFruitExtractVoucherStrategy("VEV_4500", 300, 4500),
            #"VEV_5000": VelvetFruitExtractVoucherStrategy("VEV_5000", 300, 5000),
            "VEV_5100": VelvetFruitExtractVoucherStrategy("VEV_5100", 300, 5100),
            "VEV_5200": VelvetFruitExtractVoucherStrategy("VEV_5200", 300, 5200),
            #"VEV_5300": VelvetFruitExtractVoucherStrategy("VEV_5300", 300, 5300),
            #"VEV_5400": VelvetFruitExtractVoucherStrategy("VEV_5400", 300, 5400),
            #"VEV_5500": VelvetFruitExtractVoucherStrategy("VEV_5500", 300, 5500),
            #"VEV_6000": VelvetFruitExtractVoucherStrategy("VEV_6000", 300, 6000),
            #"VEV_6500": VelvetFruitExtractVoucherStrategy("VEV_6500", 300, 6500),
            #"VELVETFRUIT_EXTRACT": VelvetFruitExtractStrategy("VELVETFRUIT_EXTRACT", 200),
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            if symbol in state.order_depths:
                strategy_orders, strategy_conversions = strategy.run(state)

                for order in strategy_orders:
                    if order.symbol not in orders:
                        orders[order.symbol] = []
                    orders[order.symbol].append(order)

                conversions += strategy_conversions

            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))


        return orders, conversions, trader_data

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

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        pass

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass


class VelvetFruitExtractStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        pass

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass


class VelvetFruitExtractVoucherStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, strike_price: int, ROUND_NUMBER = 3) -> None:
        super().__init__(symbol, limit)
        self.strike_price = strike_price
        self.round_number = ROUND_NUMBER ##################IMPORTANT TO CHANGE AFTER EACH ROUND############
        self.iv_history: deque[float] = deque(maxlen=100)

    def act(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths:
            return

        if "VELVETFRUIT_EXTRACT" not in state.order_depths:
            return

        voucher_mid = self.get_mid(state, self.symbol)
        underlying_mid = self.get_mid(state, "VELVETFRUIT_EXTRACT")

        if voucher_mid is None or underlying_mid is None:
            return

        S = underlying_mid
        K = self.strike_price
        T = max(0, 8 - self.round_number)
        r = 0.0

        market_iv = IV(voucher_mid, S, K, T, r)
        self.iv_history.append(market_iv)

        if len(self.iv_history) < 20:
            return

        fair_iv = sum(self.iv_history) / len(self.iv_history)
        fair_price = BS_CALL(S, K, T, r, fair_iv)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = self.get_position(state)
        to_buy = self.limit - position
        to_sell = self.limit + position

        edge = 2  # tune this

        # If asks are below fair value, buy
        for ask, ask_vol in sell_orders:
            if ask < fair_price - edge and to_buy > 0:
                qty = min(-ask_vol, to_buy)
                self.sell(ask, qty)
                to_buy -= qty

        # If bids are above fair value, sell
        for bid, bid_vol in buy_orders:
            if bid > fair_price + edge and to_sell > 0:
                qty = min(bid_vol, to_sell)
                self.buy(bid, qty)
                to_sell -= qty


    def get_tte_years(self, round_number: int) -> float:
        days_left = max(0, 8 - round_number)
        return days_left / 365

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
            "VELVETFRUIT_EXTRACT": VelvetFruitExtractStrategy("VELVETFRUIT_EXTRACT", 200),
            #"VEV_4000": VelvetFruitExtractVoucherStrategy("VEV_4000", 300, 4000),
            #"VEV_4500": VelvetFruitExtractVoucherStrategy("VEV_4500", 300, 4500),
            "VEV_5000": VelvetFruitExtractVoucherStrategy("VEV_5000", 300, 5000),
            "VEV_5100": VelvetFruitExtractVoucherStrategy("VEV_5100", 300, 5100),
            "VEV_5200": VelvetFruitExtractVoucherStrategy("VEV_5200", 300, 5200),
            "VEV_5300": VelvetFruitExtractVoucherStrategy("VEV_5300", 300, 5300),
            "VEV_5400": VelvetFruitExtractVoucherStrategy("VEV_5400", 300, 5400),
            "VEV_5500": VelvetFruitExtractVoucherStrategy("VEV_5500", 300, 5500),
            "VEV_6000": VelvetFruitExtractVoucherStrategy("VEV_6000", 300, 6000),
            "VEV_6500": VelvetFruitExtractVoucherStrategy("VEV_6500", 300, 6500),
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
                orders[symbol] = strategy_orders
                conversions += strategy_conversions

            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))


        return orders, conversions, trader_data

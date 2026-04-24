import json
from typing import cast
from abc import abstractmethod
from collections import deque
from typing import Any, TypeAlias

from datamodel import Order, OrderDepth, Symbol, TradingState

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

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



class AshCoatedOsmiumStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.mid_history:Any = deque(maxlen = 20)

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)   # highest bid first
        sell_orders = sorted(order_depth.sell_orders.items())               # lowest ask first

        if not buy_orders or not sell_orders:
            return

        best_bid, best_bid_vol = buy_orders[0]
        best_ask, best_ask_vol = sell_orders[0]

        mid = (best_bid + best_ask) / 2

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        max_clip = 12

        # -----------------------------
        # 1. Load / update rolling mean
        # -----------------------------

        self.mid_history.append(mid)

        rolling_mean = sum(self.mid_history) / len(self.mid_history)


        # -----------------------------
        # 2. Mean reversion signal
        # -----------------------------
        deviation = mid - rolling_mean

        # inventory penalty
        # long inventory lowers fair, short inventory raises fair
        fair = rolling_mean - 5.0 * (position / self.limit)

        # threshold for entering trades
        entry_threshold = 1.5
        exit_threshold = 0.5

        # --------------------------------
        # 1. Forced inventory unwind mode
        # --------------------------------
        extreme_long = position > 0.9 * self.limit
        extreme_short = position < -0.9 * self.limit

        if extreme_long:
            qty = min(to_sell, best_bid_vol, max_clip)
            if qty > 0:
                self.sell(best_bid, qty)
                to_sell -= qty

        if extreme_short:
            qty = min(to_buy, -best_ask_vol, max_clip)
            if qty > 0:
                self.buy(best_ask, qty)
                to_buy -= qty

        # -----------------------------
        # 3. TAKE trades if book is good
        # -----------------------------
        # If market is below mean, buy undervalued asks
        if deviation < -entry_threshold:
            for price, volume in sell_orders:
                if to_buy <= 0:
                    break
                if price <= fair:
                    qty = min(to_buy, -volume, max_clip)
                    if qty > 0:
                        self.buy(price, qty)
                        to_buy -= qty

        # If market is above mean, sell overvalued bids
        elif deviation > entry_threshold:
            for price, volume in buy_orders:
                if to_sell <= 0:
                    break
                if price >= fair:
                    qty = min(to_sell, volume, max_clip)
                    if qty > 0:
                        self.sell(price, qty)
                        to_sell -= qty

        # -----------------------------
        # 4. Passive quotes around mean
        # -----------------------------
        bid_size = max(0, min(max_clip, to_buy))
        ask_size = max(0, min(max_clip, to_sell))

        # if below mean -> want to buy lower and sell less aggressively
        if deviation < -entry_threshold:
            bid_quote = min(best_bid + 1, int(fair))
            ask_quote = max(best_ask, int(fair + 2))

        # if above mean -> want to sell higher and buy less aggressively
        elif deviation > entry_threshold:
            bid_quote = min(best_bid, int(fair - 2))
            ask_quote = max(best_ask - 1, int(fair))

        # near mean -> market make symmetrically
        else:
            bid_quote = min(best_bid + 1, int(fair - 1))
            ask_quote = max(best_ask - 1, int(fair + 1))

        # flatten inventory more aggressively when price has reverted
        if position > 0 and deviation >= -exit_threshold:
            ask_quote = min(ask_quote, best_ask)
        if position < 0 and deviation <= exit_threshold:
            bid_quote = max(bid_quote, best_bid)

        # avoid crossing yourself
        if bid_quote < ask_quote:
            if bid_size > 0:
                self.buy(bid_quote, bid_size)
            if ask_size > 0:
                self.sell(ask_quote, ask_size)

    def actWorse(self,state:TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        if not buy_orders or not sell_orders:
            return

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = max(position, 0)  # only sell existing long inventory

        buy_clip = 10
        sell_clip = 10

        bid_depth = sum(volume for _, volume in buy_orders)
        ask_depth = sum(-volume for _, volume in sell_orders)

        imbalance = (
            (bid_depth - ask_depth) / (bid_depth + ask_depth)
            if (bid_depth + ask_depth) != 0 else 0
        )

        likely_upswing = imbalance > 0.1
        likely_downswing = imbalance < -0.1

        best_bid, best_bid_vol = buy_orders[0]
        best_ask, best_ask_vol = sell_orders[0]

        mid = (best_bid + best_ask) / 2

        # buy dips only if order book pressure supports upside
        if likely_upswing:
            for price, volume in sell_orders:
                ask_qty = -volume
                if price <= mid - 5:
                    qty = min(buy_clip, to_buy, ask_qty)
                    if qty > 0:
                        self.buy(price, qty)
                        to_buy -= qty
                    break

        # rare selling only if there is clear downside pressure
        if likely_downswing and to_sell > 0:
            for price, volume in buy_orders:
                bid_qty = volume
                if price >= mid + 5:
                    qty = min(sell_clip, to_sell, bid_qty)
                    if qty > 0:
                        self.sell(price, qty)
                        to_sell -= qty
                    break









    def actTwo(self, state: TradingState) -> None:
        # true value of the item
        true_value = self.get_true_value(state)

        #reading all the orders that have occured till this point
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        # defining present inventory
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        max_clip = 10

        bid_size = max(0, min(max_clip, to_buy))
        ask_size = max(0, min(max_clip, to_sell))

        #array appending whether or not position has pinned to the limit

        #variables indicating how often we are hitting our position limit
        #soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        #hard_liquidate = len(self.window) == self.window_size and all(self.window)

        #defining our max and min buy and sell prices around true value
        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        #taking at fair value when position limit very high
        for price, volume in sell_orders:
            if to_buy<=5 and price == true_value:
                self.buy(price, volume//2)
                to_buy -= volume//2

        # picking out cheap ask prices
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        # if we have enough position to buy, and for the past 10 ticks weve hit our
        # limit then we need to buy more( reducing risk from being stuck at short limit)
        '''if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity'''

        # same as above but for less amount of previous ticks we've hit our limit
        # still need to buy more but less aggressively
        '''if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity'''

        # if we have enough position to buy, then place a bid that beats the most popular bid
        # by 1
        if to_buy > 0 and buy_orders:
            popular_buy_price = buy_orders[0][0]
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, min(bid_size,to_buy))

        #taking at fair value when position limit very high
        for price, volume in buy_orders:
            if to_sell<=5 and price == true_value:
                self.sell(price, volume//2)
                to_sell -= volume//2

        # the following is symmetric for sell side
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        '''if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity'''

        '''if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity'''

        if to_sell > 0 and sell_orders:
            popular_sell_price = sell_orders[0][0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, min(ask_size,to_sell))

    def actOne (self, state: TradingState) -> None:
        # true value of the item
        true_value = self.get_true_value(state)

        #reading all the orders that have occured till this point
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        # defining present inventory
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        max_clip = 10

        bid_size = max(0, min(max_clip, to_buy))
        ask_size = max(0, min(max_clip, to_sell))

        skew = position//self.limit
        true_value = true_value + skew
        #defining our max and min buy and sell prices around true value
        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value


        # picking out cheap ask prices
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity


        # if we have enough position to buy, then place a bid that beats the most popular bid
        # by 1
        if to_buy > 0 and buy_orders:
            popular_buy_price = buy_orders[0][0]
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, min(bid_size,to_buy))

        #taking at fair value when position limit is negative
        for price, volume in buy_orders:
            if to_sell == 0 and price == true_value:
                self.sell(price, volume//2)
                to_sell -= volume//2

        # the following is symmetric for sell side
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and sell_orders:
            popular_sell_price = sell_orders[0][0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, min(ask_size,to_sell))

    def get_true_value(self, state: TradingState) -> int:
        return 10_000

    def save(self) -> JSON:
        return {"mid_history": list(self.mid_history)}


    def load(self, data: JSON) -> None:
        from collections import deque
        self.mid_history = deque(maxlen=20)

        if isinstance(data, dict) and "mid_history" in data:
            hist = data["mid_history"]
            if isinstance(hist, list):
                for x in hist:
                    if isinstance(x, (int, float)):
                        self.mid_history.append(float(x))

class IntarianPepperRootStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.history:Any = deque(maxlen=80)
        self.midPriceSpread:Any = deque(maxlen=80)


    def actOne(self, state:TradingState) -> None:
        order_depth = state.order_depths[self.symbol]

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol,0)

        if(not buy_orders):
           self.buy(10000, 4)
        else:
            self.buy(buy_orders[0][0]+4,5)


    def actThree(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol,0)

        if(len(self.history)>2):
            difference = [self.history[i+1] -self.history[i] for i in range(len(self.history) - 1)]
            avgDifference = sum(difference)/len(difference) if len(difference) != 0 else 0

            currentBestAsk, AskSize  = sell_orders[0] if len(sell_orders) !=0 else (0,0)
            if( difference[-1] - currentBestAsk > 4):
                self.buy(currentBestAsk, AskSize)

        if(len(sell_orders) != 0):
            self.history.append(sell_orders[0][0])
        else:
            self.history.append(0)


    def actTwo(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        if not buy_orders or not sell_orders:
            return

        best_bid, bid_vol = buy_orders[0]
        best_ask, ask_vol = sell_orders[0]
        ask_size = -ask_vol

        position = state.position.get(self.symbol, 0)
        limit = self.limit

        to_buy = limit - position
        to_sell = limit + position   # assuming symmetric long/short limits

        # ---------- MID / MICROPRICE ----------
        mid = (best_bid + best_ask) / 2

        if bid_vol + ask_size == 0:
            microprice = mid
        else:
            microprice = (best_ask * bid_vol + best_bid * ask_size) / (bid_vol + ask_size)

        # ---------- HISTORY ----------
        self.history.append(float(mid))

        short_mom = 0.0
        med_mom = 0.0
        long_mom = 0.0

        if len(self.history) >= 5:
            short_mom = self.history[-1] - self.history[-5]
        if len(self.history) >= 12:
            med_mom = self.history[-1] - self.history[-12]
        if len(self.history) >= 25:
            long_mom = self.history[-1] - self.history[-25]

        momentum = 0.65 * short_mom + 0.25 * med_mom + 0.10 * long_mom

        # ---------- TREND FAIR ----------
        fair = (
            mid
            + 1.35 * momentum
            + 0.20 * (microprice - mid)
            - 0.10 * (position / limit)
        )

        # local dip detector: catches sudden cheap asks relative to recent price
        last_mid = self.history[-2] if len(self.history) >= 2 else mid
        gap_fair = fair - best_ask
        gap_local = last_mid - best_ask
        snipe_signal = 0.7 * gap_fair + 0.3 * gap_local

        # ---------- BUY CLIP ----------
        if momentum >= 8:
            buy_clip = 25
        elif momentum >= 5:
            buy_clip = 18
        elif momentum >= 2:
            buy_clip = 12
        elif momentum > 0:
            buy_clip = 8
        else:
            buy_clip = 4

        # ============================================================
        # 1. SNIPE CHEAP ASKS FIRST
        # ============================================================
        # take obvious underpriced asks, especially in an uptrend
        for price, volume in sell_orders:
            if to_buy <= 0:
                break

            ask_qty = -volume
            edge = fair - price

            # strong dip / mispricing
            if snipe_signal >= 8 and edge >= 3:
                qty = min(to_buy, ask_qty, buy_clip * 2)
                if qty > 0:
                    self.buy(price, qty)
                    to_buy -= qty
                    position += qty

            # normal trend-following take
            elif momentum > 0 and edge >= 1.5:
                qty = min(to_buy, ask_qty, buy_clip)
                if qty > 0:
                    self.buy(price, qty)
                    to_buy -= qty
                    position += qty

        # recompute after takes
        to_buy = limit - position
        to_sell = limit + position

        # ============================================================
        # 2. POST / IMPROVE BID TO BUILD LONG
        # ============================================================
        spread = best_ask - best_bid

        if to_buy > 0:
            # only build aggressively when trend is supportive
            if momentum >= 6:
                post_price = best_ask
                post_qty = min(to_buy, buy_clip)
                if post_qty > 0:
                    self.buy(int(round(post_price)), post_qty)

            elif momentum >= 2:
                post_price = best_bid + 1 if spread >= 2 else best_bid
                post_qty = min(to_buy, buy_clip)
                if post_qty > 0:
                    self.buy(int(round(post_price)), post_qty)

            elif momentum > 0:
                post_price = best_bid
                post_qty = min(to_buy, max(4, buy_clip // 2))
                if post_qty > 0:
                    self.buy(int(round(post_price)), post_qty)

        # ============================================================
        # 3. VERY LIMITED SELLING
        # ============================================================
        # do NOT keep selling into strength.
        # only bleed inventory if very full and momentum weakens.
        to_sell = limit + position

        if position >= int(0.85 * limit):
            if momentum < 1:
                qty = min(position // 3, to_sell)
                if qty > 0:
                    self.sell(best_ask + 2, qty)

            elif momentum < -2:
                qty = min(position // 2, to_sell)
                if qty > 0:
                    self.sell(best_bid, qty)

        # emergency de-risk if trend genuinely flips while heavily long
        if position > 0 and momentum < -5:
            qty = min(position // 2, to_sell)
            if qty > 0:
                self.sell(best_bid, qty)


    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol,0)
        '''
        best_bid = (0,0)
        best_ask = (0,0)
        if(buy_orders):
            best_bid = buy_orders[0]
        if(sell_orders):
            best_ask = sell_orders[0]

        spread = best_ask[0] - best_bid[0]

        if(spread <5):
            self.buy(best_ask[0], best_ask[1])
        '''
        self.actHolding(state)

        #if(state.timestamp<0):
        #    self.actMispricings(state)
        #else:
        #    self.actMispricings(state)
        #    self.actHolding(state)

    def getMid(self, state: TradingState) -> int:

            order_depth = state.order_depths[self.symbol]

            buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
            sell_orders = sorted(order_depth.sell_orders.items())
            best_bid =0
            best_ask = 0
            if(buy_orders[0]):
                best_bid  = buy_orders[0][0]
            if(sell_orders[0]):
                best_ask  = sell_orders[0][0]

            return (best_ask+best_bid)//2

    def actMispricings(self, state: TradingState) -> None:

        order_depth = state.order_depths[self.symbol]

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())


        if not buy_orders or not sell_orders:
            return

        best_bid, bid_vol = buy_orders[0]
        best_ask, ask_vol = sell_orders[0]

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position


        mid = self.getMid(state)
        spread = abs(best_ask - best_bid)
        if(spread >2 and spread <20):
            self.midPriceSpread.append((mid,spread))



        #value is is the last midprice which had a regular spread involved
        value = self.midPriceSpread[-1][0] if self.midPriceSpread else mid

        #if there is an ask between value then buy.
        for price, volume in sell_orders:
            if to_buy > 0 and price <= value:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        #if there is a bid between value then sell
        #for price, volume in buy_orders:
        #    if to_sell > 0 and price >= value:
        #        quantity = min(to_sell, volume)
        #        self.sell(price, quantity)
        #        to_sell -= quantity

        #self.actHolding(state)


    def actHolding(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        if not buy_orders or not sell_orders:
            return

        best_bid, bid_vol = buy_orders[0]
        best_ask, ask_vol = sell_orders[0]
        ask_size = -ask_vol

        mid = (best_bid + best_ask) / 2
        if bid_vol + ask_size == 0:
            microprice = mid
        else:
            microprice = (best_ask * bid_vol + best_bid * ask_size) / (bid_vol + ask_size)

        self.history.append(int(round(mid)))

        position = state.position.get(self.symbol, 0)
        limit = self.limit
        to_buy = limit - position
        to_sell = limit + position

        # -------- MOMENTUM --------
        short_mom = 0.0
        med_mom = 0.0
        long_mom = 0.0

        if len(self.history) >= 5:
            short_mom = self.history[-1] - self.history[-5]
        if len(self.history) >= 12:
            med_mom = self.history[-1] - self.history[-12]
        if len(self.history) >= 25:
            long_mom = self.history[-1] - self.history[-25]

        momentum = 0.65 * short_mom + 0.25 * med_mom + 0.10 * long_mom

        # -------- VERY AGGRESSIVE FAIR --------
        spread = abs(best_ask - best_bid)
        recentValidMid, recentValidSpread = self.midPriceSpread[-1] if self.midPriceSpread else (mid,spread)
        fair = (
            recentValidMid
            #+ 1.2 * momentum
            #+ 0.6 * (microprice - mid)
            #-   position/limit
        )



        # -------- VERY AGGRESSIVE BUY SIZE --------
        if momentum >= 6:
            buy_clip = 30
        elif momentum >= 3:
            buy_clip = 15
        elif momentum >= 1:
            buy_clip = 10
        else:
            buy_clip = 5

        # keep buying until basically full
        can_buy = position <  limit-13

        # -------- TAKE Mispricings --------

        averageSpread = sum(entry[1] for entry in self.midPriceSpread )/len(self.midPriceSpread) if len(self.midPriceSpread)!=0 else spread

        if(spread > averageSpread-2 ):
            self.midPriceSpread.append((mid,spread))


        if(spread < recentValidSpread-5):

            #value is is the last midprice which had a regular spread involved
            realMid = fair

            #if there is an ask between value then buy.
            for price, volume in sell_orders:
                if to_buy > 0 and price <= realMid - 2:
                    quantity = min(to_buy, -volume)
                    self.buy(price, quantity)
                    to_buy -= quantity

            #for price, volume in buy_orders:
            #    if to_sell > 0 and price >= realMid+2:
            #        quantity = min(to_sell, volume, 5)
            #        self.sell(price, quantity)
            #        to_sell -= quantity
            return

        # -------- TAKE ASKS HARD --------
        if can_buy:
            # willing to pay above fair in strong trend
            take_threshold = fair + 8
            if momentum >= 4:
                take_threshold = fair + 5
            if momentum >= 7:
                take_threshold = fair + 1

            for price, volume in sell_orders:
                if to_buy <= 0:
                    break

                ask_qty = -volume
                if price <= take_threshold:
                    qty = min(buy_clip, to_buy, ask_qty)
                    if qty > 0:
                        self.buy(price, qty)
                        to_buy -= qty
                        position += qty

        # recompute
        to_buy = limit - position
        to_sell = limit + position

        # -------- POST AGGRESSIVE BID --------
        if to_buy > 0 and can_buy:
            spread = best_ask - best_bid

            if momentum >= 5:
                bid_price = best_ask      # join the ask / cross next if available
            elif spread >= 2:
                bid_price = best_bid + 1  # improve bid
            else:
                bid_price = best_bid      # stay at best bid if tight

            qty = min(buy_clip, to_buy)
            if qty > 0:
                self.buy(int(round(bid_price)), qty)


        #cheap asks/bids
        # picking out cheap ask prices
        #for price, volume in sell_orders:
        #    if to_buy > 0 and price <= mid:
        #        quantity = min(to_buy, -volume)
        #        self.buy(price, quantity)
        #        to_buy -= quantity

        #for price, volume in buy_orders:
        #    if to_sell > 0 and price >= mid:
        #        quantity = min(to_sell, volume)
        #        self.sell(price, quantity)
        #       to_sell -= quantity



        # -------- ONLY TINY INVENTORY BLEED --------
        # do not meaningfully sell unless almost max long
        # to offload inventory, sell at bid spikes


        #if position >= int(0.75 * limit) and to_sell > 0:
        #    ask_price = best_ask+2
        #    qty = to_sell//2
        #    if qty > 0:
        #        self.sell(ask_price, qty)

        #if position <= -int(0.75 * limit) and to_buy > 0:
        #    bid_price = best_bid+6
        #    qty = to_sell//2
        #    if qty > 0:
        #        self.buy(bid_price, qty)

    def save(self) -> JSON:
        return {
            "history": list(self.history),
            "midPriceSpread": list(self.midPriceSpread),
        }

    def load(self, data: JSON) -> None:

        self.history = deque(data.get("history", []), maxlen=80)
        self.midPriceSpread = deque(
            [tuple(x) for x in data.get("midPriceSpread", [])],
            maxlen=80
        )
    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        if not buy_orders or not sell_orders:
            return 10_000

        best_bid, bid_vol = buy_orders[0]
        best_ask, ask_vol = sell_orders[0]

        ask_vol = -ask_vol

        fair = (best_bid + best_ask) / 2

        return round(fair)



class Trader:
    def bid(self):
        return 5

    def __init__(self) -> None:
        limits = {
            "ASH_COATED_OSMIUM": 80,
            "INTARIAN_PEPPER_ROOT": 80,
        }

        self.strategies: dict[Symbol, Strategy] = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
            "ASH_COATED_OSMIUM": AshCoatedOsmiumStrategy,
            "INTARIAN_PEPPER_ROOT": IntarianPepperRootStrategy,
        }.items()}

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

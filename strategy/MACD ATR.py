import math
import datetime
import backtrader as bt


class MomentumBreakout(bt.Strategy):
    params = (
        ('stake', 10),  # Trading unit per order (±stake each time)
        ('series_index', 0),
        ('allow_short', True),  # Enable short selling
        # MACD
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('require_cross', True),  # True: Trigger only on cross; False: DIF>DEA(Long)/DIF<DEA(Short)
        # ATR Trailing Stop
        ('use_atr_trail', True),  # ATR Trailing Stop
        ('atr_ts_mult', 2.0),  # ATR Trailing Stop Multiplier. Suggested 1.5 to 3. Higher is more conservative.
        ('atr_period', 14) # ATR calculation period. Suggested to keep unchanged.
    )

    def _dname(self, d):
        return getattr(d, '_name', str(d))

    def _log(self, d, txt):
        dt = d.datetime.date(0)
        print(f"{dt} [{self._dname(d)}] {txt}")

    # Init: Create independent indicators and states for each data feed
    def start(self):
        self.macd = {}
        self.cross = {}
        self.atr = {}
        self.entry_price = {}
        self.highest_since_entry = {}
        self.lowest_since_entry = {}

        for d in self.datas:
            self.macd[d] = bt.indicators.MACD(
                d.close,
                period_me1=int(self.p.macd_fast),
                period_me2=int(self.p.macd_slow),
                period_signal=int(self.p.macd_signal)
            )
            self.cross[d] = bt.indicators.CrossOver(self.macd[d].macd, self.macd[d].signal)
            self.atr[d] = bt.indicators.ATR(d, period=int(self.p.atr_period))

            self.entry_price[d] = None
            self.highest_since_entry[d] = None
            self.lowest_since_entry[d] = None

    # Reset extrema when opening, reversing, or closing positions
    def _reset_extrema(self, d, px, new_pos):
        if new_pos > 0:
            self.highest_since_entry[d] = px
            self.lowest_since_entry[d] = None
        elif new_pos < 0:
            self.lowest_since_entry[d] = px
            self.highest_since_entry[d] = None
        else:
            self.highest_since_entry[d] = None
            self.lowest_since_entry[d] = None


    # Helper for independent limit orders
    def _limit_order(self, data, size, anchor_price, atr_mult=0.5):
        """Attempt to send an extra limit order (supplementing the market order)"""
        # 1. Budget check
        if not self.overspend_guard([(data, size)]):
            return

        # 2. Calculate limit price (Discount based on ATR volatility)
        # Use ATR if available, otherwise default to 1% discount
        if data in self.atr:
            discount = self.atr[data][0] * atr_mult
        else:
            discount = anchor_price * 0.01

        if size > 0:  # Buy: Limit lower
            limit_price = anchor_price - discount
            limit_price = max(0.01, limit_price)
        else:  # Sell: Limit higher
            limit_price = anchor_price + discount

        # 3. Set validity to 1 day (Day Order)
        valid_until = data.datetime.date(0) + datetime.timedelta(days=1)

        # 4. Place order
        self.buy(data=data, size=size, price=limit_price, exectype=bt.Order.Limit,
                 valid=valid_until) if size > 0 else \
            self.sell(data=data, size=size, price=limit_price, exectype=bt.Order.Limit, valid=valid_until)

        self._log(data, f"EXTRA LIMIT: {size} @ {limit_price:.2f}")

    # ---------- Main Logic ----------
    def next(self):
        for d in self.datas:
            # Indicator readiness guard
            if (math.isnan(self.macd[d].macd[0]) or
                    math.isnan(self.macd[d].signal[0]) or
                    math.isnan(self.atr[d][0])):
                continue

            close_px = float(d.close[0])
            cur_pos = int(self.getposition(d).size)
            atr_now = float(self.atr[d][0])

            # Entry / Increase or Decrease position
            if self.p.require_cross:
                long_trigger = (self.cross[d][0] > 0)  # Golden Cross
                short_trigger = (self.cross[d][0] < 0)  # Death Cross
            else:
                long_trigger = (self.macd[d].macd[0] > self.macd[d].signal[0])
                short_trigger = (self.macd[d].macd[0] < self.macd[d].signal[0])

            delta = 0
            if long_trigger:
                delta = +int(self.p.stake)
            elif short_trigger:
                if self.p.allow_short:
                    delta = -int(self.p.stake)
                else:
                    if cur_pos > 0:
                        delta = -min(int(self.p.stake), cur_pos)

            if delta != 0:
                if not self.overspend_guard([(d, delta)]):
                    self._log(d, "Overspend guard; skip entry")
                    continue

                self.place_market(d, delta)
                # Determine if opening or adding position (not closing/stop loss)
                is_entry = (cur_pos == 0) or (cur_pos * delta > 0)

                if is_entry:
                    # Attempt extra limit order.
                    # atr_mult=0.3 means placing order at "Close - 0.3 * ATR"
                    # Trend strategies shouldn't limit too far, 0.3-0.5 is appropriate
                    self._limit_order(d, delta, close_px, atr_mult=0.3)

                new_pos = cur_pos + delta
                # New/Reverse: Record entry price and extrema start point
                if (cur_pos == 0 and new_pos != 0) or (cur_pos * new_pos < 0):
                    self.entry_price[d] = close_px
                    self._reset_extrema(d, close_px, new_pos)

                # Update extrema during holding (for ATR trailing stop)
                if new_pos > 0:
                    if self.highest_since_entry[d] is None or close_px > self.highest_since_entry[d]:
                        self.highest_since_entry[d] = close_px
                elif new_pos < 0:
                    if self.lowest_since_entry[d] is None or close_px < self.lowest_since_entry[d]:
                        self.lowest_since_entry[d] = close_px

                self._log(
                    d,
                    f"MACD entry DIF={float(self.macd[d].macd[0]):.4f} "
                    f"DEA={float(self.macd[d].signal[0]):.4f} ATR={atr_now:.4f} "
                    f"| pos={cur_pos:+d}, delta={delta:+d}"
                )
                continue

            if self.p.use_atr_trail and cur_pos != 0:
                if cur_pos > 0:
                    if self.highest_since_entry[d] is None or close_px > self.highest_since_entry[d]:
                        self.highest_since_entry[d] = close_px
                    trail_long = self.highest_since_entry[d] - self.p.atr_ts_mult * atr_now
                    if close_px <= trail_long:
                        delta = -cur_pos
                else:
                    if self.lowest_since_entry[d] is None or close_px < self.lowest_since_entry[d]:
                        self.lowest_since_entry[d] = close_px
                    trail_short = self.lowest_since_entry[d] + self.p.atr_ts_mult * atr_now
                    if close_px >= trail_short:
                        delta = -cur_pos

                if delta != 0:
                    if not self.overspend_guard([(d, delta)]):
                        self._log(d, "Overspend guard; skip ATR trail exit")
                        continue
                    self.place_market(d, delta)
                    self.entry_price[d] = None
                    self._reset_extrema(d, close_px, 0)
                    self._log(d, "EXIT ATR-TRAIL")
                    continue

    # ---------- Order Callback ----------
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        d = order.data
        if order.status in [order.Completed]:
            action = "BUY" if order.isbuy() else "SELL"
            self._log(d, f"{action} executed at {order.executed.price:.2f} for size {order.executed.size}")
        elif order.status in [order.Canceled]:
            self._log(d, "Order Canceled")
        elif order.status in [order.Rejected]:
            self._log(d, "Order Rejected")

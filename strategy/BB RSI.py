import math
import datetime 
import backtrader as bt


class Bollinger(bt.Strategy):
    params = (
        ('stake', 10),  # number of units to trade
        ('series_index', 0),  # which data feed to act on (0 = first CSV)
        ('n', 20),
        ('k', 2.0),
        ('allow_short', True),  # Enable short selling
        # RSI
        ('allow_RSI', True),  # Enable RSI usage
        ('RSI_period', 14),  # RSI period
        ('RSI_buy', 35),  # Buy only when RSI < 35 and breaks lower band
        ('RSI_sell', 65),  # Sell only when RSI > 65 and breaks upper band
        # Stop Loss and take profit
        ('exit_rule', True),  # Enable stop loss and take profit
        ('Stop_loss', 0.10),  # Stop loss
        ('take_profit', 0.20)  # Take profit
    )

    def start(self):
        self.d = self.datas[self.p.series_index]
        self.mid = bt.indicators.SMA(self.d.close, period=int(self.p.n))
        self.std = bt.indicators.StandardDeviation(self.d.close, period=int(self.p.n))
        self.upper = self.mid + float(self.p.k) * self.std
        self.lower = self.mid - float(self.p.k) * self.std
        self.rsi = bt.indicators.RSI(self.d.close, period=int(self.p.RSI_period)) if self.p.allow_RSI else None
        self.orders_today = 0



    def log(self, txt):
        dt = self.data.datetime.date(0)
        print(f"{dt} {txt}")

    def entry_filters_pass_RSI(self, side, price):
        if not self.p.allow_RSI or self.rsi is None:
            return True
        rsi_now = float(self.rsi[0]) if not math.isnan(self.rsi[0]) else None
        if rsi_now is None:
            return False
        if side == 'long':
            return rsi_now <= float(self.p.RSI_buy)
        else:  # short
            return rsi_now >= float(self.p.RSI_sell)

    def exit_hit(self, close_px, pos_size, pos_price):
        if not self.p.exit_rule or pos_size == 0 or pos_price is None or pos_price <= 0:
            return (False, 0)

        sl = float(self.p.Stop_loss)
        tp = float(self.p.take_profit)

        if pos_size > 0:  # Long position
            stop_loss_hit = close_px <= pos_price * (1.0 - sl)
            take_profit_hit = close_px >= pos_price * (1.0 + tp)
            if stop_loss_hit or take_profit_hit:
                return (True, -pos_size)
        else:  # Short position
            stop_loss_hit = close_px >= pos_price * (1.0 + sl)
            take_profit_hit = close_px <= pos_price * (1.0 - tp)
            if stop_loss_hit or take_profit_hit:
                return (True, -pos_size)
        return (False, 0)

    def _limit_order(self, data, size, limit_price):

        # 1. Budget check (This is an extra order, must check)
        if not self.overspend_guard([(data, size)]):
            return

        # 2. Set validity to 1 day (Day Order)
        valid_until = data.datetime.date(0) + datetime.timedelta(days=1)

        # 3. Send limit order
        if size > 0:
            self.buy(data=data, size=size, price=limit_price, exectype=bt.Order.Limit, valid=valid_until)
        elif size < 0:
            self.sell(data=data, size=size, price=limit_price, exectype=bt.Order.Limit, valid=valid_until)

        self.log(f"EXTRA LIMIT PLACED: {size} @ {limit_price:.2f}")


    def next(self):
        # 0. Get current indicator values
        BB_up = float(self.upper[0]) if not math.isnan(self.upper[0]) else None
        BB_low = float(self.lower[0]) if not math.isnan(self.lower[0]) else None

        if BB_up is None or BB_low is None:
            return

        bandwidth = BB_up - BB_low

        close_px = float(self.d.close[0])
        pos = self.getposition(self.d)
        cur_pos = int(pos.size)
        avg_px = float(pos.price) if pos.size != 0 else None

        # 1. SL/TP check (Keep as is - must use Market order for safety)
        hit, delta_exit = self.exit_hit(close_px, cur_pos, avg_px)
        if hit and delta_exit != 0:
            intents = [(self.d, delta_exit)]
            if not self.overspend_guard(intents):
                self.log("Overspend guard triggered on exit; skip order")
                return
            self.place_market(self.d, delta_exit)
            self.log(f"EXIT via SL/TP: close={close_px:.4f} avg={avg_px:.4f} | pos={cur_pos:+d}, delta={delta_exit:+d}")
            return

            # 2. Generate trading signals
        delta = 0
        # Break below lower band -> Long
        if close_px < BB_low:
            if self.entry_filters_pass_RSI('long', close_px):
                delta = +int(self.p.stake)

        # Break above upper band
        elif close_px > BB_up:
            if self.p.allow_short:
                if self.entry_filters_pass_RSI('short', close_px):
                    delta = -int(self.p.stake)
            else:
                if cur_pos > 0:
                    delta = -min(int(self.p.stake), cur_pos)

        if delta == 0:
            return

        # 3. Execute trade (Hybrid mode)

        # [A] Market order (Baseline, guarantees entry)
        intents = [(self.d, delta)]
        if not self.overspend_guard(intents):
            self.log("Overspend guard triggered; skip order")
            return
        self.place_market(self.d, delta)
        self.log(f"MARKET ORDER: {delta} @ {close_px:.2f}")

        #  Limit order
        # Attempt limit order entry only when opening a position (0 pos)
        if cur_pos == 0:
            # Strategy: Determine depth based on bandwidth (volatility)
            # Buffer distance set to 10% of bandwidth (0.1)
            # The wilder the market (wider band), the further we limit, higher safety margin
            buffer = bandwidth * 0.1

            if delta > 0:  # Buy
                # Limit price at Lower Band - Buffer
                limit_price = BB_low - buffer
            else:  # Sell
                # Limit price at Upper Band + Buffer
                limit_price = BB_up + buffer

            # Here delta is an extra unit, you can set it to self.p.stake or delta
            self._limit_order(self.d, delta, limit_price)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            action = "BUY" if order.isbuy() else "SELL"
            price = order.executed.price
            size = order.executed.size
            self.log(f"{action} executed at {price:.2f} for size {size}")
        elif order.status in [order.Canceled]:
            self.log("Order Canceled")
        elif order.status in [order.Rejected]:
            self.log("Order Rejected")

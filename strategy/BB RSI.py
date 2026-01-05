import math
import datetime
import backtrader as bt


class Bollinger(bt.Strategy):
    params = (
        ('stake', 10),  # number of units to trade
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

    
    def _dname(self, d):
        return getattr(d, '_name', str(d))

    def _log(self, d, txt):
        dt = d.datetime.date(0)
        print(f"{dt} [{self._dname(d)}] {txt}")


    def start(self):
        self.mid = {}
        self.std = {}
        self.upper = {}
        self.lower = {}
        self.rsi = {}

        for d in self.datas:
            # 为每个数据 d 计算布林带和 RSI
            self.mid[d] = bt.indicators.SMA(d.close, period=int(self.p.n))
            self.std[d] = bt.indicators.StandardDeviation(d.close, period=int(self.p.n))
            self.upper[d] = self.mid[d] + float(self.p.k) * self.std[d]
            self.lower[d] = self.mid[d] - float(self.p.k) * self.std[d]

            if self.p.allow_RSI:
                self.rsi[d] = bt.indicators.RSI(d.close, period=int(self.p.RSI_period))
            else:
                self.rsi[d] = None


    def entry_filters_pass_RSI(self, d, side):
        if not self.p.allow_RSI or self.rsi[d] is None:
            return True

        rsi_val = self.rsi[d][0]
        if math.isnan(rsi_val):
            return False

        rsi_now = float(rsi_val)
        if side == 'long':
            return rsi_now <= float(self.p.RSI_buy)
        else:  # short
            return rsi_now >= float(self.p.RSI_sell)


    def exit_hit(self, close_px, pos_size, pos_price):
        if not self.p.exit_rule or pos_size == 0 or pos_price is None or pos_price <= 0:
            return (False, 0)

        sl = float(self.p.Stop_loss)
        tp = float(self.p.take_profit)

        if pos_size > 0:  # Long
            stop_loss_hit = close_px <= pos_price * (1.0 - sl)
            take_profit_hit = close_px >= pos_price * (1.0 + tp)
            if stop_loss_hit or take_profit_hit:
                return (True, -pos_size)
        else:  # Short
            stop_loss_hit = close_px >= pos_price * (1.0 + sl)
            take_profit_hit = close_px <= pos_price * (1.0 - tp)
            if stop_loss_hit or take_profit_hit:
                return (True, -pos_size)
        return (False, 0)


    def _limit_order(self, data, size, limit_price):
        #  Budget check (This is an extra order, must check)
        if not self.overspend_guard([(data, size)]):
            return

        # Set validity to 1 day (Day Order)
        valid_until = data.datetime.date(0) + datetime.timedelta(days=1)

        # Send limit order
        if size > 0:
            self.buy(data=data, size=size, price=limit_price, exectype=bt.Order.Limit, valid=valid_until)
        elif size < 0:
            self.sell(data=data, size=size, price=limit_price, exectype=bt.Order.Limit, valid=valid_until)

        self._log(data, f"EXTRA LIMIT PLACED: {size} @ {limit_price:.2f}")

    # ---------- Main ----------
    def next(self):
        for d in self.datas:
            # Get current indicator values
            BB_up = float(self.upper[d][0]) if not math.isnan(self.upper[d][0]) else None
            BB_low = float(self.lower[d][0]) if not math.isnan(self.lower[d][0]) else None

            if BB_up is None or BB_low is None:
                continue

            bandwidth = BB_up - BB_low
            close_px = float(d.close[0])
            pos = self.getposition(d)
            cur_pos = int(pos.size)
            avg_px = float(pos.price) if pos.size != 0 else None

            # SL/TP check (Keep as is - must use Market order for safety)
            hit, delta_exit = self.exit_hit(close_px, cur_pos, avg_px)
            if hit and delta_exit != 0:
                intents = [(d, delta_exit)]
                if not self.overspend_guard(intents):
                    self._log(d, "Overspend guard triggered on exit; skip order")
                    continue
                self.place_market(d, delta_exit)
                self._log(d,
                          f"EXIT via SL/TP: close={close_px:.4f} avg={avg_px:.4f} | pos={cur_pos:+d}, delta={delta_exit:+d}")
                continue  

            # Generate trading signals
            delta = 0
            # Break below lower band -> Long
            if close_px < BB_low:
                if self.entry_filters_pass_RSI(d, 'long'):
                    delta = +int(self.p.stake)

            # Break above upper band  -> Short
            elif close_px > BB_up:
                if self.p.allow_short:
                    if self.entry_filters_pass_RSI(d, 'short'):
                        delta = -int(self.p.stake)
                else:
                    if cur_pos > 0:
                        delta = -min(int(self.p.stake), cur_pos)

            if delta == 0:
                continue

            #  Execute trade
            # Market order
            intents = [(d, delta)]
            if not self.overspend_guard(intents):
                self._log(d, "Overspend guard triggered; skip order")
                continue

            self.place_market(d, delta)
            self._log(d, f"MARKET ORDER: {delta} @ {close_px:.2f}")

            #  Limit order
            #  Attempt limit order entry only when opening a position (0 pos)
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

                self._limit_order(d, delta, limit_price)

    # ---------- 订单回调 ----------
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

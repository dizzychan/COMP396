import math
import backtrader as bt


class AbsPctChange(bt.Indicator):
    lines = ("abspct",)

    def __init__(self):
        prev = self.data0(-1)
        self.l.abspct = bt.If(prev != 0.0, abs(self.data0 / prev - 1.0), 0.0)


class EnsembleVoteB(bt.Strategy):
    """
    - Entry by weighted vote of MACBB, BBRSI, MACD signals.
    - Exit if any sub-strategy exit rule triggers.
    """

    params = dict(
        # Voting
        vote_threshold=0.34,
        weight_macbb=1.0,
        weight_bbrsi=1.0,
        weight_macd=1.0,
        allow_short=True,

        # Position sizing
        use_inv_vol_sizing=True,
        invvol_lookback=20,
        total_exposure=1_000_000.0,
        min_avg_abs_change=1e-8,
        max_units=100000,
        stake=10,

        # MACBB params
        macbb_fast=10,
        macbb_slow=30,
        macbb_bb_period=20,
        macbb_bb_dev=2.0,
        macbb_trailing_stop_pct=0.10,
        macbb_take_profit_pct=0.15,
        macbb_mid_band_buffer=0.995,

        # BBRSI params
        bbrsi_n=20,
        bbrsi_k=2.0,
        bbrsi_use_rsi=True,
        bbrsi_rsi_period=14,
        bbrsi_rsi_buy=35,
        bbrsi_rsi_sell=65,
        bbrsi_exit_rule=True,
        bbrsi_stop_loss=0.10,
        bbrsi_take_profit=0.20,
        bbrsi_atr_period=14,
        bbrsi_atr_sl_mult=2.0,

        # MACD params
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        macd_require_cross=True,
        macd_use_atr_trail=True,
        macd_atr_ts_mult=2.0,
        macd_atr_period=14,

        # Exit voting
        exit_min_votes=2,

        printlog=True,
    )

    def __init__(self):
        # Inverse vol sizing state
        self.abspct = {}
        self.avg_abspct = {}
        self.invvol_budget = {}

        # MACBB indicators
        self.macbb_sma_fast = {}
        self.macbb_sma_slow = {}
        self.macbb_cross_sma = {}
        self.macbb_bb = {}
        self.macbb_cross_top = {}
        self.macbb_cross_bot = {}

        # BBRSI indicators
        self.bbrsi_mid = {}
        self.bbrsi_std = {}
        self.bbrsi_upper = {}
        self.bbrsi_lower = {}
        self.bbrsi_rsi = {}
        self.bbrsi_atr = {}

        # MACD indicators
        self.macd = {}
        self.macd_cross = {}
        self.macd_atr = {}

        # Position state (shared across sub-strategies)
        self.entry_price = {}
        self.highest = {}
        self.lowest = {}

        for d in self.datas:
            # Inverse vol sizing
            self.abspct[d] = AbsPctChange(d.close)
            self.avg_abspct[d] = bt.ind.SMA(self.abspct[d], period=int(self.p.invvol_lookback))

            # MACBB
            self.macbb_sma_fast[d] = bt.ind.SMA(d.close, period=int(self.p.macbb_fast))
            self.macbb_sma_slow[d] = bt.ind.SMA(d.close, period=int(self.p.macbb_slow))
            self.macbb_cross_sma[d] = bt.ind.CrossOver(self.macbb_sma_fast[d], self.macbb_sma_slow[d])
            self.macbb_bb[d] = bt.ind.BollingerBands(
                d.close,
                period=int(self.p.macbb_bb_period),
                devfactor=float(self.p.macbb_bb_dev),
            )
            self.macbb_cross_top[d] = bt.ind.CrossOver(d.close, self.macbb_bb[d].top)
            self.macbb_cross_bot[d] = bt.ind.CrossOver(d.close, self.macbb_bb[d].bot)

            # BBRSI
            self.bbrsi_mid[d] = bt.ind.SMA(d.close, period=int(self.p.bbrsi_n))
            self.bbrsi_std[d] = bt.ind.StandardDeviation(d.close, period=int(self.p.bbrsi_n))
            self.bbrsi_upper[d] = self.bbrsi_mid[d] + float(self.p.bbrsi_k) * self.bbrsi_std[d]
            self.bbrsi_lower[d] = self.bbrsi_mid[d] - float(self.p.bbrsi_k) * self.bbrsi_std[d]
            self.bbrsi_atr[d] = bt.ind.ATR(d, period=int(self.p.bbrsi_atr_period))
            if self.p.bbrsi_use_rsi:
                self.bbrsi_rsi[d] = bt.ind.RSI(d.close, period=int(self.p.bbrsi_rsi_period))
            else:
                self.bbrsi_rsi[d] = None

            # MACD
            self.macd[d] = bt.ind.MACD(
                d.close,
                period_me1=int(self.p.macd_fast),
                period_me2=int(self.p.macd_slow),
                period_signal=int(self.p.macd_signal),
            )
            self.macd_cross[d] = bt.ind.CrossOver(self.macd[d].macd, self.macd[d].signal)
            self.macd_atr[d] = bt.ind.ATR(d, period=int(self.p.macd_atr_period))

            # State init
            self.entry_price[d] = None
            self.highest[d] = None
            self.lowest[d] = None

    def _dname(self, d):
        return getattr(d, "_name", str(d))

    def _log(self, d, txt):
        if not self.p.printlog:
            return
        dt = d.datetime.date(0)
        print(f"{dt} [{self._dname(d)}] {txt}")

    def _is_nan(self, v):
        try:
            return math.isnan(float(v))
        except Exception:
            return True

    def _compute_invvol_budgets(self):
        raw = {}
        for d in self.datas:
            v = float(self.avg_abspct[d][0])
            if self._is_nan(v) or v <= 0:
                continue
            v = max(float(self.p.min_avg_abs_change), v)
            raw[d] = 1.0 / v

        s = sum(raw.values())
        if s <= 0:
            return

        budgets = {}
        total = float(self.p.total_exposure)
        for d, r in raw.items():
            budgets[d] = (r / s) * total

        self.invvol_budget = budgets

    def _pos_size(self, d):
        px = float(d.close[0])
        if self._is_nan(px) or px <= 0:
            return 0

        if self.p.use_inv_vol_sizing:
            budget = self.invvol_budget.get(d, None)
            if budget is not None and budget > 0:
                units = int(budget // px)
                units = max(0, min(int(self.p.max_units), units))
                return units

        return int(self.p.stake)

    def _update_extrema(self, d, pos_size, close_px):
        if pos_size > 0:
            if self.highest[d] is None or close_px > self.highest[d]:
                self.highest[d] = close_px
        elif pos_size < 0:
            if self.lowest[d] is None or close_px < self.lowest[d]:
                self.lowest[d] = close_px
        else:
            self.highest[d] = None
            self.lowest[d] = None

    def _signal_macbb_entry(self, d, close_px):
        c_mac = float(self.macbb_cross_sma[d][0])
        c_top = float(self.macbb_cross_top[d][0])
        c_bot = float(self.macbb_cross_bot[d][0])
        sma_fast = float(self.macbb_sma_fast[d][0])
        sma_slow = float(self.macbb_sma_slow[d][0])
        bb_top = float(self.macbb_bb[d].top[0])
        bb_bot = float(self.macbb_bb[d].bot[0])

        if any(self._is_nan(v) for v in (c_mac, c_top, c_bot, sma_fast, sma_slow, bb_top, bb_bot, close_px)):
            return 0

        if c_mac > 0 and close_px > bb_top:
            return 1
        if c_mac < 0 and close_px < bb_bot:
            return -1
        if sma_fast > sma_slow and c_top > 0:
            return 1
        if sma_fast < sma_slow and c_bot < 0:
            return -1

        return 0

    def _signal_bbrsi_entry(self, d, close_px):
        up = float(self.bbrsi_upper[d][0])
        low = float(self.bbrsi_lower[d][0])

        if self._is_nan(up) or self._is_nan(low) or self._is_nan(close_px):
            return 0

        if close_px < low:
            if self.p.bbrsi_use_rsi and self.bbrsi_rsi[d] is not None:
                rsi_val = float(self.bbrsi_rsi[d][0])
                if self._is_nan(rsi_val) or rsi_val > float(self.p.bbrsi_rsi_buy):
                    return 0
            return 1

        if close_px > up:
            if self.p.bbrsi_use_rsi and self.bbrsi_rsi[d] is not None:
                rsi_val = float(self.bbrsi_rsi[d][0])
                if self._is_nan(rsi_val) or rsi_val < float(self.p.bbrsi_rsi_sell):
                    return 0
            return -1

        return 0

    def _signal_macd_entry(self, d):
        if self.p.macd_require_cross:
            cross = float(self.macd_cross[d][0])
            if self._is_nan(cross):
                return 0
            if cross > 0:
                return 1
            if cross < 0:
                return -1
            return 0

        macd_val = float(self.macd[d].macd[0])
        sig_val = float(self.macd[d].signal[0])
        if self._is_nan(macd_val) or self._is_nan(sig_val):
            return 0
        if macd_val > sig_val:
            return 1
        if macd_val < sig_val:
            return -1
        return 0

    def _exit_macbb(self, d, pos_size, close_px):
        if pos_size == 0:
            return False

        entry = self.entry_price.get(d)
        if entry is not None and entry > 0:
            if pos_size > 0:
                if self.highest[d] is not None:
                    trail = self.highest[d] * (1.0 - float(self.p.macbb_trailing_stop_pct))
                    if close_px <= trail:
                        return True
                take_profit = entry * (1.0 + float(self.p.macbb_take_profit_pct))
                if close_px >= take_profit:
                    return True
            else:
                if self.lowest[d] is not None:
                    trail = self.lowest[d] * (1.0 + float(self.p.macbb_trailing_stop_pct))
                    if close_px >= trail:
                        return True
                take_profit = entry * (1.0 - float(self.p.macbb_take_profit_pct))
                if close_px <= take_profit:
                    return True

        c_mac = float(self.macbb_cross_sma[d][0])
        if not self._is_nan(c_mac):
            if pos_size > 0 and c_mac < 0:
                return True
            if pos_size < 0 and c_mac > 0:
                return True

        bb_mid = float(self.macbb_bb[d].mid[0])
        if self._is_nan(bb_mid):
            return False

        if pos_size > 0:
            if close_px < bb_mid * float(self.p.macbb_mid_band_buffer):
                return True
        else:
            if close_px > bb_mid * (2.0 - float(self.p.macbb_mid_band_buffer)):
                return True

        return False

    def _exit_bbrsi(self, d, pos_size, close_px, pos_price):
        if not self.p.bbrsi_exit_rule or pos_size == 0:
            return False
        if pos_price is None or pos_price <= 0:
            return False

        atr = float(self.bbrsi_atr[d][0])
        if self._is_nan(atr):
            return False

        sl_pct = float(self.p.bbrsi_stop_loss)
        tp_pct = float(self.p.bbrsi_take_profit)
        atr_mult = float(self.p.bbrsi_atr_sl_mult)

        if pos_size > 0:
            sl_fixed = pos_price * (1.0 - sl_pct)
            sl_atr = pos_price - (atr * atr_mult)
            final_sl = max(sl_fixed, sl_atr)
            if close_px <= final_sl:
                return True
            if close_px >= pos_price * (1.0 + tp_pct):
                return True
        else:
            sl_fixed = pos_price * (1.0 + sl_pct)
            sl_atr = pos_price + (atr * atr_mult)
            final_sl = min(sl_fixed, sl_atr)
            if close_px >= final_sl:
                return True
            if close_px <= pos_price * (1.0 - tp_pct):
                return True

        return False

    def _exit_macd(self, d, pos_size, close_px):
        if not self.p.macd_use_atr_trail or pos_size == 0:
            return False

        atr = float(self.macd_atr[d][0])
        if self._is_nan(atr) or atr <= 0:
            return False

        mult = float(self.p.macd_atr_ts_mult)
        if pos_size > 0 and self.highest[d] is not None:
            trail = self.highest[d] - (mult * atr)
            if close_px <= trail:
                return True
        if pos_size < 0 and self.lowest[d] is not None:
            trail = self.lowest[d] + (mult * atr)
            if close_px >= trail:
                return True

        return False

    def _vote_score(self, s1, s2, s3):
        w1 = float(self.p.weight_macbb)
        w2 = float(self.p.weight_bbrsi)
        w3 = float(self.p.weight_macd)
        wsum = w1 + w2 + w3
        if wsum <= 0:
            return 0.0
        return (w1 * s1 + w2 * s2 + w3 * s3) / wsum

    def next(self):
        if self.p.use_inv_vol_sizing:
            self._compute_invvol_budgets()

        for d in self.datas:
            close_px = float(d.close[0])
            if self._is_nan(close_px):
                continue

            pos = self.getposition(d)
            pos_size = int(pos.size)

            if pos_size != 0 and (self.entry_price[d] is None or self.entry_price[d] <= 0):
                if not self._is_nan(pos.price) and pos.price > 0:
                    self.entry_price[d] = float(pos.price)
                else:
                    self.entry_price[d] = close_px

            self._update_extrema(d, pos_size, close_px)

            if pos_size != 0:
                pos_price = float(pos.price) if not self._is_nan(pos.price) and pos.price > 0 else self.entry_price[d]
                exit_votes = 0
                if self._exit_macbb(d, pos_size, close_px):
                    exit_votes += 1
                if self._exit_bbrsi(d, pos_size, close_px, pos_price):
                    exit_votes += 1
                if self._exit_macd(d, pos_size, close_px):
                    exit_votes += 1

                if exit_votes >= int(self.p.exit_min_votes):
                    delta = -pos_size
                    if self.overspend_guard([(d, delta)]):
                        self.place_market(d, delta)
                        self._log(d, f"EXIT by {exit_votes} rules delta={delta:+d}")
                    else:
                        self._log(d, "Overspend guard; skip exit")
                continue

            s1 = self._signal_macbb_entry(d, close_px)
            s2 = self._signal_bbrsi_entry(d, close_px)
            s3 = self._signal_macd_entry(d)
            score = self._vote_score(s1, s2, s3)

            if score >= float(self.p.vote_threshold):
                side = 1
            elif score <= -float(self.p.vote_threshold):
                side = -1
            else:
                continue

            if side < 0 and not self.p.allow_short:
                continue

            size = self._pos_size(d)
            if size <= 0:
                continue

            delta = side * int(size)
            if self.overspend_guard([(d, delta)]):
                self.place_market(d, delta)
                self._log(d, f"ENTRY vote={score:.2f} delta={delta:+d}")
            else:
                self._log(d, "Overspend guard; skip entry")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        d = order.data
        if order.status in [order.Completed]:
            action = "BUY" if order.isbuy() else "SELL"
            self._log(d, f"{action} executed at {order.executed.price:.2f} size {order.executed.size}")

            cur_pos = self.getposition(d).size
            if cur_pos != 0:
                entry = float(order.executed.price)
                self.entry_price[d] = entry
                if cur_pos > 0:
                    self.highest[d] = entry
                    self.lowest[d] = None
                else:
                    self.lowest[d] = entry
                    self.highest[d] = None
            else:
                self.entry_price[d] = None
                self.highest[d] = None
                self.lowest[d] = None

        elif order.status in [order.Canceled]:
            self._log(d, "Order Canceled")
        elif order.status in [order.Margin, order.Rejected]:
            self._log(d, f"Order {order.getstatusname()}")

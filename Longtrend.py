# strategies/longtrend.py
import math
import csv
from collections import defaultdict
import backtrader as bt

class LongTermTrend(bt.Strategy):
    """
    Long-term trend-following (multi-asset)

    Signals (choose one via `signal`):
      • 'sma'       : long if SMA_fast > SMA_slow, short if < (if allow_short)
      • 'donchian'  : long if close >= Highest(N_entry); short if close <= Lowest(N_entry)

    Exits:
      • Donchian    : long exit if close <= Lowest(N_exit); short exit if close >= Highest(N_exit)
      • Trailing    : optional ATR trailing stop (long: Peak - k*ATR; short: Trough + k*ATR)
      • Weekly/Monthly rebalances flatten anything that loses its signal

    Sizing (choose via `risk_model`):
      • 'risk_pct'  : true 1–2% risk per trade using ATR stop distance (units computed directly)
      • 'inv_atr'   : inverse-ATR portfolio weights with per-asset caps

    Other:
      • Optional volume confirmation gates entries
      • Breadth filter, top-K concentration, per-asset weight cap
      • Cash-aware headroom and integer rounding
      • CSV sizing logger for plotting
    """

    params = dict(
        # === Signal config ===
        signal='sma',            # 'sma' | 'donchian'
        sma_fast=50,
        sma_slow=200,
        donch_entry=100,         # entry window (high/low)
        donch_exit=25,           # exit window (low/high)

        # === Volume confirmation (optional) ===
        use_volume_filter=False, # require vol_sma_fast > vol_sma_slow to enter
        vol_fast=20,
        vol_slow=50,

        # === Volatility / ATR ===
        atr_period=20,

        # === Posture / shorting ===
        allow_short=True,

        # === Risk exit (trailing) ===
        stop_atr_k=0.0,          # 0 = off. e.g., 3.0

        # === Sizing ===
        risk_model='risk_pct',   # 'risk_pct' | 'inv_atr'
        risk_per_trade=0.01,     # 1% (use 0.02 for 2%)
        sizing_atr_k=3.0,        # stop distance = k * ATR for risk_pct sizing
        gross_leverage=0.9,      # portfolio budget
        budget_buffer=0.95,      # keep headroom
        max_weight_per_asset=0.40,
        min_alloc=0.0,           # notional floor as % of equity (inv_atr only)
        floor_units=0,           # if >0, ensure at least this many units when non-zero target

        # === Concentration / breadth ===
        max_positions=0,         # 0 = unlimited; else top-K by momentum
        rank_lookback=126,
        breadth_min=1,           # require at least this many signals to deploy

        # === Rebalancing cadence ===
        rebalance='weekly',      # 'daily' | 'weekly' | 'monthly'

        # === Logging ===
        printlog=False,
        track_sizing=True,
        sizing_csv='outputs/sizing_log.csv',
    )

    # ---------- Utility / logging ----------
    def log(self, txt):
        if self.p.printlog:
            dt = self.datas[0].datetime.date(0)
            print(f"{dt} | {txt}")

    # --- EXECUTION HELPERS ---
    def _place_target(self, d, target_units: int):
        """Place MARKET target-size orders so they fill on the bar."""
        self.order_target_size(data=d, target=int(target_units), exectype=bt.Order.Market)

    # Back-compat for places that call place_market(d, delta)
    def place_market(self, d, delta: int):
        cur = int(self.getposition(d).size) if self.getposition(d) else 0
        self._place_target(d, cur + int(delta))

    # If your framework calls this guard, keep it permissive
    def overspend_guard(self, intents):
        return True

    def notify_order(self, order):
        status_names = {
            bt.Order.Created: "Created", bt.Order.Submitted: "Submitted",
            bt.Order.Accepted: "Accepted", bt.Order.Partial: "Partial",
            bt.Order.Completed: "Completed", bt.Order.Canceled: "Canceled",
            bt.Order.Expired: "Expired", bt.Order.Margin: "Margin",
            bt.Order.Rejected: "Rejected",
        }
        st = status_names.get(order.status, str(order.status))
        act = "BUY" if order.isbuy() else "SELL"
        created_size = getattr(getattr(order, "created", None), "size", "?")
        self.log(f"ORDER {st} | {act} {created_size} exectype={order.exectype}")

    # ---------- Init ----------
    def __init__(self):
        # Indicators per data
        self.sma_f, self.sma_s, self.atr = {}, {}, {}
        self.hiE, self.loE, self.hiX, self.loX = {}, {}, {}, {}
        self.vol_f, self.vol_s = {}, {}
        for d in self.datas:
            self.sma_f[d] = bt.ind.SMA(d.close, period=self.p.sma_fast)
            self.sma_s[d] = bt.ind.SMA(d.close, period=self.p.sma_slow)
            self.atr[d]   = bt.ind.ATR(d, period=self.p.atr_period)

            # Donchian (entry/exit)
            self.hiE[d] = bt.ind.Highest(d.high, period=self.p.donch_entry)
            self.loE[d] = bt.ind.Lowest(d.low,  period=self.p.donch_entry)
            self.hiX[d] = bt.ind.Highest(d.high, period=self.p.donch_exit)
            self.loX[d] = bt.ind.Lowest(d.low,  period=self.p.donch_exit)

            # Volume confirmation
            self.vol_f[d] = bt.ind.SMA(getattr(d, 'volume', d.close*0+1), period=self.p.vol_fast)
            self.vol_s[d] = bt.ind.SMA(getattr(d, 'volume', d.close*0+1), period=self.p.vol_slow)

        # Rebalance trackers & trailing peaks/troughs
        self._last_rebalance_date = None
        self._last_rebalance_month = None
        self.peak = {}   # per-data float for long trailing
        self.trough = {} # per-data float for short trailing

        # Sizing log buffer
        self._sizing_rows = []

    # ---------- Helpers ----------
    def _is_rebalance_day(self):
        dt = self.datas[0].datetime.date(0)
        if self.p.rebalance == 'daily':
            return True
        if self.p.rebalance == 'weekly':
            if self._last_rebalance_date == dt:
                return False
            if dt.weekday() == 0:  # Monday
                self._last_rebalance_date = dt
                return True
            return False
        # monthly = first trading day of a new month
        ym = (dt.year, dt.month)
        if self._last_rebalance_month != ym:
            self._last_rebalance_month = ym
            return True
        return False

    def _volume_ok(self, d):
        if not self.p.use_volume_filter:
            return True
        return float(self.vol_f[d][0]) > float(self.vol_s[d][0])

    def _posture(self, d):
        """Return +1 (long), -1 (short), or 0 (flat) based on entry rules."""
        if self.p.signal == 'sma':
            f = float(self.sma_f[d][0]); s = float(self.sma_s[d][0])
            if not math.isfinite(f) or not math.isfinite(s):
                return 0
            if f > s and self._volume_ok(d):
                return +1
            if self.p.allow_short and f < s and self._volume_ok(d):
                return -1
            return 0

        # Donchian
        c = float(d.close[0]); hiE = float(self.hiE[d][0]); loE = float(self.loE[d][0])
        if not all(map(math.isfinite, [c, hiE, loE])):
            return 0
        if c >= hiE and self._volume_ok(d):
            return +1
        if self.p.allow_short and c <= loE and self._volume_ok(d):
            return -1
        return 0

    def _exit_signal_fired(self, d, side):
        """Return True if Donchian exit band is breached against the position."""
        if self.p.signal != 'donchian':
            return False
        c = float(d.close[0]); loX = float(self.loX[d][0]); hiX = float(self.hiX[d][0])
        if side > 0:   # long exits on N_exit low
            return math.isfinite(loX) and c <= loX
        if side < 0:   # short exits on N_exit high
            return math.isfinite(hiX) and c >= hiX
        return False

    def _momentum(self, d, L):
        if len(d) <= L:
            return float('-inf')
        c0 = float(d.close[0]); cp = float(d.close[-L])
        if cp <= 0 or not math.isfinite(cp) or not math.isfinite(c0):
            return float('-inf')
        return (c0 / cp) - 1.0

    def _log_sizing(self, dt, equity, pos_map, weights_map, total_budget, target_units_map=None):
        """Log snapshot. If target_units_map is provided, derive weights from it for logging."""
        if not self.p.track_sizing:
            return
        for d in self.datas:
            name = getattr(d, "_name", "data")
            px   = float(d.close[0]) if len(d) else 0.0
            atrv = float(self.atr[d][0]) if len(self.atr[d]) else 0.0

            if target_units_map is not None and d in target_units_map:
                units = int(target_units_map[d])
                tgt_notional = abs(units) * px
                w_signed = (tgt_notional / total_budget) if total_budget > 0 else 0.0
                w_signed = w_signed if units >= 0 else -w_signed
            else:
                w_signed = float(weights_map.get(d, 0.0))
                tgt_notional = total_budget * abs(w_signed)
                units = int(tgt_notional // px) if px > 0 else 0
                units = units if w_signed >= 0 else -units

            cur   = int(self.getposition(d).size) if self.getposition(d) else 0
            delta = units - cur
            self._sizing_rows.append(dict(
                date=str(dt),
                series=name,
                equity=float(equity),
                price=float(px),
                atr=float(atrv),
                posture=int(pos_map.get(d, 0)),
                weight=float(w_signed),
                target_notional=float(tgt_notional),
                target_units=int(units),
                current_units=int(cur),
                delta=int(delta),
            ))

    # ---------- Write CSV at end ----------
    def stop(self):
        if not (self.p.track_sizing and self._sizing_rows):
            return
        try:
            import os
            os.makedirs(os.path.dirname(self.p.sizing_csv) or ".", exist_ok=True)
        except Exception:
            pass
        with open(self.p.sizing_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self._sizing_rows[0].keys()))
            writer.writeheader()
            writer.writerows(self._sizing_rows)
        if self.p.printlog:
            print(f"[LongTermTrend] Wrote sizing log → {self.p.sizing_csv}")

    # ---------- Core loop ----------
    def next(self):
        intents = []  # kept for compatibility with overspend_guard
        deltas  = []

        # 1) Daily risk exits (trailing + Donchian exit bands)
        for d in self.datas:
            pos = self.getposition(d)
            if not pos or pos.size == 0:
                self.peak[d] = None
                self.trough[d] = None
                continue

            c = float(d.close[0])
            atrv = float(self.atr[d][0]) if len(self.atr[d]) else 0.0
            side = 1 if pos.size > 0 else -1

            # Maintain peak/trough for trailing stops
            if side > 0:
                prev = self.peak.get(d)
                self.peak[d] = c if prev is None else max(prev, c)
            else:
                prev = self.trough.get(d)
                self.trough[d] = c if prev is None else min(prev, c)

            # Trailing stop
            if self.p.stop_atr_k and atrv > 0:
                if side > 0:
                    stop = self.peak[d] - self.p.stop_atr_k * atrv
                    if c < stop:
                        qty = int(pos.size)
                        if qty:
                            deltas.append((d, -qty, 0))
                            intents.append((d, -qty))
                            self.log(f"TRAIL STOP → SELL {qty} on {getattr(d,'_name','data')}")
                            continue
                else:
                    stop = self.trough[d] + self.p.stop_atr_k * atrv
                    if c > stop:
                        qty = int(pos.size)
                        if qty:
                            deltas.append((d, +qty, 0))
                            intents.append((d, +qty))
                            self.log(f"TRAIL STOP → BUY TO COVER {qty} on {getattr(d,'_name','data')}")
                            continue

            # Donchian exit band
            if self._exit_signal_fired(d, side):
                qty = int(pos.size)
                if qty:
                    deltas.append((d, -qty if side > 0 else +qty, 0))
                    intents.append((d, -qty if side > 0 else +qty))
                    self.log(f"DONCH EXIT → FLATTEN {qty} on {getattr(d,'_name','data')}")

        # If not a rebalance day, just execute risk exits
        if not self._is_rebalance_day():
            if intents and not self.overspend_guard(intents):
                self.log("OVRSPEND: cancelling stop/exit orders today")
                return
            for d, delta, target in deltas:
                # convert to MARKET target orders
                self.place_market(d, delta)
            return

        # 2) Rebalance sizing based on current entry signals
        equity = float(self.broker.getvalue())
        total_budget = self.p.gross_leverage * equity * self.p.budget_buffer

        # Build posture map (+1/-1/0)
        pos_map = {}
        candidates = []
        for d in self.datas:
            p = self._posture(d)
            pos_map[d] = p
            if p != 0:
                candidates.append(d)

        # Breadth / Top-K ranking (by signed momentum over lookback)
        if candidates:
            if self.p.max_positions and len(candidates) > self.p.max_positions:
                L = max(1, int(self.p.rank_lookback))
                ranked = sorted(
                    candidates,
                    key=lambda dd: pos_map[dd] * self._momentum(dd, L),
                    reverse=True
                )[: self.p.max_positions]
                candidates = ranked

            if len(candidates) < max(1, int(self.p.breadth_min)):
                candidates = []

        # If no signals → flatten all
        if not candidates:
            dt = self.datas[0].datetime.date(0)
            self._log_sizing(dt, equity, {}, {}, 0.0)
            for d in self.datas:
                pos = self.getposition(d)
                if pos and pos.size:
                    self._place_target(d, 0)
                    self.log(f"FLAT → {getattr(d,'_name','data')}")
            return

        # ---------- Sizing ----------
        weights_map = defaultdict(float)
        target_units_map = None  # used only in risk_pct mode

        if self.p.risk_model == 'risk_pct':
            # Direct units from risk per trade and stop distance
            target_units_map = {}
            desired_notionals = []

            for d in candidates:
                px = float(d.close[0]) if len(d) else 0.0
                atrv = float(self.atr[d][0]) if len(self.atr[d]) else 0.0
                if px <= 0 or atrv <= 0:
                    continue
                stop_dist = self.p.sizing_atr_k * atrv    # cash risk per single unit
                cash_risk = self.p.risk_per_trade * equity
                units_abs = int(cash_risk // stop_dist)
                if self.p.floor_units and units_abs == 0:
                    units_abs = self.p.floor_units
                units = units_abs * (1 if pos_map[d] > 0 else -1)
                target_units_map[d] = units
                desired_notionals.append((d, abs(units) * px))

            # Scale down if gross would exceed portfolio budget
            gross = sum(n for _, n in desired_notionals)
            scale = min(1.0, total_budget / gross) if gross > 0 and total_budget > 0 else 0.0
            for d in list(target_units_map.keys()):
                target_units_map[d] = int(round(target_units_map[d] * scale))

            # Per-asset cap via weight (approx using notional)
            for d in list(target_units_map.keys()):
                px = float(d.close[0]) if len(d) else 0.0
                notional = abs(target_units_map[d]) * px
                if total_budget > 0 and (notional / total_budget) > self.p.max_weight_per_asset:
                    cap_units = int((self.p.max_weight_per_asset * total_budget) // max(px, 1e-9))
                    target_units_map[d] = max(-cap_units, min(cap_units, target_units_map[d]))

        else:  # 'inv_atr' (original vol-parity)
            invvol = []
            for d in candidates:
                atrv = float(self.atr[d][0]) if len(self.atr[d]) else 0.0
                invvol.append((d, (1.0 / atrv) if atrv > 0 else 0.0))
            denom = sum(w for _, w in invvol)
            if denom > 0:
                raw = [(d, w / denom) for d, w in invvol]
                # apply per-asset cap and sign
                capped = []
                for d, w in raw:
                    signed_w = w * (1 if pos_map[d] > 0 else -1)
                    capped.append((d, max(-self.p.max_weight_per_asset,
                                          min(self.p.max_weight_per_asset, signed_w))))
                capsum = sum(abs(w) for _, w in capped)
                weights = [(d, (w / capsum) if capsum > 0 else 0.0) for d, w in capped]
                for d, w in weights:
                    weights_map[d] = w

        # ---------- Build orders (MARKET target-size) ----------
        dt = self.datas[0].datetime.date(0)
        self._log_sizing(dt, equity, pos_map, weights_map, total_budget, target_units_map=target_units_map)

        active_set = set(candidates)

        if self.p.risk_model == 'risk_pct':
            for d in candidates:
                cur = int(self.getposition(d).size) if self.getposition(d) else 0
                tgt = int(target_units_map.get(d, 0)) if target_units_map else 0
                if tgt != cur:
                    self._place_target(d, tgt)
                    side = "BUY" if tgt > cur else "SELL"
                    self.log(f"{side} {abs(tgt-cur):.0f} → target {tgt} on {getattr(d,'_name','data')}")
        else:
            for d in candidates:
                w = float(weights_map.get(d, 0.0))
                px = float(d.close[0]) if len(d) else 0.0
                tgt_notional = total_budget * w
                units = int(abs(tgt_notional) // px) if px > 0 else 0
                if self.p.floor_units and units == 0 and abs(w) > 0:
                    units = self.p.floor_units
                units = units if w >= 0 else -units
                cur = int(self.getposition(d).size) if self.getposition(d) else 0
                if units != cur:
                    self._place_target(d, units)
                    side = "BUY" if units > cur else "SELL"
                    self.log(f"{side} {abs(units-cur):.0f} → target {units} on {getattr(d,'_name','data')} (w={w:+.3f})")

        # Flatten anything not in candidates
        for d in self.datas:
            if d not in active_set:
                cur = int(self.getposition(d).size) if self.getposition(d) else 0
                if cur:
                    self._place_target(d, 0)
                    self.log(f"FLAT → {getattr(d, '_name', 'data')}")

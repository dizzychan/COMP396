# strategies/template_multi.py
import backtrader as bt

class TemplateMulti(bt.Strategy):
    params = dict(stake=1, printlog=False)

    def log(self, msg):
        if self.p.printlog:
            dt = self.datas[0].datetime.date(0)
            print(f"{dt} | {msg}")

    def next(self):
        intents, deltas = [], []

        for d in self.datas:
            # example “buy once” logic per series
            if not self.getposition(d):
                target = self.p.stake
                current = self.getposition(d).size if self.getposition(d) else 0.0
                delta = target - current
                if delta != 0:
                    deltas.append((d, delta))
                    intents.append((d, delta))   # for overspend_guard

        if intents and not self.overspend_guard(intents):
            self.log("OVRSPEND: cancelling all new market orders today")
            return

        for d, delta in deltas:
            self.place_market(d, delta)
            self.log(f"{'BUY' if delta>0 else 'SELL'} {abs(delta)} on {getattr(d,'_name','data')}")

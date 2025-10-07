import numpy as np

# =============================================================
class MACross:
    """
    Generate trading signals based on the crossover of short-term (m-day)
    and long-term (n-day) Simple Moving Averages (SMA):

    Trading Rules:
    - If yesterday's short-term SMA(m) > long-term SMA(n), go long today (signal = 1);
    -Otherwise, stay flat (signal = 0).

    The run() method returns a position signal series for NAV calculation.
    """

    def __init__(self, price: np.ndarray, m: int = 5, n: int = 20):
        self.px = price.astype(float)
        self.m, self.n = m, n

    def _ma(self, w):
        """Calculate the w-day Simple Moving Average (SMA)"""
        pad = np.full(w - 1, np.nan)  # First (w-1) days cannot form a full window
        avg = np.convolve(self.px, np.ones(w) / w, "valid")  # Compute moving average using convolution
        return np.concatenate([pad, avg])

    def run(self):
        """Generate the trading signal series"""
        s = self._ma(self.m)  
        l = self._ma(self.n)  
        sig = (np.roll(s, 1) > np.roll(l, 1)).astype(int)  
        sig[: self.n] = 0  
        return sig

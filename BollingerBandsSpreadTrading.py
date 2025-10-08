import pandas as pd
import numpy as np
from datetime import datetime

class BollingerBandsSpreadTrading:
    """
    Bollinger Bands Spread Trading Model for Pairs/Multiple Assets
    This model trades the spread between two or more assets using Bollinger Bands
    """

    def __init__(self, window=20, num_std=2, entry_threshold=2.0, exit_threshold=0.5):
        """
        Initialize the Bollinger Bands Spread Trading Model

        Parameters:
        window (int): Moving average period for Bollinger Bands calculation
        num_std (float): Number of standard deviations for bands
        entry_threshold (float): Z-score threshold for entry (default 2.0，TBD)
        exit_threshold (float): Z-score threshold for exit (default 0.5,TBD)
        """
        self.window = window
        self.num_std = num_std
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.positions = []
        self.trades = []

    def calculate_spread(self, asset1_prices, asset2_prices, method='ratio'):
        """
        Calculate spread between two assets

        Parameters:
        asset1_prices (pd.Series): Prices of first asset
        asset2_prices (pd.Series): Prices of second asset
        method (str): 'ratio' for price ratio, 'difference' for price difference

        Returns:
        pd.Series: Spread series
        """
        if method == 'ratio':
            # Use log ratio to make spread more stationary
            spread = np.log(asset1_prices / asset2_prices)
        elif method == 'difference':
            # Simple price difference
            spread = asset1_prices - asset2_prices
        else:
            # Normalized difference
            spread = (asset1_prices / asset1_prices.iloc[0]) - (asset2_prices / asset2_prices.iloc[0])

        return spread

    def calculate_bollinger_bands_for_spread(self, spread):
        """
        Calculate Bollinger Bands for the spread

        Parameters:
        spread (pd.Series): Spread between assets

        Returns:
        pd.DataFrame: DataFrame with spread and Bollinger Bands
        """
        df = pd.DataFrame()
        df['Spread'] = spread

        # Calculate moving average of spread
        df['MA'] = df['Spread'].rolling(window=self.window).mean()

        # Calculate standard deviation of spread
        df['STD'] = df['Spread'].rolling(window=self.window).std()

        # Calculate Bollinger Bands
        df['Upper_Band'] = df['MA'] + (self.num_std * df['STD'])
        df['Lower_Band'] = df['MA'] - (self.num_std * df['STD'])

        # Calculate Z-score (standardized spread)
        df['Z_Score'] = (df['Spread'] - df['MA']) / df['STD']

        return df

    #the next step is to generate trading signals


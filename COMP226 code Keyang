import yfinance as yf
import pandas as pd
import backtrader as bt


def task_1(ticker_symbol, start_date, end_date):
    """
    In this task, you are asked to download the trading data of Apple (AAPL) using yfinance.
    The period of the data is from 2012-01-01 to 2024-01-01.
    So you need to specify the start and end date using downloading the data.
    Once downloaded, you are asked to create a BackTrader data container `bt.feeds.PandasData` and
    feed in the Apple data downloaded from yfinance.
    The container should include the open, high, low, close, and volume of the data.
    The close price should correspond to the adjusted close price from the trading data.
    So you need to link the columns in the container and the trading data correctly.
    Finally, you should return the container.

    :param ticker_symbol: the string ticker for the stock.
    :param start_date: the starting date of the trading data, e.g., "2024-01-01"
    :param end_date: the end date of the trading data, e.g., "2024-12-01"
    :return: the bt.feeds.PandasData object containing the trading data.

    """
    # Enter your code here.
    data = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()

    datafeed = bt.feeds.PandasData(
        dataname=data,
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume'
    )
    return datafeed


def task_2(data, cash, commission, slippage_percentage):
    """
    In this task, you are asked to create a cerebro using BackTrader.
    After initializing the cerebro, you need to add data, set initial cash and commission fee properly.
    The data, cash, and commission have been given as the argument of the function.
    Finally, you should return the cerebro.

    :param data: a bt.feeds.PandasData object;
    :param cash: initial cash
    :param commission: commission fee for the broker.
    :param slippage_percentage: percentage of slippage.
    :return: the cerebro.
    """
    # Enter your code here.
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_slippage_perc(slippage_percentage)
    return cerebro


def task_3():
    """
    You will use Backtrader to implement a triple moving average strategy by
    creating a strategy class that inherit bt.strategy.
    This strategy uses three moving averages:
        1. A short-term moving average (10 days).
        2. A medium-term moving average (50 days).
        3. A long-term moving average (100 days).
    Compute these moving averages using the close prices (self.data.close).

    The object should include two functions:
        1. def __init__(self)
        2. def next(self)

    The entry condition should follow the TMA strategy.
    The exit condition should be a trailing stop loss with trailpercent=0.02 for both long and short positions.

    :return: a strategy class that inherit bt.strategy.
    """

    class TripleMovingAverageStrategy(bt.Strategy):
        params = (
            ("short_period", 10),
            ("medium_period", 50),
            ("long_period", 100),
            ('trailpercent',0.02)
        )

        def __init__(self):
            self.short_MA = bt.ind.SMA(self.data.close, period=self.params.short_period)
            self.medium_MA = bt.ind.SMA(self.data.close, period=self.params.medium_period)
            self.long_MA = bt.ind.SMA(self.data.close, period=self.params.long_period)

        def next(self):
            if not self.position:
                if self.short_MA > self.medium_MA > self.long_MA:
                    self.buy()
                elif self.short_MA < self.medium_MA < self.long_MA:
                    self.sell()
                return
            if self.position.size > 0:
                self.sell(exectype=bt.Order.StopTrail, trailpercent=0.02)
            else:
                self.buy(exectype=bt.Order.StopTrail, trailpercent=0.02)

        ########################################
        ## DO NOT CHANGE THE FOLLOWING FUNCTION.
        ## You can selectively comment out the following function
        ## to get a clean printing for your debugging.
        ########################################

        def log(self, txt, dt=None):
            ''' Logging function for this strategy'''
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

        def notify_order(self, order):
            if order.status in [order.Submitted, order.Accepted]:
                # Buy/Sell order submitted/accepted to/by broker - Nothing to do
                return

            # Check if an order has been completed
            # Attention: broker could reject order if not enough cash
            if order.status in [order.Completed]:
                if order.isbuy():
                    self.log(
                        'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                        (order.executed.price,
                         order.executed.value,
                         order.executed.comm))

                    self.buyprice = order.executed.price
                    self.buycomm = order.executed.comm
                else:  # Sell
                    self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                             (order.executed.price,
                              order.executed.value,
                              order.executed.comm))

                self.bar_executed = len(self)

            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                self.log('Order Canceled/Margin/Rejected')

            self.order = None

        def notify_trade(self, trade):
            if not trade.isclosed:
                return

            self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                     (trade.pnl, trade.pnlcomm))

        def stop(self):
            self.log('(MA Periods are %2d, %2d, %2d) Ending Value %.2f' %
                     (self.params.short_period, self.params.medium_period, self.params.long_period, self.broker.getvalue()))

    return TripleMovingAverageStrategy


def task_4():
    """
    You will use Backtrader to implement a BollignerBands overbought and oversold strategy by
    creating a strategy class that inherit bt.strategy.
    This strategy uses:
        1. A lookback window (20 days).
        2. A standard deviation multiplier (3).

    The object should include two functions:
        1. def __init__(self)
        2. def next(self)

    Compute these BollingerBands using the close prices (self.data.close).

    You will need the indicator: bt.indicators.BollingerBands. You can find its introduction here:
    https://www.backtrader.com/docu/indautoref/#bollingerbands

    The entry condition should follow the BollignerBands overbought and oversold strategy.
    The exit condition should be:
    Exit if the price crosses above (greater than) the upper bound for a long position;
    Exit if the price crosses below (less than)  the lower bound for a short position.

    :return: a strategy class that inherit bt.strategy.
    """

    class BollingerBandsStrategy(bt.Strategy):
        params = (
            ('period', 20),
            ('devfactor', 3),
        )

        def __init__(self):
            self.bb = bt.ind.BollingerBands(period=self.params.period, devfactor=self.params.devfactor)
            self.crossup = bt.ind.CrossOver(self.data.close, self.bb.lines.top)
            self.crossdown = bt.ind.CrossOver(self.data.close, self.bb.lines.bot)
            self.order = None

        def next(self):
            if self.order:
                return
            if not self.position:
                if self.data.close[0] < self.bb.lines.bot[0]:
                    self.order = self.buy()
                elif self.data.close[0] > self.bb.lines.top[0]:
                    self.order = self.sell()
            else:
                if self.position.size > 0 and self.crossup[0] >= 1:
                    self.order = self.close()
                elif self.position.size < 0 and self.crossdown[0] <= -1:
                    self.order = self.close()



        ########################################
        ## DO NOT CHANGE THE FOLLOWING FUNCTION.
        ########################################

        def log(self, txt, dt=None):
            ''' Logging function for this strategy'''
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

        def notify_order(self, order):
            if order.status in [order.Submitted, order.Accepted]:
                # Buy/Sell order submitted/accepted to/by broker - Nothing to do
                return

            # Check if an order has been completed
            # Attention: broker could reject order if not enough cash
            if order.status in [order.Completed]:
                if order.isbuy():
                    self.log(
                        'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                        (order.executed.price,
                         order.executed.value,
                         order.executed.comm))

                    self.buyprice = order.executed.price
                    self.buycomm = order.executed.comm
                else:  # Sell
                    self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                             (order.executed.price,
                              order.executed.value,
                              order.executed.comm))

                self.bar_executed = len(self)

            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                self.log('Order Canceled/Margin/Rejected')

            self.order = None

        def notify_trade(self, trade):
            if not trade.isclosed:
                return

            self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                     (trade.pnl, trade.pnlcomm))


    return BollingerBandsStrategy



def task_5(cerebro, strategy_class, stake):
    """
    Add the strategy class to cerebro.
    Add a FixedSize sizer according to the stake.
    :param cerebro: a cerebro object
    :param strategy_class: a strategy class returned by task 3 or 4.
    :param stake: sizer.
    :return: the cerebro.
    """
    # Enter your code here.
    cerebro.addstrategy(strategy_class)
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake)
    return cerebro




def task_6(cerebro):
    """
    Run the cerebro and return the cash after the running.
    :param cerebro: a cerebro object
    :return: cash
    """
    # Enter your code here.
    cerebro.run()
    return cerebro.broker.getvalue()



def task_7(cerebro, stake):
    """
    In this task, you are asked to optimize the TMA strategy (with grid search) you have implemented in Task 3.
    Given a cerebro, you need to create a TMA strategy class (by calling the function task_3()), add the strategy to the cerebro, and set the sizer.
    The parameters you optimize are the short period, the medium period, and the long period.
    Short period can select among three values 5, 10 and 20.
    Medium period can select among three values 50, 100 and 150.
    Long period can select among three values 200, 250 and 300.
    You need to run the cerebro with maxcpu set to be 1 and see which combination yield the highest profit in the end.

    Return two objects. The first is your username. The second is a dictionary with the best parameters and the best value from the grid search.
    Please use the trading data given by the get_periods.py using your username.
    E.g.,

    best_result = {}
    best_result["short_period"] = 5
    best_result["medium_period"] = 50
    best_result["long_period"] = 200
    best_result["value"] = 95624.37

    :return: wangyzh", best_result

    :param cerebro: a cerebro object
    :param stake: sizer.
    :return: Your MWS username, best_result.
    """
    best_result = {}
    best_result["short_period"] = 5
    best_result["medium_period"] = 50
    best_result["long_period"] = 200
    best_result["value"] = 997995.59
    return "pskyan14", best_result







def task_8():
    """
    Write your own code to get the total_compound_return and the sharpe ratio of
    the Bolliger Band Overbought/Oversold strategy in task 4 using your trading period. Once you
    get these ratios, change their values in the risk_adjusted_returns dictionary.

    Return two objects. The first is your username. The second is the risk_adjusted_returns dictionary.
    Please use the trading data given by the get_periods.py using your username.
    E.g.,

    returns = {}
    returns["total_compound_return"] = 0
    returns["sharpe_ratio"] = 0

    return "wangyzh", returns

    """
    returns = {}
    returns["total_compound_return"] = -0.0005
    returns["sharpe_ratio"] = -60.0335

    return "pskyan14", returns

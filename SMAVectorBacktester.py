#### SMA Vektor Backtester from Yahoo
import numpy as np
import pandas as pd
from scipy.optimize import brute
import matplotlib.pyplot as plt
import yahoo_fin.stock_info as si
import datetime
import yfinance as yf

class SMAVectorBacktester(object):
    ''' Class for the vectorized backtesting of SMA-based trading strategies.
    Attributes
    ==========
    symbol: str
        Yahoo! Finance symbol with which to work with
    SMA1: int
        time window in days for shorter SMA
    SMA2: int
        time window in days for longer SMA
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval
    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    set_parameters:
        sets one or two new SMA parameters
    run_strategy:
        runs the backtest for the SMA-based strategy
    plot_results:
        plots the performance of the strategy compared to the symbol
    update_and_run:
        updates SMA parameters and returns the (negative) absolute performance
    optimize_parameters:
        implements a brute force optimizeation for the two SMA parameters
    '''

    def __init__(self, symbol, SMA1, SMA2, start, end):
        self.symbol = symbol
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.start = start
        self.end = end
        self.results = None
        self.get_data()

    def get_data(self):
        raw = si.get_data(self.symbol, '2017-09-29', datetime.datetime.today(), index_as_date=True, interval='1d')[
            'adjclose']
        #raw = yf.download(self.symbol, start=self.start, end=self.end)['Adj Close']
        raw = pd.DataFrame(raw)
        raw.rename(columns={'adjclose': 'price'}, inplace=True)
        raw['return'] = np.log(raw / raw.shift(1))
        raw['SMA1'] = raw['price'].rolling(self.SMA1).mean()
        raw['SMA2'] = raw['price'].rolling(self.SMA2).mean()
        self.data = raw

    def set_parameters(self, SMA1=None, SMA2=None):
        if SMA1 is not None:
            self.SMA1 = SMA1
            self.data['SMA1'] = self.data['price'].rolling(self.SMA1).mean()
        if SMA2 is not None:
            self.SMA2 = SMA2
            self.data['SMA2'] = self.data['price'].rolling(self.SMA2).mean()

    def run_strategy(self):
        data = self.data.copy().dropna()
        data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
        data['strategy'] = data['position'].shift(1) * data['return']
        data['creturns'] = data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        self.results = data
        # absolute performance of the strategy
        aperf = data['cstrategy'].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - data['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2)

    def plot_results(self):

        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        plt.style.use('ggplot')
        title = 'Price and SMAs: %s | SMA1 = %d, SMA2 = %d' % (self.symbol, self.SMA1, self.SMA2)
        self.results[['price', 'SMA1', 'SMA2']].plot(title=title, figsize=(10, 6))

        title = 'Signal: %s | SMA1 = %d, SMA2 = %d' % (self.symbol, self.SMA1, self.SMA2)
        self.results[['position']].plot(title=title, figsize=(10, 6))

        title = '%s | SMA1 = %d, SMA2 = %d' % (self.symbol, self.SMA1, self.SMA2)
        self.results[['creturns', 'cstrategy']].plot(title=title, figsize=(10, 6))

    def update_and_run(self, SMA):
        self.set_parameters(int(SMA[0]), int(SMA[1]))
        return -self.run_strategy()[0]

    def optimize_parameters(self, SMA1_range, SMA2_range):
        opt = brute(self.update_and_run, (SMA1_range, SMA2_range), finish=None)
        return opt, -self.update_and_run(opt)


if __name__ == '__main__':
    SMA1 = 56
    SMA2 = 59
    smabt = SMAVectorBacktester('EURUSD=X', SMA1, SMA2, '2020-07-06', '2021-07-06')

    #param = smabt.optimize_parameters((5, 200, 1), (5, 200, 1))
    #smas = param[0]
    #SMA1 = smas[0].astype(np.int_)
    #SMA2 = smas[1].astype(np.int_)
    print(smabt.run_strategy())
    smabt.plot_results()
    plt.show()
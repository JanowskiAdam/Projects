import yfinance as yf
import yahoo_fin.stock_info as si
import datetime
from pylab import plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import random

class DeepLearnVectorBacktester(object):

    def __init__(self, symbol, start, cutoff, end, amount, tc):
        self.symbol = symbol
        self.start = start
        self.cutoff = cutoff
        self.end = end
        self.amount = amount
        self.tc = tc
        self.results = None
        self.tresults = None
        self.get_data()

    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        raw = si.get_data(self.symbol, '2017-09-29', datetime.datetime.today(), index_as_date=True, interval='1d')[
            'adjclose']
        #raw = yf.download(self.symbol, start=self.start, end=self.end)['Adj Close']
        raw = pd.DataFrame(raw)
        raw.rename(columns={'adjclose': 'price'}, inplace=True)
        raw['return'] = np.log(raw['price'] / raw['price'].shift(1))
        raw['direction'] = np.where(raw['return']>0,1,0)
        raw['momentum'] = raw['return'].rolling(5).mean().shift(1)
        raw['volatility'] = raw['return'].rolling(20).std().shift(1)
        raw['distance'] = (raw['price'] -
                           raw['price'].rolling(50).mean()).shift(1)
        self.data = raw.dropna()

    def select_data(self, start, end):
        ''' Selects sub-sets of the financial data.
        '''
        data = self.data[(self.data.index >= start) &
                         (self.data.index <= end)].copy()
        return data

    def prepare_lags(self, start, end):
        ''' Prepares the lagged data for the regression and prediction steps.
        '''
        self.cols = []
        self.lags = 5
        for lag in range(1, self.lags + 1):
            col = f'lag_{lag}'
            self.data[col] = self.data['return'].shift(lag)
            self.cols.append(col)
        self.data.dropna(inplace=True)
        #self.lagged_data = self.data

    def set_seeds(seed=100):
        random.seed(seed)
        tf.random.set_seed(100)

    def fit_model(self, start, end):
        ''' Implements the model.
        '''
        self.prepare_lags(start, end)
        self.set_seeds()
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_shape=(len(self.cols),)))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        self.training_data = self.data[self.data.index < self.cutoff].copy()
        mu, std = self.training_data.mean(), self.training_data.std()
        self.training_data_ = (self.training_data - mu) / std
        self.test_data = self.data[self.data.index >= self.cutoff].copy()
        self.test_data_ = (self.test_data - mu) / std

        self.model.fit(self.training_data[self.cols],
                  self.training_data['direction'],
                  epochs=25, verbose=False,
                  validation_split=0.2, shuffle=False)

    def run_strategy(self):
        '''Backtests the trading strategy.
        '''
        self.fit_model(self.start, self.end)
        res = pd.DataFrame(self.model.history.history)
        res[['accuracy', 'val_accuracy']]
        self.model.evaluate(self.training_data[self.cols], self.training_data['direction'])
        pred = np.where(self.model.predict(self.training_data_[self.cols]) > 0.5, 1, 0)
        self.training_data['prediction'] = np.where(pred > 0, 1, -1)
        self.training_data['strategy'] = (self.training_data['prediction'] *
                                          self.training_data['return'])
        self.results = self.training_data

        self.model.evaluate(self.test_data[self.cols], self.test_data['direction'])
        pred = np.where(self.model.predict(self.test_data_[self.cols]) > 0.5, 1, 0)
        self.test_data['prediction'] = np.where(pred > 0, 1, -1)
        self.test_data['strategy'] = (self.test_data['prediction'] *
                                 self.test_data['return'])

        # absolute performance of the strategy
        aperf = self.test_data['strategy'].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - self.test_data['return'].iloc[-1]
        self.results = self.training_data

        return round(aperf, 2), round(operf, 2)

    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol.
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        plt.style.use('ggplot')
        title = 'Training data %s' % (self.symbol)
        self.results[['return', 'strategy']].cumsum().apply(np.exp).plot(title=title, figsize=(10, 6))
        title = 'Test data %s' % (self.symbol)
        self.test_data[['return', 'strategy']].cumsum().apply(np.exp).plot(title=title,figsize=(10, 6));

if __name__ == '__main__':
    DL = DeepLearnVectorBacktester('DNP.WA', '2017-1-1', '2019-12-31', '2021-1-1',
                                   10000, 0.000)
    print(DL.run_strategy())
    DL.plot_results()
    plt.show()
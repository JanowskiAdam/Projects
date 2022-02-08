# Trading_Stretegies_Backtesting
# This project is a backtesting of several strategies based on technical analysis.

Files
===================================================================================================
Backtesting_Base.py 
Base class for event-based backtesting of trading strategies.

LongOnlyBacktest.py 
Class backtesting allowing only long position based on an SMA strategy.

LongShortBacktest.py 
Class backtesting allowing both long and short positions based on an SMA strategy.

LRVectorBacktester.py 
Class for the vectorized backtesting of linear regression-based trading strategies.

MomVectorBacktester.py 
Class for the vectorized backtesting of momentum-based trading strategies.

MRVecorBacktester.py 
Class for the vectorized backtesting of mean reversion-based trading strategies based on a MomVectorBacktester class.

SMAVectorBacktester.py 
Class for the vectorized backtesting of SMA-based trading strategies. Enables optimize choosing SMA using brute force algorithm.

ScikitVectorBacktester.py
Class for the vectorized backtesting of Regression/Logistic strategies using ScikitLearn.

DeepLearninVectorBacktester.py
Class for the vectorized backtesting of Deep Learning strategies using Keras.

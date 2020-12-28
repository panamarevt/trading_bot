# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 21:58:19 2020

@author: Taras
"""

import numpy as np
import backtrader as bt
import datetime
import pandas as pd
import indicators

'''
This is the two-sigma strategy.
Idea. If the price goes below the open price for the current time-frame by the value of 2 standard deviations from its moving average we buy,
wait until the price to its original value (or some fraction of it) and sell.
Use take profit and stop-loss the same. The value of 2-sigma should be at least 1%.
'''

#class RSIStrategy(bt.Strategy):
#    
#    def __init__(self):
#        #self.rsi = bt.talib.RSI(self.data, period=14)
#        self.rsi = bt.ind.RSI(self.data, period=14)
#
#    def next(self):
#        if self.rsi < 30 and not self.position:
#            self.buy(size=1)
#        
#        if self.rsi > 70 and self.position:
#            self.close()


class TwoSigmaStrategy(bt.Strategy):
    
    def __init__(self):
        #self.rsi = bt.talib.RSI(self.data, period=14)
        self.mid = bt.ind.SimpleMovingAverage(self.data, period=20)
        self.sigma = bt.ind.StandardDeviation(self.data, period=20)
        self.sigma_pc = 100 * (self.mid - self.sigma) / self.mid
        self.open = self.datas[0].open
        self.close = self.datas[0].close
        self.high = self.datas[0].high
        self.low = self.datas[0].low
        self.buy_price = self.open - 1*self.sigma
        self.profit_price = self.buy_price + 0.5*self.sigma
        self.loss_price = self.buy_price - 0.5*self.sigma
               

    def next(self):
        if self.close < self.buy_price and self.sigma_pc > 1.0 and not self.position:
            
            self.buy(size=1)
        
        if (self.high > self.profit_price and self.position) or (self.low <= self.loss_price and self.position):
            self.close()

cerebro = bt.Cerebro()

fromdate = datetime.datetime.strptime('2020-11-01', '%Y-%m-%d')
todate = datetime.datetime.strptime('2020-12-25', '%Y-%m-%d')


data = bt.feeds.GenericCSVData(dataname='BTCUSDT_15MinuteBars.csv', compression=15, timeframe=bt.TimeFrame.Minutes, fromdate=fromdate, todate=todate)

cerebro.adddata(data)

cerebro.addstrategy(TwoSigmaStrategy)

cerebro.run()

cerebro.plot()
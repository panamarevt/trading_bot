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
wait until the price goes back to its original value (or some fraction of it) and sell.
Use the same take profit and stop-loss. The value of 2-sigma should be at least 1%.
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

class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        if self.dataclose[0] < self.dataclose[-1]:
            # current close less than previous close

            if self.dataclose[-1] < self.dataclose[-2]:
                # previous close less than the previous close

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.buy()

class TwoSigmaStrategy(bt.Strategy):
    
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
    
    def __init__(self):
        #self.rsi = bt.talib.RSI(self.data, period=14)
        print("Initializing the strategy...")
        self.mid = bt.ind.SimpleMovingAverage(self.data, period=20)
        self.sigma = bt.ind.StandardDeviation(self.data, period=20)
        self.sigma_pc = 100 * (self.mid - self.sigma) / self.mid
        self.open = self.datas[0].open
        self.close = self.datas[0].close
        self.high = self.datas[0].high
        self.low = self.datas[0].low
        self.down = 100*(self.open - self.low) / self.open
        self.buy_price = self.open - 1*self.sigma
        self.order = None
        #self.profit_price = self.buy_price + 0.5*self.sigma
        #self.loss_price = self.buy_price - 0.5*self.sigma
        print(self.close)
               

    def next(self):
        #if self.close[0] < self.open[0] and self.sigma_pc[0] > 1.0 and not self.position:
        self.log('Close, %.2f' % self.close[0])
        
        print(len(self))
        print(self.order )
        print(self.position)
        
        if self.down > 1.0 and not self.position:
            #print(self.buy_price[0])
            self.log('BUY CREATE, %.2f' % self.close[0])
            print(self.open[0])
            print(self.close[0])
            print(self.low[0])
            print(self.down[0])
            #print(self.sigma[0])
            
            #print("Create BUY order", self.close[0])
            print(self.position)
            self.enter = self.low[0]
            self.profit_price = (1+0.01)*self.enter
            self.loss_price = (1-0.01)*self.enter
            
            self.buy(size=1)
            print(self.position)
        
        if self.position:
            if (self.high[0] > self.profit_price):  
                print("PROFIT", self.close[0])
                self.close()
            elif (self.low[0] <= self.loss_price):
                print("LOSS")
                self.close()


cerebro = bt.Cerebro()

fromdate = datetime.datetime.strptime('2020-11-01', '%Y-%m-%d')
todate = datetime.datetime.strptime('2020-11-30', '%Y-%m-%d')

cerebro.broker.set_cash(100000)

data = bt.feeds.GenericCSVData(dataname='BTCUSDT_15MinuteBars.csv', compression=15, timeframe=bt.TimeFrame.Minutes, fromdate=fromdate, todate=todate)

cerebro.adddata(data)

cerebro.addstrategy(TwoSigmaStrategy)

cerebro.run()

cerebro.plot()
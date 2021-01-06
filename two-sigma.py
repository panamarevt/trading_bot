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
        # Initialize indicators:
        self.mid = bt.ind.SimpleMovingAverage(self.data1, period=20)
        self.sigma = bt.ind.StandardDeviation(self.data1, period=20)
        # One standard deviation 
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



class BuyDipStrategy(bt.Strategy):
    params = (
        ('pcdown', 3),
        ('take_profit', 3),
        ('stop_loss', 3),
    )
    
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.strftime("%d %b %Y %H:%M:%S"), txt))
        
    def __init__(self):
        
        # Initialize open price for higher time-frame (e.g. 1-hour)
        #self.op_1hr = self.datas[1].open
        #self.op_1hr = self.data1.open
        # Close price for lower time-frame (should be 1-minute for better realism)
        #self.close_1m = self.datas[0].close
        self.open_1m = self.data.open
        self.close_1m = self.data.close
        # Compute by how many pc the price is down
        #self.down_by = 100 * (self.op_1hr - self.close_1m) / self.op_1hr
        
        self.count = 0
        
        
    def next(self):
        
        # A stupid way to check open price for 1-hr candle
        if self.count == 60:
            self.count = 0

        if self.count == 0:
            self.op_1hr = self.open_1m[0]
                
        if not self.position:
            self.down_by = 100 * (self.close_1m[0] - self.op_1hr ) / self.op_1hr
            #self.log('Price change, %.2f' % self.down_by)
            
            if self.down_by < -1*self.params.pcdown:
                
                self.log('BUY CREATE, %.2f' % self.close_1m[0])
                self.buy_price = self.close_1m[0]
                self.buy()
        
        if self.position:
            # By how many pc the price is up or down at the moment:
            curr_up_down_pc = 100* (self.close_1m[0] - self.buy_price)/ self.buy_price
            # Conditions for take profit or stop loss
            PROFIT =  curr_up_down_pc > self.params.take_profit
            LOSS = curr_up_down_pc <= -1*self.params.stop_loss
            
            if PROFIT:            
                self.log(f'PROFIT! Buy: {self.buy_price:.2f}, Sell: {self.close_1m[0]:.2f}, UP {curr_up_down_pc:.2f}')
                #self.log('SELL CREATE, %.2f' % self.close_1m[0])
                self.close()
            elif LOSS:
                self.log(f'LOSS! Buy: {self.buy_price:.2f}, Sell: {self.close_1m[0]:.2f}, DOWN {curr_up_down_pc:.2f}')
                self.close()
                
        self.count += 1

# Instantiate the main class 
cerebro = bt.Cerebro()

fromdate = datetime.datetime.strptime('2021-01-01', '%Y-%m-%d')
todate = datetime.datetime.strptime('2021-01-06', '%Y-%m-%d')

cerebro.broker.set_cash(100000)
cerebro.broker.setcommission(commission=0.00075)

data = bt.feeds.GenericCSVData(dataname='BTCUSDT_1MinuteBars.csv',  
                               timeframe=bt.TimeFrame.Minutes, compression=1, fromdate=fromdate, todate=todate)



cerebro.adddata(data)
#cerebro.resampledata(data, timeframe = bt.TimeFrame.Minutes, compression=60)

cerebro.addstrategy(BuyDipStrategy)


print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Run over everything
cerebro.run()

# Print out the final result
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

cerebro.plot()
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
# Make it possible to buy fractional sizes of shares (in our case cryptocurrencies)
class CommInfoFractional(bt.CommissionInfo):
    def getsize(self, price, cash):
        '''Returns fractional size for cash operation @price'''
        return self.p.leverage * (cash / price)

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


class TenUpStrategy(bt.Strategy):
    params = (
        ('pcup', 10),
        ('take_profit', 10),
        ('stop_loss', 10),
        ('max_days', 1),
    )    
    
    
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
    
    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.cl = self.datas[0].close
        #self.bar_executed = 0
    
    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.cl[0])

        if not self.position and ( self.cl[0] > self.cl[-1] > self.cl[-2] ):
            # Check if price went up in 2 days at least by pcup days
            if 100*(self.cl[0] - self.cl[-2])/self.cl[-2] > self.params.pcup :
                self.log('BUY CREATE, %.2f' % self.cl[0])
                self.bar_executed = len(self)
                self.buy()                
            # current close less than previous close

        if self.position:
            if len(self) == self.bar_executed + self.params.max_days:
                self.log('SELL CREATE, %.2f' % self.cl[0])                
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



class TwoSigmaStrategy(bt.Strategy):
    params = (
        ('time_period', 60), # At which time-frames to trade (in minutes)
        ('sigma_fac', 2), # Factor to multiply std
        ('fac_back', 0.5), # Fraction of original pc pullback we hope to be rocovered
        ('take_profit', None),
        ('stop_loss', 150),
        ('max_hold', 5), # Max number of hours to hold the position
        ('trend_cond', False), # Apply trend condition on not. Trend condition means to execute the trades only during the trend
    )    
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.strftime("%d %b %Y %H:%M:%S"), txt))
    
    def __init__(self):
        self.op = self.datas[0].open
        self.cl = self.datas[0].close
        self.count = 0
        self.cl_1hr = []
        self.hr_count = 0                

    def next(self):      
        # A stupid way to check open price for 1-hr candle
        
        if self.count == self.p.time_period:
            self.cl_1hr.append(self.cl[0])
            #self.log('Close, %.2f' % self.cl[0])
            self.count = 0
            self.hr_count += 1

        if self.count == 0:
            self.op_1hr = self.op[0]

        # Check if we reached minimum number of periods to compute mean and standard deviation:
        if len(self.cl_1hr) >= 20:
            #print("Now we can compute std ..")
            cl_1hr_tmp = np.array(self.cl_1hr[-20:])
            self.sigma = np.std(cl_1hr_tmp)
            self.mid = np.mean(cl_1hr_tmp)
            self.sigma_pc = 100 * self.sigma / self.mid
            #self.log(f'2-SIGMA: {pc_buy:.1f}; DOWN: {self.down:.1f}')
        else:
            self.count += 1
            return 0            
        
        # DEBUG
#        if self.count == 0:
#            self.log(f'MEAN: {self.mid:.2f}; STD: {self.sigma:.2f}; STD_pc: {self.sigma_pc:.2f}')       
        
        if not self.position:
            self.down = 100*(self.op_1hr - self.cl[0]) / self.op_1hr
            pc_buy = self.params.sigma_fac*self.sigma_pc
            #DEBUG:
#            if self.count == 0:                 
#                self.log(f'2-SIGMA: {pc_buy:.1f}; DOWN: {self.down:.1f}')
            if self.down > pc_buy :
            #print(self.buy_price[0])
                self.log('BUY CREATE, %.2f' % self.cl[0])
                # Fractional size:
                self.size = self.broker.get_cash() / self.cl[0]
                self.buy(size=self.size)
                # Remeber at what `hour` index the trade was executed:
                self.hour_executed = self.hr_count
                self.buy_price = self.cl[0]
                self.take_profit = self.params.take_profit or self.params.fac_back*pc_buy
                self.take_profit = (1+self.take_profit*0.01)*self.buy_price
                self.stop_loss = self.params.stop_loss or self.params.fac_back*pc_buy
                self.stop_loss = (1-self.stop_loss*0.01)*self.buy_price                
                #self.stop_loss = self.params.stop_loss or (1-self.params.fac_back*pc_buy*0.01)*self.buy_price
                #print(f'2-SIGMA: {pc_buy:.1f}; DOWN: {self.down:.1f}')
                #print(f'Profit target: {self.take_profit:.2f}; Loss target: {self.stop_loss:.2f}')
        
        if self.position:
            if (self.cl[0] > self.take_profit):  
                print("PROFIT", self.cl[0])
                self.close()
            elif (self.cl[0] <= self.stop_loss):
                print("LOSS")
                self.close()
            else:
                if (self.hr_count - self.hour_executed) >= self.params.max_hold:
                    self.close()
                    profit = 100*(self.cl[0] - self.buy_price)/self.buy_price
                    print(f'PROFIT/LOSS: {profit:.2f}%')
        self.count += 1



# Instantiate the main class 
cerebro = bt.Cerebro()

fromdate = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')
todate = datetime.datetime.strptime('2021-01-12', '%Y-%m-%d')

#cerebro.broker.set_cash(100000)
#cerebro.broker.set_cash(1000)
cerebro.broker.set_cash(1000)
#cerebro.addsizer(bt.sizers.FixedSize, stake=100)
cerebro.broker.setcommission(commission=0.00075)

# Fractional size to buy:
cerebro.broker.addcommissioninfo(CommInfoFractional())

#dataname = 'BTCUSDT_1MinuteBars.csv'
#dataname = 'LINKUSDT_1MinuteBars.csv'
dataname = 'BQXETH_1MinuteBars.csv'
#dataname = 'ETHBTC_1MinuteBars.csv'

data = bt.feeds.GenericCSVData(dataname=dataname,  
                               timeframe=bt.TimeFrame.Minutes, compression=1, fromdate=fromdate, todate=todate)



cerebro.adddata(data)
#cerebro.resampledata(data, timeframe = bt.TimeFrame.Minutes, compression=60)
#cerebro.resampledata(data, timeframe = bt.TimeFrame.Days, compression=1)

cerebro.addstrategy(TwoSigmaStrategy, time_period=15, fac_back=0.5)

#Optimize strategy
#cerebro.optstrategy(TwoSigmaStrategy, time_period=15, fac_back=(0.5, 0.75, 1.0))

# Add TimeReturn Analyzers to benchmark data
#Daily (or any other period) return on investment
cerebro.addanalyzer(
    bt.analyzers.TimeReturn, _name="daily_roi", timeframe=bt.TimeFrame.Months
)
# Statistics for periods
cerebro.addanalyzer(
    bt.analyzers.PeriodStats, _name="period", timeframe=bt.TimeFrame.Days
)
# All-time ROI
cerebro.addanalyzer(
    bt.analyzers.TimeReturn, _name="alltime_roi", timeframe=bt.TimeFrame.NoTimeFrame
)
# Return on buy-and-hold strategy
cerebro.addanalyzer(
    bt.analyzers.TimeReturn,
    data=data,
    _name="benchmark",
    timeframe=bt.TimeFrame.NoTimeFrame,
)

#print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
#
## Run over everything
#cerebro.run()
#
## Print out the final result
#print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())


results = cerebro.run()
st0 = results[0]

for alyzer in st0.analyzers:
    alyzer.print()

#cerebro.plot()
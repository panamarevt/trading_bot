# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 21:58:19 2020

@author: Taras
"""

import numpy as np
import backtrader as bt
import datetime
#import pandas as pd
#import indicators
import os

import argparse
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


class BitmexComissionInfo(bt.CommissionInfo):
    params = (
        ("commission", 0.00075),
        ("mult", 1.0),
        ("margin", None),
        ("commtype", None),
        ("stocklike", False),
        ("percabs", False),
        ("interest", 0.0),
        ("interest_long", False),
        ("leverage", 1.0),
        ("automargin", False),
    )
    def getsize(self, price, cash):
        """Returns fractional size for cash operation @price"""
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
        ('symbol', 'btcusdt'),
        ('time_period', 60), # At which time-frames to trade (in minutes)
        ('sigma_fac', 2), # Factor to multiply std
        ('sigma_max', 100), # Max value for std in % (100 means no limit)
        ('sigma_min', 1),
        ('fac_back', 0.5), # Fraction of original pc pullback we hope to be rocovered
        ('take_profit', None),
        ('stop_loss', 150),
        ('max_hold', 5), # Max number of hours to hold the position
        ('trend_cond', False), # Apply trend condition on not. Trend condition means to execute the trades only during the trend
        ('save_stats', None)
    )    
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.datetime(0)
        print('%s, %s' % (dt.strftime("%d %b %Y %H:%M:%S"), txt))
    
    def __init__(self):
        self.op = self.datas[0].open
        self.cl = self.datas[0].close
        self.hi = self.datas[0].high
        self.lo = self.datas[0].low
        self.count = 0
        self.cl_1hr = []
        self.hr_count = 0      
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.val_start = self.broker.get_cash() # keep the starting cash
        self.commision = 0.00075
        self.total_comm = 0
        self.true_cash = self.broker.get_cash()
        self.sellprice = None
        self.cost = None
        self.ntrades=0

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                order.executed.comm = order.executed.value*self.commision
                self.log(
                    'BUY EXECUTED, Price: %.8f, Cost: %.8f, Comm %.8f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.total_comm += order.executed.value*self.commision
            else:  # Sell
                order.executed.comm = order.executed.value*self.commision
                self.log('SELL EXECUTED, Price: %.8f, Cost: %.8f, Comm %.8f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
                self.total_comm += order.executed.value*self.commision
                self.sellprice = order.executed.price
                self.cost = order.executed.value
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Rejected]:
            #print(order.status)
            self.log('Order Canceled/Rejected')
        elif order.status in [order.Margin]:
            self.log('Order Margin')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))
        
    
    def record_trade(self):
        '''Save basic statistics for each trade in csv format'''
        # Save trade or not?
        self.ntrades += 1
        if not self.p.save_stats: return
                
        self.cost = self.true_cash
        trade_pnl = (self.sellprice - self.buyprice)/self.buyprice
        gross = trade_pnl * self.cost
        fee = 2*self.cost*self.commision
        net = gross - fee
        self.true_cash += net
        
        with open(self.p.save_stats, 'a') as f:
            empty = os.path.getsize(self.p.save_stats) == 0
            if empty:
                f.write("buytime,symbol,buyprice,pnl,gross,net,fee,cash,sigma,mindown,maxup\n")
            f.write(f"{self.buy_time},{self.p.symbol},{self.buyprice:.8f},")
            f.write(f"{100*trade_pnl:.2f},{gross:.2f},{net:.2f},{fee:.2f},")
            f.write(f"{self.true_cash:.8f},{self.sigma_pc_exec:.2f},{self.mindown:.2f},{self.maxup:.2f}")
            f.write("\n")
        
    

    def next(self):      
        # A stupid way to check open price for 1-hr candle
        if len(self) == 1:
            self.start_day = self.datas[0].datetime.date(0).isoformat()
        if self.order:
            return
        
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
        
        if not self.position:
            self.down = 100*(self.op_1hr - self.cl[0]) / self.op_1hr
            pc_buy = self.params.sigma_fac*self.sigma_pc
            #DEBUG:
#            if self.count == 0:                 
#                self.log(f'2-SIGMA: {pc_buy:.1f}; DOWN: {self.down:.1f}')
            if (self.down > pc_buy) and (self.p.sigma_min < self.sigma_pc < self.p.sigma_max) :
            #print(self.buy_price[0])
                print(f"Current cash: {self.true_cash:.2f}")    
                self.log('BUY CREATE, %.8f' % self.cl[0])
                # Fractional size:
                self.size = 0.99*self.broker.get_cash() / self.cl[0] # mult. by 0.99 to save for commission
                #self.size = self.true_cash / self.cl[0]
                self.order = self.buy(size=self.size, exectype=bt.Order.Market)
                self.buy_time = self.datas[0].datetime.datetime(1).strftime("%d %b %Y %H:%M:%S")
                self.buy_price = self.op[1]
                self.bar_executed = len(self)
                trade_tmp = 60*self.p.max_hold
                # Form temporary arrays of highs and lows (slicing thhrough self.hi[:] doesn't work for some reason)
                high_tmp = np.array([self.hi[i] for i in range(1,trade_tmp)])
                #low_tmp = np.array([self.lo[i] for i in range(1,trade_tmp)])
                self.max_price = high_tmp.max()
                #self.min_price = low_tmp.min()
                self.maxup = 100*(self.max_price - self.buy_price)/self.buy_price
                #self.mindown = 100*(self.min_price - self.buy_price)/self.buy_price
                self.sigma_pc_exec = self.sigma_pc
                # Remeber at what `hour` index the trade was executed:
                self.hour_executed = self.hr_count
                #self.buy_price = self.cl[0]           
                take_profit = self.params.take_profit or self.params.fac_back*pc_buy
                self.profit_target = (1+take_profit*0.01)*self.buy_price
                stop_loss = self.params.stop_loss or self.params.fac_back*pc_buy
                
                self.loss_target = (1-stop_loss*0.01)*self.buy_price
                print(f"Profit target: {self.profit_target:.8f}; Loss target:{self.loss_target:.8f}")                
        
        else:
            if (self.cl[0] > self.profit_target):  
                print("PROFIT", self.cl[0])                
                trade_tmp = len(self) - self.bar_executed
                #print(trade_tmp)
                if trade_tmp <= 1:
                    low_tmp = self.lo[0]
                else:
                    low_tmp = np.array([self.lo[-i] for i in range(0,trade_tmp)]).min()
                self.mindown = 100*(low_tmp - self.buy_price)/self.buy_price
                self.close()
                self.sellprice = self.profit_target
                self.record_trade()
            elif (self.cl[0] <= self.loss_target):
                print("LOSS")                
                trade_tmp = len(self) - self.bar_executed
                if trade_tmp <= 1:
                    low_tmp = self.lo[0]
                else:
                    low_tmp = np.array([self.lo[-i] for i in range(0,trade_tmp)]).min()
                self.mindown = 100*(low_tmp - self.buy_price)/self.buy_price             
                self.close()
                self.sellprice = self.loss_target
                self.record_trade()
            else:
                if (self.hr_count - self.hour_executed) >= self.params.max_hold: 
                    trade_tmp = len(self) - self.bar_executed
                    if trade_tmp <= 1:
                        low_tmp = self.lo[0]
                    else:
                        low_tmp = np.array([self.lo[-i] for i in range(0,trade_tmp)]).min()
                    self.mindown = 100*(low_tmp - self.buy_price)/self.buy_price
                    self.close()
                    self.sellprice = self.cl[0]
                    self.record_trade()
                    profit = 100*(self.cl[0] - self.buy_price)/self.buy_price
                    print(f'PROFIT/LOSS: {profit:.2f}%')
        self.count += 1

    def stop(self):
        """ Calculate the actual returns """
        #val_end = self.true_cash - self.val_start
        self.roi = ( self.true_cash / self.val_start) - 1.0
        
        print(
            f"ROI: {100.0 * self.roi:.2f}%, Start cash {self.val_start:.2f}, "
            f"End cash: {self.true_cash:.2f},  Total comm: {self.total_comm:.2f}"
        )
        
        summary_file = 'summary_all.dat'
        with open(summary_file, 'a') as f:
            empty = os.path.getsize(summary_file) == 0
            if empty:
                f.write(f"symbol,interval,maxhold,sigma_min,facback,ntrades,ROI,start,fees\n")
            f.write(f"{self.p.symbol},{self.p.time_period},{self.p.max_hold},{self.p.sigma_min},")
            f.write(f"{self.p.fac_back:.2f},{self.ntrades},{100*self.roi:.2f}%,")
            f.write(f"{self.start_day},{self.total_comm:.2f}\n")



def parse_args():
    ''' Main parameters of the strategy:
    sigma_fac
    fac_back
    max_hold
    '''
    parser = argparse.ArgumentParser(
        description='Backtest for the 2-Sigma Strategy',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        '--symbol', '-s',
        default='btcusdt',
        help='Trading symbol, str, e.g.: btcusdt')

    parser.add_argument(
        '--interval', '-i',
        default=15, type=int,
        help='Period interval to trade on, in minutes. Accepted values:1,3,5,15,30,60')

    parser.add_argument(
        '--sigma', '-x',
        default=2.0, type=float,
        help='Factor to multiply one standard deviation')    
    
    parser.add_argument(
        '--fback', '-f',
        default=0.5, type=float,
        help='Factor for expected recovery from the buy price')

    parser.add_argument(
        '--maxhold', '-m',
        default=5, type=int,
        help='Maximum number of bars to hold the position')

    parser.add_argument(
        '--cash', '-c',
        default=1000, type=float,
        help='Amount of cash to start trading with')

    parser.add_argument(
        '--log', '-l',
        default=None,
        help='Filename to save logging data. If None the default filename is used: {symbol}_{interval}.log')         

    return parser.parse_args()


def run(args, symb=None):
    
    # Instantiate the main class 
    cerebro = bt.Cerebro()
    
    fromdate = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')
    todate = datetime.datetime.strptime('2021-01-12', '%Y-%m-%d')
    
    cerebro.broker.set_cash(1000)
    
    # Fractional size to buy:
    cerebro.broker.addcommissioninfo(CommInfoFractional())
    
    #symbol='linkusdt'
    ##symbol='bqxeth'
    ##symbol='dotbnb'
    ##symbol='cvcbtc'
    #
    #time_period = 30
    #fac_back=0.5
    #max_hold=5
    
    sigma_max = 100
    time_period = args.interval
    symbol = symb or args.symbol # Check if we want to pass another symbol to the test
    fac_back = args.fback
    max_hold = args.maxhold
    
    
    save_stats = f"{symbol}_{time_period}m_fback{fac_back}_hold{max_hold}_smax{sigma_max}.csv"
    #dataname = 'BTCUSDT_1MinuteBars.csv'
    dataname = f'{symbol.upper()}_1MinuteBars.csv'
    #dataname = 'BQXBTC_1MinuteBars.csv'
    #dataname = 'ETHBTC_1MinuteBars.csv'
    
    data = bt.feeds.GenericCSVData(dataname=dataname,  
                                   timeframe=bt.TimeFrame.Minutes, compression=1, fromdate=fromdate, todate=todate)
    
    
    
    cerebro.adddata(data)
    #cerebro.resampledata(data, timeframe = bt.TimeFrame.Minutes, compression=60)
    #cerebro.resampledata(data, timeframe = bt.TimeFrame.Days, compression=1)
    
    cerebro.addstrategy(TwoSigmaStrategy, time_period=time_period, fac_back=fac_back, max_hold=max_hold, stop_loss=None,
                        sigma_max = sigma_max, save_stats=save_stats, symbol=symbol)
    #cerebro.broker.addcommissioninfo(CommInfoFractional())
    
    #cerebro.broker.addcommissioninfo(BitmexComissionInfo())
    #Optimize strategy
    #cerebro.optstrategy(TwoSigmaStrategy, time_period=15, fac_back=(0.5, 0.75, 1.0))
    
    # Add TimeReturn Analyzers to benchmark data
    #Daily (or any other period) return on investment
    cerebro.addanalyzer(
        bt.analyzers.SQN, _name="sqn")
    
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
    
    cerebro.broker.set_checksubmit(False)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    #
    
    results = cerebro.run()
    
    print('----------------------------------------------------------------------------')
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    st0 = results[0]
    
    for alyzer in st0.analyzers:
        alyzer.print()

#cerebro.plot()
        
        
if __name__=='__main__':
    
    #Get Args
    args = parse_args()
    
    #Run the whole thing
    run(args, symb=args.symbol)    
        
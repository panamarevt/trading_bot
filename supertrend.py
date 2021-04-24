# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 16:29:11 2021

@author: Taras
"""

import numpy as np
import backtrader as bt
import backtrader.indicators as btind
import datetime

import time

# Make it possible to buy fractional sizes of shares (in our case cryptocurrencies)
class BitmexComissionInfo(bt.CommissionInfo):
    params = (
        ("commission", 0.075),
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

# Define SuperTrend indicator
#class SuperTrend(bt.Indicator):
#
#    lines = ('top', 'bot')
#    params = (
#            ('period', 10),
#            ('mult', 3), 
#    )
#    plotinfo = dict(subplot=False)
#
#    def __init__(self):
#        
#        atr = btind.ATR(self.data, period=self.p.period)
#
#        self.lines.top = (self.data.high + self.data.low) / 2.0 + self.p.mult * atr
#        self.lines.bot = (self.data.high + self.data.low) / 2.0 - self.p.mult * atr


# SuperTrend indicators taken from: 
# https://github.com/mementum/backtrader/pull/374/files
class SuperTrendBand(bt.Indicator):
    """
    Helper inidcator for Supertrend indicator
    
    """
    params = (('period',10),('multiplier',3))
    lines = ('basic_ub','basic_lb','final_ub','final_lb')
    plotinfo = dict(subplot=False)

    def __init__(self):
        self.atr = bt.indicators.AverageTrueRange(period=self.p.period)
        self.l.basic_ub = ((self.data.high + self.data.low) / 2) + (self.atr * self.p.multiplier)
        self.l.basic_lb = ((self.data.high + self.data.low) / 2) - (self.atr * self.p.multiplier)

    def next(self):
        if len(self)-1 == self.p.period:
            self.l.final_ub[0] = self.l.basic_ub[0]
            self.l.final_lb[0] = self.l.basic_lb[0]
        else:
            #=IF(OR(basic_ub<final_ub*,close*>final_ub*),basic_ub,final_ub*)
            if self.l.basic_ub[0] < self.l.final_ub[-1] or self.data.close[-1] > self.l.final_ub[-1]:
                self.l.final_ub[0] = self.l.basic_ub[0]
            else:
                self.l.final_ub[0] = self.l.final_ub[-1]

            #=IF(OR(baisc_lb > final_lb *, close * < final_lb *), basic_lb *, final_lb *)
            if self.l.basic_lb[0] > self.l.final_lb[-1] or self.data.close[-1] < self.l.final_lb[-1]:
                self.l.final_lb[0] = self.l.basic_lb[0]
            else:
                self.l.final_lb[0] = self.l.final_lb[-1]


class SuperTrend(bt.Indicator):
    """
    Super Trend indicator
    """
    params = (('period', 10), ('multiplier', 3))
    lines = ('super_trend',)
    plotinfo = dict(subplot=False)

    def __init__(self):
        self.stb = SuperTrendBand(period = self.p.period, multiplier = self.p.multiplier)

    def next(self):
        if len(self) - 1 == self.p.period:
            self.l.super_trend[0] = self.stb.final_ub[0]
            return

        if self.l.super_trend[-1] == self.stb.final_ub[-1]:
            if self.data.close[0] <= self.stb.final_ub[0]:
                self.l.super_trend[0] = self.stb.final_ub[0]
            else:
                self.l.super_trend[0] = self.stb.final_lb[0]

        if self.l.super_trend[-1] == self.stb.final_lb[-1]:
            if self.data.close[0] >= self.stb.final_lb[0]:
                self.l.super_trend[0] = self.stb.final_lb[0]
            else:
                self.l.super_trend[0] = self.stb.final_ub[0]        
 

       
class SuperTrendStrategy(bt.Strategy):    
    params = (('strendperiod', 10),
              ('strendmult', 3),
              ('take_profit', 0.016),
              ('atr_fac_prof', 1.0),
              ('stop_loss', 0.03),
              ('atr_fac_loss', 1.0),
              ('side', 'long'),
              )

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.op = self.datas[0].open
        self.cl = self.datas[0].close
        self.hi = self.datas[0].high
        self.lo = self.datas[0].low
        
        #self.supertrend_band = SuperTrendBand(self.data, period=self.p.strendperiod, multiplier=self.p.strendmult)
        self.supertrend = SuperTrend(self.data, period=self.p.strendperiod, multiplier=self.p.strendmult)
        self.atr = bt.indicators.AverageTrueRange(period=self.p.strendperiod)
        self.uptrend = self.op > self.supertrend.lines.super_trend
        
        self.profit_target = None
        self.loss_target = None
        self.trade_type = None # Mark if we are in short or in long position


    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.datetime(0)
        print('%s, %s' % (dt.strftime("%d %b %Y %H:%M:%S"), txt))        

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.5f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                #self.total_comm += self.buycomm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.5f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

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

    def trend_signal(self, cond='touch', side='long'):
        '''Check if there is a BUY/SELL signal for a SuperTrend strategy
        returns True if the signal conditions are met, otherwise False.
        ------------------------
        parameters:
        cond - condition for the signal 
             = 'touch' - if last candle touched supertrend line, buy on open of current candle
             = 'green' - current candle closed green (red for short) after supertrend line, previous candle touched the supertrend line
        side - long or short strategy
             = 'long' - to check for LONG signal
             = 'short' to check for SHORT signal'''
        if cond == 'touch':
            if side=='long':
                signal = self.lo < self.supertrend.l.super_trend and self.cl > self.supertrend.l.super_trend
            if side=='short':
                signal = self.hi > self.supertrend.l.super_trend and self.cl < self.supertrend.l.super_trend
        if cond == 'green':
            if side=='long':
                signal = (self.lo[-1] < self.supertrend.l.super_trend[-1] and self.cl[-1] > self.supertrend.l.super_trend[-1]) \
                    and (self.cl > self.op and self.cl > self.supertrend.l.super_trend)
            if side=='short':
                signal = (self.hi[-1] > self.supertrend.l.super_trend[-1] and self.cl[-1] < self.supertrend.l.super_trend[-1]) \
                    and (self.cl < self.op and self.cl < self.supertrend.l.super_trend)                
        return signal
        
        
    def next(self):
        if not self.position:
            if self.uptrend and (self.p.side == 'long' or self.p.side == 'both'):
                #if self.lo < self.supertrend.l.super_trend and self.cl > self.supertrend.l.super_trend :
#                if self.lo[-1] < self.supertrend.l.super_trend[-1] and self.cl[-1] > self.supertrend.l.super_trend[-1] :
#                    if self.cl > self.op and self.cl > self.supertrend.l.super_trend:
                signal_long = self.trend_signal(cond='touch', side='long')
                if signal_long:
                        self.log(f"Price signal!  Buy at {self.cl[0]}")
                        if self.p.take_profit != 'atr' : # if take profit is a fixed number, just use it as %
                            print(self.p.take_profit)
                            print(type(self.p.take_profit))
                            self.profit_target = (1+self.p.take_profit)*self.cl[0]
                        else: # if profit depends on ATR:
                            self.profit_target = self.cl[0] + self.p.atr_fac_prof*self.atr
                        if self.p.stop_loss != 'atr' :
                            self.loss_target = (1-self.p.stop_loss)*self.cl[0]
                        else:
                            self.loss_target = self.cl[0] - self.p.atr_fac_loss*self.atr
                        self.size = 0.99*self.broker.get_cash() / self.cl[0] # mult. by 0.99 to save for commission
                        self.buy(size=self.size, exectype=bt.Order.Market)
                        self.trade_type = 'long'
                        #if self.p.side == 'long': return                        
            if (not self.uptrend) and (self.p.side == 'short'or self.p.side == 'both'):
                signal_short = self.trend_signal(cond='touch', side='short')
                if signal_short:
                        self.log(f"Price signal!  Sell at {self.cl[0]}")
                        if self.p.take_profit != 'atr' : # if take profit is a fixed number, just use it as %
                            self.profit_target = (1-self.p.take_profit)*self.cl[0]                        
                        else:
                            self.profit_target = self.cl[0] - self.p.atr_fac_prof*self.atr
                        if self.p.stop_loss != 'atr' :
                            self.loss_target = (1+self.p.stop_loss)*self.cl[0]
                        else:
                            self.loss_target = self.cl[0] + self.p.atr_fac_loss*self.atr
                        self.size = 0.99*self.broker.get_cash() / self.cl[0] # mult. by 0.99 to save for commission
                        self.sell(size=self.size, exectype=bt.Order.Market)
                        self.trade_type = 'short'
                             
        else:
            if self.trade_type == 'long': # if we are in a `long` position
                if (self.cl[0] > self.profit_target):  
                    self.log(f"PROFIT! {self.cl[0]}")   
                    self.close()
                    self.profit_target = None
                elif (self.cl[0] <= self.loss_target): 
                    self.log(f"LOSS! {self.cl[0]}")   
                    self.close()
                    self.loss_target = None      
            if self.trade_type == 'short': # if we are in a `short` position
                if (self.cl[0] < self.profit_target):  
                    self.log(f"PROFIT! {self.cl[0]}")   
                    self.close()
                    self.profit_target = None
                elif (self.cl[0] >= self.loss_target): 
                    self.log(f"LOSS! {self.cl[0]}")   
                    self.close()
                    self.loss_target = None                 


def run(symbol, interval, side='long', strendmult=3, take_profit=0.015, stop_loss=0.03,
        atr_fac_prof=1.5, atr_fac_loss=1.5):
    
    print("<-!->")
    print("Starting new strategy...")
    print('Parameters:')
    print(f'symbol:{symbol}')
    print(f'interval:{interval}')
    print(f'side:{side}')
    print(f'strendmult:{strendmult}')
    print(f'take_profit:{take_profit}')
    print(f'stop_loss:{stop_loss}')
    print(f'atr_fac_prof:{atr_fac_prof}')
    print(f'atr_fac_loss:{atr_fac_loss}')
    
    cerebro = bt.Cerebro()
    
    fromdate = datetime.datetime.strptime('2021-04-23', '%Y-%m-%d')
    todate = datetime.datetime.strptime('2021-04-25', '%Y-%m-%d')
    
    cerebro.broker.set_cash(1000)
    
    #symbol = 'BTCUSDT'
    #symbol = 'BNBBTC'
    #symbol = 'ETHBTC'
    #symbol = 'BNBETH'
    #symbol = 'LINKETH'
    #symbol = 'BNBUSDT'
    #symbol = 'ETHUSDT'
    #symbol = 'AAVEBTC'
    #symbol = 'ADABNB'
    #symbol = 'CAKEBNB'
    #symbol = 'DOTBNB'
    
    dataname = f'{symbol.upper()}_1MinuteBars.csv'
    #dataname = 'BQXBTC_1MinuteBars.csv'
    #dataname = 'ETHBTC_1MinuteBars.csv'
    
    data = bt.feeds.GenericCSVData(dataname=dataname,  
                                   timeframe=bt.TimeFrame.Minutes, compression=1, fromdate=fromdate, todate=todate)
    
     
    #cerebro.adddata(data)
    cerebro.resampledata(data, timeframe = bt.TimeFrame.Minutes, compression=interval)
    
    #cerebro.addstrategy(SuperTrendStrategy, side='short', strendmult=2, take_profit = 0.03, stop_loss = 100, atr_fac_prof = 1.5, atr_fac_loss = 1)
    cerebro.addstrategy(SuperTrendStrategy, side=side, strendmult=strendmult, take_profit = take_profit, 
                        stop_loss = stop_loss, atr_fac_prof = atr_fac_prof, atr_fac_loss = atr_fac_loss)
    # Run over everything
    #cerebro.run()
    
    cerebro.broker.addcommissioninfo(BitmexComissionInfo())
    
    # Analyzers:
    cerebro.addanalyzer(
        bt.analyzers.SQN, _name="sqn")
    
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn, _name="daily_roi", timeframe=bt.TimeFrame.Months
    )
    # Statistics for periods
    cerebro.addanalyzer(
        bt.analyzers.PeriodStats, _name="period", timeframe=bt.TimeFrame.Months
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
    
    
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    #
    
    results = cerebro.run()
    
    print('----------------------------------------------------------------------------')
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    st0 = results[0]
    
    for alyzer in st0.analyzers:
        alyzer.print()
    
    # Plot the result
    cerebro.plot(style='candlestick')

def run_multitest(symbols, intervals, sides, mults, profits, losses, atr_profits, atr_losses):
    count = 0
    t_start = time.time()
    #Here comes our UGLY loop :-)
    for symbol in symbols:
        for interval in intervals:
            for mult in mults:
                for prof in profits:
                    for loss in losses:
                        for side in sides:
                            if prof == 'atr' or loss == 'atr':
                                for atr_prof in atr_profits:
                                    for atr_loss in atr_losses:
                                        count += 1
                                        print(f'###{count}###')
                                        run(symbol,interval,side=side,strendmult=mult,take_profit=prof,stop_loss=loss,
                                            atr_fac_prof=atr_prof,atr_fac_loss=atr_loss)
                            else:
                                count += 1
                                print(f'###{count}###')
                                run(symbol,interval,side=side,strendmult=mult,take_profit=prof,stop_loss=loss)
                                t_1st_loop = time.time() - t_start
                                print(f'Time taken for 1 loop (s): {t_1st_loop}')
        t_1st_symbol = time.time() - t_start
        print(f'Time taken for 1 symbol (min): {t_1st_symbol/60}')
    #Run the whole thing
    t_tot = time.time() - t_start  
    print(f'Total number of combinations: {count}')
    print(f'Total time taken (hr): {t_tot/3600}')    


if __name__=='__main__':
    
#    symbols = [f'{item}USDT' for item in ['BTC', 'ETH', 'BNB'] ]
#    symbols += [f'{item}ETH' for item in ['BNB', 'ADA', 'LINK', 'AAVE'] ]
#    symbols += [f'{item}BNB' for item in ['AAVE', 'DOT', 'ADA', 'BAND', 'CAKE'] ]
#    symbols += [f'{item}BTC' for item in ['ETH', 'BNB', 'ADA', 'DOT', 'AAVE', 'LINK', 'BAND', 'CAKE'] ]
#    
#    intervals = [15,30,60]
#    mults = [2,3]
#    profits = [0.015, 0.03, 0.05, 0.1, 'atr']
#    losses = [0.015, 0.03, 0.05, 100, 'atr']
#    atr_profits = [1, 2]
#    atr_losses = [1, 2]
#    sides = ['long','short','both']
    
    #run_multitest(symbols, intervals, sides, mults, profits, losses, atr_profits, atr_losses)
#    

    
    #run('DOTBTC', 15, side='both', strendmult=2, take_profit=0.05, stop_loss=100)
    #run('BANDBNB', 30, side='both', strendmult=2, take_profit=0.1, stop_loss=100, atr_fac_prof=1, atr_fac_loss=1)
    run('BNBUSDT', 15, side='both', strendmult=2, take_profit=0.015, stop_loss=100, atr_fac_prof=1, atr_fac_loss=2)
    #run('ETHUSDT', 15, side='long', strendmult=2, take_profit=0.05, stop_loss=0.05, atr_fac_prof=1, atr_fac_loss=2)
    
    
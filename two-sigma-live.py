# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:46:10 2021

@author: Taras
"""

import numpy as np
#import backtrader as bt
import datetime
#import pandas as pd
#import indicators

import websocket
import json
#import pprint
import time
import argparse

import binance_endpoints
'''
This is the two-sigma strategy.
Idea. If the price goes below the open price for the current time-frame by the value of 2 standard deviations from its moving average we buy,
wait until the price goes back to its original value (or some fraction of it) and sell.
Use the same take profit and stop-loss. 
'''

from binance.client import Client
from strategies import weighted_avg_orderBook

client = Client()

class TwoSigmaStrategy():
#    params = (
#        ('time_period', 60), # At which time-frames to trade (in minutes)
#        ('sigma_fac', 2), # Factor to multiply std
#        ('fac_back', 0.5), # Fraction of original pc pullback we hope to be rocovered
#        ('take_profit', None),
#        ('stop_loss', 150),
#        ('max_hold', 5), # Max number of hours to hold the position
#        ('trend_cond', False), # Apply trend condition on not. Trend condition means to execute the trades only during the trend
#    )    
#    
    def __init__(self, time_period=60, sigma_fac=2.0, fac_back=0.5,
                 take_profit=None, stop_loss=150, max_hold=5, 
                 trade_type='PAPER', deposit_fraction=0.1, symbol='ETHBTC',logfile=None):
        #self.op = self.datas[0].open
        #self.cl = self.datas[0].close
        # Main parameters of the strategy:
        self.time_period = time_period # At which time-frames to trade (in minutes)
        self.sigma_fac = sigma_fac # Factor to multiply std
        self.fac_back = fac_back # Fraction of original pc pullback we hope to be rocovered
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.max_hold = max_hold # Max number of candles to hold the position
        self.trade_type = trade_type
        self.deposit_fraction = deposit_fraction
        self.symbol = symbol
        self.logfile = logfile
        # Auxillary vars:
        self.count = 0
        self.cl_1hr = []
        self.hr_count = 0
        self.position = False
        self.cumulative_profit = 0
        self.total_fee = 0
        self.fee = 0.075*0.01
    
    def log(self, txt, dt=None, fname=None):
        ''' Logging function fot this strategy'''
        dt = dt or datetime.datetime.now()
        print('%s, %s' % (dt.strftime("%d %b %Y %H:%M:%S"), txt))           
        if fname:
            with open(fname, 'a') as f:
                f.write('%s, %s\n' % (dt.strftime("%d %b %Y %H:%M:%S"), txt)) 

    def market_buy_order(self):
        pass
    
    def market_sell_order(self):
        pass
    
    def place_sell_order(self):
        pass
    
    def check_order_status(self):
        pass
    
    def next(self, op, hi, lo, cl, op_time, last_20_closes):      
        '''Strategy decision-making for the received candlestick data
        params:
        ------------    
        op - open price for the current time-frame
        cl - close price for the current time-frame
        last_20_closes- list of last 20 close prices'''
        self.op_1hr = op
        self.cl = cl
        self.hi = hi
        self.lo = lo        
        
#        if self.count == self.time_period:
#            self.cl_1hr.append(self.cl[0])
#            #self.log('Close, %.2f' % self.cl[0])
#            self.count = 0
#            self.hr_count += 1
#
#        if self.count == 0:
#            self.op_1hr = self.op[0]

        # Check if we reached minimum number of periods to compute mean and standard deviation:
        #if len(self.cl_1hr) >= 20:
            #print("Now we can compute std ..")
#        cl_1hr_tmp = last_20_closes.append(self.cl)
#        cl_1hr_tmp = np.array(self.cl_1hr[-20:])
#        self.sigma = np.std(cl_1hr_tmp)
#        self.mid = np.mean(cl_1hr_tmp)
#        self.sigma_pc = 100 * self.sigma / self.mid
            #self.log(f'2-SIGMA: {pc_buy:.1f}; DOWN: {self.down:.1f}')
#        else:
#            self.count += 1
#            return 0            
        
        # DEBUG
#        if self.count == 0:
#            self.log(f'MEAN: {self.mid:.2f}; STD: {self.sigma:.2f}; STD_pc: {self.sigma_pc:.2f}')       
        
        if not self.position:
            #print("Not in position...")
            #print(f'self.cl = {self.cl}')
            #print(f'last 20 = {last_20_closes}')
            cl_1hr_tmp = last_20_closes + [self.cl]
            #print(cl_1hr_tmp)
            cl_1hr_tmp = np.array(cl_1hr_tmp[-20:])
            #print(cl_1hr_tmp)
            self.sigma = np.std(cl_1hr_tmp)
            #print(self.sigma)
            self.mid = np.mean(cl_1hr_tmp)
            #print(self.mid)
            self.sigma_pc = 100 * self.sigma / self.mid
            #print(self.sigma_pc)
            self.down = 100*(self.op_1hr - self.cl) / self.op_1hr
            #print(self.down)
            pc_buy = self.sigma_fac*self.sigma_pc
            #print(f' pc_buy = {pc_buy}')
            #DEBUG:
#            if self.count == 0:                 
#                self.log(f'2-SIGMA: {pc_buy:.1f}; DOWN: {self.down:.1f}')
            if self.down > pc_buy :
            #print(self.buy_price[0])
                self.time_executed = time.time()
                book = client.get_order_book(symbol=self.symbol)
                self.buy_price = weighted_avg_orderBook(book['asks'], self.deposit_fraction)                
                self.log(f'BUY CREATE, {self.buy_price:.8f}; DOWN: {self.down:.2f}', fname=self.logfile )
                self.market_buy_order()
                #self.order,_ = binance_endpoints.place_buy_order(self.symbol,1,1,'MARKET',0.1,trade_type=self.trade_type,buy_below=0)
                self.position = True
                # Fractional size:
                #self.size = self.broker.get_cash() / self.cl[0]
                #self.buy(size=self.size)
                # Remeber at what `hour` index the trade was executed:

                
                #self.buy_price = self.cl
                take_profit = self.take_profit or self.fac_back*pc_buy
                self.profit_target = (1+take_profit*0.01)*self.buy_price
                stop_loss = self.stop_loss or self.fac_back*pc_buy
                self.loss_target = (1-stop_loss*0.01)*self.buy_price
                
                self.market_sell_order()
                self.log(f'Profit at: {self.profit_target:.8f}, Loss at: {self.loss_target:.8f}')              
                #self.stop_loss = self.params.stop_loss or (1-self.params.fac_back*pc_buy*0.01)*self.buy_price
                #print(f'2-SIGMA: {pc_buy:.1f}; DOWN: {self.down:.1f}')
                #print(f'Profit target: {self.take_profit:.2f}; Loss target: {self.stop_loss:.2f}')
        
        else:
            # If we are still on the same candle where buy was executed
            # don't use high and low before the trade
            if self.time_executed > op_time/1000:
                self.hi = self.cl
                self.lo = self.cl
                
            self.log(f'Hi: {self.hi:.8f}, Profit: {self.profit_target:.8f}, Loss: {self.loss_target:.8f}')
            if (self.hi > self.profit_target):  
                profit = 100*(self.profit_target - self.buy_price)/self.buy_price
                self.log(f"PROFIT: {profit:.2f}%, Price: {self.cl:.8f}, Take profit: {self.profit_target:.8f}", fname=self.logfile)
                self.position = False
                #self.close()
            elif (self.lo <= self.loss_target):
                profit = 100*(self.loss_target - self.buy_price)/self.buy_price
                self.log(f"LOSS: {profit:.2f}%, Price: {self.cl:.8f}, Stop loss: {self.loss_target:.8f}", fname=self.logfile)
                self.position = False
                #self.close()
            else:
                time_diff = (time.time() - self.time_executed)/60.0 # Time past in minutes simse the order has been executed
                if time_diff >= self.max_hold*self.time_period:
                    #self.close()
                    # For now do just MARKET SELL:
                    self.log("Reached max holding time! Market sell.", fname=self.logfile)
                    book = client.get_order_book(symbol=self.symbol)
                    sell_price = weighted_avg_orderBook(book['bids'], self.deposit_fraction)
                    profit = 100*(sell_price - self.buy_price)/self.buy_price
                    if profit > 0:
                        self.log(f"SELL: {sell_price:.8f}, PROFIT: {profit:.2f}%", fname=self.logfile)
                    else:
                        self.log(f'SELL: {sell_price:.8f}, LOSS: {profit:.2f}%', fname=self.logfile)
                    self.position = False
        #self.count += 1


def on_message(ws, message):
    global closes
    
    #print('received message')
    json_message = json.loads(message)
    #pprint.pprint(json_message)

    candle = json_message['k']
    
    print(candle['c'])
    is_candle_closed = candle['x']
    close = float(candle['c'])
    op = float(candle['o'])
    hi = float(candle['h'])
    lo = float(candle['l'])
    op_time = candle['t']

    if is_candle_closed:
        print(f"candle closed at {close:.8f}")
        closes.append(float(close))
        #print("closes")
        #print(closes)

    if len(closes) >= 20: # Make sure that we have at leat 20 elements to compute 20-period MA and std
        closes = closes[-20:] # Update to have only 20 elements (to avoid too large arrays)

        # Call the strategy
        Strategy.next(op, hi, lo, close, op_time, closes)
        



def on_open(ws):
    print('opened connection')

def on_close(ws):
    print('closed connection')


def parse_args():
    ''' Main parameters of the strategy:
    sigma_fac
    fac_back
    max_hold
    '''
    parser = argparse.ArgumentParser(
        description='2-Sigma Strategy',
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
        
def starting_summary(args):
    print("Starting summary of the strategy: ")
    for var in vars(args):
        print(f"{var}: {vars(args)[var]}")
        
if __name__=='__main__':
    
    #Get Args
    args = parse_args()
    
    time_period = args.interval
    symbol = args.symbol
    #symbol='linkusdt'
    if time_period == 60:
        interval='1h'
    else:
        interval = f'{time_period}m'
    args.log = args.log or f'{symbol}_{interval}.log' 
    logfile = args.log
    
    starting_summary(args)
    

    #closes = binance_endpoints.GetKlines(symbol.upper(), interval=f'{time_period}m', limit=100)
    closes = binance_endpoints.GetKlines(symbol.upper(), interval=interval, limit=100)
    closes = list(closes.close.iloc[-20:])
    
    SOCKET = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}"
    
    
    sigma_fac = args.sigma
    fac_back = args.fback
    max_hold = args.maxhold
    cash = args.cash
    
    #closes = []
    Strategy = TwoSigmaStrategy(time_period=time_period, sigma_fac=sigma_fac, fac_back=fac_back, symbol=symbol.upper(), 
                                logfile=logfile, max_hold=max_hold, deposit_fraction=cash)
    
    ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
    
    ws.run_forever()
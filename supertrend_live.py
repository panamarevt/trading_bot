# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 15:35:06 2021

@author: Taras
"""

#import numpy as np
#import backtrader as bt
import datetime
import pandas as pd
#import indicators

import websocket
import json
#import pprint
import time
import argparse

import binance_endpoints

from binance.client import Client
#from strategies import weighted_avg_orderBook
import indicators

import keys
client = Client(api_key=keys.Pkey, api_secret=keys.Skey)

class SuperTrendStrategy():

    def __init__(self, strendperiod, strendmult, interval=15, side = 'long', take_profit=0.05, stop_loss=100, 
                 atr_fac_prof = 1, atr_fac_loss = 1, ambuy=100.0, amsell=0.1, symbol='ETHUSDT', logfile=None):
        #self.op = self.datas[0].open
        #self.cl = self.datas[0].close
        # Main parameters of the strategy:
        self.strendperiod = strendperiod # period to compute the ATR
        self.strendmult = strendmult # multiplier for the supertrend
        self.interval = interval # At which time-frames to trade (in minutes)
        self.take_profit = take_profit # profit in fraction
        self.stop_loss = stop_loss # loss in fraction
        self.side = side # short, long or both
        self.atr_fac_prof = atr_fac_prof # factor to multiply ATR if take profit is ATR
        self.atr_fac_loss = atr_fac_loss
        self.ambuy = ambuy # amount of 'cash' in base asset to trade with
        self.amsell = amsell # amount of cash in quote asset
        self.symbol = symbol # trading pair
        self.logfile = logfile # filename to save log
        # Auxillary vars:
        self.count = 0
        self.position = False
        self.cumulative_profit = 0
        self.total_fee = 0
        self.fee = 0.075*0.01
        
        self.price_prec = binance_endpoints.get_price_precision(self.symbol)
        self.lot_prec = binance_endpoints.get_lot_precision(self.symbol)
    
        self.profit_order = False
    
    def log(self, txt, dt=None, fname=None):
        ''' Logging function fot this strategy'''
        dt = dt or datetime.datetime.now()
        print('%s, %s' % (dt.strftime("%d %b %Y %H:%M:%S"), txt))           
        fname = fname or self.logfile
        if fname:
            with open(fname, 'a') as f:
                f.write('%s, %s\n' % (dt.strftime("%d %b %Y %H:%M:%S"), txt)) 


    def place_buy_order(self, qty=None, price=None):
        '''Place BUY order and get determine the price if not specified'''
        price = price or self.cl.iloc[-1] # if not specified - take current close price
        qty = qty or self.ambuy / price # go all-in
        self.qty = float(binance_endpoints.truncate(qty, self.lot_prec)) # truncate quantity to match the lot precision
        prec = self.price_prec # get price precision
        BuyPrice = f'{price:.{prec}f}' # buy price should be a string
        self.new_order = client.order_limit_buy(symbol=self.symbol, quantity=self.qty, price=BuyPrice)
        self.trade_type = 'long' # global class variable to define whether this is short or long trade
        self.position = True  # Indicate that we open the position       
        return self.new_order
    
    def place_sell_order(self, qty=None, price=None):
        '''Place SELL order and get determine the price if not specified'''
        price = price or self.cl.iloc[-1] # if not specified - take current close price
        qty = qty or self.amsell  #/ price # go all-in
        self.qty = float(binance_endpoints.truncate(qty, self.lot_prec)) # truncate quantity to match the lot precision
        prec = self.price_prec # get price precision
        SellPrice = f'{price:.{prec}f}' # buy price should be a string
        self.new_order = client.order_limit_sell(symbol=self.symbol, quantity=self.qty, price=SellPrice)
        self.trade_type = 'short' # global class variable to define whether this is short or long trade
        self.position = True  # Indicate that we open the position       
        return self.new_order

    def check_position(self):
        self.new_order_status = binance_endpoints.check_order_status(self.new_order)['status']
        if (self.new_order_status == 'FILLED') and (not self.profit_order):
            if self.trade_type == 'long':
                SellPrice = f'{self.profit_target:{self.price_prec}.f}'
                self.profit_order = client.order_limit_sell(symbol=self.symbol, quantity=self.qty, price=SellPrice)
            if self.trade_type == 'short':
                BuyPrice = f'{self.profit_target:{self.price_prec}.f}'
                self.profit_order = client.order_limit_buy(symbol=self.symbol, quantity=self.qty, price=BuyPrice)           
        if (self.new_order_status != 'FILLED'):
            self.log(f"New order status: {self.new_order_status}")
        if self.profit_order:
            self.prof_order_status = binance_endpoints.check_order_status(self.profit_order)['status']
            if self.prof_order_status == 'FILLED':
                self.log("Profit!")
                self.position = False
                self.profit_order = False
            else:
                self.log(f"Exit order status: {self.prof_order_status}")        


    def next(self, op, hi, lo, cl, op_time):      
        '''Strategy decision-making for the received candlestick data
        params:
        ------------    
        op - open price for the current time-frame
        hi, lo - high and low prices
        cl - close price for the current time-frame (assume to receive `finished` candle)
        last_20_closes- list of last 20 close prices'''
        self.op = op
        self.cl = cl
        self.hi = hi
        self.lo = lo  
        
        price = self.cl.iloc[-1]
        
        if not self.position:
            self.log("Searching for signal ...")
            if (self.side == 'long') or (self.side == 'both'):
                #if self.lo < self.supertrend.l.super_trend and self.cl > self.supertrend.l.super_trend :
#                if self.lo[-1] < self.supertrend.l.super_trend[-1] and self.cl[-1] > self.supertrend.l.super_trend[-1] :
#                    if self.cl > self.op and self.cl > self.supertrend.l.super_trend:
                signal_long = indicators.supertrend_signal(self.op,self.hi,self.lo,self.cl,self.strendperiod, self.strendmult, cond='touch', side='long')
                self.log(f"Long signal: {signal_long}")
                if signal_long:
                        self.log(f"Price signal!  Buy at {price}")
                        if self.take_profit != 'atr' : # if take profit is a fixed number, just use it as %
                            self.profit_target = (1+self.take_profit)*price
                        else: # if profit depends on ATR:
                            #self.profit_target = price + self.atr_fac_prof*self.atr
                            pass
                        if self.p.stop_loss != 'atr' :
                            self.loss_target = (1-self.stop_loss)*price
                        else:
                            #self.loss_target = self.cl[0] - self.p.atr_fac_loss*self.atr
                            pass
                        #----------------------
                        #place buy order here
                        self.place_buy_order()
                        #if self.p.side == 'long': return                        
            if ( (self.side == 'short') or (self.side == 'both') ) and (not self.position) :
                signal_short = indicators.supertrend_signal(self.op,self.hi,self.lo,self.cl,self.strendperiod, self.strendmult, cond='touch', side='short')
                self.log(f"Short signal: {signal_short}")
                if signal_short:
                        self.log(f"Price signal!  Sell at {price}")
                        if self.take_profit != 'atr' : # if take profit is a fixed number, just use it as %
                            self.profit_target = (1-self.take_profit)*price                        
                        else:
                            #self.profit_target = self.cl[0] - self.p.atr_fac_prof*self.atr
                            pass
                        if self.stop_loss != 'atr' :
                            self.loss_target = (1+self.stop_loss)*price
                        else:
                            #self.loss_target = price + self.p.atr_fac_loss*self.atr
                            pass
                        #----------------------
                        #place sell order here
                        self.place_buy_order()
                             
        else:
            self.check_position()
            #            self.new_order_status = binance_endpoints.check_order_status(self.new_order)['status']
#            if (self.new_order_status == 'FILLED') and (not self.profit_order):
#                if self.trade_type == 'long':
#                    SellPrice = f'{self.profit_target:{self.price_prec}.f}'
#                    self.profit_order = client.order_limit_sell(symbol=self.symbol, quantity=self.qty, price=SellPrice)
#                if self.trade_type == 'short':
#                    BuyPrice = f'{self.profit_target:{self.price_prec}.f}'
#                    self.profit_order = client.order_limit_buy(symbol=self.symbol, quantity=self.qty, price=BuyPrice)           
#            if (self.new_order_status != 'FILLED'):
#                self.log(f"New order status: {self.new_order_status}")
#            if self.profit_order:
#                self.prof_order_status = binance_endpoints.check_order_status(self.profit_order)['status']
#                if self.prof_order_status == 'FILLED':
#                    self.log("Profit!")
#                    self.position = False
#                    self.profit_order = False
#                else:
#                    self.log(f"Exit order status: {self.prof_order_status}")
            
            
#            if self.trade_type == 'long': # if we are in a `long` position
#                if (self.cl[0] > self.profit_target):  
#                    self.log(f"PROFIT! {self.cl[0]}")   
#                    self.close()
#                    self.profit_target = None
#                elif (self.cl[0] <= self.loss_target): 
#                    self.log(f"LOSS! {self.cl[0]}")   
#                    self.close()
#                    self.loss_target = None      
#            if self.trade_type == 'short': # if we are in a `short` position
#                if (self.cl[0] < self.profit_target):  
#                    self.log(f"PROFIT! {self.cl[0]}")   
#                    self.close()
#                    self.profit_target = None
#                elif (self.cl[0] >= self.loss_target): 
#                    self.log(f"LOSS! {self.cl[0]}")   
#                    self.close()
#                    self.loss_target = None            
        

#indicators.supertrend_signal(Strategy.op,Strategy.hi,Strategy.lo,Strategy.cl,Strategy.strendperiod, Strategy.strendmult, cond='touch', side='long')

def on_message(ws, message):
    global closes, opens, highs, lows
    
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
        Strategy.log(f"candle closed at {close:.8f}")
        closes.append(float(close))
        opens.append(float(op))
        highs.append(float(hi))
        lows.append(float(lo))
        #print("closes")
        #print(closes)
        # Call the strategy
        Strategy.next(pd.Series(opens), pd.Series(highs), pd.Series(lows), pd.Series(closes), op_time)
        if Strategy.position: 
            update_position = time.time()
    if Strategy.position:
        update_position = time.time() - update_position
        if update_position > 30:
            Strategy.check_position()
            update_position = time.time()
        pass
    max_len = 50
    if len(closes) >= max_len: # Make sure that we have at leat 20 elements to compute 20-period MA and std
        closes = closes[-max_len:] # Update to have only 20 elements (to avoid too large arrays)
        opens = opens[-max_len:]
        highs = highs[-max_len:]
        lows = lows[-max_len:]


        


def on_open(ws):
    print('opened connection')

def on_close(ws):
    print(f'Closed connection! Retry {time.ctime()}...')
    time.sleep(10)
    ws.run_forever()

def on_error(ws, error):
    print(error)

def starting_summary(args):
    print("Starting summary of the strategy: ")
    for var in vars(args):
        print(f"{var}: {vars(args)[var]}")
 

#---------------------------------------------------------------
       
if __name__=='__main__':
    
    #Get Args
#    args = parse_args()
#    
#    time_period = args.interval
#    symbol = args.symbol
#    #symbol='linkusdt'
#    if time_period == 60:
#        interval='1h'
#    else:
#        interval = f'{time_period}m'
#    args.log = args.log or f'{symbol}_{interval}.log' 
#    logfile = args.log
#    
#    starting_summary(args)
    
    #symbol = 'BANDBNB'
    symbol = 'BNBUSDT'
    interval = '15m'
    side = 'long'
    
    #closes = binance_endpoints.GetKlines(symbol.upper(), interval=f'{time_period}m', limit=100)
    closes = binance_endpoints.GetKlines(symbol.upper(), interval=interval, limit=100)    
    opens = list(closes.open.iloc[-20:])
    highs = list(closes.high.iloc[-20:])
    lows = list(closes.low.iloc[-20:])
    closes = list(closes.close.iloc[-20:])
    
    
    SOCKET = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}"
    logfile = f"{symbol}_{interval}_10_3_{side}.log"
    
#    sigma_fac = args.sigma
#    fac_back = args.fback
#    max_hold = args.maxhold
#    cash = args.cash
#    
    #closes = []
    Strategy = SuperTrendStrategy(10, 3, interval=interval, side = side, take_profit=0.05, stop_loss=None, 
                 atr_fac_prof = 1, atr_fac_loss = 1, ambuy=15.0, amsell=0.1, symbol=symbol, logfile=logfile)
    
    ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
    
    ws.run_forever()
        
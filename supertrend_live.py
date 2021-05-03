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

from loguru import logger

import keys
client = Client(api_key=keys.Pkey, api_secret=keys.Skey)

logger.add('debug.log', format="{time} {level} {message}", level='DEBUG')


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
    
        self.new_order = False
        self.profit_order = False
    
    def log(self, txt, dt=None, fname=None):
        ''' Logging function for this strategy'''
        dt = dt or datetime.datetime.now()
        print('%s, %s' % (dt.strftime("%d %b %Y %H:%M:%S"), txt))           
        fname = fname or self.logfile
        if fname:
            with open(fname, 'a') as f:
                f.write('%s, %s\n' % (dt.strftime("%d %b %Y %H:%M:%S"), txt)) 


    def place_buy_order(self, qty=None, price=None):
        '''Place BUY order and determine the price if not specified'''
        price = price or self.cl.iloc[-1] # if not specified - take current close price
        qty = qty or self.ambuy / price # go all-in
        self.qty = float(binance_endpoints.truncate(qty, self.lot_prec)) # truncate quantity to match the lot precision
        prec = self.price_prec # get price precision
        BuyPrice = f'{price:.{prec}f}' # buy price should be a string
        logger.info(f"Placing BUY order: symbol:{self.symbol}; price:{BuyPrice}; qty:{self.qty}")
        self.new_order = client.order_limit_buy(symbol=self.symbol, quantity=self.qty, price=BuyPrice)
        self.trade_type = 'long' # global class variable to define whether this is short or long trade
        self.position = True  # Indicate that we open the position       
        return self.new_order
    
    def place_sell_order(self, qty=None, price=None):
        '''Place SELL order and determine the price if not specified'''
        price = price or self.cl.iloc[-1] # if not specified - take current close price
        qty = qty or self.amsell  #/ price # go all-in
        self.qty = float(binance_endpoints.truncate(qty, self.lot_prec)) # truncate quantity to match the lot precision
        prec = self.price_prec # get price precision
        SellPrice = f'{price:.{prec}f}' # buy price should be a string
        logger.info(f"Placing SELL order: symbol:{self.symbol}; price:{SellPrice}; qty:{self.qty}")
        self.new_order = client.order_limit_sell(symbol=self.symbol, quantity=self.qty, price=SellPrice)
        self.trade_type = 'short' # global class variable to define whether this is short or long trade
        self.position = True  # Indicate that we open the position       
        return self.new_order

    def check_position(self):        
        #self.new_order_status = binance_endpoints.check_order_status(self.new_order)['status']
        # If entry order was filled already before:
        if self.profit_order:
            self.prof_order_status = binance_endpoints.check_order_status(self.profit_order)['status']
            if self.prof_order_status == 'FILLED':
                #self.log("Profit!")
                logger.info("Profit!")
                #!!! TODO: Call function to analyze trade
                self.position = False
                self.profit_order = False
            else:
                #self.log(f"Exit order status: {self.prof_order_status}")
                logger.info(f"EXIT order status: {self.prof_order_status}")
        else:        
            # We have just submitted the ENTRY order:
            self.new_order_status = self.check_entry_order(BUY_TIME_LIMIT=self.interval)
            if (self.new_order_status == 'FILLED') and (not self.profit_order):
                logger.info("ENTRY order FILLED")
                if self.trade_type == 'long':
                    SellPrice = f'{self.profit_target:.{self.price_prec}f}'
                    logger.info(f"Place profit order: symbol:{self.symbol}; price:{SellPrice}; qty:{self.qty}")
                    self.profit_order = client.order_limit_sell(symbol=self.symbol, quantity=self.qty, price=SellPrice)
                if self.trade_type == 'short':
                    BuyPrice = f'{self.profit_target:.{self.price_prec}f}'
                    logger.info(f"Place profit order: symbol:{self.symbol}; price:{BuyPrice}; qty:{self.qty}")
                    self.profit_order = client.order_limit_buy(symbol=self.symbol, quantity=self.qty, price=BuyPrice)           
            if (self.new_order_status != 'FILLED'):
                #self.log(f"New order status: {self.new_order_status}")
                logger.info(f"ENTRY order status: {self.new_order_status}")


    def check_entry_order(self, BUY_TIME_LIMIT='30m'):    
        '''
        BUY_TIME_LIMIT maximum time in minutes for a pending entry order
        default is the candlestick interval used for the trading strategy
        If pending time exceeds - cancel order, if order is partially filled continue trade with the executed quantity
            order status response:
           {'symbol': 'WRXBTC',
             'orderId': 7895233,
             'orderListId': -1,
             'clientOrderId': 'beMu1tPLtTpJlEYjuXSzGR',
             'price': '0.00002132',
             'origQty': '1111.00000000',
             'executedQty': '1111.00000000',
             'cummulativeQuoteQty': '0.02368652',
             'status': 'FILLED',
             'timeInForce': 'GTC',
             'type': 'LIMIT',
             'side': 'BUY',
             'stopPrice': '0.00000000',
             'icebergQty': '0.00000000',
             'time': 1583698931057,
             'updateTime': 1583699099624,
             'isWorking': True,
             'origQuoteOrderQty': '0.00000000'}
        '''
        #!!!TODO: Improve Partial filling. What if executed quantitity is too small, etc
        logger.info("Check ENTRY Order status...")
        BUY_TIME_LIMIT = pd.Timedelta(BUY_TIME_LIMIT)
        try:
            order = binance_endpoints.check_order_status(self.new_order)
            self.new_order = order
            status = self.new_order['status']            
        except Exception as e:
            print("Warning didn't manage to check order status! (Called from fuction 'check buy order')")
            print(e)
            return self.new_order['status']
        try:
            place_time = order['time']/1000 #Time in Binance is given in milliseconds, convert to seconds here.
        except KeyError:
            place_time = order['transactionTime']/1000
    
        if status == "FILLED":        
            if self.stop_loss :
                try:
                    # ATTENTION! The OCO order function below for now works only for LONG trades
                    self.profit_order = binance_endpoints.place_oco_order(self.symbol, order['price'], 
                                 take_profit=self.take_profit, stop_loss=self.stop_loss, trade_type='REAL')
                except Exception as e:
                    logger.exception(f"WARNING!!!! OCO order didn't place! {e}")
            else:
                logger.info(f"Entry order for {self.symbol} filled! Now start evaluating the order...")
                return status
        
        time_curr = time.time()
        elapsed_min = (time_curr - place_time)/60
        # convert to Timedelta object
        elapsed_min = pd.Timedelta(f'{elapsed_min}m')
        if elapsed_min > BUY_TIME_LIMIT:
           #if trade_type == 'REAL': order = cancel_order(order)
           order = binance_endpoints.cancel_order(order)
           logger.info(f"ENTRY order didn't fill in {BUY_TIME_LIMIT}")
           executedQty = float(order['executedQty'])
           if executedQty > 0:
               logger.info("Order Partially Filled!")
               # Continue with the executed quantity
               logger.info("Perform evaluation with executed quantity...")
               status = "PARTIALLY_FILLED"
               return status
            
           status = "CANCELLED"
           logger.info("Cancel order, start searching for new opportunities!")
           self.position = False
           
        return status


    @logger.catch
    def next(self, op, hi, lo, cl, op_time, trade_type='LIVE'):      
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
        logger.debug(f"In position: {self.position}")
        
        if not self.position:
            #self.log("Searching for signal ...")
            logger.debug("Searching for signal ...")
            if (self.side == 'long') or (self.side == 'both'):
                #if self.lo < self.supertrend.l.super_trend and self.cl > self.supertrend.l.super_trend :
#                if self.lo[-1] < self.supertrend.l.super_trend[-1] and self.cl[-1] > self.supertrend.l.super_trend[-1] :
#                    if self.cl > self.op and self.cl > self.supertrend.l.super_trend:
                signal_long = indicators.supertrend_signal(self.op,self.hi,self.lo,self.cl,self.strendperiod, self.strendmult, cond='touch', side='long')
                #self.log(f"Long signal: {signal_long}")
                logger.debug(f"Long signal: {signal_long}")
                if signal_long:
                        #self.log(f"Price signal!  Buy at {price}")
                        logger.info(f"Price signal!  Buy {self.symbol} at {price}")
                        if trade_type == 'ALERT': return
                        if self.take_profit != 'atr' : # if take profit is a fixed number, just use it as %
                            self.profit_target = (1+self.take_profit)*price
                        else: # if profit depends on ATR:
                            #self.profit_target = price + self.atr_fac_prof*self.atr
                            pass
                        if (self.stop_loss) and (self.stop_loss != 'atr') :
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
                #self.log(f"Short signal: {signal_short}")
                logger.debug(f"Short signal: {signal_short}")
                if signal_short:
                        #self.log(f"Price signal!  Sell at {price}")
                        #logger.info(f"Price signal!  Sell at {price}")
                        logger.info(f"Price signal!  Sell {self.symbol} at {price}")
                        if trade_type == 'ALERT': return
                        if self.take_profit != 'atr' : # if take profit is a fixed number, just use it as %
                            self.profit_target = (1-self.take_profit)*price                        
                        else:
                            #self.profit_target = self.cl[0] - self.p.atr_fac_prof*self.atr
                            pass
                        if (self.stop_loss) and (self.stop_loss != 'atr') :
                            self.loss_target = (1+self.stop_loss)*price
                        else:
                            #self.loss_target = price + self.p.atr_fac_loss*self.atr
                            pass
                        #----------------------
                        #place sell order here
                        self.place_sell_order()
                             
        else:
            self.check_position()
            #            self.new_order_status = binance_endpoints.check_order_status(self.new_order)['status']



def on_message(ws, message):
    global closes, opens, highs, lows, count, elapsed, start
    
    #t_start = time.time()
    #print('received message')
    json_message = json.loads(message)
    #pprint.pprint(json_message)

    candle = json_message['k']
    count += 1
    
    #print(candle['c'])
    #logger.debug(candle['c'])
    is_candle_closed = candle['x']
    close = float(candle['c'])
    op = float(candle['o'])
    hi = float(candle['h'])
    lo = float(candle['l'])
    op_time = candle['t']

    if is_candle_closed:
        #Strategy.log(f"candle closed at {close:.8f}")
        logger.info((f"candle closed at {close:.8f}"))
        closes.append(float(close))
        opens.append(float(op))
        highs.append(float(hi))
        lows.append(float(lo))
        Strategy.next(pd.Series(opens), pd.Series(highs), pd.Series(lows), pd.Series(closes), op_time)
        if Strategy.position:
            logger.debug("Check position status")
            Strategy.check_position()
            update_position = time.time()
            start = time.time()
        count = 0
        
    elapsed = time.time() - start
    #logger.debug(f"elapsed = {elapsed}")
    if elapsed >= 60:
        logger.debug(f"elapsed = {elapsed}")
        #print("closes")
        #print(closes)
        # Call the strategy
        #Strategy.next(pd.Series(opens+[op]), pd.Series(highs+[hi]), pd.Series(lows+[lo]), pd.Series(closes+[close]), op_time)
        Strategy.next(pd.Series(opens[:-1] + [op]), pd.Series(highs[:-1]+[hi]), pd.Series(lows[:-1]+[lo]), pd.Series(closes[:-1]+[close]), op_time, trade_type='ALERT')
        start = time.time()
        count = 0

    elapsed = time.time() - start
    #logger.debug(f"In position: {Strategy.position}")
    if Strategy.position:
        update_position = time.time() - update_position
        logger.debug(f"Last time since checked the position status ...{elapsed} s")
        if elapsed > 60:
        #if update_position > 30:
            logger.debug("Check position status")
            Strategy.check_position()
            update_position = time.time()
            start = time.time()
        update_position = time.time() - update_position
    max_len = 50
    if len(closes) >= max_len: # Make sure that we have at leat 20 elements to compute 20-period MA and std
        closes = closes[-max_len:] # Update to have only 20 elements (to avoid too large arrays)
        opens = opens[-max_len:]
        highs = highs[-max_len:]
        lows = lows[-max_len:]


        


def on_open(ws):
    #print('opened connection')
    logger.info('opened connection')

def on_close(ws):
    #print(f'Closed connection! Retry {time.ctime()}...')
    logger.info(f'Closed connection! Retry {time.ctime()}...')
    time.sleep(10)
    ws.run_forever()

def on_error(ws, error):
    print(error)

def starting_summary(args):
    print("Starting summary of the strategy: ")
    for var in vars(args):
        print(f"{var}: {vars(args)[var]}")
 


def parse_args():
    ''' Main parameters of the strategy:
    '''
    parser = argparse.ArgumentParser(
        description='SuperTrend Strategy',
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
        '--period', '-p',
        default=10, type=float,
        help='Period to compute ATR for SuperTrend ')    
    
    parser.add_argument(
        '--mult', '-m',
        default=3, type=float,
        help='Factor to multiply ATR')

    parser.add_argument(
        '--side', 
        default='long', type=str,
        help='Accepted values: long, short, both')

    parser.add_argument(
        '--cashbuy', 
        default=1000, type=float,
        help='Amount of cash for LONG trades, in units of base asset')

    parser.add_argument(
        '--cashsell', 
        default=1000, type=float,
        help='Amount of cash for SHORT trades, in units of quote asset')
    
    parser.add_argument(
        '--profit', type=float, 
        default=0.05,
        help='Take profit as a fraction e.g. 1% profit should be 0.01')

    parser.add_argument(
        '--loss', 
        default=None, type=float,
        help='Stop loss as a fraction e.g. 1% loss should be 0.01. Can be None (dangerous!)')    

    parser.add_argument(
        '--log', '-l',
        default=None,
        help='Filename to save logging data. If None the default filename is used: {symbol}_{interval}.log')         

    return parser.parse_args()

#---------------------------------------------------------------
       
if __name__=='__main__':
    
    #Get Args
    args = parse_args()
    
    interval = args.interval
    symbol = args.symbol
    side = args.side
    period = args.period
    mult = args.mult
    ambuy = args.cashbuy
    amsell = args.cashsell
    profit = args.profit
    loss = args.loss
    
    #symbol='linkusdt'
    if interval == 60:
        interval='1h'
    else:
        interval = f'{interval}m'
    args.log = args.log or f'{symbol}_{interval}_{side}_{period}_{mult}.log' 
    logfile = args.log
    
    starting_summary(args)
    
    logger.add(logfile, format="{time} {level} {message}", level='DEBUG')
    
    #symbol = 'BANDBNB'
#    symbol = 'BNBUSDT'
#    interval = '15m'
#    side = 'short'
    
    #closes = binance_endpoints.GetKlines(symbol.upper(), interval=f'{time_period}m', limit=100)
    closes = binance_endpoints.GetKlines(symbol.upper(), interval=interval, limit=100)    
    opens = list(closes.open.iloc[-20:])
    highs = list(closes.high.iloc[-20:])
    lows = list(closes.low.iloc[-20:])
    closes = list(closes.close.iloc[-20:])
    
    count = 0
    SOCKET = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}"
    logfile = f"{symbol}_{interval}_10_3_{side}.log"
    
#    sigma_fac = args.sigma
#    fac_back = args.fback
#    max_hold = args.maxhold
#    cash = args.cash
#    
    #closes = []
    elapsed = 0
    start = time.time()
    Strategy = SuperTrendStrategy(period, mult, interval=interval, side = side, take_profit=profit, stop_loss=loss, 
                 atr_fac_prof = 2, atr_fac_loss = 1, ambuy=ambuy, amsell=amsell, symbol=symbol.upper(), logfile=logfile)
    
    end = time.time()
    elapsed += end - start
    ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
    
    ws.run_forever()
        

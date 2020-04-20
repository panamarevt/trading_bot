# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 04:22:13 2020

@author: Taras
"""

# import all necessary modules
from binance.client import Client
#from datetime import datetime
from pandas import DataFrame as df
import numpy as np
import pandas as pd
#import keys

#import requests
#import json
#import os
import time
#import logging
import os
#from threading import Thread

#from binance_endpoints import get_symbols_BTC
import binance_endpoints
import indicators
#from binance.enums import *

#from binance_api import Binance

# Threading and multiprocessing
import concurrent.futures

# Techincal indicators library by bablofil
# https://bablofil.ru/raschet-indikatorov-SMMA-EWMA-MMA-RMA-RSI-STOCH-STOCHRSI/
# https://bablofil.ru/bot-dlya-binance-s-indikatorami/
#import bablofil_ta as ta

# Technical indicators from TA library
# https://technical-analysis-library-in-python.readthedocs.io/en/latest/
# https://github.com/bukosabino/ta
#import ta

#logging.basicConfig(filename='volume_strategy_new.log', level=logging.DEBUG,
#                    format='%(asctime)s:%(levelname)s:%(message)s')

# Configute logging settings:
##logger = logging.get#logger(__name__)
##logger.setLevel(logging.INFO)
#formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
#
#file_handler = logging.FileHandler('volume_strategy_new_2.log')
#file_handler.setLevel(logging.INFO)
#file_handler.setFormatter(formatter)
#
#stream_handler = logging.StreamHandler()
#stream_handler.setLevel(logging.INFO)
#stream_handler.setFormatter(formatter)
#
##logger.addHandler(file_handler)
##logger.addHandler(stream_handler)

client = Client()
# Initialize main constants for the strategy
# Minimum and maximum price
MIN_PRICE = 0.00000200 # 200 Satoshi
MAX_PRICE = 0.009
# Stop loss percentage
STOP_LOSS = 0.05
# Number of red candles to sell after. Backtests show that it is better to sell after the first red candle
N_RED = 1
# Interval in minutes to campare with the current volume.
MINS_BEFORE = 10
# Minimum volume in BTC for current 1 minute candle
QUOTE_AV_MIN = 3
# Price change criterion for the current candle
PRICE_CHANGE_PC = 1

# List of BTC quantities (only for testing reasons)
AMOUNTS = [0.05, 0.1, 0.2, 1]

#def get_active_volume(symb, active):
#    '''input: symbol, list of symbols
#    output: None
#    Check if the trading pair satisfies the conditions: 24h volume > 150 BTC and price > 200 Satoshi
#    If yes, append the pair in list of active coins
#    '''
#    #global active
#    try:
#        tmp = client.get_ticker(symbol=symb)
#    except:
#        #logger.exception(f"Didn't manage to get 24 hr tickers for {symb}.")
#    price = np.float64(tmp['lastPrice'])
#    volume = np.float64(tmp['quoteVolume'])
#    # For the volume strategy we use only price criterion
#    if (0.00000200 < price < 0.009) :
#        active.append([symb, price, volume])
#
#
#def refresh_active_volume(symbols):
#    '''Create a list of active trading pairs
#    Return list of active pairs as a pandas.DataFrame with 3 columns: symbol, price, 24h volume
#    '''
#    active = []
#    for symbol in symbols:
#        #Thread(target=get_active, args=(symbol,)).start()
#        get_active_volume(symbol, active)
#    #    tmp = client.get_ticker(symbol=symbol)
#    #    price = float(tmp['lastPrice'])
#    #    volume = float(tmp['quoteVolume'])
#    #    if (0.00000200 < price < 0.001) & (volume > 150) :
#    #        active.append([symbol, price, volume])
#    active = df(active, columns=['symbol', 'Price', "Volume"])
#    print("Active list formed")
#    print(active)
#    return active        

def weighted_avg_orderBook(book, qty):
    '''book is list of asks or bids with prices and quatities, e.g.:
        asks': [['0.00000644', '6749.00000000'],
      ['0.00000645', '25059.00000000'],
      ['0.00000646', '33069.00000000'],
      ['0.00000647', '1007.00000000'],]'''
    book = np.array(book)   
    pr = book[:,0].astype(float) # list pf prices
    qt = book[:,1].astype(float) # list of corresponding quantities
    StartQty = qt[0]
    #If first quantity in the order book is larger then our quantity, return the first price:
    if StartQty >= qty: return pr[0]
    qty_tmp = qty # We will reduce this value in the while loop below
    price_times_qty = 0
    j = 0
    while qty_tmp > StartQty:
        price_times_qty += pr[j]*qt[j]
        qty_tmp -= StartQty # We reduce our quantity here
        j += 1
        StartQty = qt[j] # Set start quantity to the next in the list
        # If StartQty happens to exceed our remaining quantity the while loop will terminate
    # Now we have to compute the final product: last price times remaining quantity
    price_times_qty += pr[j]*qty_tmp
    # The total weighted average price is :
    avg_price = price_times_qty/qty
    return avg_price         
            

def volume_strategy(symbol):
    global in_trade
#for symbol in symbols:
    ##logger.debug(f"Checking {symbol} ...")
    #print(f"Checking {symbol} ...")
    if symbol in list(in_trade) : 
        # Don't buy the same coin multiple times
        #logger.info(f"This coin is already in trade: {symbol}")
        #continue
        return 0
    try:
        candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=100)
    except Exception as e:
        #logger.exception(f"Didn't get kline intervals for {symbol}")
        print(f"Didn't get kline intervals for {symbol}")
        print(e)
        #continue
        return 0
    candles_df = df(candles, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    candles_df = candles_df.astype('float')

    #close = candles_df['close']
    price_curr = candles_df['close'].iloc[-1]
    
    min_price_cond = (price_curr  > MIN_PRICE) #If minimum price condition is not satisfied, we don't need to do anything here:
    if not min_price_cond: 
        #continue
        return 0
    
    #print("Check status!")
    #save_volumes(symbol, candles_df)
    
    prev_index = -1*MINS_BEFORE-1 # Subtract one to exclude current candle with index -1
    vol_prev = candles_df['volume'].iloc[prev_index : -1].sum()
    #vol_24h = candles_df['quote_av'].iloc[i-1440 : i].sum()
    #candle_color = candle_params(candles_df['open'].iloc[i], candles_df['high'].iloc[i],candles_df['low'].iloc[i],candles_df['close'].iloc[i])[-1]
    vol_curr = candles_df['volume'].iloc[-1]
    q_vol = candles_df['quote_av'].iloc[-1]
    
    trades = int(candles_df['trades'].iloc[-1])
    
    vol_curr_last = candles_df['volume'].iloc[-2]
    q_vol_last = candles_df['quote_av'].iloc[-2]
    vol_prev_last = candles_df['volume'].iloc[prev_index-1 : -2].sum()
    trades_last = int(candles_df['trades'].iloc[-2])
    
    # Conditions:
    vol_cond = (vol_curr > vol_prev)
    quote_col_cond = (q_vol > QUOTE_AV_MIN)
    #green_candle = (candle_color == 'green')       
    
    vol_cond_last = (vol_curr_last > vol_prev_last)
    quote_col_cond_last = (q_vol_last > QUOTE_AV_MIN)
    #green_candle = (candle_color == 'green')

    if vol_cond_last and (not vol_cond):
        price_last = candles_df['close'].iloc[-3]
    else:
        price_last = candles_df['close'].iloc[-2]
    
    # Price conditions:
    price_change = 100*(price_curr - price_last)/price_last        
    price_cond = (price_change > PRICE_CHANGE_PC) 
   
    if (vol_cond or vol_cond_last) and (quote_col_cond or quote_col_cond_last) :
        #logger.info(f"Volume condition! {symbol}:{vol_cond}:{vol_cond_last}:{vol_curr}:{vol_prev}:{price_change:.1f}:{q_vol:.2f}")
        print(f"Volume condition! {symbol}:{vol_cond}:{vol_cond_last}:{vol_curr}:{vol_prev}:{price_change:.1f}:{q_vol:.2f}")
    if price_cond:
        #logger.info(f"Price condition! {symbol}:{vol_cond}:{vol_cond_last}:{vol_curr}:{vol_prev}:{price_change:.1f}:{q_vol:.2f}")
        print(f"Price condition! {symbol}:{vol_cond}:{vol_cond_last}:{vol_curr}:{vol_prev}:{price_change:.1f}:{q_vol:.2f}")
    if (vol_cond and quote_col_cond and price_cond and min_price_cond) or \
            (vol_cond_last and quote_col_cond_last and price_cond and min_price_cond):
        
        buy_time = time.time()
        book = client.get_order_book(symbol=symbol, limit=1000)
        buy_price = book['asks'][0][0]
        #buy_price = binance_endpoints.get_buy_amount(symbol, "MARKET")
        qties = [am/float(buy_price) for am in AMOUNTS]
        deals = [weighted_avg_orderBook(book['asks'], qty) for qty in qties]
        #logger.info(f"Buy {symbol} for {buy_price}!")
        print(f"Buy {symbol} for {buy_price}!")
        if vol_cond_last and (not vol_cond) :
            vol_curr = vol_curr_last
            trades = trades_last
            q_vol = q_vol_last
        in_trade[symbol] = {'symbol':symbol, 'buy_time':buy_time, 'vol_curr':vol_curr, 'q_vol':q_vol, 'price_change':price_change,'trades':trades,
                'quantities':qties, 'buy_deals':deals}


def evaluate_volume(symbol):
    global in_trade
    try:
        candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=50)
        candles_df = df(candles, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
        candles_df = candles_df.astype('float')
    except:
        #logger.exception(f"WARNING: Didn't get kline intervals for {symbol} during evaluation!")
        print(f"WARNING: Didn't get kline intervals for {symbol} during evaluation!")
        return 0
        #continue
    op, hi, lo, cl = candles_df['open'].iloc[-2], candles_df['high'].iloc[-2],candles_df['low'].iloc[-2],candles_df['close'].iloc[-2]
    last_candle_color = indicators.candle_params(op, hi, lo, cl)[-1]
    print(last_candle_color)
    time_now = time.time()
    elapsed = time_now - in_trade[symbol]['buy_time']
    if (last_candle_color == 'red') and (elapsed > 60): # Here elapsed time is measured in seconds
        # We wait at least 1 minute before we can sell. Maybe it is better to sell imediately if the candle where we bought closes red? 
        # We can use 'close_time' of the buy candle to check whether that candle has been closed or not.
        # Another idea for this strategy: don't wait until candle closes red, check only the sell volume.
        book = client.get_order_book(symbol=symbol)
        sell_deals = [ weighted_avg_orderBook(book['bids'], qty) for qty in in_trade[symbol]['quantities'] ]                
        in_trade[symbol]['sell_deals'] = sell_deals
        in_trade[symbol]['elapsed'] = elapsed/60
        in_trade[symbol]['profits'] = [100*(sell - buy)/buy for sell, buy in zip(in_trade[symbol]['sell_deals'], in_trade[symbol]['buy_deals'] )]
        ##logger.info(f"Sell {symbol} for {sell_deals[0]:.8f}! Profit: {in_trade[symbol]['profits'][0]:.2f}")
        print(f"Sell {symbol} for {sell_deals[0]:.8f}! Profit: {in_trade[symbol]['profits'][0]:.2f}")
        save_trade(in_trade, symbol)
        del in_trade[symbol]

def save_trade(in_trade, symbol):
    fname = 'trades_volume_strategy_demo3_1-10.dat'
    print(f"Saving trade info for {symbol} ...")
    with open(fname, 'a') as f:
        empty = os.path.getsize(fname) == 0
        if empty:
            f.write("buy_time, symbol, buy1, buy2, buy3, buy4, qty1, qty2, qty3, qty4, sell1, sell2, sell3, sell4, profit1, profit2, profit3, profit4, elapsed, vol_curr, q_vol, price_change, trades\n")
        buy_time = pd.to_datetime(in_trade[symbol]['buy_time'], unit='s')
        buy1,buy2,buy3,buy4 = in_trade[symbol]['buy_deals']
        qty1,qty2,qty3,qty4 = in_trade[symbol]['quantities']        
        sell1,sell2,sell3,sell4 = in_trade[symbol]['sell_deals']
        p1,p2,p3,p4 = in_trade[symbol]['profits']
        f.write(f"{buy_time:%Y-%m-%d %H:%M:%S},{symbol},{buy1:.8f},{buy2:.8f},{buy3:.8f},{buy4:.8f},{qty1:.2f},{qty2:.2f},{qty3:.2f},{qty4:.2f},{sell1:.8f},{sell2:.8f},{sell3:.8f},{sell3:.8f},{p1:.2f},{p2:.2f},{p3:.2f},{p4:.2f},{in_trade[symbol]['elapsed']:.1f},{in_trade[symbol]['vol_curr']:.1f},{in_trade[symbol]['q_vol']:.2f},{in_trade[symbol]['price_change']:.1f},{in_trade[symbol]['trades']}\n")


def save_volumes(symbol, candles):
    '''Record real-time volume data
    canldes -- a DataFrame containing the candlestick information
    '''
    b_vol = candles['volume']
    q_vol = candles['quote_av']
    
    vol_curr = b_vol.iloc[-1]
    q_vol_curr = q_vol.iloc[-1]    
    vol_prev = b_vol.iloc[-2]
    q_vol_prev = q_vol.iloc[-2]
    vol_prev_15 = b_vol.iloc[-16:-1].sum()
    q_vol_prev_15 = q_vol.iloc[-16:-1].sum()
    
    fname= f'VolumeData/{symbol}_volume.dat'
    with open(fname, 'a') as f:
        if os.path.getsize(fname) == 0:
            f.write("timestamp,vol_curr,q_vol_curr,vol_prev,q_vol_prev,vol_prev_15,q_vol_prev_15\n")
        timestamp = candles['timestamp'].iloc[-1]
        f.write(f"{timestamp},{vol_curr},{q_vol_curr},{vol_prev},{q_vol_prev},{vol_prev_15},{q_vol_prev_15}\n")
        

if __name__ == '__main__':
    
    # Get all trading symbols from the exchange
    #logger.debug("Starting the program...")
    symbols = binance_endpoints.get_symbols_BTC()
    
    in_trade = {}
    # start main infinite loop
         
    #logger.debug("Start searching for buy signals!")
    
    while True:
        i = 0
        t1 = time.time()
        while i < len(symbols):
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                #executor.map(binanceBarExtractor, symbols[i:i+8])
                if i + 8 <= len(symbols):
                    executor.map(volume_strategy, symbols[i:i+8])
                else:
                    executor.map(volume_strategy, symbols[i:])
            i += 8
        
        t2 = time.time()
        elpsd = t2 - t1
        #logger.debug(f"Checked all symbols in {elpsd} sec.")
        print(f"Checked all symbols in {elpsd} sec.")                

        if len(in_trade) > 0:
        
            ##logger.debug(f"{len(in_trade.keys())} coins are in trade")
            print(in_trade.keys())
            then = time.time()                
            #logger.debug("Start evaluating strategy")
            
            for symbol in list(in_trade):
                evaluate_volume(symbol)
            
            now = time.time()
            elpsd = now - then


#    while True:            
#        t1 = time.time()
#        with concurrent.futures.ProcessPoolExecutor() as executor:
#            executor.map(volume_strategy, [symbols[:50], symbols[50:100], symbols[100:150], symbols[150:]])        
##        volume_strategy(symbols[:50])
##        volume_strategy(symbols[50:100])
##        volume_strategy(symbols[100:150])
##        volume_strategy(symbols[150:])
#        t2 = time.time()
#        elpsd = t2 - t1
#        #logger.debug(f"Checked all symbols in {elpsd} sec.")
#        print(f"Checked all symbols in {elpsd} sec.")
#        ##logger.info("Are there any coins in trade?...")
#        if len(in_trade) > 0:
#            
#            ##logger.debug(f"{len(in_trade.keys())} coins are in trade")
#            print(in_trade.keys())
#            then = time.time()                
#            #logger.debug("Start evaluating strategy")
#            
#            for symbol in list(in_trade):
#                evaluate_volume(symbol)
#            
#            now = time.time()
#            elpsd = now - then
##            if elpsd < 10:
##                time.sleep(5)    
#    


    
#    with concurrent.futures.ProcessPoolExecutor() as executor:        
#        #logger.debug("Start serching for buy signals!")
#        while True:            
#            executor.map(volume_strategy, [symbols[:50], symbols[50:100], symbols[100:150], symbols[150:]])
#            ##logger.debug("Are there any coins in trade?...")
#            if len(in_trade) > 0:
#                
#                #logger.debug(f"{len(in_trade.keys())} coins are in trade")
#                then = time.time()                
#                #logger.debug("Start evaluating strategy")
#                
#                evaluate_volume(in_trade.keys()[0])
#                
#                now = time.time()
#                elpsd = now - then
#                if elpsd < 10:
#                    time.sleep(5)
        
    
    
    
            
            
            
            
        
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 8 23:58:43 2020

@author: Taras
"""

# Trading bot for binance using C1M strategy
# Author Taras Panamarev
# Date 29.01.2020
# Last redaction: 29.01.2020

# import all necessary modules
from binance.client import Client
from datetime import datetime
from pandas import DataFrame as df
import numpy as np
import pandas as pd
import keys

import requests
import json
import os
import time
from threading import Thread

from binance.enums import *
#from binance_api import Binance

# Techincal indicators library by bablofil
# https://bablofil.ru/raschet-indikatorov-SMMA-EWMA-MMA-RMA-RSI-STOCH-STOCHRSI/
# https://bablofil.ru/bot-dlya-binance-s-indikatorami/
#import bablofil_ta as ta

# Technical indicators from TA library
# https://technical-analysis-library-in-python.readthedocs.io/en/latest/
# https://github.com/bukosabino/ta
import ta



def get_symbols_BTC():
    import json
    #import requests
    #import time
    BASE_URL = 'https://api.binance.com'
    symbols = []
    try:
        resp = requests.get(BASE_URL + '/api/v1/ticker/allBookTickers')
    except:
        print("Time out error. No internet connection?")
        #print('Wait 10 sec. and retry...')
        start = time.time()
        done = start
        elapsed = done - start
        interval = 3600*5 #  seconds
        while elapsed < interval:
            print("Connecting....")
            try:
                resp = requests.get(BASE_URL + '/api/v1/ticker/allBookTickers')
                print("Connection established")
                break
            except:
                done = time.time()
                elapsed = done - start
                if elapsed < 10 : time.sleep(10)
            
    tickers_list = json.loads(resp.content)
    
    # Select only pairs with BTC
    for ticker in tickers_list:
        if str(ticker['symbol'])[-3:] == 'BTC':
            symbols.append(ticker['symbol'])
    return symbols


#def test_func(func, *args, **kwargs):
#    #print( args)
#    #print (kwargs)
#    #res1 = func(args)
#    #print len(*args)
#    res = func(*args, **kwargs)
#    return res

def try_func(func, step=10, duration=3600, *args, **kwargs):
    '''Allows to run any function 'func' with one argument arg inside the code in the body of the function.
    Trying to execute the function, if there is an exception, 
    assume that it is timeout error and call the fucntion func again 
    after every 'step' seconds during 'duration' period
    *args means that any number of positional arguments may be passed (can be empty)
    **kwargs means that any number of keyword arguments may be passed (can be empty)
    '''
    try:
        resp = func(*args, **kwargs)
    except Exception as e:
#        if (type(e) == 'ReadTimeout') or (type(e) == 'ConnectTimeout') or type(e) == 'ConnectionError':
#            print("Time out error. Retry connection...")
        print("Exception occured : ", func, e)    
        #print('Wait 10 sec. and retry...')
        start = time.time()
        done = start
        elapsed = done - start
        interval = duration #  seconds
        while elapsed < interval:
            print("Connecting....")
            try:
                resp = func(*args, **kwargs)
                print("Connection established")
                break
            except:
                done = time.time()
                elapsed = done - start
                if elapsed < step : time.sleep(step)
#        else:
#            print(func, '\n Unexpected Error:', e)
#            func(*args, **kwargs)
    return resp

def get_active(symb, active):
    '''input: symbol, list of symbols
    output: None
    Check if the trading pair satisfies the conditions: 24h volume > 150 BTC and price > 200 Satoshi
    If yes, append the pair in list of active coins
    '''
    #global active
    tmp = client.get_ticker(symbol=symb)
    price = np.float64(tmp['lastPrice'])
    volume = np.float64(tmp['quoteVolume'])
    if (0.00000200 < price < 0.009) & (volume > 150) :
        active.append([symb, price, volume])



def refresh_active():
    '''Create a list of active trading pairs
    Return list of active pairs as a pandas.DataFrame with 3 columns: symbol, price, 24h volume
    '''
    active = []
    for symbol in symbols:
        #Thread(target=get_active, args=(symbol,)).start()
        get_active(symbol, active)
    #    tmp = client.get_ticker(symbol=symbol)
    #    price = float(tmp['lastPrice'])
    #    volume = float(tmp['quoteVolume'])
    #    if (0.00000200 < price < 0.001) & (volume > 150) :
    #        active.append([symbol, price, volume])
    active = df(active, columns=['symbol', 'Price', "Volume"])
    print("Active list formed")
    print(active)
    return active
#fet = client.get_ticker(symbol='FETBTC')

#def test_kwargs(**kwargs):
#    candles = client.get_klines(**kwargs)
#    return df(candles, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])

#active_list = refresh_active()

def get_promising(active_list): 
    '''Select list of promising coins from the active list
    Inut: active coins
    Return: promising coins
    Promising coins satisfy the condition that distanse between lower and upper Bollinger bands > 4% (relative to lower)
    '''
    promising = {}
    symbols_tmp = []
    vol_tmp = []
    ranging_curr = []
    #count = 0
    for symbol in active_list["symbol"]:
        #Get 15 minute candles
        candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=200)
        #Convert the candles to DataFrame object:
        candles_df = df(candles, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
        #By defauls the values are 'str' -> convert them to float: 
        #close = np.float64(candles_df['close'])
        #low = np.float64(candles_df['low'])
        # Remove the BTC from symbols since we trade only in BTC:
        
        #Create another dataframe with time and open, high, low, close (ohlc) values only
        ohlc = candles_df.loc[:, ['timestamp', 'open', 'high', 'low', 'close']]
        ohlc = ohlc.astype(float)
        
        quote_av = candles_df.loc[-96:,'quote_av'].astype(float).sum()
        
        coin = symbol[:-3]
        #Compute Bollinger bands using TA library from bablofil
        #upper, middle, lower = ta.BBANDS(close)
        #Compute Bollinger bands using TA-lib from bukosabino
        
        # Get technical indicators:

        # Bollinger Bands
        BBands = ta.volatility.BollingerBands(ohlc['close'])
        #ohlc.values
        #middle = BBands.bollinger_mavg()
        #upper = BBands.bollinger_hband()
        lower = BBands.bollinger_lband()
        width = BBands.bollinger_wband()
        
        #RSI and stochastic:
        #RSI = ta.momentum.RSIIndicator(ohlc['close']).rsi()
        ##stoch = ta.momentum.StochasticOscillator(ohlc['high'],ohlc['low'],ohlc['close']).stoch()
        
        # Stochastic RSI 
#        stoch_RSI = ta.momentum.StochasticOscillator(RSI,RSI,RSI)
#        #fastk
#        fast_k = stoch_RSI.stoch_signal()
#        #slow_d is 3-day SMA of fast_k
#        slow_d = ta.volatility.bollinger_mavg(fast_k, n=3)
        
        #Conver the resulting values to numpy arrays
        #upper, middle, lower = np.array(upper), np.array(middle), np.array(lower)
        #ranging = (upper[-1] - lower[-1])/lower[-1]
        # Distance benween bands in '%'
        #ranging =  (upper - lower)/lower
        ranging= width/lower
        ## DEBUG!
        ##print(ranging)
        # Compute mean distance in case current ranging is small, but the coin was ranging high recently
        mean_range = np.mean(ranging[-96:]) # 96 15 minute candles give 24 hours
        max_range = np.max(ranging[-96:])
        
        #The condition for a promising coin:
        ##if  (max_range > 0.04) or (ranging[-1] > 0.04) : 
        if  (max_range > 0.04) :        
            print("Promising: %s, mean range: %.1f, max: %.1f, current: %.1f" % (coin, 100*mean_range,100*max_range, 100*ranging.iloc[-1])  )
            #promising.append([symbol:quote_av])
            #count += 1
            symbols_tmp.append(symbol)
            vol_tmp.append(quote_av)
            ranging_curr.append(100*ranging.iloc[-1])
            #promising[symbol] = quote_av
    promising = {'symbol':symbols_tmp, 'volume':vol_tmp, 'ranging':ranging_curr}
    print("Promising list formed")
    #promising = {k: v for k, v in sorted(promising.items(), key=lambda item: item[1])}
    promising = df(promising)#, columns=['symbol', 'volume'])
    promising.sort_values(by='volume', inplace=True, ascending=False)
    print(promising)
    return promising
    # Check of price is hitting lower Bollinger Bands:
    #if close[-1] < lower[-1] or low[-1] < lower[-1] : print("Hitting Bollinger: %s at %.8f" % (coin, close[-1]) ) 
    

def watch(promising):
    '''Watch the promising coins:
    Check if the price is hitting lower Bollinger bands, stochastic RSI k>d and both going up
    If coin fills conditions of BBands and stoch RSI, buy signal is triggered, coin is stored in the dictionary of triggered coins
    If after 5 minutes the coin is still satisfying the conditions, buy signal is triggered again
    '''
    global triggered, trading_coins #trade_modeling 
    
    #for symbol in promising:
    for symbol in promising.loc[:,'symbol']:
        candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=200)
        candles_df = df(candles, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
        candles_df = candles_df.astype('float')

        close = candles_df['close']
        low = candles_df['low']
        Open = candles_df['open']
        high = candles_df['high']
        volume = candles_df['quote_av']
        
        
        coin = symbol[:-3]
        
        if coin in list(trading_coins):
            if 'orderId' in trading_coins[coin]['order'].keys():
                check_buy_order(trading_coins[coin]['order'], trading_coins, coin)
            else:
                check_oco_order(trading_coins[coin]['order'], trading_coins, coin)
                            
        
        if coin in triggered.keys():
            #status = check_profit(triggered[coin]['coin'], close.iloc[-1], triggered[coin]['start_signal'], triggered[coin]['buy_price'], triggered[coin]['buy_time'] )
            status = check_profit_new(triggered[coin], close.iloc[-1])
            #status = check_profit_new(triggered[coin], candles_df.loc[:,'timestamp':'close'])
            if status == 0:
                continue
            else: 
                del triggered[coin]        
            
        # Get technical indicators:

        # Bollinger Bands
        BBands = ta.volatility.BollingerBands(close)
        #ohlc.values
        middle = BBands.bollinger_mavg()
        upper = BBands.bollinger_hband()
        lower = BBands.bollinger_lband()
        width = BBands.bollinger_wband()
        
        ranging= width/lower *100 # in %

        lower_BB_signal = ((close.iloc[-1] < lower.iloc[-1] or low.iloc[-1] < lower.iloc[-1]) \
        or (close.iloc[-2] < lower.iloc[-2] or low.iloc[-2] < lower.iloc[-2]) \
        or (close.iloc[-3] < lower.iloc[-3] or low.iloc[-3] < lower.iloc[-3]))

        middle_BB_signal = (  ( ( (close.iloc[-2] < middle.iloc[-2]) and (Open.iloc[-2] > middle.iloc[-2]) ) \
        or ( (low.iloc[-2] < middle.iloc[-2]) and (Open.iloc[-2] > middle.iloc[-2]) and (close.iloc[-2] > middle.iloc[-2]) ) )\
        and (close.iloc[-1] > middle.iloc[-1])  )       
       # Check if range is > 4% and price is hitting lower Bollinger Bands:
        #!!! Check Bollinger signal provided by ta library!
        if ranging.iloc[-1] > 4.0 :
            # Note. Use iloc() to get access to position (same as indexing in numpy arrays). Reason that there is no negative indexing in pandas.Series        
            if lower_BB_signal or middle_BB_signal: #print("Hitting Bollinger: %s at %.8f" % (coin, close[-1]) ) 
                # If Bollingers are hit: compute stochastic RSI            
                
                #RSI and stochastic:
                RSI = ta.momentum.RSIIndicator(close).rsi()
                ##stoch = ta.momentum.StochasticOscillator(ohlc['high'],ohlc['low'],ohlc['close']).stoch()            
                # Stochastic RSI 
                stoch_RSI = ta.momentum.StochasticOscillator(RSI,RSI,RSI)
                #fastk; convert to numpy arrays to allow negative index -- DON't do that!
                fastk =  stoch_RSI.stoch_signal() 
                #slow_d is 3-day SMA of fast_k
                slowd = ta.volatility.bollinger_mavg(fastk, n=3) 
                #Stochastic RSI signal fask > slowd, both going up and 10 < fastk < 30
                stochRSI_signal = (fastk.iloc[-1] > slowd.iloc[-1]) and (fastk.iloc[-1] > fastk.iloc[-2]) and (slowd.iloc[-1] > slowd.iloc[-2]) and (10 < fastk.iloc[-1] < 30)
                #Condition that distance from buy price to mid(up) bollinder band is greater than 1.5% 
                if stochRSI_signal:    
                    buy_price = float(get_buy_price(symbol,BUY_METHOD))
                    if lower_BB_signal:
                        dist_to_BB = 100*(middle.iloc[-1] - buy_price)/buy_price
                        price_to_range = (buy_price - lower.iloc[-1])/(middle.iloc[-1] - lower.iloc[-1])
                    else:
                        dist_to_BB = 100*(upper.iloc[-1] - buy_price)/buy_price
                        price_to_range = (buy_price - middle.iloc[-1])/(upper.iloc[-1] - middle.iloc[-1])
                    price_cond = (dist_to_BB > 1.5)
                    
                    #price_dist_cond = (price_to_range < 0.1)
                    price_dist_cond = True 
                    
                if stochRSI_signal and price_cond and price_dist_cond:
                    # Get the current time for output:                   
                    now = datetime.now()
                    current_time = now.strftime("%d/%m/%y %H:%M:%S")
                    #current_time = now.strftime("%D %H:%M:%S")  # With Day. !!! Figure out how to adjust format
                    # About to annouce buy signal.
                    # But first check if the coin was already signalling less than 5 minutes ago                
                    #counter = 0
                    if coin in triggered.keys(): 
                        # Get current time to measure time intervals between buy signals of a coin
                        time_now = time.time()

#                        # If the coin was signalling more than 5 minutes (300 sec) ago, remove it from the list
                        if time_now - triggered[coin]['start_signal'] > 300: 
                            buy_price = triggered[coin]['buy_price']
                            triggered[coin]['counter'] += 1
                            
                            save_signal(current_time, coin, buy_price, triggered[coin]['counter'])
                            #del triggered[coin]
                        else:
                            continue
                           
                    else:
                        #buy_price = close.iloc[-1]
                        buy_price = float(get_buy_price(symbol,BUY_METHOD))
                        min_price = buy_price
                        print(current_time, "BUY signal! %s at %.8f Stoch RSI:" % (coin, buy_price), fastk.iloc[-1], slowd.iloc[-1])
                        start_signal = time.time()
                        #save signaling coins to file:
                        counter = 0
                        # Check if there was a candle pattern 1 or 2 canclesticks before:
                        last = [Open.iloc[-2], high.iloc[-2], low.iloc[-2], close.iloc[-2]]
                        before_last = [Open.iloc[-3], high.iloc[-3], low.iloc[-3], close.iloc[-3]]
                        before_before_last = [Open.iloc[-4], high.iloc[-4], low.iloc[-4], close.iloc[-4]]
                        pattern = candle_pattern(last, before_last)
                        if pattern == 'no': pattern = candle_pattern(before_last, before_before_last)
                        save_signal(current_time, coin, buy_price, counter, pattern)
                        # Create a new item in the dictionary of signalling coins, where coin is the key and time at the 1st signal is the item
                        status = 0
                        if lower_BB_signal:
                            origin = 'lower'
                        else: origin = 'upper'
                        vol_1hr = volume[-4:].sum()
                        #triggered[coin] = [start_signal, buy_price, current_time, status]
                        #Create a dictionary with all coins that triggered a buy signal:
                        triggered[coin] = {'coin':coin, 'start_signal':start_signal, 'buy_price':buy_price, 'buy_time':current_time, 'status':status, 'counter':counter, \
                                 'vol_1hr':vol_1hr, 'pattern':pattern, 'origin':origin, 'ranging':ranging.iloc[-1], 'fastk15': fastk.iloc[-1], 'min_price':min_price }
                        n_trades = len(trading_coins)
                        if (n_trades < MAX_TRADES) and (coin not in trading_coins.keys() ):
                            #in_trade = len(trading_coins)
                        
                            print("Placing Buy Order")
                            try:
                                order, am_btc = place_buy_order(symbol,MAX_TRADES,n_trades,BUY_METHOD)
                            except Exception as e:
                                print("Exception during pacing BUY order occured", e)
                                continue                             
                                
                            if BUY_METHOD == 'MARKET':
                                buy_price = order['price']
                                print("Placing OCO order", symbol)
                                try:                                    
                                    order = place_oco_order(symbol, buy_price, take_profit=0.017, stop_loss = 0.02)
                                except Exception as e:
                                    print("Exception during placing OCO order:", symbol, buy_price)
                                    print(e)
                            else:
                                #If we didn't use market buy then we have to check the order status                                    
                                print("Check order status")
                                try:
                                    order = check_order_status(order)
                                except Exception as e:
                                    print("Didn't manage to check order status", e)
                            
                            trading_coins[coin] = {'coin':coin, 'start_signal':start_signal, 'buy_price':buy_price, \
                                      'buy_time':current_time, 'status':status, 'counter':counter, \
                                      'vol_1hr':vol_1hr, 'pattern':pattern, 'origin':origin,\
                                      'ranging':ranging.iloc[-1], 'fastk15': fastk.iloc[-1], 'min_price':min_price, 'order':order, 'am_btc': am_btc,'n_oco':0 }
              

def save_signal(t_curr, coin, price, counter, pattern):
    with open('buy_signals_pattern_new.dat', 'a') as f:
        f.write("%s   %5s    %.8f   %2d  %12s\n" % (t_curr, coin, price, counter, pattern))

def save_coin_prev(coin, bought_time, bought_price, profit, went_down_by, elapsed):
    with open('bought_coins.dat', 'a') as f:
        f.write("%s  %5s  %.8f   %.2f  %.2f  %.1f\n" % (bought_time, coin, bought_price, profit, went_down_by, elapsed))

def save_coin(coin, bought_time, bought_price, profit, elapsed):
    with open('bought_coins_new.dat', 'a') as f:
        f.write("%s  %5s  %.8f   %.2f  %.1f\n" % (bought_time, coin, bought_price, profit, elapsed))

def save_coin_full(coin, bought_time, bought_price, profit, elapsed, pattern, origin, ranging, vol_1hr, fastk15, min_price, trading):
    with open('bought_coins_full_market.dat', 'a') as f:
        f.write("%s,%s,%.8f,%.2f,%.1f,%s,%s,%.2f,%.3f,%.2f,%.8f,%s\n" % (bought_time, coin, bought_price, profit, elapsed,\
                                                                      pattern,origin,ranging,vol_1hr,fastk15,min_price,trading))

def save_filled_oco(coin_dict, profit):
    #!!!TODO in the morning!!!!!!!!
    elapsed = (time.time() - coin_dict['start_signal'])/60
    bought_time, coin, bought_price, pattern,origin,ranging,vol_1hr,fastk15,min_price = \
    coin_dict['buy_time'],coin_dict['coin'],float(coin_dict['buy_price']),coin_dict['pattern'], coin_dict['origin'],\
    coin_dict['ranging'],coin_dict['vol_1hr'],coin_dict['fastk15'],coin_dict['min_price']
    am_btc, g_profit = coin_dict['am_btc'], coin_dict['g_profit']
    with open('filled_oco_orders_market.dat', 'a') as f:
        f.write("%s,%s,%.8f,%.2f,%.1f,%s,%s,%.2f,%.3f,%.2f,%.8f,%.8f,%.8f\n" % (bought_time, coin, bought_price, profit, elapsed,\
                                                                      pattern,origin,ranging,vol_1hr,fastk15,min_price,\
                                                                      am_btc,g_profit))
    
def check_profit(coin, price_curr, bought_time, bought_price, bought_time_fmt, take_profit = 0.015, stop_loss = 0.03):
    if price_curr >= (1+take_profit)*bought_price:
        time_now = time.time()
        elapsed = (time_now - bought_time)/60
        status = 1
    elif price_curr <= (1-stop_loss)*bought_price:
        time_now = time.time()
        elapsed = (time_now - bought_time)/60
        status = -1
    else:
        status = 0
    
    if status != 0 :    
        profit = (price_curr - bought_price)/bought_price * 100
        save_coin(coin, bought_time_fmt, bought_price, profit, elapsed)
    return status

def check_profit_new(coin_dict, price_curr, take_profit = 0.015, stop_loss = 0.03, trading="NO"):
    coin = coin_dict['coin']
    bought_price = coin_dict['buy_price']
    bought_time = coin_dict['start_signal']
    #Check minimum price:
    if price_curr < bought_price : coin_dict['min_price'] = price_curr
    #Check stop loss or take profit
    if price_curr >= (1+take_profit)*bought_price:
        time_now = time.time()
        elapsed = (time_now - bought_time)/60
        status = 1
    elif price_curr <= (1-stop_loss)*bought_price:
        time_now = time.time()
        elapsed = (time_now - bought_time)/60
        status = -1
    else:
        status = 0
    #if trading complete, save coin data:
    if status != 0 :    
        bought_time_fmt = coin_dict['buy_time']
        profit = (price_curr - bought_price)/bought_price * 100
        save_coin_full(coin, bought_time_fmt, bought_price, profit, elapsed, \
                       coin_dict['pattern'], coin_dict['origin'], coin_dict['ranging'], \
                       coin_dict['vol_1hr'], coin_dict['fastk15'], coin_dict['min_price'], trading)
    return status    


        
def candle_params(Open, High, Low, Close):
    #import math
    green = Close >= Open
    red = Close < Open
    #doji = Close == Open
    color = 'green'
    if green:
        low_shadow = Open - Low
        high_shadow = High - Close
        color = 'green'
    #if red:
    else:
        low_shadow = Close - Low
        high_shadow = High - Open
        color = 'red'
    body = np.abs(Close - Open)
    return body, low_shadow, high_shadow, color

def is_hammer(body, low_shadow, high_shadow):
    hammer = False
    if body == 0: hammer = False
    elif low_shadow/body >= 2.0 and high_shadow/body <= 1.5:
        hammer = True
    else: hammer = False
    return hammer

def is_doji(body, low_shadow, high_shadow):
    doji = False
    if body == 0:  
        doji = True # Any type of doji
    elif high_shadow != 0:
        if (0.9 < low_shadow/high_shadow < 1.1) and (low_shadow/body > 5):
            doji = True # standard doji, but body is not zero
        if (low_shadow/high_shadow > 5) and (low_shadow/body > 5) : doji = True  # Dragonfly doji
    elif low_shadow != 0:
        if (high_shadow/low_shadow > 5) and (high_shadow/body > 5): doji = True # gravestone doji
    return doji
    

def candle_pattern(last, before_last):
    '''Start counting from last candle
    '''
    #import math
    pattern = {'last':'no', 'blast':'no', 'total':'no'}
    Open_l, High_l, Low_l, Close_l = [item for item in last]
    Open_bl, High_bl, Low_bl, Close_bl = [item for item in before_last]
    Body_l, high_shadow_l, low_shadow_l, color_l = candle_params(Open_l, High_l, Low_l, Close_l)
    Body_bl, high_shadow_bl, low_shadow_bl, color_bl = candle_params(Open_bl, High_bl, Low_bl, Close_bl)
    if is_doji(Body_l, high_shadow_l, low_shadow_l):
        pattern['last'] = 'Doji'
        return pattern['last']
    elif is_hammer(Body_l, high_shadow_l, low_shadow_l):
        pattern['last'] = 'Hammer'
        return pattern['last']
    else: pattern['last'] = 'no'
    
    if is_doji(Body_bl, high_shadow_bl, low_shadow_bl):
        pattern['blast'] = 'Doji'
        return pattern['blast']
    elif is_hammer(Body_bl, high_shadow_bl, low_shadow_bl):
        pattern['blast'] = 'Hammer'
        return pattern['blast']
    else: pattern['blast'] = 'no'
    
    # Check for Bullish engulfing and Harami
    if color_l == 'green' and color_bl == 'red':
        Bullish_eng = Open_l < Close_bl and Close_l > Open_bl
        Harami = Open_l > Close_bl and Close_l < Open_bl
    else: 
        Bullish_eng = False
        Harami = False
    if Bullish_eng: pattern['total'] = 'Bullish eng.'
    if Harami: pattern['total'] = 'Harami'
    
    if pattern['total'] != 'no': 
        pattern = pattern['total']
    elif pattern['last'] != 'no' :
        pattern = pattern['last']
    else:
        pattern = 'no'
    
    return pattern

def get_buy_price(symbol, BUY_METHOD):
    '''For now use very basic approach - place order at 2nd place in the order book for a limit order
    For a market order take the first price. For market order we need price to determine the quatity of the coin we are going to buy
    '''
    order_book = client.get_order_book(symbol=symbol)
    if BUY_METHOD == 'MARKET':
        buy_price = order_book['asks'][0][0]
    else:        
        buy_price = order_book['bids'][1][0]
    return buy_price

def weighted_avg(fills, symbol):
    '''Computes weghted average price of a market order
    fills is array of dictionaries with prices and quantities (from market order API response)
    "fills": [
        {
            "price": "4000.00000000",
            "qty": "1.00000000",
            "commission": "4.00000000",
            "commissionAsset": "USDT"
        },
        {
            "price": "3999.00000000",
            "qty": "5.00000000",
            "commission": "19.99500000",
            "commissionAsset": "USDT"
        }]
    '''
    price_times_qty = 0
    qty = 0    
    #
    for fill in fills:       
        price_times_qty += float(fill['price']) * float(fill['qty'])
        qty += float(fill['qty'])
    #
    avg_price = price_times_qty/qty
    # Round the price to the nearest step
    price_prec = get_price_precision(symbol)
    avg_price = np.round( avg_price, price_prec)
    avg_price = "%.8f" % avg_price
    
    return  avg_price
    
       

#Orders:

def place_market_sell_order(symbol, qty):
    market_order = client.order_market_sell(symbol=symbol, quantity=qty)
    return market_order

def place_market_buy_order(symbol, qty):
    market_order = client.order_market_buy(symbol=symbol, quantity=qty)
    return market_order
    
def place_buy_order(symbol,MAX_TRADES,in_trade, BUY_METHOD):
    #amount_btc = 0.0058 #Our BTC amount for each trade (about $50 as of 01:40 29.02.2020)
    BuyPrice = get_buy_price(symbol, BUY_METHOD)
    #qty = amount_btc/np.float(BuyPrice)
    #qty = np.round(qty,0)
    qty = get_buy_amount(BuyPrice,MAX_TRADES,in_trade)
    prec = get_lot_precision(symbol)
    qty = float(truncate(qty, prec))
    
    if BUY_METHOD == 'MARKET':
        order = place_market_buy_order(symbol, qty)
        BuyPrice = weighted_avg(order['fills'], symbol)
        order['price'] = BuyPrice # !!! Check if market order API response returns weighted average price!!!!!!!
    else:
        order = client.order_limit_buy(symbol=symbol, quantity=qty, price=BuyPrice)
    
    with open('placed_buy_orders.dat', 'a') as f:
        now = datetime.now()
        time_curr = now.strftime("%d/%m/%y %H:%M:%S")
        am_btc = qty*float(BuyPrice)                        
        f.write("%s,%s,%s,%.3f,%.8f\n" % (time_curr, symbol, BuyPrice, qty, am_btc))

    return order, am_btc

def get_buy_amount(BuyPrice, MAX_TRADES, in_trade, asset='BTC'):
    '''Determine the amount of coin to buy. 
    MAX_TRADES - number of maximum simulteneous trades
    in_trade - number of coins tradind at the moment
    Takes total available amount in BTC and devides it by MAX_TRADES
    Algorithm: take 'free' balance of BTC and devide by (MAX_TRADES - in_trade)
    Note, in trade is computed before the coin is added to the trading list
    '''
    #amount_btc = float(client.get_asset_balance(asset)['free'])+float(client.get_asset_balance(asset)['locked'])
    amount_btc = float(client.get_asset_balance(asset)['free'])
    #in_trade = len(trading_coins)
    amount_btc = 0.1*amount_btc/(MAX_TRADES - in_trade)
    qty = amount_btc/float(BuyPrice)
    return qty

def place_oco_order(symbol, buy_price, take_profit = 0.015, stop_loss = 0.03):
    #from binance.enums import *
    buy_price = float(buy_price)
    #price = np.round( (1+take_profit)*buy_price, 8)
    #Get precision for price value of the symbol:
    price_prec = get_price_precision(symbol)
    price = np.round( (1+take_profit)*buy_price, price_prec)
    price = "%.8f" % price
    #print('price = ", price)
    #stop_price = np.round( (1-stop_loss/1.2)*buy_price, 8)
    stop_price = np.round( (1-stop_loss/1.2)*buy_price, price_prec)
    #stop_price = str(stop_price)
    stop_price = "%.8f" % stop_price
    #stop_limit_price = np.round( (1-stop_loss)*buy_price, 8)
    stop_limit_price = np.round( (1-stop_loss)*buy_price, price_prec)
    #stop_limit_price = str(stop_limit_price)
    stop_limit_price = "%.8f" % stop_limit_price
    qntity = float( client.get_asset_balance(asset=symbol[:-3])['free'] )
    #qntity = int(qntity//1) # take integer part from the balance
    prec = get_lot_precision(symbol)
    qntity = float(truncate(qntity, prec)) #It's important to truncate qunatity, NOT to round it, otherwise may be not enough balance
    #print("Place OCO order for %s for %s with stop price %s and limit price %s" % (symbol, price, stop_price, stop_limit_price) )    
    oco_order = client.create_oco_order(symbol=symbol,side=SIDE_SELL, stopLimitTimeInForce=TIME_IN_FORCE_GTC,\
                                        quantity=qntity, stopPrice=stop_price, price=price,\
                                        stopLimitPrice=stop_limit_price)
    now = datetime.now()
    time_curr = now.strftime("%d/%m/%y %H:%M:%S")
    with open('placed_oco_orders.dat', 'a') as f:
        f.write('%s,%s,%s,%s,%s\n' % (time_curr, symbol, price, stop_price, stop_limit_price))
    #sell_limit = client.order_limit_sell(symbol=symbol, quantity=qntity, price=price)
    #return sell_limit
    return oco_order

def check_order_status(order):
    try:
        order_status = client.get_order(symbol=order['symbol'], orderId=order['orderId'])
    except Exception as e:
        print("Warning! Didn't manage to check order status", order['symbol'])
        print(e)
        order_status = order
    #client.order_market_sell(symbol='XTZBTC',quantity=qntity)
    return order_status

def cancel_order(order):
    result = client.cancel_order(symbol=order['symbol'], orderId=order['orderId'])
    return result

def check_buy_order(order, trading_coins, coin):    
    '''order status response:
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
    #!!!TODO: Improce Partial filling
    BUY_TIME_LIMIT = 10
    try:
        trading_coins[coin]['order'] = check_order_status(order)
    except Exception as e:
        print("Warning didn't manage to check order status! (Called from fuction 'check buy order')")
        print(e)
    status = trading_coins[coin]['order']['status']
    try:
        place_time = trading_coins[coin]['order']['time']/1000 #Time in Binance is given in milliseconds, convert to seconds here.
    except KeyError:
        place_time = trading_coins[coin]['order']['transactTime']/1000
    time_curr = time.time()
    elapsed_min = (time_curr - place_time)/60
    if elapsed_min > BUY_TIME_LIMIT:
       order = cancel_order(order)
       print("Buy order didn't fill in %d minutes" % BUY_TIME_LIMIT)
       executedQty = float(trading_coins[coin]['order']['executedQty'])
       if executedQty > 0:
           print("Order Partially Filled", coin)
           try:
               sell_leftovers(trading_coins[coin]['order']['symbol'])    
           except Exception as e:
               print("Warning! Order partially filled, but it was not sold!", coin, executedQty)
               print(e)
               del trading_coins[coin]
       status = "CANCELLED"
       del trading_coins[coin]
    if status == "FILLED":
        try:
            trading_coins[coin]['order'] = place_oco_order(order['symbol'], order['price'])
        except Exception as e:
            print("WARNING!!!! OCO order didn't place!", e)
        
    return status
        
def check_oco_order(order, trading_coins, coin, time_limit = 10):
    '''OCO order response:
    {'orderListId': 2804760,
         'contingencyType': 'OCO',
         'listStatusType': 'EXEC_STARTED',
         'listOrderStatus': 'EXECUTING',
         'listClientOrderId': '4vtKEEteQ4sMxeeJdedcFY',
         'transactionTime': 1583699174031,
         'symbol': 'WRXBTC',
         'orders': [{'symbol': 'WRXBTC',
           'orderId': 7897973,
           'clientOrderId': '6RTVZNxjMTBBt4UOL350Xm'},
          {'symbol': 'WRXBTC',
           'orderId': 7897974,
           'clientOrderId': 'jAdBKYiZNYTeGP2lB34P2t'}],
         'orderReports': [{'symbol': 'WRXBTC',
           'orderId': 7897973,
           'orderListId': 2804760,
           'clientOrderId': '6RTVZNxjMTBBt4UOL350Xm',
           'transactTime': 1583699174031,
           'price': '0.00002068',
           'origQty': '1109.00000000',
           'executedQty': '0.00000000',
           'cummulativeQuoteQty': '0.00000000',
           'status': 'NEW',
           'timeInForce': 'GTC',
           'type': 'STOP_LOSS_LIMIT',
           'side': 'SELL',
           'stopPrice': '0.00002079'},
          {'symbol': 'WRXBTC',
           'orderId': 7897974,
           'orderListId': 2804760,
           'clientOrderId': 'jAdBKYiZNYTeGP2lB34P2t',
           'transactTime': 1583699174031,
           'price': '0.00002164',
           'origQty': '1109.00000000',
           'executedQty': '0.00000000',
           'cummulativeQuoteQty': '0.00000000',
           'status': 'NEW',
           'timeInForce': 'GTC',
           'type': 'LIMIT_MAKER',
           'side': 'SELL'}]}
    '''
    #OCO_TIME_LIMIT = 240
    #trading_coins[coin]['order'] = check_order_status(order)
    #OCO order consists of 2 orders. Check their status separetely:
    #Grab order responses:
    stop_loss_order = trading_coins[coin]['order']['orderReports'][0]
    limit_maker = trading_coins[coin]['order']['orderReports'][1]
    #Check their status
    stop_loss_order = client.get_order(symbol = stop_loss_order['symbol'], orderId=stop_loss_order['orderId'])    
    limit_maker = client.get_order(symbol = limit_maker['symbol'], orderId=limit_maker['orderId'])    
    if (limit_maker['status'] == 'EXPIRED') or (stop_loss_order['status'] == 'EXPIRED'):       
        now = datetime.now().strftime("%d/%m/%y %H:%M:%S")
        print("OCO starts filling: %s, place time: %s, execute time: %s" % (coin, trading_coins[coin]['buy_time'], now) )
        limit_filled = (limit_maker['status'] == 'FILLED')
        loss_filled = (stop_loss_order['status'] == 'FILLED')
        if limit_filled :
            now = datetime.now().strftime("%d/%m/%y %H:%M:%S")
            print("OCO LIMIT filled: %s, place time: %s, execute time: %s" % (coin, trading_coins[coin]['buy_time'], now) )
            profit = 100*(float(limit_maker['price']) - float(trading_coins[coin]['buy_price']))/float(trading_coins[coin]['buy_price'])
            trading_coins[coin]['profit'] = profit
            trading_coins[coin]['g_profit'] = 0.01*profit*trading_coins[coin]['am_btc']
            try:
                save_filled_oco(trading_coins[coin], profit)
            except Exception as e:
                print("File save error: ", e)
                with open("save_filled_oco.dat", 'a') as f:
                    f.write(str(trading_coins[coin]))
                    f.write("\n")
            #del trading_coins[coin]
        if loss_filled:
            now = datetime.now().strftime("%d/%m/%y %H:%M:%S")
            print("OCO STOP LOSS filled: %s, place time: %s, execute time: %s" % (coin, trading_coins[coin]['buy_time'], now) )
            profit = 100*(float(stop_loss_order['price']) - float(trading_coins[coin]['buy_price']))/float(trading_coins[coin]['buy_price'])
            trading_coins[coin]['profit'] = profit
            trading_coins[coin]['g_profit'] = 0.01*profit*trading_coins[coin]['am_btc']
            try:
                save_filled_oco(trading_coins[coin], profit)
            except Exception as e:
                print("File save error: ", e)
                with open("save_filled_oco.dat", 'a') as f:
                    f.write(trading_coins[coin])
                    f.write("\n")
            #del trading_coins[coin]   
        if limit_filled or loss_filled :
            #Make sure that the coin has been completely sold:
            try:
                sell_leftovers(trading_coins[coin]['order']['symbol'])
            except Exception as e:
                print("Warning! Didn't manage to sell leftovers!", e)
            del trading_coins[coin]
            #print (trading_coins)
        return "FILLED"
    elif (limit_maker['status'] == 'CANCELED') or (stop_loss_order['status'] == 'CANCELED'):
        status = 'CANCELED'
        del trading_coins[coin]
    else:
        trading_coins[coin]['order']['orderReports'][0] = stop_loss_order
        trading_coins[coin]['order']['orderReports'][1] = limit_maker
        status = "EXECUTING"
        #last_price = 
        print("BUY: %s, SELL: %s" % (trading_coins[coin]['buy_price'],limit_maker['price']) )
        place_time = order['transactionTime']/1000
        time_curr = time.time()
        elapsed_min = (time_curr - place_time)/60
        if elapsed_min > time_limit and trading_coins[coin]['n_oco'] == 0 :
            #qty = float(stop_loss_order["origQty"])
            symbol = stop_loss_order['symbol']
            cancel_order(stop_loss_order)
            try: 
                trading_coins[coin]['order'] = place_oco_order(symbol,trading_coins[coin]['buy_price'], take_profit=0.015,stop_loss=0.03)
                trading_coins[coin]['n_oco'] += 1
                print("Placed 2nd OCO order", symbol)
            except Exception as e:
                print("Error occured while placing 2nd OCO order!", e)
                sell_leftovers(symbol)
                del trading_coins[coin]
            
            #market_sell = place_market_sell_order(symbol, qty)            
            #price = weighted_avg(market_sell['fills'], symbol)
            #profit = 100*(float(price) - float(trading_coins[coin]['buy_price']))/float(trading_coins[coin]['buy_price']) 
            #trading_coins[coin]['profit'] = profit
            #trading_coins[coin]['g_profit'] = 0.01*profit*trading_coins[coin]['am_btc']
            #status = 'SOLD'
            #save_filled_oco(trading_coins[coin], profit)
            #place market sell order
            #poka bez etogo oboidemsya

    return status

def sell_leftovers(symbol):
    coin = symbol[:-3] 
    info = client.get_symbol_info(symbol)
    stepSize = float(info['filters'][2]['stepSize'])
    leftover = float(client.get_asset_balance(coin)['free'])
    if leftover > stepSize:
        print("Not everything has been sold for some reason!")
        qty = leftover
        prec = get_lot_precision(symbol)
        qty = float(truncate(qty, prec))
        try:
            print("Place market sell order", symbol, qty)
            place_market_sell_order(symbol, qty)
            
            #del trading_coins[coin]
        except Exception as e:
            print("Warning! Market sell didn't execute!" , e, symbol )
        
    
def follow_buy_order(order):
    status = ''
    start_time = time.time()
    MAX_PENDING_TIME = 30 #minutes
    while status != 'FILLED':
        status = check_order_status(order)['status']
        elapsed = (time.time() - start_time)/60
        print(status)
        if elapsed > MAX_PENDING_TIME : 
            cancel_order(order)
            print( "Cancel order, didn't fill in %.1f minutes" % elapsed )
            break

def follow_oco_order(order):
    pass

def get_lot_precision(symbol):
    '''return precision of the lot for the order
    algorithm: take step size from the exchange, then split it by '.', after that find '1' in the part after '.'
    It returns the position of digit 1 after '.'. If it is not found then it returns -1.
    When we add 1 to the result it also handles the case when the digit 1 is not there (zero precision)
    Works for BTC trading pairs, but have to be tested on other trading assests'''
    info = client.get_symbol_info(symbol)
    stepSize = info['filters'][2]['stepSize']
    prec = stepSize.split('.')[-1].find('1') + 1
    return prec

def get_price_precision(symbol):
    '''return precision of the price for the order
    algorithm: take step size from the exchange, then split it by '.', after that find '1' in the part after '.'
    It returns the position of digit 1 after '.'. If it is not found then it returns -1.
    When we add 1 to the result it also handles the case when the digit 1 is not there (zero precision)
    Works for BTC trading pairs, but have to be tested on other trading assests'''
    info = client.get_symbol_info(symbol)
    stepSize = info['filters'][0]['tickSize']
    prec = stepSize.split('.')[-1].find('1') + 1
    return prec

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding
    Taken from: https://stackoverflow.com/questions/783897/truncating-floats-in-python
    '''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


        

def follow_coin(bought_coin, take_profit = 0.015, stop_loss = 0.03):
    '''bought_coin should be a dictionary with coin name as a key and item a list of the following parameters:
    bought time, bought price, %down, %up, elapsed time from buy to sell
    %down is by how much in % the price went down relative to the buy price
    %up is by how much % the price went up in total (relative to the buy price)
    !!! Modify function later or deprecate.
    '''
#    symbol = bought_coin.keys()[0] + 'BTC'
#    candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=200)
#    candles_df = df(candles, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
#    candles_df = candles_df.astype('float')
#    ##close = np.float64(candles_df['close'])
#    close = candles_df['close']
#    ##low = np.float64(candles_df['low'])
#    low = candles_df['low']
#    coin = symbol[:-3]
#    curr_price = close
#    if curr_price >= (1+take_profit)*bought_price:
#        time_curr = time.time()
#        elapsed = time_curr - bought_time



if __name__ == '__main__':

    
    # Authorise to binance using public and private key pairs stored in different file
    client = Client(api_key=keys.Pkey, api_secret=keys.Skey)
    
    # Some test code here:   --------------------------------------------     
#    symbol = 'PERLBTC'    
#    candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=200)
#    candles_df = df(candles, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
#    candles_df = candles_df.astype('float')
#    
#    length = len(candles_df['open'])
#    for i in range(1,length):
#        items = ['open', 'high', 'low', 'close']
#        last = [candles_df[item].iloc[i] for item in items]
#        before_last = [candles_df[item].iloc[i-1] for item in items]
#        pattern = candle_pattern(last, before_last)
#        timestamp = pd.Timestamp(candles_df['timestamp'][i], unit='ms', tz='Asia/Almaty')
#        #timestamp = datetime.fromtimestamp(timestamp)#.strptime("%d/%m/%Y, %H:%M:%S")
#        print(timestamp.strftime("%d/%m/%y %H:%M:%S"), "   %12s\n" % (pattern) )
    ## End test piece of code -------------------------------------------------------------------------------------

    #MAX_TRADES = 4 # Maximum number of simultaneous trades, if 0 then it doesn't trade, just shows signals
    MAX_TRADES = 0  # 
    #BUY_METHOD = 'MARKET'
    BUY_METHOD = 'LIMIT'
    
    symbols = get_symbols_BTC()    
    ##########     Main Body  ------------------------
    #create list of 'active' coins
    #active_list = refresh_active()
    active_list = try_func(refresh_active)
    ##print("Active list formed")
    ##print(active_list)
    #create list of 'promising' coins which are ranging more than 4%
    args = (active_list,)
    #promising = get_promising(active_list)
    
    promising = try_func(get_promising, 10, 3600, *args)
    ##print("Promising list formed")
    ##print(promising)
    # while True: watch(promising)
    #Initialise dictionary for the array of coins that triggered BUY signal:
    triggered = {}
    #Initialise dictionaries for trading coins
    trading_coins={}
#    trading_buy = {}
#    trading_OCO = {}
    #Dictionary for trading simulation
    #trade_modeling = {}
    #Initialise watch time counter
    total_watch = 0
    last_active_uptade = 0
    
    # Set intervals for updating lists of promising and active coins:
    promise_update_interval = 300 # in seconds
    active_update_interval = 600 # in seconds
    
    # start infiinte loop:
    start_loop = time.time()
    while True:        
        #create list of 'active' coins
        #active_list = refresh_active()
        #create list of 'promising' coins which are ranging more than 4%
        #promising = get_promising(active_list)
        # Watch promising coins
        start_watch = time.time()   
        args = (promising,)
        try_func(watch, 10,3600, *args)
        end_watch = time.time()
        
        are_trading = len(trading_coins)
        #If there are coins in trade, check their status
        if are_trading > 0: 
            #print("%d coins are in trade:" % are_trading)
            for item in list(trading_coins): #here better to use list(trading_coins) instead of trading_coins.keys()
                if 'orderId' in trading_coins[item]['order'].keys():
                    print("BUY PENDING:", item)
                    try:
                        check_buy_order(trading_coins[item]['order'], trading_coins, item)
                    except Exception as e:
                        print("Didn't check BUY order", item)
                        print(e)
                else:
                    print("OCO:", item)
                    try:
                        check_oco_order(trading_coins[item]['order'], trading_coins, item)
                    except Exception as e:
                        print("Didn't check OCO order", item)
                        print(e)
                        

        # Measure total time spent for watching:
        total_watch += end_watch - start_watch
        # If watching time is more than 5 minutes, update promising list: 
        if total_watch > promise_update_interval: 
            args = (active_list,)
            #promising = get_promising(active_list)
            promising = try_func(get_promising, 10, 3600, *args)
            last_active_uptade += total_watch
            total_watch = 0
        # Update active list every 10 minutes
        if last_active_uptade > active_update_interval:
            #active_list = refresh_active()
            active_list = try_func(refresh_active)
            args = (active_list,)
            #promising = get_promising(active_list)
            promising = try_func(get_promising, 10, 3600, *args)
            #are_trading = len(trade_modeling)can

            are_trading = len(trading_coins)
            print("%d coins are in trade:" % are_trading)            
            if are_trading > 0:                 
                for item in trading_coins.keys():
                    print(item, trading_coins[item]['buy_time'])
                    
            total_watch = 0
            last_active_uptade = 0
        
# !!!TODO: If the coin is no longer 'promising', but still in triggered
# !!!              
    


    
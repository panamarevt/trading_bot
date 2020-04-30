# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:58:31 2020

@author: Taras
"""

import numpy as np
from pandas import DataFrame as df
import pandas as pd

import time
from datetime import datetime

#from binance_endpoints import get_symbols_BTC, get_ticker
#from binance_endpoints import *   # quick and dirtly solution to check consisntency with previous version. to be changed!
import binance_endpoints
from binance.client import Client

import indicators

def save_signal(t_curr, coin, price, counter, pattern):
    with open('buy_signals_pattern_new.dat', 'a') as f:
        f.write("%s   %5s    %.8f   %2d  %12s\n" % (t_curr, coin, price, counter, pattern))

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
#        save_coin_full(coin, bought_time_fmt, bought_price, profit, elapsed, \
#                       coin_dict['pattern'], coin_dict['origin'], coin_dict['ranging'], \
#                       coin_dict['vol_1hr'], coin_dict['fastk15'], coin_dict['min_price'], trading)
    return status    

client = Client() # to be moved to binance_endpoints

class C1M:
    '''Body of the C1M strategy'''      
    def __init__(self, min_24h_volume = 150, min_price = 0.00000200, max_price = 0.009, min_ranging = 0.04):
        self.min_24h_volume = min_24h_volume
        self.min_price = min_price
        self.max_price = max_price
        self.min_ranging = min_ranging
        #Initialize empty dictionaries for coins in trade:
        self.triggered = {}
        self.trading_coins = {}
    
    def get_active(self, symb, active):
    #def get_active(symb, active):
        '''input: symbol, list of symbols
        output: None
        Check if the trading pair satisfies the conditions: 24h volume > 150 BTC and price > 200 Satoshi
        If yes, append the pair in list of active coins
        TODO: Merge with refresh active!
        '''
        #global active
        tmp = binance_endpoints.get_ticker(symb)
        price = np.float64(tmp['lastPrice'])
        volume = np.float64(tmp['quoteVolume'])
        
        if (self.min_price < price < self.max_price) & (volume > self.min_24h_volume) :
            active.append([symb, price, volume])


    def refresh_active(self):
        '''Create a list of active trading pairs
        Return list of active pairs as a pandas.DataFrame with 3 columns: symbol, price, 24h volume
        '''
        active = []
        symbols = binance_endpoints.get_symbols_BTC()
        
        for symbol in symbols:
            #Thread(target=get_active, args=(symbol,)).start()
            self.get_active(symbol, active)
        #    tmp = client.get_ticker(symbol=symbol)
        #    price = float(tmp['lastPrice'])
        #    volume = float(tmp['quoteVolume'])
        #    if (0.00000200 < price < 0.001) & (volume > 150) :
        #        active.append([symbol, price, volume])
        active = pd.DataFrame(active, columns=['symbol', 'Price', "Volume"])
        print("Active list formed")
        print(active)
        return active
    
    
    def get_promising(self, active_list): 
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
            
            candles_df = binance_endpoints.GetKlines(symbol, interval='15m')
            #Create a dataframe with time and open, high, low, close (ohlc) values only
            ohlc = candles_df.loc[:, ['timestamp', 'open', 'high', 'low', 'close']]
            
            #Get 24h volume:
            quote_av = candles_df.loc[-96:,'quote_av'].astype(float).sum()
            
            coin = symbol[:-3]                  
            # Get technical indicators:    
            # Bollinger Bands
            lower, _, _, width = indicators.get_BBands(ohlc['close'])
            #ranging =  (upper - lower)/lower
            ranging= width/lower
    #        slow_d = ta.volatility.bollinger_mavg(fast_k, n=3)
            mean_range = np.mean(ranging[-96:]) # 96 15 minute candles give 24 hours
            max_range = np.max(ranging[-96:])
            
            #The condition for a promising coin:
            if  (max_range > self.min_ranging ) :        
                print("Promising: %s, mean range: %.1f, max: %.1f, current: %.1f" % (coin, 100*mean_range,100*max_range, 100*ranging.iloc[-1])  )
                symbols_tmp.append(symbol)
                vol_tmp.append(quote_av)
                ranging_curr.append(100*ranging.iloc[-1])
                #promising[symbol] = quote_av
        promising = {'symbol':symbols_tmp, 'volume':vol_tmp, 'ranging':ranging_curr}
        print("Promising list formed")
        promising = df(promising)#, columns=['symbol', 'volume'])
        #promising.sort_values(by='volume', inplace=True, ascending=False)
        promising.sort_values(by='ranging', inplace=True, ascending=False)
        print(promising)
        return promising
    
    def search_signals(self, promising):
        '''Put here all the conditions...'''
        '''Watch the promising coins:
        Check if the price is hitting lower Bollinger bands, stochastic RSI k>d and both going up
        If coin fills conditions of BBands and stoch RSI, buy signal is triggered, coin is stored in the dictionary of triggered coins
        If after 5 minutes the coin is still satisfying the conditions, buy signal is triggered again
        '''
        #global triggered, trading_coins #trade_modeling 
        
        #for symbol in promising:
        for symbol in promising.loc[:,'symbol']:
            
            try:
                candles_df = binance_endpoints.GetKlines(symbol)
            except Exception as e:
                print(f"Warning: didn't get klines for {symbol}!", e)
                continue
            
            Open, high = candles_df['open'], candles_df['high']
            low, close = candles_df['low'], candles_df['close']
            volume = candles_df['quote_av']
            
            coin = symbol[:-3]
            
            # Check if coin was bought in previous step of the loop
            if coin in list(self.trading_coins):
                if 'orderId' in self.trading_coins[coin]['order'].keys():
                    binance_endpoints.check_buy_order(self.trading_coins[coin]['order'], self.trading_coins, coin)
                else:
                    binance_endpoints.check_oco_order(self.trading_coins[coin]['order'], self.trading_coins, coin)
            
            # Check if coin signalled   in previous step of the loop                              
            if coin in self.triggered.keys():
                #status = check_profit(triggered[coin]['coin'], close.iloc[-1], triggered[coin]['start_signal'], triggered[coin]['buy_price'], triggered[coin]['buy_time'] )
                status = check_profit_new(self.triggered[coin], close.iloc[-1])
                #status = check_profit_new(triggered[coin], candles_df.loc[:,'timestamp':'close'])
                if status == 0:
                    continue
                else: 
                    del self.triggered[coin]        
                
            # Get technical indicators:    
            # Bollinger Bands
            lower, middle, upper, width = indicators.get_BBands(close) 
            ranging= width/lower *100 # in %
            # Check for Bollinger band breach (lower or middle):
            lower_BB_signal = indicators.check_lower_BB(close, low, lower)
            middle_BB_signal = indicators.check_middle_BB(Open, close, low, middle)
      
           # Check if range is > 4% and price is hitting lower Bollinger Bands:
            if ranging.iloc[-1] > self.min_ranging :
                # Note. Use iloc() to get access to position (same as indexing in numpy arrays). Reason that there is no negative indexing in pandas.Series        
                if lower_BB_signal or middle_BB_signal: 
                    # If Bollingers are hit: compute stochastic RSI                                
                    #StochasticRSI:
                    fastk, slowd = indicators.get_StochRSI(close)
                    #Stochastic RSI signal fask > slowd, both going up and 10 < fastk < 30
                    stochRSI_signal = indicators.check_stochRSI_signal(fastk, slowd, kmin=10, kmax=30)
                    #Condition that distance from buy price to mid(up) bollinder band is greater than 1.5% 
                    if stochRSI_signal:    
                        buy_price = float(binance_endpoints.get_buy_price(symbol,self.BUY_METHOD))
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
                        # Check if the coin was already signalling less than 5 minutes ago:           
                        #counter = 0
                        if coin in self.triggered.keys(): 
                            # Get current time to measure time intervals between buy signals of a coin
                            time_now = time.time()    
    #                        # If the coin was signalling more than 5 minutes (300 sec) ago, remove it from the list
                            if time_now - self.triggered[coin]['start_signal'] > 300: 
                                buy_price = self.triggered[coin]['buy_price']
                                self.triggered[coin]['counter'] += 1
                                
                                save_signal(current_time, coin, buy_price, self.triggered[coin]['counter'])
                                #del triggered[coin]
                            else:
                                continue
                               
                        else:
                            #buy_price = close.iloc[-1]
                            buy_price = float(binance_endpoints.get_buy_price(symbol,self.BUY_METHOD))
                            min_price = buy_price
                            print(current_time, "BUY signal! %s at %.8f Stoch RSI:" % (coin, buy_price), fastk.iloc[-1], slowd.iloc[-1])
                            start_signal = time.time()
                            #save signaling coins to file:
                            counter = 0
                            # Check if there was a candle pattern 1 or 2 canclesticks before:
                            pattern = indicators.check_candle_patterns(Open, high, low, close)
                            
                            save_signal(current_time, coin, buy_price, counter, pattern)
                            # Create a new item in the dictionary of signalling coins, where coin is the key and time at the 1st signal is the item
                            status = 0
                            if lower_BB_signal:
                                origin = 'lower'
                            else: origin = 'upper'
                            vol_1hr = volume[-4:].sum()
                            #triggered[coin] = [start_signal, buy_price, current_time, status]
                            #Create a dictionary with all coins that triggered a buy signal:
                            self.triggered[coin] = {'coin':coin, 'start_signal':start_signal, 'buy_price':buy_price, 'buy_time':current_time, 'status':status, 'counter':counter, \
                                     'vol_1hr':vol_1hr, 'pattern':pattern, 'origin':origin, 'ranging':ranging.iloc[-1], 'fastk15': fastk.iloc[-1], 'min_price':min_price }
                            n_trades = len(self.trading_coins)
                            if (n_trades < self.MAX_TRADES) and (coin not in self.trading_coins.keys() ):
                                #in_trade = len(trading_coins)
                            
                                print("Placing Buy Order")
                                try:
                                    order, am_btc = binance_endpoints.place_buy_order(symbol,self.MAX_TRADES,n_trades,self.BUY_METHOD,self.DEPOSIT_FRACTION)
                                except Exception as e:
                                    print("Exception during pacing BUY order occured", e)
                                    continue                             
                                    
                                if self.BUY_METHOD == 'MARKET':
                                    buy_price = order['price']
                                    print("Placing OCO order", symbol)
                                    try:                                    
                                        order = binance_endpoints.place_oco_order(symbol, buy_price, take_profit=0.017, stop_loss = 0.02)
                                    except Exception as e:
                                        print("Exception during placing OCO order:", symbol, buy_price)
                                        print(e)
                                else:
                                    #If we didn't use market buy then we have to check the order status                                    
                                    print("Check order status")
                                    try:
                                        order = binance_endpoints.check_order_status(order)
                                    except Exception as e:
                                        print("Didn't manage to check order status", e)
                                
                                self.trading_coins[coin] = {'coin':coin, 'start_signal':start_signal, 'buy_price':buy_price, \
                                          'buy_time':current_time, 'status':status, 'counter':counter, \
                                          'vol_1hr':vol_1hr, 'pattern':pattern, 'origin':origin,\
                                          'ranging':ranging.iloc[-1], 'fastk15': fastk.iloc[-1], 'min_price':min_price, 'order':order, 'am_btc': am_btc,'n_oco':0 }
     
        
    
    def c1m_flow(self, active_update_interval = 600, promise_update_interval = 300, BUY_METHOD='LIMIT', MAX_TRADES=4, DEPOSIT_FRACTION=0.1):
        '''Main flow of the strategy: 
        get_active -> get_promising -> search_signals -> repeat'''
        
        self.BUY_METHOD = BUY_METHOD
        self.MAX_TRADES = MAX_TRADES
        self.DEPOSIT_FRACTION = DEPOSIT_FRACTION
        
        active_list = self.refresh_active()
        promising = self.get_promising(active_list)
        
        total_watch = 0
        last_active_uptade = 0
        
        #start_loop = time.time()
        
        while True:        
            # Watch promising coins
            start_watch = time.time()   
            #args = (promising,)
            #try_func(watch, 10,3600, *args)
            self.search_signals(promising)
            end_watch = time.time()
            #Check how many coins are in trade at the moment
            are_trading = len(self.trading_coins)
            #If there are coins in trade, check their status
            if are_trading > 0: 
                #print("%d coins are in trade:" % are_trading)
                for item in list(self.trading_coins): #here better to use list(trading_coins) instead of trading_coins.keys()
                    if 'orderId' in self.trading_coins[item]['order'].keys():
                        print("BUY PENDING:", item)
                        try:
                            binance_endpoints.check_buy_order(self.trading_coins[item]['order'], self.trading_coins, item)
                        except Exception as e:
                            print("Didn't check BUY order", item)
                            print(e)
                    else:
                        print("OCO:", item)
                        try:
                            binance_endpoints.check_oco_order(self.trading_coins[item]['order'], self.trading_coins, item)
                        except Exception as e:
                            print("Didn't check OCO order", item)
                            print(e)                            
    
            # Measure total time spent for watching:
            total_watch += end_watch - start_watch
            # If watching time is more than 5 minutes, update promising list: 
            if total_watch > promise_update_interval: 
                #args = (active_list,)
                #promising = try_func(get_promising, 10, 3600, *args)
                promising = self.get_promising(active_list)
                last_active_uptade += total_watch
                total_watch = 0
            # Update active list every 10 minutes
            if last_active_uptade > active_update_interval:
                #active_list = try_func(refresh_active)
                active_list = self.refresh_active()
                #args = (active_list,)
                promising = self.get_promising(active_list)
                #promising = try_func(get_promising, 10, 3600, *args)
    
                are_trading = len(self.trading_coins)
                print("%d coins are in trade:" % are_trading)            
                if are_trading > 0:                 
                    for item in self.trading_coins.keys():
                        print(item, self.trading_coins[item]['buy_time'])
                        
                total_watch = 0
                last_active_uptade = 0
            
        